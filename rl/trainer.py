from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback

from rl.buffers import BLBuffer, HLBuffer, HLSegment


class HRLCallback(BaseCallback):
    def __init__(self, trainer, verbose: int = 0):
        super().__init__(verbose)
        self.trainer = trainer
        self.step_count = 0

    def _unwrap(self):
        env = self.training_env.envs[0]

        # Walk through wrappers until we find the custom HRL wrapper
        while env is not None:
            if hasattr(env, "pending_bl") and hasattr(env, "pending_hl"):
                return env
            if hasattr(env, "env"):
                env = env.env
            else:
                break

        raise RuntimeError(f"Could not locate HRLWrapper in env chain, got type={type(env)}")

    def _on_step(self) -> bool:
        env = self._unwrap()

        for item in env.pending_bl:
            self.trainer.bl_buffer.add(**item)
        env.pending_bl.clear()

        for seg in env.pending_hl:
            self.trainer.hl_buffer.add(seg)
        env.pending_hl.clear()

        self.step_count += 1

        if self.step_count % int(self.trainer.cfg.hl_update_every_steps) == 0:
            self.trainer.update_selector(self.trainer.ll_model)

        if self.step_count % int(self.trainer.cfg.hl_update_every_steps) == 0:
            self.trainer.update_reward_model()

        return True


class Trainer:
    def __init__(self, cfg, ll_model, selector, state_encoder, instr_adapter, reward_model, bl_buffer: BLBuffer, hl_buffer: HLBuffer, minilm):
        self.cfg = cfg
        self.ll_model = ll_model
        self.selector = selector
        self.state_encoder = state_encoder
        self.instr_adapter = instr_adapter
        self.reward_model = reward_model
        self.bl_buffer = bl_buffer
        self.hl_buffer = hl_buffer
        self.minilm = minilm
        params = list(self.selector.parameters()) + list(self.instr_adapter.parameters()) + list(self.state_encoder.parameters())
        self.opt_h = torch.optim.Adam(params, lr=cfg.hl_lr)
        self.opt_rm = torch.optim.Adam(self.reward_model.parameters(), lr=cfg.rm_lr)
        self.device = cfg.device

    def _obs_to_device(self, obs: Dict) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in obs.items():
            if k == "z":
                continue
            if isinstance(v, torch.Tensor):
                t = v.to(self.device)
            else:
                t = torch.as_tensor(v, device=self.device)
            if t.dtype == torch.float64:
                t = t.float()
            if t.dim() == len(t.shape):
                out[k] = t.unsqueeze(0)
            else:
                out[k] = t
        return out

    def _encode_state(self, obs: Dict, grad: bool = True) -> torch.Tensor:
        fn = self.state_encoder if grad else torch.no_grad()
        if grad:
            return self.state_encoder(self._obs_to_device(obs))
        with torch.no_grad():
            return self.state_encoder(self._obs_to_device(obs))

    def _value_of_obs(self, ll_model, obs: Dict) -> float:
        wrapped = {k: (v if k == "z" else np.asarray(v)) for k, v in obs.items()}
        wrapped_t = ll_model.policy.obs_to_tensor(wrapped)[0]
        with torch.no_grad():
            v = ll_model.policy.predict_values(wrapped_t)
        return float(v.squeeze().cpu().item())

    def update_reward_model(self) -> None:
        if len(self.bl_buffer) < self.cfg.rm_batch_size:
            return
        self.reward_model.train()
        for _ in range(int(self.cfg.rm_updates_per_call)):
            batch = self.bl_buffer.sample(self.cfg.rm_batch_size)
            if batch is None:
                break
            hs_list = []
            hsp1_list = []
            a_list = []
            r_list = []
            z_list = []
            for item in batch:
                hs_list.append(self._encode_state(item.obs, grad=False).squeeze(0))
                hsp1_list.append(self._encode_state(item.next_obs, grad=False).squeeze(0))
                a_list.append(item.action)
                r_list.append(item.reward)
                z_list.append(item.z.to(self.device))
            h_s = torch.stack(hs_list, dim=0)
            h_sp1 = torch.stack(hsp1_list, dim=0)
            a = torch.tensor(a_list, device=self.device)
            z = torch.stack(z_list, dim=0)
            pred = self.reward_model.predict_reward(
                h_s,
                a,
                h_sp1=h_sp1 if self.cfg.rm_variant in {"sas", "sasz"} else None,
                z=z if self.cfg.rm_variant == "sasz" else None,
            )
            target = torch.tensor(r_list, device=self.device, dtype=torch.float32)
            loss = F.mse_loss(pred, target)
            self.opt_rm.zero_grad(set_to_none=True)
            loss.backward()
            self.opt_rm.step()
        self.reward_model.eval()

    def update_selector(self, ll_model) -> None:
        self.selector.train()
        self.state_encoder.train()
        self.instr_adapter.train()

        segments = self.hl_buffer.pop_all()
        if not segments:
            return

        by_episode: Dict[int, List[HLSegment]] = defaultdict(list)
        for seg in segments:
            by_episode[seg.episode_id].append(seg)

        losses = []
        for _, ep_segments in by_episode.items():
            ep_segments.sort(key=lambda x: x.seg_idx)
            total_terms = []
            running = 0.0
            returns = [0.0] * len(ep_segments)
            seg_rewards = []
            for seg in ep_segments:
                aux = float(seg.aux_reward)
                if self.cfg.hl_aux_type == "v_diff":
                    start_obs = dict(seg.obs_start)
                    end_obs = dict(seg.obs_end)
                    start_obs["z"] = seg.z.numpy().astype(np.float32)
                    end_obs["z"] = seg.z.numpy().astype(np.float32)
                    aux = self._value_of_obs(ll_model, end_obs) - self._value_of_obs(ll_model, start_obs)
                seg_rewards.append(float(seg.base_return) + float(self.cfg.hl_aux_lambda) * aux)
            for i in reversed(range(len(ep_segments))):
                running = seg_rewards[i] + float(self.cfg.hl_gamma) * running
                returns[i] = running

            for seg, ret in zip(ep_segments, returns):
                h_s = self._encode_state(seg.obs_start, grad=True)
                z = seg.z.to(self.device).unsqueeze(0)
                tips = self.minilm.encode([seg.obs_start.get("_tip_text", "")]) if False else None
                # Recompute instruction embeddings from the current episode text.
                # The wrapper stores only the selected index, so we rebuild the candidate set from obs_start.
                from rl.env_utils import get_instruction_paragraph
                from rl.instruction_split import split_with_lm, split_with_parser
                paragraph = get_instruction_paragraph(None, seg.obs_start)
                if self.cfg.split_mode == "lm":
                    instructions = split_with_lm(paragraph)
                else:
                    instructions = split_with_parser(paragraph)
                instructions = [s for s in instructions if s.strip()][: self.cfg.max_instructions] or ["(no instruction)"]
                H = self.instr_adapter(self.minilm.encode(instructions)).unsqueeze(0)
                out = self.selector(h_s, H, mode=self.cfg.selector_mode)
                chosen_logp = out.logp.squeeze(0)
                total_terms.append(-chosen_logp * float(ret))
            if total_terms:
                losses.append(torch.stack(total_terms).mean())

        if not losses:
            return
        loss = self.cfg.xi_H * torch.stack(losses).mean()
        self.opt_h.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_h.step()

    def train(self, total_timesteps: int):
        cb = HRLCallback(self, verbose=0)
        self.ll_model.learn(total_timesteps=int(total_timesteps), reset_num_timesteps=False, callback=cb)
