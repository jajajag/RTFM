from __future__ import annotations

from typing import Any, Dict, List, Optional

import gym
import numpy as np
import torch

from gym import spaces
from rl.buffers import HLSegment
from rl.env_utils import clone_obs, get_instruction_paragraph, normalize_reset, normalize_step
from rl.instruction_split import split_with_lm, split_with_parser


class HRLWrapper(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, env, cfg, minilm, instr_adapter, state_encoder, selector, reward_model, action_dim: int):
        super().__init__()
        self.env = env
        self.cfg = cfg
        self.minilm = minilm
        self.instr_adapter = instr_adapter
        self.state_encoder = state_encoder
        self.selector = selector
        self.reward_model = reward_model
        self.action_dim = int(action_dim)
        self.device = cfg.device

        #self.action_space = env.action_space
        self.action_space = spaces.Discrete(action_dim)
        self.observation_space = None

        self.current_raw_obs: Optional[Dict[str, Any]] = None
        self.current_wrapped_obs: Optional[Dict[str, Any]] = None
        self.current_z: Optional[torch.Tensor] = None
        self.current_logp: Optional[torch.Tensor] = None
        self.current_instr_idx: Optional[int] = None
        self.current_instr_embs: Optional[torch.Tensor] = None
        self.step_in_seg = 0
        self.total_steps = 0
        self.episode_id = 0
        self.seg_idx = 0
        self.seg_start_obs: Optional[Dict[str, Any]] = None
        self.seg_env_rewards: List[float] = []
        self.seg_rm_rewards: List[float] = []
        self.seg_state_embs: List[torch.Tensor] = []
        self.pending_bl: List[Dict[str, Any]] = []
        self.pending_hl: List[HLSegment] = []

    def _to_device_obs(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                t = v.to(self.device)
            else:
                t = torch.as_tensor(v, device=self.device)
            if t.dtype == torch.float64:
                t = t.float()
            out[k] = t
        return out

    def _split_instructions(self, obs: Dict[str, Any]) -> List[str]:
        paragraph = get_instruction_paragraph(self.env, obs)
        if self.cfg.split_mode == "lm":
            items = split_with_lm(paragraph)
        else:
            items = split_with_parser(paragraph)
        items = [s.strip() for s in items if s and s.strip()]
        if not items:
            items = ["(no instruction)"]
        return items[: self.cfg.max_instructions]

    @torch.no_grad()
    def _instruction_embeddings(self, obs: Dict[str, Any]) -> torch.Tensor:
        tips = self._split_instructions(obs)
        base = self.minilm.encode(tips)
        return self.instr_adapter(base)

    @torch.no_grad()
    def _state_embedding(self, obs: Dict[str, Any]) -> torch.Tensor:
        obs_t = self._to_device_obs(obs)
        return self.state_encoder(obs_t).squeeze(0)

    def _selector_step(self, obs: Dict[str, Any]) -> None:
        obs_t = self._to_device_obs(obs)
        h_s = self.state_encoder(obs_t)
        self.current_instr_embs = self._instruction_embeddings(obs)
        H = self.current_instr_embs.unsqueeze(0)
        out = self.selector(h_s, H, mode=self.cfg.selector_mode)
        self.current_instr_idx = int(out.idx.item())
        self.current_logp = out.logp.squeeze(0)
        self.current_z = out.selected.squeeze(0).detach()
        self.seg_start_obs = clone_obs(obs)
        self.seg_env_rewards = []
        self.seg_rm_rewards = []
        self.seg_state_embs = [h_s.squeeze(0).detach()]

    def _wrap_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        wrapped = clone_obs(obs)
        wrapped["z"] = self.current_z.detach().cpu().numpy().astype(np.float32)
        return wrapped

    def _rm_predict(self, obs: Dict[str, Any], action: int, next_obs: Dict[str, Any]) -> float:
        with torch.no_grad():
            h_s = self._state_embedding(obs).unsqueeze(0)
            h_sp1 = self._state_embedding(next_obs).unsqueeze(0)
            a = torch.tensor([action], device=self.device)
            z = self.current_z.unsqueeze(0).to(self.device) if self.current_z is not None else None
            pred = self.reward_model.predict_reward(h_s, a, h_sp1=h_sp1 if self.cfg.rm_variant in {"sas", "sasz"} else None, z=z if self.cfg.rm_variant == "sasz" else None)
            return float(pred.item())

    def _ll_reward(self, r_env: float, r_rm: float) -> float:
        if self.cfg.ll_reward == "env":
            return float(r_env)
        if self.cfg.ll_reward == "rm":
            return float(r_rm)
        return float(r_env + self.cfg.ll_lambda * r_rm)

    def _aux_reward(self, obs_end: Dict[str, Any]) -> float:
        if self.cfg.hl_aux_type == "none":
            return 0.0
        if self.cfg.hl_aux_type == "cos":
            if self.current_z is None or not self.seg_state_embs:
                return 0.0
            mean_h = torch.stack(self.seg_state_embs, dim=0).mean(dim=0)
            z = self.current_z.to(mean_h.device)
            return float(torch.nn.functional.cosine_similarity(mean_h.unsqueeze(0), z.unsqueeze(0), dim=-1).item())
        if self.cfg.hl_aux_type == "v_diff":
            return 0.0
        raise ValueError(f"Unknown aux type: {self.cfg.hl_aux_type}")

    def _finalize_segment(self, obs_end: Dict[str, Any], done: bool) -> None:
        if self.seg_start_obs is None or self.current_logp is None or self.current_z is None or self.current_instr_idx is None:
            return
        base_return = sum(self.seg_rm_rewards) if self.cfg.hl_return_source == "rm" else sum(self.seg_env_rewards)
        aux_reward = self._aux_reward(obs_end)
        seg = HLSegment(
            episode_id=self.episode_id,
            seg_idx=self.seg_idx,
            obs_start=clone_obs(self.seg_start_obs),
            obs_end=clone_obs(obs_end),
            action_idx=int(self.current_instr_idx),
            logp=self.current_logp,
            z=self.current_z.detach().cpu(),
            base_return=float(base_return),
            aux_reward=float(aux_reward),
            done=bool(done),
        )
        self.pending_hl.append(seg)
        self.seg_idx += 1

    def reset(self, **kwargs):
        obs, info = normalize_reset(self.env.reset(**kwargs))
        self.episode_id += 1
        self.seg_idx = 0
        self.step_in_seg = 0
        self.total_steps = 0
        self.current_raw_obs = clone_obs(obs)
        self._selector_step(self.current_raw_obs)
        wrapped = self._wrap_obs(self.current_raw_obs)
        self.current_wrapped_obs = clone_obs(wrapped)
        return wrapped

    def step(self, action):
        prev_raw = clone_obs(self.current_raw_obs)
        out = self.env.step(int(action))
        next_obs, r_env, done, info = normalize_step(out)
        r_rm = self._rm_predict(prev_raw, int(action), next_obs)
        reward = self._ll_reward(r_env, r_rm)

        self.seg_env_rewards.append(float(r_env))
        self.seg_rm_rewards.append(float(r_rm))
        self.seg_state_embs.append(self._state_embedding(next_obs).detach())

        self.pending_bl.append(
            {
                "obs": clone_obs(prev_raw),
                "action": int(action),
                "reward": float(r_env),
                "next_obs": clone_obs(next_obs),
                "done": bool(done),
                "z": self.current_z.detach().cpu().clone(),
            }
        )

        self.step_in_seg += 1
        self.total_steps += 1
        segment_done = done or (self.step_in_seg >= int(self.cfg.hl_T))
        if segment_done:
            self._finalize_segment(next_obs, done)
            self.step_in_seg = 0
            if not done:
                self._selector_step(next_obs)

        self.current_raw_obs = clone_obs(next_obs)
        wrapped = self._wrap_obs(self.current_raw_obs)
        self.current_wrapped_obs = clone_obs(wrapped)
        info = dict(info)
        info.update(
            {
                "r_env": float(r_env),
                "r_rm": float(r_rm),
                "segment_done": bool(segment_done),
            }
        )
        return wrapped, float(reward), bool(done), info
