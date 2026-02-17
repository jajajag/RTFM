# rl/trainer.py
from __future__ import annotations
from typing import List
import torch
import torch.nn.functional as F

from stable_baselines3.common.callbacks import BaseCallback

from .buffers import RMBuffer, SelSegment

class RTFMCallback(BaseCallback):
    """SB3 callback to:
      - collect RMBuffer data every step
      - periodically update reward model / selector
    """
    def __init__(self, trainer: "Trainer", verbose: int = 0):
        super().__init__(verbose=verbose)
        self.trainer = trainer
        self._step = 0

    def _on_step(self) -> bool:
        self._step += 1
        env0 = self.training_env.envs[0]

        info = self.locals.get("infos", [{}])[0]
        action = self.locals.get("actions", [0])[0]

        # --- RM buffer ---
        try:
            var = getattr(self.trainer.cfg, "rm_variant", "sa")

            # Wrapper stores these per transition (set in LLWrapper.step)
            h_s = env0.last_sel.get("h_s_prev", env0.last_sel.get("h_s", None))
            h_sp1 = env0.last_sel.get("h_s_next", None)
            z = env0.last_sel.get("z_k", None)

            if h_s is None:
                raise RuntimeError("LLWrapper did not provide h_s / h_s_prev in last_sel")

            h_s = h_s.to(self.trainer.device)
            if h_sp1 is not None:
                h_sp1 = h_sp1.to(self.trainer.device)
            if z is not None:
                z = z.to(self.trainer.device)

            a = self.trainer.onehot(int(action), env0.n_actions, self.trainer.device)
            r_env = float(info.get("r_env", 0.0))

            if var == "sa":
                self.trainer.rm_buffer.add(h_s, a, r_env)
            elif var == "sas":
                if h_sp1 is None:
                    raise RuntimeError("rm_variant='sas' requires h_s_next but it's missing")
                self.trainer.rm_buffer.add(h_s, a, r_env, h_sp1=h_sp1)
            elif var == "sasz":
                if h_sp1 is None:
                    raise RuntimeError("rm_variant='sasz' requires h_s_next but it's missing")
                if z is None:
                    # robustness: still train, but z is zero
                    z = torch.zeros(self.trainer.cfg.z_dim, device=self.trainer.device)
                self.trainer.rm_buffer.add(h_s, a, r_env, h_sp1=h_sp1, z=z)
            else:
                raise RuntimeError(f"Unknown rm_variant={var}")
        except Exception:
            pass

        # selector update
        if self._step % int(self.trainer.cfg.hl_update_every_steps) == 0:
            self.trainer.update_selector_from_env(env0)

        # reward-model update
        if self._step % int(self.trainer.cfg.ll_update_every_steps) == 0:
            self.trainer.update_reward_model(self.trainer.cfg.rm_updates_per_call)

        return True

class Trainer:
    def __init__(self, cfg, ll_env, ll_model, pi_sel, reward_model, rm_buffer: RMBuffer, device: str,
                 state_adapter=None, instr_adapter=None):
        self.cfg = cfg
        self.env = ll_env
        self.ll_model = ll_model
        self.pi_sel = pi_sel
        self.rm = reward_model
        self.rm_buffer = rm_buffer
        self.device = device
        self.state_adapter = state_adapter
        self.instr_adapter = instr_adapter

        # build optimizers with the requested gradient routing
        sel_params = list(self.pi_sel.parameters())
        if self.instr_adapter is not None:
            sel_params += list(self.instr_adapter.parameters())
        if getattr(self.cfg, "state_encoder_update", "both") == "both" and (self.state_adapter is not None):
            sel_params += list(self.state_adapter.parameters())
        self.opt_sel = torch.optim.Adam(sel_params, lr=cfg.lr_sel)

        rm_params = list(self.rm.parameters())
        if getattr(self.cfg, "state_encoder_update", "both") in ["low", "both"] and (self.state_adapter is not None):
            rm_params += list(self.state_adapter.parameters())
        self.opt_rm  = torch.optim.Adam(rm_params, lr=cfg.lr_rm)

    @staticmethod
    def onehot(action: int, n: int, device: str):
        a = torch.zeros(n, device=device, dtype=torch.float32)
        a[int(action)] = 1.0
        return a

    def update_reward_model(self, iters: int):
        """Regression fit reward model on env reward."""
        self.rm.train()
        if self.state_adapter is not None:
            self.state_adapter.train()

        var = getattr(self.cfg, "rm_variant", "sa")

        for _ in range(int(iters)):
            batch = self.rm_buffer.sample(self.cfg.rm_batch, device=self.device)
            if batch is None:
                break

            h_s = batch["h_s"]
            a   = batch["a"]
            r   = batch["r"]  # (B,1)

            if var == "sa":
                pred = self.rm(h_s, a)
            elif var == "sas":
                pred = self.rm(h_s, a, batch["h_sp1"])
            elif var == "sasz":
                z = batch["z"]
                if z is None:
                    z = torch.zeros((h_s.shape[0], self.cfg.z_dim), device=self.device, dtype=h_s.dtype)
                pred = self.rm(h_s, a, batch["h_sp1"], z)
            else:
                raise RuntimeError(f"Unknown rm_variant={var}")

            loss = F.mse_loss(pred, r)

            self.opt_rm.zero_grad(set_to_none=True)
            loss.backward()
            self.opt_rm.step()

        self.rm.eval()
        if self.state_adapter is not None:
            self.state_adapter.eval()

    def _value(self, obs_cpu: torch.Tensor) -> float:
        """Try to query low-level value function V(s) from SB3 (PPO)."""
        try:
            policy = getattr(self.ll_model, "policy", None)
            if policy is None:
                return 0.0
            obs = obs_cpu.to(self.device).unsqueeze(0)
            with torch.no_grad():
                v = policy.predict_values(obs).squeeze().cpu().item()
            return float(v)
        except Exception:
            return 0.0

    def update_selector_from_env(self, env0):
        """Consume finished T-step segments from env wrapper and run REINFORCE update."""
        segs: List[SelSegment] = list(getattr(env0, "finished_segments", []))
        if not segs:
            return
        env0.finished_segments = []

        aux_type = getattr(self.cfg, "hl_aux_type", "none")

        Rs = []
        logps = []
        for seg in segs:
            R = float(seg.R)
            if aux_type == "v_diff" and (seg.obs_start is not None) and (seg.obs_end is not None):
                R += float(self.cfg.hl_aux_scale) * (self._value(seg.obs_end) - self._value(seg.obs_start))
            Rs.append(R)
            logps.append(seg.logp)

        logps_t = torch.stack(logps).to(self.device)
        R_t = torch.tensor(Rs, device=self.device, dtype=torch.float32)

        loss = -(logps_t * R_t).mean()
        self.opt_sel.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_sel.step()

    def train(self, total_timesteps: int):
        cb = RTFMCallback(self, verbose=0)
        self.ll_model.learn(total_timesteps=int(total_timesteps), reset_num_timesteps=False, callback=cb)
