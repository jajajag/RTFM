# rl/trainer.py
from __future__ import annotations
from typing import Dict
import time
import torch
import torch.nn.functional as F

from .buffers import RMBuffer, SelTraj
from .env_utils import get_n_actions

def onehot(action: int, n: int, device: str):
    a = torch.zeros(n, device=device, dtype=torch.float32)
    a[int(action)] = 1.0
    return a

class Trainer:
    def __init__(self, cfg, ll_env, ll_model, pi_sel, reward_model, rm_buffer: RMBuffer, device: str):
        self.cfg = cfg
        self.env = ll_env          # wrapped env
        self.ll_model = ll_model   # SB3 PPO/SAC
        self.pi_sel = pi_sel
        self.rm = reward_model
        self.rm_buffer = rm_buffer
        self.device = device

        self.n_actions = get_n_actions(ll_env.env)

        self.opt_sel = torch.optim.Adam(self.pi_sel.parameters(), lr=cfg.lr_sel)
        self.opt_rm  = torch.optim.Adam(self.rm.parameters(), lr=cfg.lr_rm)

    def collect_and_fill_buffers(self, steps: int):
        """
        Roll out with current low-level policy (on env reward),
        collect (h_s, a, r_env) for reward model,
        and collect selector logp for pi_sel update signals.
        """
        obs = self.env.reset()
        traj = SelTraj()
        t0 = time.time()

        for t in range(steps):
            #if t % 200 == 0:
            #    dt = time.time() - t0
            #    fps = (t / dt) if dt > 0 else 0.0
            #    print(f"[collect] t={t}/{steps} fps={fps:.1f}", flush=True)

            action, _ = self.ll_model.predict(obs, deterministic=False)
            obs2, r_env, done, info = self.env.step(action)

            # reward model data
            h_s = self.env.last_sel["h_s"].to(self.device)                 # (state_dim,)
            a = onehot(int(action), self.n_actions, self.device)           # (A,)
            self.rm_buffer.add(h_s, a, float(info.get("r_env", r_env)))

            # reward model prediction for selector reward bookkeeping
            with torch.no_grad():
                logit = self.rm(h_s.unsqueeze(0), a.unsqueeze(0)).squeeze(0)
                r_rm = torch.sigmoid(logit).item()

            traj.add(
                logp=self.env.last_sel["sel_logp_t"],
                r_env=float(info.get("r_env", r_env)),
                r_rm=float(r_rm),
            )

            obs = obs2
            if done:
                # update selector on episode end if traj-wise mode
                self.update_selector(traj)
                traj = SelTraj()
                obs = self.env.reset()

        # partial traj update (optional)
        if traj.steps:
            self.update_selector(traj)

    def update_reward_model(self, iters: int):
        """
        Supervised fit reward model on sparse env reward.
        """
        self.rm.train()
        for _ in range(iters):
            batch = self.rm_buffer.sample(self.cfg.rm_batch, device=self.device)
            if batch is None:
                break
            h_s = batch["h_s"]
            a   = batch["a"]
            r   = batch["r"]  # (B,1)

            logit = self.rm(h_s, a)
            pred = torch.sigmoid(logit)

            # BCE works well if rewards are sparse {0,1}; otherwise switch to MSE
            loss = F.mse_loss(pred, r.clamp(0, 1))

            self.opt_rm.zero_grad()
            loss.backward()
            self.opt_rm.step()

    def update_selector(self, traj: SelTraj):
        """
        REINFORCE-style update for pi_sel.
        sel_reward:
          - one_step_rm: sum_t logp_t * r_rm_t
          - traj_env: logp_t * sum r_env
          - traj_rm : logp_t * sum r_rm
        NOTE: We use stored logp scalars from env wrapper; for exact gradients,
              you'd store tensors from forward pass. This is a lightweight baseline.
              If you want exact gradients, we can store logp tensors in wrapper.
        """
        if not traj.steps:
            return

        mode = self.cfg.sel_reward
        logps = torch.stack([s.logp for s in traj.steps]).to(self.device)

        if mode == "traj_env":
            G = sum(s.r_env for s in traj.steps)
            R = torch.full_like(logps, float(G))
        elif mode == "traj_rm":
            G = sum(s.r_rm for s in traj.steps)
            R = torch.full_like(logps, float(G))
        else:  # one_step_rm
            R = torch.tensor([s.r_rm for s in traj.steps], 
                             device=self.device, dtype=torch.float32)
        loss = -(logps * R).mean()

        self.opt_sel.zero_grad()
        loss.backward()
        self.opt_sel.step()

    def train_low_level(self, total_timesteps: int):
        """
        Train SB3 low-level model. Reward comes from env wrapper's returned reward.
        You can later change wrapper to return env/rm/mix using cfg.ll_reward.
        """
        self.ll_model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)

