# rl/replay_buffer.py
from __future__ import annotations
import numpy as np
import torch


class RewardReplayBuffer:
    """
    只给 reward model / distance model 用的 buffer。
    存 h_s^t, h_s^{t+1}, z_goal^t, h_g_bar, action, env_r, (T - t), success 标记。
    HER: 对于成功的 trajectory，所有步的目标奖励都加一个 her_coef。
    """
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        goal_dim: int,
        her_coef: float = 1.0,
    ):
        self.capacity = int(capacity)
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.her_coef = her_coef

        self.ptr = 0
        self.full = False

        self.h_s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.h_s_next = np.zeros((capacity, state_dim), dtype=np.float32)
        self.z_goal = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.h_g_bar = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.action = np.zeros((capacity,), dtype=np.int64)
        self.env_r = np.zeros((capacity,), dtype=np.float32)
        self.remaining = np.zeros((capacity,), dtype=np.float32)  # T - t
        self.success = np.zeros((capacity,), dtype=np.bool_)

    # ---- 写入整个 trajectory ----
    def add_trajectory(self, traj_steps):
        """
        traj_steps: list[dict]，每个 dict 里至少有：
          h_s, h_s_next, z_goal, h_g_bar: torch.Tensor / np.array (D,)
          action: int
          env_r: float
        """
        if not traj_steps:
            return
        T_idx = len(traj_steps) - 1
        success = any(step["env_r"] > 0.0 for step in traj_steps)

        for t_idx, step in enumerate(traj_steps):
            idx = self.ptr

            self.h_s[idx] = self._to_numpy(step["h_s"])
            self.h_s_next[idx] = self._to_numpy(step["h_s_next"])
            self.z_goal[idx] = self._to_numpy(step["z_goal"])
            self.h_g_bar[idx] = self._to_numpy(step["h_g_bar"])
            self.action[idx] = int(step["action"])
            self.env_r[idx] = float(step["env_r"])
            self.remaining[idx] = float(T_idx - t_idx)
            self.success[idx] = bool(success)

            self.ptr = (self.ptr + 1) % self.capacity
            if self.ptr == 0:
                self.full = True

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    # ---- 采样 ----
    def size(self) -> int:
        return self.capacity if self.full else self.ptr

    def sample(self, batch_size: int, device: str):
        n = self.size()
        if n == 0:
            return None
        idx = np.random.randint(0, n, size=batch_size)

        h_s = torch.from_numpy(self.h_s[idx]).to(device)
        h_s_next = torch.from_numpy(self.h_s_next[idx]).to(device)
        z_goal = torch.from_numpy(self.z_goal[idx]).to(device)
        h_g_bar = torch.from_numpy(self.h_g_bar[idx]).to(device)
        action = torch.from_numpy(self.action[idx]).to(device)
        env_r = torch.from_numpy(self.env_r[idx]).to(device)
        remaining = torch.from_numpy(self.remaining[idx]).to(device)
        success = torch.from_numpy(self.success[idx].astype(np.float32)).to(device)

        # HER: 成功轨迹中所有步都额外得到 her_coef 的正奖励
        r_target = env_r + self.her_coef * success

        d_target = remaining  # 直接用 T - t，当作回归标签
        return {
            "h_s": h_s,
            "h_s_next": h_s_next,
            "z_goal": z_goal,
            "h_g_bar": h_g_bar,
            "action": action,
            "r_target": r_target,
            "d_target": d_target,
        }

