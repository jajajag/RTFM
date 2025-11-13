# rl/reward_heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    """
    r_t = f(h_s^t, a_t, h_s^{t+1}, h_g_bar, z_goal^t)
    用回归拟合标量奖励。
    """
    def __init__(self, state_dim: int, goal_dim: int, n_actions: int,
                 hidden_dim: int = 256):
        super().__init__()
        self.action_emb = nn.Embedding(n_actions, state_dim)

        in_dim = state_dim + state_dim + state_dim + goal_dim + goal_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self,
                h_s_t: torch.Tensor,      # (B, D)
                a_t: torch.Tensor,        # (B,)
                h_s_tp1: torch.Tensor,    # (B, D)
                h_g_bar: torch.Tensor,    # (B, Dg)
                z_goal_t: torch.Tensor,   # (B, Dg)
                ) -> torch.Tensor:
        a_emb = self.action_emb(a_t)              # (B, D)
        x = torch.cat([h_s_t, h_s_tp1, a_emb, h_g_bar, z_goal_t], dim=-1)
        r = self.mlp(x).squeeze(-1)               # (B,)
        return r


class DistanceModel(nn.Module):
    """
    d(h_s, h_g_bar) ≈ T - t  （remaining timesteps to go）
    """
    def __init__(self, state_dim: int, goal_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self,
                h_s: torch.Tensor,        # (B, D)
                h_g_bar: torch.Tensor,    # (B, Dg)
                ) -> torch.Tensor:
        x = torch.cat([h_s, h_g_bar], dim=-1)
        # softplus 保证非负
        d = F.softplus(self.mlp(x).squeeze(-1))
        return d

