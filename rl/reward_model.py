# rl/reward_model.py
from __future__ import annotations
import torch
import torch.nn as nn

class RewardModel(nn.Module):
    """
    Predict sparse reward from (state embedding, action one-hot).
    Output is a logit; use sigmoid if you want probability-like shaping.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_s: torch.Tensor, a_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_s, a_onehot], dim=-1)
        return self.net(x)  # (B,1)

