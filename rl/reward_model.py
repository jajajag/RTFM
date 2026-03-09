from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256, z_dim: Optional[int] = None, rm_variant: str = "sas"):
        super().__init__()
        assert rm_variant in {"sa", "sas", "sasz"}
        self.rm_variant = rm_variant
        self.action_dim = int(action_dim)
        self.z_dim = int(z_dim or state_dim)

        in_dim = state_dim + self.action_dim
        if rm_variant in {"sas", "sasz"}:
            in_dim += state_dim
        if rm_variant == "sasz":
            in_dim += self.z_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_s: torch.Tensor, a_onehot: torch.Tensor, h_sp1: Optional[torch.Tensor] = None, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        pieces = [h_s, a_onehot]
        if self.rm_variant in {"sas", "sasz"}:
            if h_sp1 is None:
                raise ValueError("h_sp1 is required for rm_variant in {'sas', 'sasz'}")
            pieces.append(h_sp1)
        if self.rm_variant == "sasz":
            if z is None:
                raise ValueError("z is required for rm_variant='sasz'")
            pieces.append(z)
        x = torch.cat(pieces, dim=-1)
        return self.net(x)

    def predict_reward(self, h_s: torch.Tensor, action_idx: torch.Tensor, h_sp1: Optional[torch.Tensor] = None, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        a_onehot = F.one_hot(action_idx.long(), num_classes=self.action_dim).float()
        return self.forward(h_s, a_onehot, h_sp1, z).squeeze(-1)
