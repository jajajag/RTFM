# rl/reward_model.py
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class _FiLMVec(nn.Module):
    """Vector FiLM: x -> (1+gamma)*x + beta, where (gamma,beta)=MLP(cond)."""
    def __init__(self, x_dim: int, cond_dim: int, hidden: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * x_dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.proj(cond)
        gamma, beta = torch.chunk(gb, 2, dim=-1)
        return (1.0 + gamma) * x + beta

class _DoubleFiLMVec(nn.Module):
    """txt2π-inspired *double FiLM* on vectors.

    We cross-modulate two streams:
        x' = FiLM(x | y), y' = FiLM(y | x), then fuse.
    """
    def __init__(self, x_dim: int, y_dim: int, d: int, hidden: int):
        super().__init__()
        self.x_in = nn.Linear(x_dim, d)
        self.y_in = nn.Linear(y_dim, d)
        self.film_x = _FiLMVec(d, d, hidden)
        self.film_y = _FiLMVec(d, d, hidden)
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        hx = self.x_in(x)
        hy = self.y_in(y)
        hx2 = F.relu(self.film_x(hx, hy))
        hy2 = F.relu(self.film_y(hy, hx))
        return self.out(hx2 + hy2)

class RewardModel(nn.Module):
    """Regression reward model with txt2π-style cross-modulation backbone.

    Supports three variants (set by rm_variant):

      - 'sa'   : r(s_t, a_t)
      - 'sas'  : r(s_t, a_t, s_{t+1})
      - 'sasz' : r(s_t, a_t, s_{t+1}, z_t)   (z_t = selected instruction embedding)

    Notes:
      - Output is a scalar (B,1) without sigmoid (regression).
      - This is designed for your model-free training loop (SB3) and does NOT require a transition model.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        *,
        z_dim: Optional[int] = None,
        rm_variant: str = "sa",
        d: Optional[int] = None,
    ):
        super().__init__()
        assert rm_variant in ["sa", "sas", "sasz"], f"Unknown rm_variant={rm_variant}"
        self.rm_variant = rm_variant
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.z_dim = int(z_dim) if z_dim is not None else state_dim
        d = int(d) if d is not None else int(hidden)

        # base stream: (s,a)
        self.sa = _DoubleFiLMVec(x_dim=state_dim, y_dim=action_dim, d=d, hidden=hidden)

        # optional: fuse with s'
        if rm_variant in ["sas", "sasz"]:
            self.sp1 = nn.Sequential(
                nn.Linear(state_dim, d),
                nn.ReLU(),
                nn.Linear(d, d),
            )
            self.fuse_sa_sp1 = _DoubleFiLMVec(x_dim=d, y_dim=d, d=d, hidden=hidden)

        # optional: fuse with z
        if rm_variant == "sasz":
            self.z = nn.Sequential(
                nn.Linear(self.z_dim, d),
                nn.ReLU(),
                nn.Linear(d, d),
            )
            self.fuse_with_z = _DoubleFiLMVec(x_dim=d, y_dim=d, d=d, hidden=hidden)

        self.head = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        h_s: torch.Tensor,
        a_onehot: torch.Tensor,
        h_sp1: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat = self.sa(h_s, a_onehot)  # (B,d)

        if self.rm_variant in ["sas", "sasz"]:
            if h_sp1 is None:
                raise ValueError("rm_variant requires h_sp1 (next-state embedding), got None")
            sp1 = self.sp1(h_sp1)
            feat = self.fuse_sa_sp1(feat, sp1)

        if self.rm_variant == "sasz":
            if z is None:
                raise ValueError("rm_variant='sasz' requires z (instruction embedding), got None")
            zf = self.z(z)
            feat = self.fuse_with_z(feat, zf)

        return self.head(feat)  # (B,1)
