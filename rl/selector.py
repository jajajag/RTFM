# rl/selector.py
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class SelOut:
    idx: torch.Tensor    # (B,)
    logp: torch.Tensor   # (B,)
    scores: torch.Tensor # (B,N)
    probs: torch.Tensor  # (B,N)

class PiSel(nn.Module):
    """
    pi_sel(i | h_s, H_i)
    - h_s: (B,D)
    - H:   (B,N,D)
    """
    def __init__(self, d: int, hidden: int = 256):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(2*d, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_s: torch.Tensor, H: torch.Tensor, greedy: bool = False) -> SelOut:
        B, N, D = H.shape
        hs = h_s.unsqueeze(1).expand(B, N, D)
        x = torch.cat([hs, H], dim=-1)              # (B,N,2D)
        scores = self.scorer(x).squeeze(-1)         # (B,N)
        probs = F.softmax(scores, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        idx = probs.argmax(dim=-1) if greedy else dist.sample()
        logp = dist.log_prob(idx)
        return SelOut(idx=idx, logp=logp, scores=scores, probs=probs)

