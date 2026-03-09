from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SelectorOutput:
    idx: torch.Tensor
    logp: torch.Tensor
    probs: torch.Tensor
    scores: torch.Tensor
    selected: torch.Tensor


class PiSel(nn.Module):
    def __init__(self, d: int, hidden: int = 256):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(2 * d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_s: torch.Tensor, H: torch.Tensor, mode: str = "sample") -> SelectorOutput:
        # h_s: (B, D), H: (B, N, D)
        bsz, n_instr, dim = H.shape
        hs = h_s.unsqueeze(1).expand(bsz, n_instr, dim)
        x = torch.cat([hs, H], dim=-1)
        scores = self.score(x).squeeze(-1)
        probs = F.softmax(scores, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        if mode == "hard":
            idx = probs.argmax(dim=-1)
        elif mode == "sample":
            idx = dist.sample()
        else:
            raise ValueError(f"Unknown selector mode: {mode}")
        logp = dist.log_prob(idx)
        selected = H[torch.arange(bsz, device=H.device), idx]
        return SelectorOutput(idx=idx, logp=logp, probs=probs, scores=scores, selected=selected)
