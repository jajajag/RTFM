# rl/retrieval.py
from __future__ import annotations
import torch
import torch.nn.functional as F

def z_single(H: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # H: (B,N,D), idx: (B,) -> (B,D)
    return H[torch.arange(H.shape[0], device=H.device), idx]

def z_topk(H: torch.Tensor, scores: torch.Tensor, k: int = 4, pool: str = "mean") -> torch.Tensor:
    # scores: (B,N)
    k = min(k, H.shape[1])
    top = torch.topk(scores, k=k, dim=-1)
    idx = top.indices  # (B,k)
    Hk = H.gather(1, idx.unsqueeze(-1).expand(-1, -1, H.shape[-1]))  # (B,k,D)
    if pool == "mean":
        return Hk.mean(dim=1)
    w = F.softmax(top.values, dim=-1).unsqueeze(-1)  # (B,k,1)
    return (Hk * w).sum(dim=1)

