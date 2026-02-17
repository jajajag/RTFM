# rl/buffers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import random
import torch

@dataclass
class RMItem:
    h_s: torch.Tensor            # (D,)
    a: torch.Tensor              # (A,)
    r: float
    h_sp1: Optional[torch.Tensor] = None  # (D,) next-state embedding (optional)
    z: Optional[torch.Tensor] = None      # (Z,) instruction embedding (optional)

class RMBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: List[RMItem] = []

    def add(
        self,
        h_s: torch.Tensor,
        a: torch.Tensor,
        r: float,
        *,
        h_sp1: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ):
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append(
            RMItem(
                h_s=h_s.detach().cpu(),
                a=a.detach().cpu(),
                r=float(r),
                h_sp1=(h_sp1.detach().cpu() if h_sp1 is not None else None),
                z=(z.detach().cpu() if z is not None else None),
            )
        )

    def sample(self, batch: int, device: str) -> Optional[dict]:
        if len(self.data) < batch:
            return None
        items = random.sample(self.data, batch)

        h = torch.stack([it.h_s for it in items]).to(device)
        a = torch.stack([it.a for it in items]).to(device)
        r = torch.tensor([it.r for it in items], dtype=torch.float32, device=device).unsqueeze(-1)

        h_sp1 = None
        if items[0].h_sp1 is not None:
            h_sp1 = torch.stack([it.h_sp1 for it in items]).to(device)

        z = None
        if items[0].z is not None:
            z = torch.stack([it.z for it in items]).to(device)

        return {"h_s": h, "a": a, "r": r, "h_sp1": h_sp1, "z": z}

@dataclass
class SelSegment:
    # REINFORCE term
    logp: torch.Tensor           # scalar tensor with graph
    R: float                     # scalar return for this segment (already R + R_aux)
    # extra info for optional aux computation in trainer
    obs_start: Optional[torch.Tensor] = None  # (obs_dim,) torch float32 on CPU
    obs_end: Optional[torch.Tensor] = None    # (obs_dim,) torch float32 on CPU
