# rl/buffers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import random
import torch

@dataclass
class RMItem:
    h_s: torch.Tensor      # (D,)
    a: torch.Tensor        # (A,)
    r: float

class RMBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: List[RMItem] = []

    def add(self, h_s: torch.Tensor, a: torch.Tensor, r: float):
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append(RMItem(h_s.detach().cpu(), a.detach().cpu(), float(r)))

    def sample(self, batch: int, device: str) -> Optional[dict]:
        if len(self.data) < batch:
            return None
        items = random.sample(self.data, batch)
        h = torch.stack([it.h_s for it in items]).to(device)
        a = torch.stack([it.a for it in items]).to(device)
        r = torch.tensor([it.r for it in items], dtype=torch.float32, device=device).unsqueeze(-1)
        return {"h_s": h, "a": a, "r": r}

@dataclass
class SelStep:
    logp: torch.Tensor
    r_env: float
    r_rm: float

class SelTraj:
    def __init__(self):
        self.steps: List[SelStep] = []
    def add(self, logp: float, r_env: float, r_rm: float):
        self.steps.append(SelStep(logp=logp, r_env=r_env, r_rm=r_rm))

