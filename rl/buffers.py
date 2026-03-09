from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import random

import torch


RawObs = Dict[str, Any]


def _clone_obs_to_cpu(obs: RawObs) -> RawObs:
    out: RawObs = {}
    for k, v in obs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = v
    return out


@dataclass
class BLTransition:
    obs: RawObs
    action: int
    reward: float
    next_obs: RawObs
    done: bool
    z: torch.Tensor


class BLBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data: List[BLTransition] = []

    def add(self, obs: RawObs, action: int, reward: float, next_obs: RawObs, done: bool, z: torch.Tensor) -> None:
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append(
            BLTransition(
                obs=_clone_obs_to_cpu(obs),
                action=int(action),
                reward=float(reward),
                next_obs=_clone_obs_to_cpu(next_obs),
                done=bool(done),
                z=z.detach().cpu().clone(),
            )
        )

    def sample(self, batch_size: int) -> Optional[List[BLTransition]]:
        if len(self.data) < batch_size:
            return None
        return random.sample(self.data, batch_size)

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class HLSegment:
    episode_id: int
    seg_idx: int
    obs_start: RawObs
    obs_end: RawObs
    action_idx: int
    logp: torch.Tensor
    z: torch.Tensor
    base_return: float
    aux_reward: float
    done: bool


class HLBuffer:
    def __init__(self):
        self.data: List[HLSegment] = []

    def add(self, seg: HLSegment) -> None:
        self.data.append(seg)

    def pop_all(self) -> List[HLSegment]:
        data = self.data
        self.data = []
        return data
