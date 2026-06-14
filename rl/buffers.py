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
    success: float | None = None


class BLBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data: List[BLTransition] = []

    def add(self, obs: RawObs, action: int, reward: float, next_obs: RawObs, done: bool, z: torch.Tensor, success: float | None = None) -> None:
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
                success=None if success is None else float(success),
            )
        )

    def sample(self, batch_size: int) -> Optional[List[BLTransition]]:
        if len(self.data) < batch_size:
            return None
        return random.sample(self.data, batch_size)

    def sample_balanced(self, batch_size: int, nonzero_fraction: float) -> Optional[List[BLTransition]]:
        if len(self.data) < batch_size:
            return None

        nonzero = [x for x in self.data if abs(float(x.reward)) > 1e-8]
        zero = [x for x in self.data if abs(float(x.reward)) <= 1e-8]
        if not nonzero or not zero or nonzero_fraction <= 0:
            return self.sample(batch_size)

        n_nonzero = min(len(nonzero), int(round(batch_size * float(nonzero_fraction))))
        n_zero = batch_size - n_nonzero
        if n_zero > len(zero):
            n_zero = len(zero)
            n_nonzero = batch_size - n_zero

        batch = random.sample(nonzero, n_nonzero) + random.sample(zero, n_zero)
        random.shuffle(batch)
        return batch

    def sample_success_balanced(self, batch_size: int, positive_fraction: float = 0.5) -> Optional[List[BLTransition]]:
        labeled = [x for x in self.data if x.success is not None]
        if len(labeled) < batch_size:
            return None

        positive = [x for x in labeled if float(x.success) > 0.5]
        negative = [x for x in labeled if float(x.success) <= 0.5]
        if not positive or not negative or positive_fraction <= 0:
            return random.sample(labeled, batch_size)

        n_positive = min(len(positive), int(round(batch_size * float(positive_fraction))))
        n_negative = batch_size - n_positive
        if n_negative > len(negative):
            n_negative = len(negative)
            n_positive = batch_size - n_negative

        batch = random.sample(positive, n_positive) + random.sample(negative, n_negative)
        random.shuffle(batch)
        return batch

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class HLSegment:
    episode_id: int
    seg_idx: int
    obs_start: RawObs
    obs_end: RawObs
    instructions: List[str]
    action_idx: int
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
