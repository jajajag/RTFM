from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class FrozenMiniLM(nn.Module):
    """Frozen sentence encoder with a CPU cache for base embeddings only."""

    def __init__(self, name: str, device: str):
        super().__init__()
        self.model = SentenceTransformer(name, device=device)
        self.device = device
        self.cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def encode(self, texts: Iterable[str]) -> torch.Tensor:
        texts = [str(t) for t in texts]
        missing = [t for t in texts if t not in self.cache]
        if missing:
            embs = self.model.encode(
                missing,
                convert_to_tensor=True,
                normalize_embeddings=True,
            ).detach().cpu()
            for text, emb in zip(missing, embs):
                self.cache[text] = emb
        stacked = torch.stack([self.cache[t] for t in texts], dim=0)
        return stacked.to(self.device)


class Adapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
