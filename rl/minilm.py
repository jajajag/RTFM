# rl/minilm.py
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class FrozenMiniLM(nn.Module):
    """
    Frozen sentence transformer encoder.
    """
    def __init__(self, name: str, device: str):
        super().__init__()
        self.model = SentenceTransformer(name, device=device)
        self.device = device

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        # (N, emb_dim)
        return self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

class Adapter(nn.Module):
    """
    Trainable adapter on top of frozen MiniLM embeddings.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

