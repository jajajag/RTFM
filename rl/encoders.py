from typing import List
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# —— 简单词表 / 分词（仅给 BiLSTM 用）——————————————
class Vocab:
    def __init__(self):
        self.idx = {"<pad>": 0, "<unk>": 1}
        self.inv = ["<pad>", "<unk>"]

    def add_sentence(self, s: str):
        for w in s.strip().split():
            if w not in self.idx:
                self.idx[w] = len(self.inv)
                self.inv.append(w)

    def encode(self, s: str, max_len: int = 64):
        ids = [self.idx.get(w, 1) for w in s.strip().split()][:max_len]
        return ids + [0] * (max_len - len(ids))

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size=50000, dim=512, pad_id=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(dim, dim // 2, batch_first=True, bidirectional=True)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: (B, T)
        x = self.embed(ids)
        h, _ = self.lstm(x)
        return self.out_proj(h.mean(dim=1))  # (B, dim)

class MiniLMEncoder(nn.Module):
    def __init__(self, name="nreimers/MiniLM-L6-H384-uncased", out_dim=512):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        self.proj = nn.Linear(self.model.config.hidden_size, out_dim)

    def forward(self, texts: List[str]) -> torch.Tensor:
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True)
        out = self.model(**enc).last_hidden_state.mean(dim=1)
        return self.proj(out)  # (B, out_dim)
