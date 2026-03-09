from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces


class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    return GradScale.apply(x, float(scale))


class TokenEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, padding_idx: int, hidden_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.scorer = nn.Linear(2 * hidden_dim, 1)
        self.hidden_dim = hidden_dim

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # tokens: (B, L), lengths: (B,)
        lengths = lengths.clamp(min=1).long()
        emb = self.emb(tokens.long())
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        max_len = out.size(1)
        mask = (torch.arange(max_len, device=out.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
        scores = self.scorer(out).squeeze(-1) - (1.0 - mask) * 1e9
        attn = torch.softmax(scores, dim=1)
        return (attn.unsqueeze(-1) * out).sum(dim=1)


class BaseStateEncoder(nn.Module):
    def __init__(self, vocab_size: int, out_dim: int, token_emb_dim: int, text_rnn_dim: int, hidden_dim: int, padding_idx: int = 0):
        super().__init__()
        self.out_dim = int(out_dim)
        self.text_enc = TokenEncoder(vocab_size, token_emb_dim, padding_idx, text_rnn_dim)
        self.hidden_dim = hidden_dim
        self.text_dim = 2 * text_rnn_dim

    @staticmethod
    def make_observation_space(sample_obs: Dict[str, torch.Tensor], z_dim: int) -> spaces.Dict:
        obs_spaces: Dict[str, spaces.Space] = {}
        for k, v in sample_obs.items():
            arr = v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            if np.issubdtype(arr.dtype, np.floating):
                obs_spaces[k] = spaces.Box(low=-np.inf, high=np.inf, shape=arr.shape, dtype=np.float32)
            else:
                obs_spaces[k] = spaces.Box(low=0, high=np.iinfo(np.int64).max, shape=arr.shape, dtype=np.int64)
        obs_spaces["z"] = spaces.Box(low=-np.inf, high=np.inf, shape=(int(z_dim),), dtype=np.float32)
        return spaces.Dict(obs_spaces)

    def _to_device(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        dev = next(self.parameters()).device
        out: Dict[str, torch.Tensor] = {}
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                t = v.to(dev)
            else:
                t = torch.as_tensor(v, device=dev)
            if t.dtype == torch.float64:
                t = t.float()
            out[k] = t
        return out

    def _batch(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs = self._to_device(obs)
        out: Dict[str, torch.Tensor] = {}
        for k, v in obs.items():
            out[k] = v.unsqueeze(0) if v.dim() == len(v.shape) else v
        if obs["name"].dim() == 5:
            return {k: (v.unsqueeze(0) if v.dim() >= 1 and k != "z" else v) for k, v in obs.items()}
        return obs

    def _prepare(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs = self._to_device(obs)
        out: Dict[str, torch.Tensor] = {}

        for k, v in obs.items():
            t = v

            if k == "name":
                # [H, W, P, L] -> [B, H, W, P, L]
                if t.dim() == 4:
                    t = t.unsqueeze(0)
                elif t.dim() != 5:
                    raise RuntimeError(f"Unexpected obs['name'] shape in _prepare: {tuple(t.shape)}")

            elif k == "name_len":
                # [H, W, P] -> [B, H, W, P]
                if t.dim() == 3:
                    t = t.unsqueeze(0)
                elif t.dim() != 4:
                    raise RuntimeError(f"Unexpected obs['name_len'] shape in _prepare: {tuple(t.shape)}")

            elif k in {"inv", "wiki", "task"}:
                # [L] -> [B, L]
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                elif t.dim() != 2:
                    raise RuntimeError(f"Unexpected obs['{k}'] shape in _prepare: {tuple(t.shape)}")

            elif k in {"inv_len", "wiki_len", "task_len"}:
                # Accept:
                # []      -> [1]
                # [B]     -> [B]
                # [B, 1]  -> [B]
                if t.dim() == 0:
                    t = t.unsqueeze(0)
                elif t.dim() == 2 and t.shape[-1] == 1:
                    t = t.view(-1)
                elif t.dim() != 1:
                    raise RuntimeError(f"Unexpected obs['{k}'] shape in _prepare: {tuple(t.shape)}")

            elif k == "rel_pos":
                # [H, W, D] -> [B, H, W, D]
                if t.dim() == 3:
                    t = t.unsqueeze(0)
                elif t.dim() != 4:
                    raise RuntimeError(f"Unexpected obs['rel_pos'] shape in _prepare: {tuple(t.shape)}")

            elif k in {"valid", "progress"}:
                # [D] -> [B, D]
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                elif t.dim() != 2:
                    raise RuntimeError(f"Unexpected obs['{k}'] shape in _prepare: {tuple(t.shape)}")

            out[k] = t

        return out

    def encode_text_field(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # tokens: [T] or [B, T]
        # lengths: scalar / [1] / [B]
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        elif tokens.dim() != 2:
            raise RuntimeError(f"Expected text tokens with 1 or 2 dims, got shape {tuple(tokens.shape)}")

        if lengths.dim() == 0:
            lengths = lengths.unsqueeze(0)
        elif lengths.dim() != 1:
            lengths = lengths.view(-1)

        return self.text_enc(tokens, lengths)

    def encode_cell(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # name: (B,H,W,P,L), name_len: (B,H,W,P)
        name = obs["name"].long()

        # Accept both:
        # unbatched: [H, W, P, L]
        # batched:   [B, H, W, P, L]
        if name.dim() == 4:
            name = name.unsqueeze(0)
        elif name.dim() != 5:
            raise RuntimeError(f"Unexpected obs['name'] shape: {tuple(name.shape)}")
        
        bsz, h, w, p, l = name.shape
        name_len = obs["name_len"].long().clamp(min=1)
        flat_tokens = name.view(bsz * h * w * p, l)
        flat_lens = name_len.view(bsz * h * w * p)
        enc = self.encode_text_field(flat_tokens, flat_lens)
        enc = enc.view(bsz, h, w, p, self.text_dim).sum(dim=3)
        return enc

    def encode_inv(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encode_text_field(obs["inv"].long(), obs["inv_len"].view(-1).long())

    def encode_wiki(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encode_text_field(obs["wiki"].long(), obs["wiki_len"].view(-1).long())

    def encode_task(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encode_text_field(obs["task"].long(), obs["task_len"].view(-1).long())


class MLPStateEncoder(BaseStateEncoder):
    def __init__(self, vocab_size: int, out_dim: int, token_emb_dim: int, text_rnn_dim: int, hidden_dim: int, padding_idx: int = 0):
        super().__init__(vocab_size, out_dim, token_emb_dim, text_rnn_dim, hidden_dim, padding_idx)
        self.mlp = nn.Sequential(
            nn.Linear(self.text_dim * 4 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh(),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs = self._prepare(obs)
        cell = self.encode_cell(obs).mean(dim=(1, 2))
        inv = self.encode_inv(obs)
        wiki = self.encode_wiki(obs)
        task = self.encode_task(obs)
        progress = obs.get("progress", torch.zeros(cell.size(0), 1, device=cell.device)).view(cell.size(0), -1)
        x = torch.cat([cell, inv, wiki, task, progress], dim=-1)
        return self.mlp(x)


class ConvStateEncoder(BaseStateEncoder):
    def __init__(self, vocab_size: int, out_dim: int, token_emb_dim: int, text_rnn_dim: int, hidden_dim: int, padding_idx: int = 0):
        super().__init__(vocab_size, out_dim, token_emb_dim, text_rnn_dim, hidden_dim, padding_idx)
        self.conv = nn.Sequential(
            nn.Conv2d(self.text_dim * 3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, out_dim),
            nn.Tanh(),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs = self._prepare(obs)
        cell = self.encode_cell(obs)
        inv = self.encode_inv(obs)
        wiki = self.encode_wiki(obs)
        task = self.encode_task(obs)
        bsz, h, w, _ = cell.shape
        inv_grid = inv.unsqueeze(1).unsqueeze(1).expand(bsz, h, w, -1)
        text = (wiki + task).unsqueeze(1).unsqueeze(1).expand(bsz, h, w, -1)
        grid = torch.cat([cell, inv_grid, text], dim=-1).permute(0, 3, 1, 2)
        feat = self.conv(grid).amax(dim=(2, 3))
        return self.fc(feat)


class FiLMBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.gamma_beta = nn.Linear(cond_dim, 2 * out_ch)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        gamma, beta = torch.chunk(self.gamma_beta(cond), 2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return F.relu((1.0 + gamma) * h + beta)


class FiLMStateEncoder(BaseStateEncoder):
    def __init__(self, vocab_size: int, out_dim: int, token_emb_dim: int, text_rnn_dim: int, hidden_dim: int, padding_idx: int = 0):
        super().__init__(vocab_size, out_dim, token_emb_dim, text_rnn_dim, hidden_dim, padding_idx)
        self.init = nn.Conv2d(self.text_dim + 2, 32, kernel_size=3, padding=1)
        self.f1 = FiLMBlock(32, 64, self.text_dim * 3)
        self.f2 = FiLMBlock(64, 64, self.text_dim * 3)
        self.fc = nn.Sequential(nn.Linear(64, out_dim), nn.Tanh())

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs = self._prepare(obs)
        cell = self.encode_cell(obs)
        inv = self.encode_inv(obs)
        wiki = self.encode_wiki(obs)
        task = self.encode_task(obs)
        rel_pos = obs.get("rel_pos")
        if rel_pos is None:
            bsz, h, w, _ = cell.shape
            rel_pos = torch.zeros(bsz, h, w, 2, device=cell.device)
        grid = torch.cat([cell, rel_pos.float()], dim=-1).permute(0, 3, 1, 2)
        cond = torch.cat([wiki, inv, task], dim=-1)
        x = F.relu(self.init(grid))
        x = self.f1(x, cond)
        x = self.f2(x, cond)
        feat = x.amax(dim=(2, 3))
        return self.fc(feat)


class Txt2PiStateEncoder(BaseStateEncoder):
    def __init__(self, vocab_size: int, out_dim: int, token_emb_dim: int, text_rnn_dim: int, hidden_dim: int, padding_idx: int = 0):
        super().__init__(vocab_size, out_dim, token_emb_dim, text_rnn_dim, hidden_dim, padding_idx)
        self.c0 = nn.Conv2d(self.text_dim + 2, 64, kernel_size=3, padding=1)
        self.f1 = FiLMBlock(64, 64, self.text_dim)
        self.f2 = FiLMBlock(64, 64, self.text_dim)
        self.query = nn.Linear(64, self.text_dim)
        self.fc = nn.Sequential(
            nn.Linear(64 + self.text_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh(),
        )

    def _wiki_attention(self, wiki_tokens: torch.Tensor, wiki_len: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Accept:
        # wiki_tokens: [L] or [B, L]
        # wiki_len: [] or [1] or [B]
        # cond: [D] or [B, D]

        if wiki_tokens.dim() == 1:
            wiki_tokens = wiki_tokens.unsqueeze(0)
        elif wiki_tokens.dim() != 2:
            raise RuntimeError(f"Expected wiki_tokens with 1 or 2 dims, got shape {tuple(wiki_tokens.shape)}")

        if wiki_len.dim() == 0:
            wiki_len = wiki_len.unsqueeze(0)
        else:
            wiki_len = wiki_len.view(-1)

        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        elif cond.dim() != 2:
            raise RuntimeError(f"Expected cond with 1 or 2 dims, got shape {tuple(cond.shape)}")

        wiki_len = wiki_len.clamp(min=1).long()

        emb = self.text_enc.emb(wiki_tokens.long())  # [B, L, E]
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, wiki_len.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.text_enc.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # [B, L, 2H]

        scores = (out * cond.unsqueeze(1)).sum(dim=-1)
        mask = (
            torch.arange(out.size(1), device=out.device).unsqueeze(0)
            < wiki_len.unsqueeze(1)
        ).float()
        scores = scores - (1.0 - mask) * 1e9
        attn = torch.softmax(scores, dim=1)
        return (attn.unsqueeze(-1) * out).sum(dim=1)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs = self._prepare(obs)
        cell = self.encode_cell(obs)
        inv = self.encode_inv(obs)
        task = self.encode_task(obs)
        wiki = self.encode_wiki(obs)
        rel_pos = obs.get("rel_pos")
        if rel_pos is None:
            bsz, h, w, _ = cell.shape
            rel_pos = torch.zeros(bsz, h, w, 2, device=cell.device)
        else:
            rel_pos = rel_pos.float().to(cell.device)
            if rel_pos.dim() == 3:
                # [H, W, D] -> [B, H, W, D]
                rel_pos = rel_pos.unsqueeze(0)
            elif rel_pos.dim() != 4:
                raise RuntimeError(f"Unexpected rel_pos shape: {tuple(rel_pos.shape)}")

            if rel_pos.shape[:3] != cell.shape[:3]:
                raise RuntimeError(
                    f"cell and rel_pos shape mismatch: cell={tuple(cell.shape)}, rel_pos={tuple(rel_pos.shape)}"
                )
        grid = torch.cat([cell, rel_pos.float()], dim=-1).permute(0, 3, 1, 2)
        x = F.relu(self.c0(grid))
        q0 = self.query(x.amax(dim=(2, 3)))
        wiki_attn = self._wiki_attention(obs["wiki"].long(), obs["wiki_len"].view(-1).long(), q0)
        cond1 = wiki_attn + inv
        x = self.f1(x, cond1)
        q1 = self.query(x.amax(dim=(2, 3)))
        wiki_attn2 = self._wiki_attention(obs["wiki"].long(), obs["wiki_len"].view(-1).long(), q1)
        cond2 = wiki_attn2 + task
        x = self.f2(x, cond2)
        pooled = x.amax(dim=(2, 3))
        return self.fc(torch.cat([pooled, inv, task], dim=-1))


class StateZExtractor(nn.Module):
    def __init__(self, state_encoder: BaseStateEncoder, z_dim: int, xi_L: float):
        super().__init__()
        self.state_encoder = state_encoder
        self.z_dim = int(z_dim)
        self.xi_L = float(xi_L)
        self.features_dim = int(state_encoder.out_dim + z_dim)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        state_dict = {k: v for k, v in obs.items() if k != "z"}
        h_s = self.state_encoder(state_dict)
        h_s = grad_scale(h_s, self.xi_L)
        z = obs["z"].float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return torch.cat([h_s, z], dim=-1)


def build_state_encoder(name: str, vocab_size: int, out_dim: int, token_emb_dim: int, text_rnn_dim: int, hidden_dim: int, padding_idx: int = 0) -> BaseStateEncoder:
    name = name.lower()
    if name == "mlp":
        return MLPStateEncoder(vocab_size, out_dim, token_emb_dim, text_rnn_dim, hidden_dim, padding_idx)
    if name == "conv":
        return ConvStateEncoder(vocab_size, out_dim, token_emb_dim, text_rnn_dim, hidden_dim, padding_idx)
    if name == "film":
        return FiLMStateEncoder(vocab_size, out_dim, token_emb_dim, text_rnn_dim, hidden_dim, padding_idx)
    if name == "txt2pi":
        return Txt2PiStateEncoder(vocab_size, out_dim, token_emb_dim, text_rnn_dim, hidden_dim, padding_idx)
    raise ValueError(f"Unknown state encoder type: {name}")
