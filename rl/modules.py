import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self, dim=512, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(p),
        )
    def forward(self, x): return self.net(x)

class CrossAttention(nn.Module):
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
    def forward(self, q, k, v):
        # q: (B, dim) ; k,v: (B, N, dim)
        z, w = self.attn(q.unsqueeze(1), k, v)  # (B,1,dim),(B,1,N)
        return z.squeeze(1), w.squeeze(1)       # (B,dim),(B,N)

class FiLM(nn.Module):
    def __init__(self, cond_dim=512, feat_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim * 2),
        )
    def forward(self, H, zc):
        # H: (B, N, D) or (B, D) ; zc: (B, D)
        gb = self.mlp(zc)                       # (B, 2D)
        gamma, beta = gb.chunk(2, dim=-1)       # (B, D)
        if H.dim() == 3:
            gamma, beta = gamma.unsqueeze(1), beta.unsqueeze(1)  # (B,1,D)
        return gamma * H + beta

class PolicyNet(nn.Module):
    def __init__(self, in_dim=1024, hidden=512, n_actions=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
    def forward(self, x): return self.net(x)

class ActorCritic(nn.Module):
    """离散动作 PPO 头：输出 policy logits 和状态价值"""
    def __init__(self, in_dim=1024, hidden=512, n_actions=6):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self.v = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        logits = self.pi(x)          # (B, A)
        value  = self.v(x).squeeze(-1)  # (B,)
        return logits, value
