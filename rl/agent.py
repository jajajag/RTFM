from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import BiLSTMEncoder, MiniLMEncoder, Vocab
from .modules import Adapter, CrossAttention, FiLM, ActorCritic

class Agent(nn.Module):
    def __init__(
        self,
        text_encoder="bilstm",            # "bilstm" | "minilm"
        hidden=512,
        mod_mode="film",                  # "film" | "xattn"
        heads=8,
        n_actions=6,
        vocab: Vocab = None,
    ):
        super().__init__()
        self.hidden = hidden
        self.mod_mode = mod_mode

        if text_encoder == "bilstm":
            assert vocab is not None, "BiLSTM 需要传入构建好的 Vocab"
            self.vocab = vocab
            self.encoder = BiLSTMEncoder(vocab_size=len(vocab.inv), dim=hidden)
            self.encode_text = self._encode_bilstm
        else:
            self.encoder = MiniLMEncoder(out_dim=hidden)
            self.encode_text = self._encode_minilm

        self.state_adapter = Adapter(hidden)
        self.goal_adapter  = Adapter(hidden)
        self.tip_adapter   = Adapter(hidden)

        self.xattn = CrossAttention(hidden, heads=heads)
        self.film  = FiLM(hidden, hidden)

        #self.policy = PolicyNet(in_dim=hidden*2, hidden=hidden, n_actions=n_actions)
        self.policy = ActorCritic(in_dim=hidden*2, hidden=hidden, n_actions=n_actions)

    # —— 编码接口 ——————————————————————————
    def _encode_bilstm(self, texts: List[str]) -> torch.Tensor:
        ids = [self.vocab.encode(t, max_len=64) for t in texts]
        ids = torch.tensor(ids, dtype=torch.long)
        return self.encoder(ids)

    def _encode_minilm(self, texts: List[str]) -> torch.Tensor:
        return self.encoder(texts)

    # —— 前向 ——————————————————————————————
    def forward(
        self,
        state_texts: List[str],
        goal_texts: List[str],
        tip_texts: List[str],
        return_attn=False,
    ):
        # enc & adapt
        h_s  = self.state_adapter(self.encode_text(state_texts))          # (B,D)
        H_g  = self.goal_adapter(self.encode_text(goal_texts))            # (G,D)
        H_t  = self.tip_adapter(self.encode_text(tip_texts))              # (T,D)

        # batch 拼法：统一 batch 维（简单做法：都当 batch=1）
        if h_s.dim()==1:  h_s = h_s.unsqueeze(0)
        H_g = H_g.unsqueeze(0)  # (1,G,D)
        H_t = H_t.unsqueeze(0)  # (1,T,D)

        # z_goal = CrossAttn(h_s, H_g)
        z_goal, A_goal = self.xattn(h_s, H_g, H_g)                        # (1,D),(1,G)

        # z_t：FiLM 或 CrossAttn 到 tip
        if self.mod_mode == "film":
            # 先把 tips 池化一个向量再 FiLM（也可先 FiLM 再池化）
            H_t_mod = self.film(H_t, z_goal)                              # (1,T,D)
            z_t = H_t_mod.mean(dim=1)                                     # (1,D)
            A_tip = None
        else:
            z_t, A_tip = self.xattn(z_goal, H_t, H_t)                     # (1,D),(1,T)

        # policy
        pi_in = torch.cat([h_s, z_t], dim=-1)                             # (1,2D)
        logits, value = self.policy(pi_in)                                       # (1,A)
        if return_attn:
            return logits, value, A_goal, A_tip, z_goal, z_t, h_s, H_g
        return logits, value

