# rl/sb3_wrap.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import gym
import torch
import torch.nn.functional as F

from .env_utils import normalize_reset, normalize_step, get_n_actions
from .env_utils import get_instruction_paragraph, obs_to_text
from .instruction_split import split_with_parser, split_with_lm
from .retrieval import z_single, z_topk

class LLWrapper(gym.Wrapper):
    """
    Produces low-level observation = concat(h_s, z_t) as numpy float32.
    Also stores selector info in self.last_sel for trainer to update pi_sel.
    """
    def __init__(
        self,
        env,
        cfg,
        minilm,
        state_adapter,
        instr_adapter,
        pi_sel,
        reward_model,
        parse_instructions_fn=None,
    ):
        super().__init__(env)
        self.cfg = cfg
        self.minilm = minilm
        self.state_adapter = state_adapter
        self.instr_adapter = instr_adapter
        self.pi_sel = pi_sel
        self.rm = reward_model
        self.parse_instructions_fn = parse_instructions_fn

        self.n_actions = get_n_actions(env)
        self.action_space = gym.spaces.Discrete(self.n_actions)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(cfg.state_dim + cfg.z_dim,), dtype=np.float32
        )

        self._cached_paragraph = None
        self._cached_candidates: List[str] = []
        self.last_sel: Dict[str, Any] = {}

    def _get_candidates(self, paragraph: str) -> List[str]:
        if self.cfg.split_mode == "lm":
            return split_with_lm(paragraph, self.cfg.max_instructions)
        return split_with_parser(paragraph, self.cfg.max_instructions, self.parse_instructions_fn)

    #@torch.no_grad()
    def _build(self, obs: Dict[str, Any], greedy: bool = False):
        paragraph = get_instruction_paragraph(self.env, obs)
        if paragraph != self._cached_paragraph:
            self._cached_paragraph = paragraph
            self._cached_candidates = self._get_candidates(paragraph)

        cands = self._cached_candidates
        if not cands:
            cands = ["(no instruction)"]

        # --- encode state/instructions with frozen MiniLM (no grad) ---
        with torch.no_grad():
            st = obs_to_text(obs)
            e_s = self.minilm.encode([st])      # (1,emb)
            e_i = self.minilm.encode(cands)     # (N,emb)
        
        # adapters are trainable -> keep grad
        h_s = self.state_adapter(e_s)           # (1,state_dim) grad OK
        H   = self.instr_adapter(e_i).unsqueeze(0)  # (1,N,z_dim) grad OK
        
        # selector must keep grad
        sel = self.pi_sel(h_s, H, greedy=greedy)

        # build z_t
        if self.cfg.z_mode == "single":
            z = z_single(H, sel.idx)               # (1,z_dim)
        else:
            z = z_topk(H, sel.scores, k=self.cfg.topk, pool=self.cfg.topk_pool)

        obs_ll = torch.cat([h_s, z], dim=-1).squeeze(0).detach().cpu().numpy(
                ).astype(np.float32)

        # reward model prediction (for sel/ll rewards)
        # action not known yet, so we store h_s and z
        self.last_sel = {
            "sel_logp_t": sel.logp.squeeze(0),
            "h_s": h_s.squeeze(0),
            "z_t": z.squeeze(0),
            "cands": cands,
        }
        return obs_ll

    def reset(self, **kwargs):
        obs, _ = normalize_reset(self.env.reset(**kwargs))
        self._cached_paragraph = None
        self._cached_candidates = []
        return self._build(obs, greedy=False)

    def step(self, action: int):
        obs2, r_env, done, info = normalize_step(self.env.step(int(action)))
        info = dict(info) if isinstance(info, dict) else {}
    
        # --- compute reward model prediction r_model (0..1) for this (s,a) ---
        # uses the *previous* state's embedding stored in last_sel during _build()
        h_s = self.last_sel["h_s"].to(self.cfg.device)  # (state_dim,)
        a = torch.zeros(self.n_actions, device=self.cfg.device, 
                        dtype=torch.float32)
        a[int(action)] = 1.0
    
        with torch.no_grad():
            logit = self.rm(h_s.unsqueeze(0), a.unsqueeze(0)).squeeze(0)  
            # (1,) -> ()
            r_model = torch.sigmoid(logit).item()
    
        # --- choose low-level reward mode ---
        if self.cfg.ll_reward == "rm":
            r_ll = float(r_model)
        elif self.cfg.ll_reward == "mix":
            r_ll = float(r_env) + float(self.cfg.ll_lambda) * float(r_model)
        else:  # "env"
            r_ll = float(r_env)
    
        # log for debugging/selector training
        info["r_env"] = float(r_env)
        info["r_model"] = float(r_model)
        info["r_ll"] = float(r_ll)
        info["sel_logp"] = float(self.last_sel.get("sel_logp_t", 
                                torch.tensor(0.0)).detach().item())
    
        obs_ll = self._build(obs2, greedy=False)
        return obs_ll, float(r_ll), bool(done), info
