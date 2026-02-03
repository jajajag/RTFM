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
from .buffers import SelSegment

def onehot(action: int, n: int, device: str):
    a = torch.zeros(n, device=device, dtype=torch.float32)
    a[int(action)] = 1.0
    return a

class LLWrapper(gym.Wrapper):
    """
    Low-level wrapper that:
      1) Builds observation = concat(h_s(t), z_k) where z_k is chosen by high-level selector every T steps.
      2) Computes low-level reward r_ll = r_env / r_model / (r_env + lambda * r_model) per cfg.ll_reward.
      3) Collects selector segments for REINFORCE updates (every T steps).
      4) Collects reward-model supervised data (h_s, a_onehot, r_env).

    Trainer reads:
      - self.last_sel: per-step bookkeeping (h_s, z_k, etc.)
      - self.finished_segments: list[SelSegment] finalized since last trainer pull()
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

        self._cached_paragraph: Optional[str] = None
        self._cached_candidates: List[str] = []
        self._cached_H: Optional[torch.Tensor] = None  # (1,N,z_dim)

        # high-level segment state
        self._t_in_ep: int = 0
        self._current_z: Optional[torch.Tensor] = None     # (z_dim,)
        self._current_hi: Optional[torch.Tensor] = None    # (z_dim,)
        self._current_logp: Optional[torch.Tensor] = None  # scalar tensor
        self._current_probs: Optional[torch.Tensor] = None # (N,)
        self._current_score_start: Optional[torch.Tensor] = None # scalar tensor
        self._segment_rewards: List[float] = []
        self._segment_hs: List[torch.Tensor] = []          # list[(state_dim,)] for cosine aux
        self._segment_obs_start: Optional[torch.Tensor] = None  # (obs_dim,) CPU
        self._segment_obs_end: Optional[torch.Tensor] = None    # (obs_dim,) CPU

        self.last_sel: Dict[str, Any] = {}
        self.finished_segments: List[SelSegment] = []

    def _get_candidates(self, paragraph: str) -> List[str]:
        if self.cfg.split_mode == "lm":
            return split_with_lm(paragraph, self.cfg.max_instructions)
        return split_with_parser(paragraph, self.cfg.max_instructions, self.parse_instructions_fn)

    def _encode_instructions(self, paragraph: str) -> List[str]:
        if paragraph != self._cached_paragraph:
            self._cached_paragraph = paragraph
            self._cached_candidates = self._get_candidates(paragraph)
            self._cached_H = None
        return self._cached_candidates

    def _get_H(self, cands: List[str]) -> torch.Tensor:
        # cache instruction embeddings across steps/segments (same paragraph)
        if self._cached_H is None:
            if not cands:
                cands = ["(no instruction)"]
            with torch.no_grad():
                e_i = self.minilm.encode(cands)  # (N,emb)
            H = self.instr_adapter(e_i).unsqueeze(0)  # (1,N,z_dim) (trainable)
            self._cached_H = H
        return self._cached_H

    def _encode_state(self, obs: Dict[str, Any]) -> torch.Tensor:
        with torch.no_grad():
            st = obs_to_text(obs)
            e_s = self.minilm.encode([st])  # (1,emb)
        h_s = self.state_adapter(e_s)       # (1,state_dim)
        return h_s

    def _h_s_for_selector(self, h_s: torch.Tensor) -> torch.Tensor:
        # control whether selector gradients flow into state encoder
        if getattr(self.cfg, "state_encoder_update", "both") == "low":
            return h_s.detach()
        return h_s

    def _obs_ll(self, h_s: torch.Tensor, z: torch.Tensor) -> np.ndarray:
        obs_ll = torch.cat([h_s, z.unsqueeze(0)], dim=-1).squeeze(0).detach().cpu().numpy().astype(np.float32)
        return obs_ll

    def _discounted_sum(self, rewards: List[float]) -> float:
        g = 0.0
        p = 1.0
        for r in rewards:
            g += p * float(r)
            p *= float(self.cfg.hl_gamma)
        return float(g)

    def _finalize_segment_with_boundary(self, h_s_end: torch.Tensor, probs_end: Optional[torch.Tensor]):
        """Finalize the current segment using the boundary state (end of segment)."""
        if self._current_logp is None:
            return
        R = self._discounted_sum(self._segment_rewards)

        # --- R_aux computed here for types that don't need value function ---
        aux_type = getattr(self.cfg, "hl_aux_type", "none")
        R_aux = 0.0

        # score difference: score(h_s_start, h_i*) - score(h_s_end, h_i*)
        if aux_type == "score_diff" and self._current_hi is not None:
            hs_end = self._h_s_for_selector(h_s_end)  # (1,D)
            hi = self._current_hi.unsqueeze(0)        # (1,D)
            with torch.no_grad():
                s_end = self.pi_sel.scorer(torch.cat([hs_end, hi], dim=-1)).squeeze()
                s_start = (self._current_score_start.squeeze() if self._current_score_start is not None else 0.0)
                R_aux = float((s_start - s_end).detach().cpu().item())

        # KL: KL(p_start || p_end) (pos or neg)
        if aux_type in ["kl_pos", "kl_neg"] and (self._current_probs is not None) and (probs_end is not None):
            p = self._current_probs.clamp_min(1e-8)
            q = probs_end.clamp_min(1e-8)
            with torch.no_grad():
                kl = (p * (p.log() - q.log())).sum()
                sign = 1.0 if aux_type == "kl_pos" else -1.0
                R_aux = float((sign * kl).detach().cpu().item())

        # cosine: cos(mean_t h_s(t), h_i*)
        if aux_type == "cos" and (self._current_hi is not None) and self._segment_hs:
            with torch.no_grad():
                u = torch.stack([h.detach() for h in self._segment_hs], dim=0).mean(dim=0)
                hi = self._current_hi.detach()
                R_aux = float(F.cosine_similarity(u.unsqueeze(0), hi.unsqueeze(0), dim=-1).squeeze().cpu().item())

        R_total = float(R + getattr(self.cfg, "hl_aux_scale", 1.0) * R_aux)

        # store obs start/end for optional v-diff aux in trainer
        seg = SelSegment(
            logp=self._current_logp,
            R=R_total,
            obs_start=self._segment_obs_start,
            obs_end=self._segment_obs_end,
        )
        self.finished_segments.append(seg)

        # reset segment accumulators
        self._current_logp = None
        self._current_probs = None
        self._current_hi = None
        self._current_score_start = None
        self._segment_rewards = []
        self._segment_hs = []
        self._segment_obs_start = None
        self._segment_obs_end = None

    def _select_if_needed(self, obs: Dict[str, Any]):
        """Run selector at the start of an episode or at segment boundaries."""
        paragraph = get_instruction_paragraph(self.env, obs)
        cands = self._encode_instructions(paragraph)
        H = self._get_H(cands)  # (1,N,D)

        h_s = self._encode_state(obs)              # (1,D)
        h_s_sel = self._h_s_for_selector(h_s)

        # if we are at a boundary AND already have a running segment, we need the new probs for KL
        sel = self.pi_sel(h_s_sel, H, greedy=False)  # keeps graph for logp

        # decide z
        if self.cfg.z_mode == "single":
            z = z_single(H, sel.idx)  # (1,D)
        else:
            z = z_topk(H, sel.scores, k=self.cfg.topk, pool=self.cfg.topk_pool)

        # update current selection
        idx0 = int(sel.idx.squeeze(0).detach().cpu().item())
        self._current_z = z.squeeze(0)
        self._current_hi = H[0, idx0]  # (D,)
        self._current_logp = sel.logp.squeeze(0)  # scalar tensor with graph
        self._current_probs = sel.probs.squeeze(0).detach()  # detach to avoid graph bloat
        self._current_score_start = sel.scores[0, idx0].detach()

        # start segment accumulators
        self._segment_rewards = []
        self._segment_hs = []
        obs_ll = self._obs_ll(h_s, self._current_z)
        self._segment_obs_start = torch.from_numpy(obs_ll).float()  # CPU tensor

        # bookkeeping
        self.last_sel = {
            "h_s": h_s.squeeze(0),
            "z_k": self._current_z,
            "cands": cands,
            "sel_logp": float(self._current_logp.detach().cpu().item()),
            "sel_idx": idx0,
        }

        return obs_ll, h_s, sel.probs.squeeze(0).detach()

    def reset(self, **kwargs):
        obs, _ = normalize_reset(self.env.reset(**kwargs))
        self._cached_paragraph = None
        self._cached_candidates = []
        self._cached_H = None

        self._t_in_ep = 0
        self.finished_segments = []

        obs_ll, h_s, _ = self._select_if_needed(obs)
        # include current h_s for cosine
        self._segment_hs.append(h_s.squeeze(0))
        return obs_ll

    def step(self, action: int):
        # --- compute reward model prediction r_model for (s_t, a_t) using last_sel h_s ---
        h_s_prev = self.last_sel["h_s"].to(self.cfg.device)  # (state_dim,)
        a_oh = onehot(int(action), self.n_actions, self.cfg.device)

        with torch.no_grad():
            r_model = float(self.rm(h_s_prev.unsqueeze(0), a_oh.unsqueeze(0)).squeeze().cpu().item())

        # step env
        obs2, r_env, done, info = normalize_step(self.env.step(int(action)))
        info = dict(info) if isinstance(info, dict) else {}

        # low-level reward
        if self.cfg.ll_reward == "rm":
            r_ll = float(r_model)
        elif self.cfg.ll_reward == "mix":
            r_ll = float(r_env) + float(self.cfg.ll_lambda) * float(r_model)
        else:
            r_ll = float(r_env)

        info["r_env"] = float(r_env)
        info["r_model"] = float(r_model)
        info["r_ll"] = float(r_ll)

        # accumulate segment rewards / hs
        self._segment_rewards.append(float(r_ll))
        # we will append h_s for next obs after encoding below (so segment has T states)
        # build next obs (might finalize + reselection)
        end_of_segment = bool(done) or ((self._t_in_ep + 1) % int(self.cfg.hl_T) == 0)

        # encode boundary state and next probs for KL/score
        paragraph2 = get_instruction_paragraph(self.env, obs2)
        cands2 = self._encode_instructions(paragraph2)
        H2 = self._get_H(cands2)
        h_s2 = self._encode_state(obs2)
        h_s2_sel = self._h_s_for_selector(h_s2)
        sel2 = self.pi_sel(h_s2_sel, H2, greedy=False)
        probs2 = sel2.probs.squeeze(0).detach()

        if end_of_segment:
            # store obs_end for optional v-diff
            obs_ll_end = self._obs_ll(h_s2, self._current_z if self._current_z is not None else torch.zeros(self.cfg.z_dim, device=h_s2.device))
            self._segment_obs_end = torch.from_numpy(obs_ll_end).float()
            self._finalize_segment_with_boundary(h_s2, probs2)

            if not done:
                # start new segment with selection at boundary
                obs_ll, h_s_new, _ = self._select_if_needed(obs2)
                self._segment_hs.append(h_s_new.squeeze(0))
                self._t_in_ep += 1
                return obs_ll, float(r_ll), bool(done), info

        # not end-of-segment: keep z, just update observation with new h_s
        if self._current_z is None:
            # safety fallback
            self._current_z = torch.zeros(self.cfg.z_dim, device=h_s2.device)

        obs_ll = self._obs_ll(h_s2, self._current_z)
        self.last_sel["h_s"] = h_s2.squeeze(0)
        self._segment_hs.append(h_s2.squeeze(0))

        self._t_in_ep += 1
        return obs_ll, float(r_ll), bool(done), info
