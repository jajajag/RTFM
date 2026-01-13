# rl/env_utils.py
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, List
import random
import numpy as np

def get_n_actions(env) -> int:
    A = getattr(env, "action_space", None)
    if hasattr(A, "n"):
        return int(A.n)
    if isinstance(A, list):
        return len(A)
    # fallback
    try:
        import rtfm.dynamics.monster as M
        return len(M.QueuedAgent.valid_moves)
    except Exception as e:
        raise RuntimeError(f"Cannot infer n_actions: {e}")

def sample_action(env) -> int:
    A = getattr(env, "action_space", None)
    if hasattr(A, "sample"):
        return A.sample()
    if isinstance(A, list):
        return random.randrange(len(A))
    return 0

def normalize_reset(out):
    # gymnasium: (obs, info), gym: obs
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], dict):
        return out[0], out[1]
    return out, {}

def normalize_step(out):
    """
    Return (obs2, r, done, info)
    - gymnasium: (obs, r, terminated, truncated, info)
    - some RTFM env: (obs, r, done, truncated_bool)  # 4th is bool, not dict
    - gym: (obs, r, done, info)
    """
    if not isinstance(out, tuple):
        raise RuntimeError(f"step() returned non-tuple: {type(out)}")

    if len(out) == 5:
        obs2, r, term, trunc, info = out
        done = bool(term or trunc)
        if info is None: info = {}
        return obs2, float(r), done, dict(info)
    if len(out) == 4:
        obs2, r, a3, a4 = out
        # RTFM Task: (obs, r, done, win_bool)
        # In RTFM, the 4th item is "win"/"success", NOT truncated.
        if isinstance(a4, (bool, np.bool_)) and isinstance(a3, (bool, np.bool_)):
            done = bool(a3)
            info = {"win": bool(a4), "success": bool(a4)}
            return obs2, float(r), done, info
        # Gym classic: (obs, r, done, info_dict)
        done = bool(a3)
        info = a4 if isinstance(a4, dict) else {}
        return obs2, float(r), done, dict(info)
    if len(out) == 3:
        obs2, r, done = out
        return obs2, float(r), bool(done), {}
    raise RuntimeError(f"Unexpected step tuple len={len(out)}")

def get_instruction_paragraph(env, obs: Dict[str, Any]) -> str:
    """
    Instructions may not be in obs for some RTFM envs.
    Try obs first, then env.unwrapped, then env.unwrapped.task/_task.
    """
    if isinstance(obs, dict):
        for k in ("instructions", "manual", "instruction", "prompt", "text"):
            v = obs.get(k, None)
            if isinstance(v, str) and v.strip():
                return v

    U = getattr(env, "unwrapped", env)
    for k in ("instructions", "manual", "instruction", "prompt", "text"):
        v = getattr(U, k, None)
        if isinstance(v, str) and v.strip():
            return v

    task = getattr(U, "task", None) or getattr(U, "_task", None)
    if task is not None:
        for k in ("instructions", "manual", "prompt", "text"):
            v = getattr(task, k, None)
            if isinstance(v, str) and v.strip():
                return v

    return ""

def obs_to_text(obs: Dict[str, Any]) -> str:
    """
    Minimal robust serialization. Works even for groups_nl-v0 (progress only).
    You can customize later without touching the RL stack.
    """
    if not isinstance(obs, dict):
        return str(obs)

    parts: List[str] = []
    # common text fields if exist
    for k in ("grid_text", "observation", "text", "inventory", "goal", "description"):
        v = obs.get(k, None)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())

    # progress-only envs
    if "progress" in obs:
        try:
            p = obs["progress"]
            val = float(p.item()) if hasattr(p, "item") else float(p)
            parts.append(f"progress={val:.4f}")
        except Exception:
            parts.append(f"progress={obs['progress']}")

    if not parts:
        return "(empty state)"
    return " ".join(parts)

