from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def get_n_actions(env) -> int:
    if hasattr(env.action_space, "n"):
        return int(env.action_space.n)
    return len(env.action_space)


def normalize_reset(out):
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], dict):
        return out[0], out[1]
    return out, {}


def normalize_step(out):
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated or truncated), info or {}
    if len(out) == 4:
        obs, reward, d3, d4 = out
        if isinstance(d4, (bool, np.bool_)) and isinstance(d3, (bool, np.bool_)):
            return obs, float(reward), bool(d3), {"success": bool(d4), "win": bool(d4)}
        return obs, float(reward), bool(d3), d4 if isinstance(d4, dict) else {}
    raise RuntimeError(f"Unexpected step output: {out}")


def get_instruction_paragraph(env, obs: Dict[str, Any]) -> str:
    if isinstance(obs, dict):
        for key in ("instructions", "manual", "instruction", "prompt", "text"):
            val = obs.get(key, None)
            if isinstance(val, str) and val.strip():
                return val
    unwrapped = getattr(env, "unwrapped", env)
    task = getattr(unwrapped, "task", None) or getattr(unwrapped, "_task", None)
    wiki = ""
    goal = ""
    if task is not None and hasattr(task, "get_wiki"):
        wiki = task.get_wiki() or ""
    if task is not None and hasattr(task, "get_task"):
        goal = task.get_task() or ""
    return " ".join([x.strip() for x in [wiki, goal] if isinstance(x, str) and x.strip()])


def clone_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in obs.items():
        if hasattr(v, "clone"):
            out[k] = v.clone()
        else:
            out[k] = v
    return out
