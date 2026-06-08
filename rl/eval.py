from __future__ import annotations

from typing import Dict

import numpy as np


def evaluate(model, env, n_episodes: int = 20) -> Dict[str, float]:
    old_mode = getattr(env.cfg, "selector_train_mode", None) if hasattr(env, "cfg") else None
    old_collect = getattr(env, "collect_experience", None)
    if old_mode is not None:
        env.cfg.selector_train_mode = env.cfg.selector_eval_mode
    if old_collect is not None:
        env.collect_experience = False
    returns = []
    success = []
    try:
        for _ in range(int(n_episodes)):
            obs = env.reset()
            done = False
            ep_ret = 0.0
            info = {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                ep_ret += float(reward)
            returns.append(ep_ret)
            success.append(float(info.get("success", info.get("win", ep_ret > 0))))
    finally:
        if old_mode is not None:
            env.cfg.selector_train_mode = old_mode
        if old_collect is not None:
            env.collect_experience = old_collect
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "success_rate": float(np.mean(success)),
    }
