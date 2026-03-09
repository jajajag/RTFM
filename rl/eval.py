from __future__ import annotations

from typing import Dict

import numpy as np

from rl.env_utils import normalize_reset, normalize_step


def evaluate(model, env, n_episodes: int = 20) -> Dict[str, float]:
    returns = []
    success = []
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
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "success_rate": float(np.mean(success)),
    }
