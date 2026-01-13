# rl/eval.py
from __future__ import annotations
import numpy as np

def evaluate(ll_model, env, n_episodes: int = 50):
    rets = []
    succ = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = ll_model.predict(obs, deterministic=True)
            obs, r, done, info = env.step(action)
            ep_ret += float(r)
        rets.append(ep_ret)

        # generic success heuristic: if env provides info["success"] use it; else positive return
        if isinstance(info, dict) and "success" in info:
            succ.append(float(info["success"]))
        else:
            succ.append(1.0 if ep_ret > 0 else 0.0)

    return {
        "return_mean": float(np.mean(rets)),
        "return_std": float(np.std(rets)),
        "success_rate": float(np.mean(succ)),
    }

