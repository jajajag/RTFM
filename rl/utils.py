from typing import List
from .encoders import Vocab

def obs_to_text(obs) -> List[str]:
    """
    尝试把 RTFM 的观测转换成一行文本。
    不同版本 key 可能不同：你可按自己环境改这里。
    """
    # 常见字段兜底
    grid = obs.get("grid_text") or obs.get("observation") or obs.get("text") or ""
    inv  = obs.get("inventory") or ""
    goal = obs.get("goal") or ""
    # 拼一句；也可以返回多句
    line = f"{grid} {inv} {goal}".strip()
    return [line if line else "(empty state)."]

def build_vocab_from_env(env, episodes: int = 200) -> Vocab:
    """
    只用于 BiLSTM：在线滚动收集一些句子建词表（最小可用版）。
    真实项目建议预扫训练集/使用更好的 tokenizer。
    """
    v = Vocab()
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    for _ in range(episodes):
        for k in ("grid_text", "observation", "text", "inventory", "goal", "instructions", "manual"):
            if isinstance(obs.get(k), str):
                v.add_sentence(obs[k])
        obs, *_ = env.step(getattr(env.action_space, "sample", lambda: 0)())
    return v

