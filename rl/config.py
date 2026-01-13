# rl/config.py
from dataclasses import dataclass

@dataclass
class Config:
    # env
    level: str = "rtfm:groups_nl-v0"
    seed: int = 0
    device: str = "cuda"

    # instruction split
    split_mode: str = "parser"   # ["parser", "lm"]
    max_instructions: int = 64

    # frozen MiniLM + adapters
    minilm_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    emb_dim: int = 384
    state_dim: int = 256
    z_dim: int = 256
    adapter_hidden: int = 256

    # pi_sel options
    sel_reward: str = "one_step_rm"  # ["one_step_rm", "traj_env", "traj_rm"]
    z_mode: str = "single"           # ["single", "topk"]
    topk: int = 4
    topk_pool: str = "mean"          # ["mean", "attn"]

    # low-level RL options
    ll_algo: str = "ppo"             # ["ppo", "sac"]
    ll_reward: str = "mix"           # ["env", "rm", "mix"]
    ll_lambda: float = 0.1           # lambda for mix: r_env + lambda * r_model

    # training schedule
    total_iters: int = 200
    steps_per_iter: int = 5000
    eval_every: int = 1
    eval_episodes: int = 50

    # optimization
    lr_sel: float = 3e-4
    lr_rm: float = 3e-4

    # reward model
    rm_hidden: int = 256
    rm_updates_per_iter: int = 200
    rm_batch: int = 256
    rm_buffer_capacity: int = 200000

