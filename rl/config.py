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

    # high-level (selector) horizon
    hl_T: int = 5                     # call pi_sel every T env steps
    hl_gamma: float = 0.99            # discount inside each T-step segment
    hl_update_every_steps: int = 1000 # update selector every N env steps

    # high-level auxiliary reward
    hl_aux_type: str = "none"         # ["none","v_diff","score_diff","kl_pos","kl_neg","cos"]
    hl_aux_scale: float = 1.0         # scale for R_aux

    # state encoder update mode
    # "low": updated only by reward-model (low-level side supervision)
    # "both": updated by reward-model and selector losses
    state_encoder_update: str = "both"  # ["low","both"]

    # z / retrieval options
    z_mode: str = "single"           # ["single", "topk"]
    topk: int = 4
    topk_pool: str = "mean"          # ["mean", "attn"]

    # low-level RL options
    ll_algo: str = "ppo"             # ["ppo", "sac"]
    ll_reward: str = "mix"           # ["env", "rm", "mix"]
    ll_lambda: float = 0.1           # lambda for mix: r_env + lambda * r_model
    ll_update_every_steps: int = 1000  # PPO n_steps (and intended update cadence)

    # training schedule
    total_steps: int = 1_000_000
    eval_every_steps: int = 50_000
    eval_episodes: int = 50

    # optimization
    lr_sel: float = 3e-4
    lr_rm: float = 3e-4

    # reward model
    rm_hidden: int = 256
    rm_updates_per_call: int = 200    # gradient steps per scheduled update
    rm_batch: int = 256
    rm_buffer_capacity: int = 200000
