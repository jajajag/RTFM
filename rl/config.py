from dataclasses import dataclass


@dataclass
class Config:
    level: str = "rtfm:groups_nl-v0"
    seed: int = 0
    device: str = "cpu"

    # Environment / observations
    room_shape: int = 6
    partially_observable: bool = False
    max_placement: int = 1
    shuffle_wiki: bool = False
    time_penalty: float = 0.0

    # Instruction processing
    split_mode: str = "parser"
    max_instructions: int = 64
    minilm_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    emb_dim: int = 384
    instr_dim: int = 256
    state_dim: int = 256
    adapter_hidden: int = 256

    # State encoder
    state_encoder_type: str = "conv"  # mlp | conv | film | txt2pi
    token_emb_dim: int = 64
    text_rnn_dim: int = 64
    state_hidden: int = 128

    # High-level selector
    hl_T: int = 5
    hl_gamma: float = 0.99
    hl_lr: float = 3e-4
    hl_update_every_steps: int = 1000
    selector_mode: str = "sample"  # hard | sample
    hl_return_source: str = "rm"   # rm | env
    hl_aux_type: str = "cos"       # none | cos | v_diff
    hl_aux_lambda: float = 1.0

    # Low-level policy / shaping
    ll_algo: str = "ppo"
    ll_lr: float = 3e-4
    ll_gamma: float = 0.99
    ll_reward: str = "mix"  # env | rm | mix
    ll_lambda: float = 0.1
    ll_update_every_steps: int = 1024

    # Shared state encoder trade-offs
    xi_H: float = 1.0
    xi_L: float = 1.0

    # Reward model
    rm_variant: str = "sas"  # sa | sas | sasz
    rm_hidden: int = 256
    rm_lr: float = 3e-4
    rm_batch_size: int = 256
    rm_buffer_capacity: int = 200000
    rm_updates_per_call: int = 100

    # Training / eval
    total_steps: int = 1000000
    eval_every_steps: int = 5000
    eval_episodes: int = 20
