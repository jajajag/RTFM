# train_rtfm.py
import os
import argparse
import numpy as np
import torch

from rl.config import Config
from rl.minilm import FrozenMiniLM, Adapter
from rl.selector import PiSel
from rl.reward_model import RewardModel
from rl.buffers import RMBuffer
from rl.sb3_wrap import LLWrapper
from rl.trainer import Trainer
from rl.eval import evaluate
from rl.env_utils import get_n_actions

def make_env(level: str):
    import gym
    import rtfm.tasks  # IMPORTANT for env registration
    return gym.make(level)

def main():
    ap = argparse.ArgumentParser()

    # env / split
    ap.add_argument("--level", type=str, default="rtfm:groups_nl-v0")
    ap.add_argument("--split-mode", choices=["parser", "lm"], default="parser")
    ap.add_argument("--max-instructions", type=int, default=64)

    # high-level schedule + aux
    ap.add_argument("--hl-T", type=int, default=5)
    ap.add_argument("--hl-gamma", type=float, default=0.99)
    ap.add_argument("--hl-update-every-steps", type=int, default=1000)
    ap.add_argument("--hl-aux-type", choices=["none","v_diff","score_diff","kl_pos","kl_neg","cos"], default="none")
    ap.add_argument("--hl-aux-scale", type=float, default=1.0)

    # state encoder update routing
    ap.add_argument("--state-encoder-update", choices=["low","both"], default="both")

    # retrieval / z
    ap.add_argument("--z-mode", choices=["single", "topk"], default="single")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--topk-pool", choices=["mean", "attn"], default="mean")

    # low-level RL
    ap.add_argument("--ll-algo", choices=["ppo", "sac"], default="ppo")
    ap.add_argument("--ll-reward", choices=["env", "rm", "mix"], default="mix")
    ap.add_argument("--ll-lambda", type=float, default=0.1)
    ap.add_argument("--ll-update-every-steps", type=int, default=1000)

    # training schedule
    ap.add_argument("--total-steps", type=int, default=1_000_000)
    ap.add_argument("--eval-every-steps", type=int, default=50_000)
    ap.add_argument("--eval-episodes", type=int, default=50)

    # dims / opt
    ap.add_argument("--state-dim", type=int, default=256)
    ap.add_argument("--z-dim", type=int, default=256)
    ap.add_argument("--rm-hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")

    args = ap.parse_args()

    cfg = Config(
        level=args.level,
        split_mode=args.split_mode,
        max_instructions=args.max_instructions,

        hl_T=args.hl_T,
        hl_gamma=args.hl_gamma,
        hl_update_every_steps=args.hl_update_every_steps,
        hl_aux_type=args.hl_aux_type,
        hl_aux_scale=args.hl_aux_scale,
        state_encoder_update=args.state_encoder_update,

        z_mode=args.z_mode,
        topk=args.topk,
        topk_pool=args.topk_pool,

        ll_algo=args.ll_algo,
        ll_reward=args.ll_reward,
        ll_lambda=args.ll_lambda,
        ll_update_every_steps=args.ll_update_every_steps,

        total_steps=args.total_steps,
        eval_every_steps=args.eval_every_steps,
        eval_episodes=args.eval_episodes,

        state_dim=args.state_dim,
        z_dim=args.z_dim,
        rm_hidden=args.rm_hidden,
        seed=args.seed,
        device=args.device,
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = make_env(cfg.level)
    n_actions = get_n_actions(env)

    # Optional: plug your existing parsers.py if you have one in repo
    parse_instructions_fn = None
    try:
        from rl.parsers import parse_instructions  # if you later add your parser here
        parse_instructions_fn = parse_instructions
    except Exception:
        parse_instructions_fn = None

    # frozen MiniLM + adapters
    minilm = FrozenMiniLM(cfg.minilm_name, device=cfg.device)
    state_adapter = Adapter(cfg.emb_dim, cfg.state_dim, hidden=cfg.adapter_hidden).to(cfg.device)
    instr_adapter = Adapter(cfg.emb_dim, cfg.z_dim, hidden=cfg.adapter_hidden).to(cfg.device)

    # pi_sel + reward model (regression)
    pi_sel = PiSel(d=cfg.z_dim, hidden=cfg.adapter_hidden).to(cfg.device)
    rm = RewardModel(state_dim=cfg.state_dim, action_dim=n_actions, hidden=cfg.rm_hidden).to(cfg.device)

    # wrap env to produce (h_s, z_k) obs
    ll_env = LLWrapper(
        env=env,
        cfg=cfg,
        minilm=minilm,
        state_adapter=state_adapter,
        instr_adapter=instr_adapter,
        pi_sel=pi_sel,
        reward_model=rm,
        parse_instructions_fn=parse_instructions_fn,
    )

    # SB3 low-level model
    if cfg.ll_algo == "ppo":
        from stable_baselines3 import PPO
        ll_model = PPO(
            "MlpPolicy",
            ll_env,
            seed=cfg.seed,
            verbose=0,
            n_steps=int(cfg.ll_update_every_steps),
        )
    else:
        from stable_baselines3 import SAC
        ll_model = SAC("MlpPolicy", ll_env, seed=cfg.seed, verbose=0)

    rm_buffer = RMBuffer(capacity=cfg.rm_buffer_capacity)
    trainer = Trainer(
        cfg, ll_env, ll_model, pi_sel, rm, rm_buffer,
        device=cfg.device, state_adapter=state_adapter, instr_adapter=instr_adapter
    )

    # train in chunks so we can evaluate
    steps_done = 0
    while steps_done < cfg.total_steps:
        chunk = min(cfg.eval_every_steps, cfg.total_steps - steps_done)
        trainer.train(chunk)
        steps_done += chunk

        metrics = evaluate(ll_model, ll_env, n_episodes=cfg.eval_episodes)
        print(
            f"[steps={steps_done}] "
            f"return={metrics['return_mean']:.3f}±{metrics['return_std']:.3f} "
            f"success={metrics['success_rate']:.3f}",
            flush=True,
        )

if __name__ == "__main__":
    main()
