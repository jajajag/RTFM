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
    ap.add_argument("--level", type=str, default="rtfm:groups_nl-v0")

    ap.add_argument("--split-mode", choices=["parser", "lm"], default="parser")
    ap.add_argument("--sel-reward", choices=["one_step_rm", "traj_env", "traj_rm"], default="one_step_rm")
    ap.add_argument("--z-mode", choices=["single", "topk"], default="single")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--topk-pool", choices=["mean", "attn"], default="mean")

    ap.add_argument("--ll-algo", choices=["ppo", "sac"], default="ppo")
    ap.add_argument("--ll-reward", choices=["env", "rm", "mix"], default="mix")
    ap.add_argument("--ll-lambda", type=float, default=0.1)

    ap.add_argument("--total-iters", type=int, default=200)
    ap.add_argument("--steps-per-iter", type=int, default=5000)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--eval-episodes", type=int, default=50)

    ap.add_argument("--state-dim", type=int, default=256)
    ap.add_argument("--z-dim", type=int, default=256)
    ap.add_argument("--rm-hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    args = ap.parse_args()

    cfg = Config(
        level=args.level,
        split_mode=args.split_mode,
        sel_reward=args.sel_reward,
        z_mode=args.z_mode,
        topk=args.topk,
        topk_pool=args.topk_pool,
        ll_algo=args.ll_algo,
        ll_reward=args.ll_reward,
        ll_lambda=args.ll_lambda,
        total_iters=args.total_iters,
        steps_per_iter=args.steps_per_iter,
        eval_every=args.eval_every,
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

    # pi_sel + reward model
    pi_sel = PiSel(d=cfg.z_dim, hidden=cfg.adapter_hidden).to(cfg.device)
    rm = RewardModel(state_dim=cfg.state_dim, action_dim=n_actions, hidden=cfg.rm_hidden).to(cfg.device)

    # wrap env to produce (h_s, z_t) obs
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
        ll_model = PPO("MlpPolicy", ll_env, seed=cfg.seed, verbose=0)
    else:
        from stable_baselines3 import SAC
        ll_model = SAC("MlpPolicy", ll_env, seed=cfg.seed, verbose=0)

    rm_buffer = RMBuffer(capacity=cfg.rm_buffer_capacity)
    trainer = Trainer(cfg, ll_env, ll_model, pi_sel, rm, rm_buffer, device=cfg.device)

    for it in range(cfg.total_iters):
        print(f"[Iter {it+1}/{cfg.total_iters}] starting...", flush=True)
        trainer.collect_and_fill_buffers(cfg.steps_per_iter)
        trainer.update_reward_model(cfg.rm_updates_per_iter)
        trainer.train_low_level(cfg.steps_per_iter)

        if (it + 1) % cfg.eval_every == 0:
            metrics = evaluate(ll_model, ll_env, n_episodes=cfg.eval_episodes)
            print(
                f"[Iter {it+1}] "
                f"return={metrics['return_mean']:.3f}±{metrics['return_std']:.3f} "
                f"success={metrics['success_rate']:.3f}",
                flush=True,
            )

if __name__ == "__main__":
    main()

