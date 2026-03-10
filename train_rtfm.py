import argparse
import os
import random

import numpy as np
import torch

from rl.buffers import BLBuffer, HLBuffer
from rl.config import Config
from rl.eval import evaluate
from rl.ll_policy import SharedExtractor
from rl.minilm import Adapter, FrozenMiniLM
from rl.reward_model import RewardModel
from rl.sb3_wrap import HRLWrapper
from rl.selector import PiSel
from rl.state_encoder import StateZExtractor, build_state_encoder

os.makedirs("checkpoints", exist_ok=True)

def _unwrap_env(env):
    """
    RTFM / gym wrappers may sometimes return nested list/tuple containers.
    Keep unwrapping until we get the actual env object.
    """
    while isinstance(env, (list, tuple)):
        if len(env) == 0:
            raise RuntimeError("Environment factory returned an empty container.")
        env = env[0]
    return env


def make_env(
    level: str,
    room_shape: int,
    partially_observable: bool,
    max_placement: int,
    shuffle_wiki: bool,
    time_penalty: float,
):
    import gym
    import rtfm.tasks
    from rtfm import featurizer as X

    feat = X.Concat([
        X.Text(),
        X.ValidMoves(),
        X.RelativePosition(),
        X.Progress(),
    ])

    env = gym.make(
        level,
        room_shape=(room_shape, room_shape),
        partially_observable=partially_observable,
        max_placement=max_placement,
        featurizer=feat,
        shuffle_wiki=shuffle_wiki,
        time_penalty=time_penalty,
    )
    env = _unwrap_env(env)
    return env


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--level", type=str, default="rtfm:groups_nl-v0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--room-shape", type=int, default=6)
    ap.add_argument("--partially-observable", action="store_true")
    ap.add_argument("--max-placement", type=int, default=1)
    ap.add_argument("--shuffle-wiki", action="store_true")
    ap.add_argument("--time-penalty", type=float, default=0.0)

    ap.add_argument("--split-mode", choices=["parser", "lm"], default="lm")
    ap.add_argument("--max-instructions", type=int, default=64)

    ap.add_argument("--state-encoder-type", choices=["mlp", "conv", "film", "txt2pi"], default="conv")
    ap.add_argument("--selector-mode", choices=["hard", "sample"], default="hard")

    ap.add_argument("--hl-T", type=int, default=5)
    ap.add_argument("--hl-gamma", type=float, default=0.99)
    ap.add_argument("--hl-update-every-steps", type=int, default=1000)
    ap.add_argument("--hl-return-source", choices=["rm", "env"], default="rm")
    ap.add_argument("--hl-aux-type", choices=["none", "cos", "v_diff"], default="cos")
    ap.add_argument("--hl-aux-lambda", type=float, default=1.0)

    ap.add_argument("--ll-algo", choices=["ppo"], default="ppo")
    ap.add_argument("--ll-reward", choices=["env", "rm", "mix"], default="mix")
    ap.add_argument("--ll-lambda", type=float, default=0.1)
    ap.add_argument("--ll-update-every-steps", type=int, default=1024)

    ap.add_argument("--xi-h", type=float, default=1.0)
    ap.add_argument("--xi-l", type=float, default=1.0)

    ap.add_argument("--rm-variant", choices=["sa", "sas", "sasz"], default="sas")

    ap.add_argument("--state-dim", type=int, default=256)
    ap.add_argument("--instr-dim", type=int, default=256)
    ap.add_argument("--token-emb-dim", type=int, default=64)
    ap.add_argument("--text-rnn-dim", type=int, default=64)
    ap.add_argument("--state-hidden", type=int, default=128)
    ap.add_argument("--adapter-hidden", type=int, default=256)
    ap.add_argument("--rm-hidden", type=int, default=256)

    ap.add_argument("--total-steps", type=int, default=1000000)
    ap.add_argument("--eval-every-steps", type=int, default=5000)
    ap.add_argument("--eval-episodes", type=int, default=20)

    return ap.parse_args()


def main():
    args = parse_args()

    cfg = Config(
        level=args.level,
        seed=args.seed,
        device=args.device,
        room_shape=args.room_shape,
        partially_observable=args.partially_observable,
        max_placement=args.max_placement,
        shuffle_wiki=args.shuffle_wiki,
        time_penalty=args.time_penalty,
        split_mode=args.split_mode,
        max_instructions=args.max_instructions,
        state_encoder_type=args.state_encoder_type,
        selector_mode=args.selector_mode,
        hl_T=args.hl_T,
        hl_gamma=args.hl_gamma,
        hl_update_every_steps=args.hl_update_every_steps,
        hl_return_source=args.hl_return_source,
        hl_aux_type=args.hl_aux_type,
        hl_aux_lambda=args.hl_aux_lambda,
        ll_algo=args.ll_algo,
        ll_reward=args.ll_reward,
        ll_lambda=args.ll_lambda,
        ll_update_every_steps=args.ll_update_every_steps,
        xi_H=args.xi_h,
        xi_L=args.xi_l,
        rm_variant=args.rm_variant,
        state_dim=args.state_dim,
        instr_dim=args.instr_dim,
        token_emb_dim=args.token_emb_dim,
        text_rnn_dim=args.text_rnn_dim,
        state_hidden=args.state_hidden,
        adapter_hidden=args.adapter_hidden,
        rm_hidden=args.rm_hidden,
        total_steps=args.total_steps,
        eval_every_steps=args.eval_every_steps,
        eval_episodes=args.eval_episodes,
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Build base env once for metadata / spaces / vocab
    base_env = make_env(
        cfg.level,
        cfg.room_shape,
        cfg.partially_observable,
        cfg.max_placement,
        cfg.shuffle_wiki,
        cfg.time_penalty,
    )

    print("base_env type:", type(base_env), flush=True)

    sample_obs = base_env.reset()
    if isinstance(sample_obs, tuple):
        sample_obs = sample_obs[0]
    sample_obs = _unwrap_env(sample_obs)

    vocab = getattr(base_env, "vocab", None)
    if vocab is None:
        raise RuntimeError("RTFM env did not expose env.vocab, cannot build token-based state encoder.")

    vocab_size = len(vocab)
    padding_idx = vocab.word2index("pad") if hasattr(vocab, "word2index") else 0

    action_space = getattr(base_env, "action_space", None)
    if action_space is None:
        raise RuntimeError(f"base_env has no action_space. type(base_env)={type(base_env)}")
    
    if hasattr(action_space, "n"):
        action_dim = int(action_space.n)
    elif isinstance(action_space, (list, tuple)):
        action_dim = len(action_space)
    else:
        raise RuntimeError(
            f"Unsupported action_space type {type(action_space)}: {action_space}"
        )

    minilm = FrozenMiniLM(cfg.minilm_name, device=cfg.device)
    instr_adapter = Adapter(cfg.emb_dim, cfg.instr_dim, hidden=cfg.adapter_hidden).to(cfg.device)

    state_encoder = build_state_encoder(
        cfg.state_encoder_type,
        vocab_size=vocab_size,
        out_dim=cfg.state_dim,
        token_emb_dim=cfg.token_emb_dim,
        text_rnn_dim=cfg.text_rnn_dim,
        hidden_dim=cfg.state_hidden,
        padding_idx=padding_idx,
    ).to(cfg.device)

    selector = PiSel(d=cfg.instr_dim, hidden=cfg.adapter_hidden).to(cfg.device)

    reward_model = RewardModel(
        cfg.state_dim,
        action_dim,
        hidden=cfg.rm_hidden,
        z_dim=cfg.instr_dim,
        rm_variant=cfg.rm_variant,
    ).to(cfg.device)

    hrl_env = HRLWrapper(
        env=base_env,
        cfg=cfg,
        minilm=minilm,
        instr_adapter=instr_adapter,
        state_encoder=state_encoder,
        selector=selector,
        reward_model=reward_model,
        action_dim=action_dim,
    )

    first_wrapped = hrl_env.reset()
    if isinstance(first_wrapped, tuple):
        first_wrapped = first_wrapped[0]
    first_wrapped = _unwrap_env(first_wrapped)

    raw_obs_dict = {}
    for k, v in first_wrapped.items():
        if k == "z":
            continue
        raw_obs_dict[k] = torch.as_tensor(v)

    hrl_env.observation_space = state_encoder.make_observation_space(
        raw_obs_dict,
        cfg.instr_dim,
    )

    from stable_baselines3 import PPO

    state_z_extractor = StateZExtractor(
        state_encoder=state_encoder,
        z_dim=cfg.instr_dim,
        xi_L=cfg.xi_L,
    )

    policy_kwargs = dict(
        features_extractor_class=SharedExtractor,
        features_extractor_kwargs=dict(state_z_extractor=state_z_extractor),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    ll_model = PPO(
        "MultiInputPolicy",
        hrl_env,
        verbose=1,
        seed=cfg.seed,
        learning_rate=cfg.ll_lr,
        n_steps=cfg.ll_update_every_steps,
        gamma=cfg.ll_gamma,
        policy_kwargs=policy_kwargs,
        device=cfg.device,
    )

    from rl.trainer import Trainer

    trainer = Trainer(
        cfg=cfg,
        ll_model=ll_model,
        selector=selector,
        state_encoder=state_encoder,
        instr_adapter=instr_adapter,
        reward_model=reward_model,
        bl_buffer=BLBuffer(cfg.rm_buffer_capacity),
        hl_buffer=HLBuffer(),
        minilm=minilm,
    )

    steps_done = 0
    while steps_done < cfg.total_steps:
        chunk = min(cfg.eval_every_steps, cfg.total_steps - steps_done)
        trainer.train(chunk)
        steps_done += chunk

        metrics = evaluate(ll_model, hrl_env, n_episodes=cfg.eval_episodes)
        print(
            f"[steps={steps_done}] "
            f"return={metrics.get('return_mean', 0.0):.3f} "
            f"success={metrics.get('success_rate', 0.0):.3f}",
            flush=True,
        )
        ll_model.save(f"checkpoints/ll_model_{steps_done}.zip")

        torch.save(
            {
                "selector": selector.state_dict(),
                "instr_adapter": instr_adapter.state_dict(),
                "state_encoder": state_encoder.state_dict(),
                "reward_model": reward_model.state_dict(),
                "opt_h": trainer.opt_h.state_dict(),
                "opt_rm": trainer.opt_rm.state_dict(),
                "steps_done": steps_done,
                "cfg": vars(args),
            },
            f"checkpoints/hrl_modules_{steps_done}.pt",
        )

    final_metrics = evaluate(ll_model, hrl_env, n_episodes=cfg.eval_episodes)
    print(final_metrics, flush=True)


if __name__ == "__main__":
    main()
