import os
import argparse
from rl.trainer import Trainer

def make_env(level="rtfm:groups_nl-v0", **kwargs):
    import gym
    import rtfm.tasks  # ⚠️ 必须 import，否则注册不会触发

    env = gym.make(level, **kwargs)
    return env

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=str, default="rtfm:groups_nl-v0")
    ap.add_argument("--text-encoder", choices=["bilstm", "minilm"], default="minilm")
    ap.add_argument("--mod-mode", choices=["film", "xattn"], default="film")
    ap.add_argument("--reward-mode", choices=["attn", "attn_diff", "align", "align_delta"], default="attn")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    args = ap.parse_args()

    env = make_env(args.level)

    trainer = Trainer(
        env=env,
        text_encoder=args.text_encoder,
        mod_mode=args.mod_mode,
        reward_mode=args.reward_mode,
        hidden=args.hidden,
        heads=args.heads,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        device=args.device,
    )
    trainer.train(epochs=args.epochs)

if __name__ == "__main__":
    main()
