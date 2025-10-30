from typing import Dict, List, Optional
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .parsers import parse_instructions
from .encoders import Vocab
from .agent import Agent
from .utils import build_vocab_from_env, obs_to_text

# ---------------- Reward 计算 ----------------
def compute_reward(
    mode: str,
    env_r: float,
    A_goal: Optional[torch.Tensor] = None,
    prev_A_goal: Optional[torch.Tensor] = None,
    h_s: Optional[torch.Tensor] = None,
    H_g: Optional[torch.Tensor] = None,
    lam: float = 0.1,
) -> torch.Tensor:
    env_r = torch.as_tensor(env_r).float()
    if mode == "attn" and A_goal is not None:
        p = A_goal.clamp_min(1e-8)
        r = -(p * p.log()).sum()
        return env_r + lam * r
    elif mode == "attn_diff" and A_goal is not None and prev_A_goal is not None:
        r = (A_goal - prev_A_goal).abs().sum()
        return env_r + lam * r
    elif mode == "align" and h_s is not None and H_g is not None:
        r = F.cosine_similarity(h_s, H_g.mean(dim=1), dim=-1).mean()
        return env_r + lam * r
    elif mode == "align_delta" and h_s is not None and H_g is not None and prev_A_goal is not None:
        # 这里简化为和上一步注意力权重的相关调整；如需真正 delta( cos ) 可在外部维护历史 cos
        r = F.cosine_similarity(h_s, H_g.mean(dim=1), dim=-1).mean()
        return env_r + lam * r
    else:
        return env_r

# ---------------- Rollout Buffer ----------------
@dataclass
class StepData:
    state_text: List[str]
    tip_texts: List[str]
    goal_texts: List[str]
    action: int
    logp: float
    value: float
    reward: float
    done: float

class RolloutBuffer:
    def __init__(self):
        self.data: List[StepData] = []

    def add(self, **kwargs):
        self.data.append(StepData(**kwargs))

    def clear(self):
        self.data.clear()

    def to_tensors(self, device):
        actions = torch.tensor([d.action for d in self.data], dtype=torch.long, device=device)
        logps   = torch.tensor([d.logp   for d in self.data], dtype=torch.float, device=device)
        values  = torch.tensor([d.value  for d in self.data], dtype=torch.float, device=device)
        rewards = torch.tensor([d.reward for d in self.data], dtype=torch.float, device=device)
        dones   = torch.tensor([d.done   for d in self.data], dtype=torch.float, device=device)
        # 文本保留原样（逐条前向）
        states  = [d.state_text for d in self.data]
        tips    = [d.tip_texts  for d in self.data]
        goals   = [d.goal_texts for d in self.data]
        return states, tips, goals, actions, logps, values, rewards, dones

# ---------------- PPO Trainer ----------------
class Trainer:
    def __init__(
        self,
        env,
        text_encoder="bilstm",
        mod_mode="film",
        reward_mode="attn",
        hidden=512,
        heads=8,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        lam=0.1,
        rollout_steps=256,
        update_epochs=4,
        minibatch_size=64,
        device="cpu",
    ):
        self.env = env
        self.device = device

        self.reward_mode = reward_mode
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.lam = lam
        self.rollout_steps = rollout_steps
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

        # vocab for BiLSTM
        self.vocab = None
        if text_encoder == "bilstm":
            self.vocab = build_vocab_from_env(env)

        if hasattr(env.action_space, "n"):
            n_actions = env.action_space.n
        else:
            import rtfm.dynamics.monster as M
            n_actions = len(M.QueuedAgent.valid_moves)

        self.agent = Agent(
            text_encoder=text_encoder,
            hidden=hidden,
            mod_mode=mod_mode,
            heads=heads,
            n_actions=n_actions,
            vocab=self.vocab,
        ).to(device)

        self.opt = torch.optim.Adam(self.agent.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    # --------- 采样一个 step ----------
    def _step_env(self, obs, prev_A_goal=None):
        # 1) 取文本
        state_text = obs_to_text(obs)
        instructions = obs.get("instructions", "") or obs.get("manual", "")
        parts = parse_instructions(instructions)
        tips, goals = parts["tips"], parts["goals"]
        if len(goals) == 0: goals = ["(no goal)."]
        if len(tips)  == 0: tips  = ["(no tip)."]

        # 2) 前向
        out = self.agent(state_text, goals, tips, return_attn=True)
        logits, value, A_goal, A_tip, z_goal, z_t, h_s, H_g = out
        probs = F.softmax(logits, dim=-1)
        dist  = torch.distributions.Categorical(probs=probs)
        action = dist.sample().item()
        logp   = dist.log_prob(torch.tensor(action)).item()
        value  = value.item()

        # 3) 与环境交互
        out = self.env.step(action)
        if len(out) == 5:
            # Gymnasium 格式: (obs, reward, terminated, truncated, info)
            obs2, env_r, terminated, truncated, info = out
            done = float(terminated or truncated)
        elif len(out) == 4:
            # RTFM 格式: (obs, reward, done, truncated)
            obs2, env_r, done, truncated = out
            done = float(done or truncated)
            info = {}
        else:
            # 极端旧版 Gym: (obs, reward, done, info)
            obs2, env_r, done, info = out
            done = float(done)

        # 4) shaped reward
        R = compute_reward(
            self.reward_mode, env_r,
            A_goal=A_goal, prev_A_goal=prev_A_goal,
            h_s=h_s, H_g=H_g, lam=self.lam
        ).item()

        return obs2, action, logp, value, R, done, A_goal

    # --------- 一次 rollout ----------
    def collect_rollout(self):
        self.buffer.clear()
        out = self.env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        prev_A_goal = None
        for _ in range(self.rollout_steps):
            obs2, act, logp, val, rew, done, A_goal = self._step_env(obs, prev_A_goal)
            self.buffer.add(
                state_text=obs_to_text(obs),
                tip_texts=parse_instructions(obs.get("instructions","") or obs.get("manual",""))["tips"],
                goal_texts=parse_instructions(obs.get("instructions","") or obs.get("manual",""))["goals"] or ["(no goal)."],
                action=act, logp=logp, value=val, reward=rew, done=done
            )
            obs = obs2
            prev_A_goal = A_goal.detach()
            if done:  # 新一局
                out = self.env.reset()
                obs = out[0] if isinstance(out, tuple) else out
                prev_A_goal = None

    # --------- 计算 GAE ----------
    def _compute_gae(self, rewards, values, dones, next_value=0.0):
        T = len(rewards)
        adv = torch.zeros(T, device=self.device)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
            lastgaelam = delta + self.gamma * self.gae_lambda * nonterminal * lastgaelam
            adv[t] = lastgaelam
            next_value = values[t]
        returns = adv + values
        # 标准化优势
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns

    # --------- PPO 更新 ----------
    def update(self):
        states, tips, goals, actions, old_logps, values, rewards, dones = \
            self.buffer.to_tensors(self.device)

        # 需要再次前向得到新的 logp/value（用于 next_value / gae 尾项）
        # 简化处理：用最后一个 step 的 value 作为 next_value（也可取新一帧前向）
        next_value = values[-1].detach()

        advantages, returns = self._compute_gae(rewards, values, dones, next_value)

        # 制作索引打乱
        idx = torch.randperm(len(actions))
        for _ in range(self.update_epochs):
            for start in range(0, len(actions), self.minibatch_size):
                mb = idx[start:start+self.minibatch_size]

                # 小批次重前向
                mb_logits, mb_values = [], []
                mb_logps_new = []
                for i in mb.tolist():
                    # 每条重新 encode（最直接；如需更快可缓存中间向量）
                    st = states[i]; tp = tips[i]; gl = goals[i]
                    logits, value = self.agent(st, gl if len(gl)>0 else ["(no goal)."], tp if len(tp)>0 else ["(no tip)."])
                    probs = F.softmax(logits, dim=-1)
                    dist  = torch.distributions.Categorical(probs=probs)
                    mb_logps_new.append(dist.log_prob(actions[i]).unsqueeze(0))
                    mb_logits.append(logits)
                    mb_values.append(value)

                mb_logps_new = torch.cat(mb_logps_new, dim=0).to(self.device)  # (B,)
                mb_values    = torch.cat(mb_values, dim=0).to(self.device)     # (B,)
                mb_oldlogps  = old_logps[mb]
                mb_adv       = advantages[mb]
                mb_ret       = returns[mb]

                ratio = torch.exp(mb_logps_new - mb_oldlogps)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss  = F.mse_loss(mb_values, mb_ret)

                # policy entropy（用当前分布）
                with torch.no_grad():
                    # 直接用 logit 近似熵：H = -sum p log p
                    p = F.softmax(torch.cat(mb_logits, dim=0), dim=-1)
                    ent = -(p * (p.clamp_min(1e-8).log())).sum(dim=-1).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                self.opt.step()

        # 统计
        return {
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(ent.detach().cpu()),
            "return_mean": float(returns.mean().cpu())
        }

    # --------- 训练主循环 ----------
    '''
    def train(self, epochs=1000):
        for e in tqdm(range(1, epochs+1)):
            self.collect_rollout()
            stats = self.update()
            if e % 5 == 0:
                print(f"[PPO Ep {e}] pi={stats['policy_loss']:.3f} "
                      f"v={stats['value_loss']:.3f} "
                      f"H={stats['entropy']:.3f} "
                      f"ret={stats['return_mean']:.2f}")
    '''

    def evaluate(self, episodes=5):
        """Evaluate current policy performance without gradient updates."""
        self.agent.eval()
        total_ret, total_win = 0.0, 0
    
        for ep in range(episodes):
            out = self.env.reset()
            obs = out[0] if isinstance(out, tuple) else out
            done = False
            ep_ret = 0.0
            prev_A_goal = None
    
            while not done:
                with torch.no_grad():
                    out = self.agent(obs, ["(no goal)."], ["(no tip)."], return_attn=True)
                    logits, value, A_goal, A_tip, z_goal, z_t, h_s, H_g = out
                    probs = torch.softmax(logits, dim=-1)
                    action = probs.argmax(dim=-1).item()  # greedy action
                out = self.env.step(action)
                if len(out) == 5:
                    obs, env_r, term, trunc, _ = out
                    done = term or trunc
                elif len(out) == 4:
                    obs, env_r, done, trunc = out
                    done = done or trunc
                else:
                    obs, env_r, done, _ = out
                R = compute_reward(self.reward_mode, env_r, A_goal=A_goal,
                                   prev_A_goal=prev_A_goal, h_s=h_s, H_g=H_g, lam=self.lam)
                ep_ret += R.item()
                prev_A_goal = A_goal
    
            total_ret += ep_ret
            total_win += (ep_ret > 0)
    
        self.agent.train()  # 切回训练模式
        avg_ret = total_ret / episodes
        win_rate = total_win / episodes
        return avg_ret, win_rate

    def train(self, epochs=1000, test_interval=5, test_episodes=5):
        for e in tqdm(range(1, epochs+1)):
            self.collect_rollout()
            stats = self.update()
    
            # 打印训练损失
            if e % test_interval == 0:
                print(f"[PPO Ep {e}] pi={stats['policy_loss']:.3f} "
                      f"v={stats['value_loss']:.3f} "
                      f"H={stats['entropy']:.3f} "
                      f"ret={stats['return_mean']:.2f}")
    
                # ---- 测试当前 policy ----
                test_ret, win_rate = self.evaluate(test_episodes)
                print(f"[Eval] avg_return={test_ret:.2f}, win_rate={win_rate:.2f}")
    
