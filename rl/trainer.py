from typing import Dict, List, Optional
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import sys

from .parsers import parse_instructions
from .encoders import Vocab
from .agent import Agent
from .utils import build_vocab_from_env, obs_to_text
from .reward_heads import RewardModel, DistanceModel
from .replay_buffer import RewardReplayBuffer


# ---------------- Reward 计算（仅做 attention / align 辅助项） ----------------
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
        r = F.cosine_similarity(h_s, H_g.mean(dim=1), dim=-1).mean()
        return env_r + lam * r
    else:
        return env_r


# ---------------- On-policy Rollout Buffer（给 PPO 用） ----------------
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
        # ---- 新增：reward / distance 设置 ----
        rm_hidden=256,
        dm_hidden=256,
        lambda_dist=0.01,          # r_dist 的权重
        rm_loss_coef=1.0,          # reward model loss 权重
        dm_loss_coef=0.1,          # distance model loss 权重
        her_coef=1.0,              # HER 强度
        rm_buffer_capacity=100000, # replay buffer 容量
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

        self.lambda_dist = lambda_dist
        self.rm_loss_coef = rm_loss_coef
        self.dm_loss_coef = dm_loss_coef

        # vocab for BiLSTM
        self.vocab = None
        if text_encoder == "bilstm":
            self.vocab = build_vocab_from_env(env)

        if hasattr(env.action_space, "n"):
            n_actions = env.action_space.n
        else:
            import rtfm.dynamics.monster as M
            n_actions = len(M.QueuedAgent.valid_moves)
        self.n_actions = n_actions

        # 主 Agent
        self.agent = Agent(
            text_encoder=text_encoder,
            hidden=hidden,
            mod_mode=mod_mode,
            heads=heads,
            n_actions=n_actions,
            vocab=self.vocab,
        ).to(device)

        # PPO optimizer
        self.opt = torch.optim.Adam(self.agent.parameters(), lr=lr)

        # On-policy buffer
        self.buffer = RolloutBuffer()

        # Reward model & Distance model
        self.reward_model = RewardModel(
            state_dim=hidden,
            goal_dim=hidden,
            n_actions=n_actions,
            hidden_dim=rm_hidden,
        ).to(device)

        self.distance_model = DistanceModel(
            state_dim=hidden,
            goal_dim=hidden,
            hidden_dim=dm_hidden,
        ).to(device)

        self.aux_opt = torch.optim.Adam(
            list(self.reward_model.parameters()) +
            list(self.distance_model.parameters()),
            lr=lr,
        )

        # Off-policy replay buffer for RM/DM
        self.rm_buffer = RewardReplayBuffer(
            capacity=rm_buffer_capacity,
            state_dim=hidden,
            goal_dim=hidden,
            her_coef=her_coef,
        )

    # --------- 采样一个 step ----------
    def _step_env(self, obs, prev_A_goal=None):
        # 1) 当前观测 -> 文本
        state_text = obs_to_text(obs)
        instructions = obs.get("instructions", "") or obs.get("manual", "")
        parts = parse_instructions(instructions)
        tips, goals = parts["tips"], parts["goals"]
        if len(goals) == 0:
            goals = ["(no goal)."]
        if len(tips) == 0:
            tips = ["(no tip)."]

        # 2) 前向，拿到各种 embedding
        out = self.agent(state_text, goals, tips, return_attn=True)
        logits, value, A_goal, A_tip, z_goal, z_t, h_s, H_g = out
        probs = F.softmax(logits, dim=-1)
        dist  = torch.distributions.Categorical(probs=probs)
        action = dist.sample().item()
        logp   = dist.log_prob(torch.tensor(action)).item()
        value  = value.item()

        # 当前步的全局 goal 向量（句子平均）
        h_g_bar = H_g.mean(dim=1)  # (1, D)

        # 3) 与环境交互
        out = self.env.step(action)
        if len(out) == 5:
            # Gymnasium: (obs, reward, terminated, truncated, info)
            obs2, env_r, terminated, truncated, info = out
            done = float(terminated or truncated)
        elif len(out) == 4:
            # RTFM: (obs, reward, done, truncated)
            obs2, env_r, done, truncated = out
            done = float(done or truncated)
            info = {}
        else:
            # 旧 Gym: (obs, reward, done, info)
            obs2, env_r, done, info = out
            done = float(done)

        # 4) 辅助 reward（只用 attention / align 部分，env_r=0）
        aux_r = compute_reward(
            self.reward_mode,
            env_r=0.0,
            A_goal=A_goal,
            prev_A_goal=prev_A_goal,
            h_s=h_s,
            H_g=H_g,
            lam=self.lam,
        ).to(self.device)  # 标量 tensor

        # 5) 下一 state 的 embedding h_s^{t+1}
        state_text2 = obs_to_text(obs2)
        instructions2 = obs2.get("instructions", "") or obs2.get("manual", "")
        parts2 = parse_instructions(instructions2)
        tips2  = parts2["tips"]  or ["(no tip)."]
        goals2 = parts2["goals"] or ["(no goal)."]

        with torch.no_grad():
            _, _, _, _, _, _, h_s_next, _ = self.agent(
                state_text2, goals2, tips2, return_attn=True
            )

            a_tensor = torch.tensor([action], dtype=torch.long, device=self.device)

            # Reward model 输出 r_t
            r_pred = self.reward_model(
                h_s, a_tensor, h_s_next, h_g_bar, z_goal
            ).squeeze(0)

            # Distance model 输出 d(h_s, h_g_bar)
            d_pred = self.distance_model(h_s, h_g_bar).squeeze(0)
            r_dist = -d_pred

            # 总 reward
            r_mod = r_pred + aux_r + self.lambda_dist * r_dist

        R = float(r_mod.item())

        # 给 RM/DM buffer 用的记录
        step_record = {
            "h_s": h_s.squeeze(0),
            "h_s_next": h_s_next.squeeze(0),
            "z_goal": z_goal.squeeze(0),
            "h_g_bar": h_g_bar.squeeze(0),
            "action": action,
            "env_r": float(env_r),
        }

        return obs2, action, logp, value, R, done, A_goal, step_record

    # --------- 一次 rollout ----------
    def collect_rollout(self):
        self.buffer.clear()
        out = self.env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        prev_A_goal = None

        current_traj = []

        for _ in range(self.rollout_steps):
            obs2, act, logp, val, rew, done, A_goal, step_rec = \
                self._step_env(obs, prev_A_goal)

            # 给 PPO buffer
            instr = obs.get("instructions", "") or obs.get("manual", "")
            parts = parse_instructions(instr)
            tips = parts["tips"]
            goals = parts["goals"] or ["(no goal)."]

            self.buffer.add(
                state_text=obs_to_text(obs),
                tip_texts=tips if len(tips) > 0 else ["(no tip)."],
                goal_texts=goals,
                action=act,
                logp=logp,
                value=val,
                reward=rew,
                done=done,
            )

            # 给 RM/DM 的 traj buffer
            current_traj.append(step_rec)

            obs = obs2
            prev_A_goal = A_goal.detach()

            if done:
                # episode 结束，写入 RM buffer
                self.rm_buffer.add_trajectory(current_traj)
                current_traj = []

                out = self.env.reset()
                obs = out[0] if isinstance(out, tuple) else out
                prev_A_goal = None

        # rollout 结束但 episode 还没结束，也写入一次
        if current_traj:
            self.rm_buffer.add_trajectory(current_traj)

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
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns

    # --------- Reward / Distance 模型更新 ----------
    def _update_aux(self):
        batch = self.rm_buffer.sample(self.minibatch_size, device=self.device)
        if batch is None:
            return {"reward_loss": 0.0, "distance_loss": 0.0}

        h_s = batch["h_s"]
        h_s_next = batch["h_s_next"]
        z_goal = batch["z_goal"]
        h_g_bar = batch["h_g_bar"]
        action = batch["action"]
        r_target = batch["r_target"]
        d_target = batch["d_target"]

        r_pred = self.reward_model(h_s, action, h_s_next, h_g_bar, z_goal)
        d_pred = self.distance_model(h_s, h_g_bar)

        reward_loss = F.mse_loss(r_pred, r_target)
        distance_loss = F.mse_loss(d_pred, d_target)

        loss = self.rm_loss_coef * reward_loss + self.dm_loss_coef * distance_loss

        self.aux_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.reward_model.parameters()) +
            list(self.distance_model.parameters()),
            0.5,
        )
        self.aux_opt.step()

        return {
            "reward_loss": float(reward_loss.detach().cpu()),
            "distance_loss": float(distance_loss.detach().cpu()),
        }

    # --------- PPO 更新 ----------
    def update(self):
        states, tips, goals, actions, old_logps, values, rewards, dones = \
            self.buffer.to_tensors(self.device)

        next_value = values[-1].detach()
        advantages, returns = self._compute_gae(rewards, values, dones, next_value)

        idx = torch.randperm(len(actions))
        for _ in range(self.update_epochs):
            for start in range(0, len(actions), self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]

                mb_logits, mb_values = [], []
                mb_logps_new = []
                for i in mb.tolist():
                    st = states[i]
                    tp = tips[i]
                    gl = goals[i]
                    logits, value = self.agent(
                        st,
                        gl if len(gl) > 0 else ["(no goal)."],
                        tp if len(tp) > 0 else ["(no tip)."],
                    )
                    probs = F.softmax(logits, dim=-1)
                    dist  = torch.distributions.Categorical(probs=probs)
                    mb_logps_new.append(dist.log_prob(actions[i]).unsqueeze(0))
                    mb_logits.append(logits)
                    mb_values.append(value)

                mb_logps_new = torch.cat(mb_logps_new, dim=0).to(self.device)
                mb_values    = torch.cat(mb_values, dim=0).to(self.device)
                mb_oldlogps  = old_logps[mb]
                mb_adv       = advantages[mb]
                mb_ret       = returns[mb]

                ratio = torch.exp(mb_logps_new - mb_oldlogps)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                ) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(mb_values, mb_ret)

                with torch.no_grad():
                    p = F.softmax(torch.cat(mb_logits, dim=0), dim=-1)
                    ent = -(p * (p.clamp_min(1e-8).log())).sum(dim=-1).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                self.opt.step()

        stats = {
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(ent.detach().cpu()),
            "return_mean": float(returns.mean().cpu()),
        }

        aux_stats = self._update_aux()
        stats.update(aux_stats)
        return stats

    # --------- Evaluate（用 r_mod，而不是纯 env_r） ----------
    def evaluate(self, episodes=5):
        self.agent.eval()
        total_ret, total_win = 0.0, 0

        for _ in range(episodes):
            out = self.env.reset()
            obs = out[0] if isinstance(out, tuple) else out
            done = False
            ep_ret = 0.0
            prev_A_goal = None

            while not done:
                state_text = obs_to_text(obs)
                instructions = obs.get("instructions", "") or obs.get("manual", "")
                parts = parse_instructions(instructions)
                tips  = parts["tips"]  or ["(no tip)."]
                goals = parts["goals"] or ["(no goal)."]

                with torch.no_grad():
                    logits, value, A_goal, A_tip, z_goal, z_t, h_s, H_g = \
                        self.agent(state_text, goals, tips, return_attn=True)
                    probs = torch.softmax(logits, dim=-1)
                    action = probs.argmax(dim=-1).item()
                    h_g_bar = H_g.mean(dim=1)

                out = self.env.step(action)
                if len(out) == 5:
                    obs2, env_r, term, trunc, _ = out
                    done = term or trunc
                elif len(out) == 4:
                    obs2, env_r, done, trunc = out
                    done = done or trunc
                else:
                    obs2, env_r, done, _ = out

                # 下一 state embedding
                state_text2 = obs_to_text(obs2)
                instructions2 = obs2.get("instructions", "") or obs2.get("manual", "")
                parts2 = parse_instructions(instructions2)
                tips2  = parts2["tips"]  or ["(no tip)."]
                goals2 = parts2["goals"] or ["(no goal)."]

                with torch.no_grad():
                    _, _, _, _, _, _, h_s_next, _ = self.agent(
                        state_text2, goals2, tips2, return_attn=True
                    )
                    aux_r = compute_reward(
                        self.reward_mode,
                        env_r=0.0,
                        A_goal=A_goal,
                        prev_A_goal=prev_A_goal,
                        h_s=h_s,
                        H_g=H_g,
                        lam=self.lam,
                    ).to(self.device)
                    a_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
                    r_pred = self.reward_model(
                        h_s, a_tensor, h_s_next, h_g_bar, z_goal
                    ).squeeze(0)
                    d_pred = self.distance_model(h_s, h_g_bar).squeeze(0)
                    r_dist = -d_pred
                    R = r_pred + aux_r + self.lambda_dist * r_dist

                ep_ret += R.item()
                prev_A_goal = A_goal
                obs = obs2

            total_ret += ep_ret
            total_win += (ep_ret > 0)

        self.agent.train()
        return total_ret / episodes, total_win / episodes

    # --------- 训练主循环 ----------
    def train(self, epochs=1000, test_interval=5, test_episodes=5):
        for e in tqdm(
            range(1, epochs + 1),
            dynamic_ncols=True,
            disable=not sys.stdout.isatty(),
        ):
            self.collect_rollout()
            stats = self.update()

            if e % test_interval == 0:
                print(
                    f"[PPO Ep {e}] pi={stats['policy_loss']:.3f} "
                    f"v={stats['value_loss']:.3f} "
                    f"H={stats['entropy']:.3f} "
                    f"ret={stats['return_mean']:.2f} "
                    f"rL={stats['reward_loss']:.3f} "
                    f"dL={stats['distance_loss']:.3f}",
                    flush=True,
                )

                test_ret, win_rate = self.evaluate(test_episodes)
                print(
                    f"[Eval] avg_return={test_ret:.2f}, win_rate={win_rate:.2f}",
                    flush=True,
                )

