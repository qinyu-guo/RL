import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Categorical
from collections import deque

# ---- 超参数 ----
LR = 3e-4          # 学习率
GAMMA = 0.99       # 折扣因子
LAMBDA = 0.95      # GAE 参数
EPS_CLIP = 0.2     # PPO 裁剪范围
K_EPOCHS = 4       # 每轮数据训练几遍
T_HORIZON = 2048    # 每次采样多少步
BATCH = 64         # 小批量大小
HIDDEN = 64

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
        )
        self.pi = nn.Linear(HIDDEN, act_dim)   # Actor：输出各动作 logit
        self.v  = nn.Linear(HIDDEN, 1)         # Critic：输出状态价值

    def forward(self, x):
        h = self.fc(x)
        return self.pi(h), self.v(h)

def compute_gae(rewards, values, dones, next_value):
    advantages, gae = [], 0.0
    for t in reversed(range(len(rewards))):
        nv = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + GAMMA * nv * (1 - dones[t]) - values[t]
        gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns

def train():
    env = gym.make("CartPole-v1")
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    recent = deque(maxlen=20)   # 记录最近 20 局奖励，算平均
    ep_reward, update = 0.0, 0
    s, _ = env.reset()

    while update < 1500:
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        # ---- 采样：让当前策略跟环境交互 2048 步 ----
        for _ in range(T_HORIZON):
            logits, v = model(torch.FloatTensor(s))
            dist = Categorical(logits=logits)
            a = dist.sample()
            s_next, r, terminated, truncated, _ = env.step(a.item())
            done = terminated or truncated

            states.append(s); actions.append(a); rewards.append(r)
            dones.append(done); log_probs.append(dist.log_prob(a)); values.append(v)

            ep_reward += r
            s = s_next
            if done:
                recent.append(ep_reward); ep_reward = 0.0
                s, _ = env.reset()

        # ---- 算优势（GAE）----
        with torch.no_grad():
            _, next_v = model(torch.FloatTensor(s))
        advantages, returns = compute_gae(rewards, [v.item() for v in values], dones, next_v.item())

        # ---- 转 tensor + 归一化优势 ----
        states = torch.FloatTensor(np.array(states))
        actions = torch.stack(actions)
        old_log_probs = torch.stack(log_probs).detach()
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- PPO 更新（核心：裁剪目标）----
        for _ in range(K_EPOCHS):
            for i in range(0, T_HORIZON, BATCH):
                b = slice(i, i + BATCH)
                logits, v = model(states[b])
                dist = Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(actions[b]) - old_log_probs[b])
                adv = advantages[b]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * adv
                loss_pi = -torch.min(surr1, surr2).mean()
                loss_v = F.mse_loss(v.squeeze(), returns[b])
                loss = loss_pi + 0.5 * loss_v - 0.01 * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); opt.step()

        update += 1
        if update % 10 == 0:
            avg = np.mean(recent) if recent else 0.0
            print(f"update {update:3d} | 近20局平均奖励 {avg:6.1f}")

    env.close()

if __name__ == "__main__":
    train()