# dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os
from env import CarAvoidEnv

# Hyperparams
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 5000
MIN_REPLAY = 500
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TRAIN_EPISODES = 1000  # increase for better performance
TARGET_UPDATE = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)
    def push(self, s,a,r,ns,d):
        self.buf.append((s,a,r,ns,d))
    def sample(self, n):
        batch = random.sample(self.buf, n)
        s,a,r,ns,d = map(np.array, zip(*batch))
        return s, a, r, ns, d
    def __len__(self):
        return len(self.buf)

def train():
    env = CarAvoidEnv(lanes=5, npc_count=3, archetype="aggressive", max_steps=200)
    n_actions = env.action_space()
    state_dim = env.observation_space_dim()

    policy_net = Net(state_dim, n_actions).to(device)
    target_net = Net(state_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)

    eps = EPS_START
    best_score = -1e9
    os.makedirs("models", exist_ok=True)

    for ep in range(1, TRAIN_EPISODES + 1):
        state = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            # ε-greedy
            if random.random() < eps:
                action = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    q_vals = policy_net(s_t)
                    action = int(torch.argmax(q_vals).item())

            next_state, reward, done, info = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            # learn
            if len(buffer) >= MIN_REPLAY:
                s_batch, a_batch, r_batch, ns_batch, d_batch = buffer.sample(BATCH_SIZE)
                s_tensor = torch.tensor(s_batch, dtype=torch.float32).to(device)
                a_tensor = torch.tensor(a_batch, dtype=torch.int64).unsqueeze(1).to(device)
                r_tensor = torch.tensor(r_batch, dtype=torch.float32).unsqueeze(1).to(device)
                ns_tensor = torch.tensor(ns_batch, dtype=torch.float32).to(device)
                d_tensor = torch.tensor(d_batch, dtype=torch.float32).unsqueeze(1).to(device)

                q_values = policy_net(s_tensor).gather(1, a_tensor)
                with torch.no_grad():
                    next_q = target_net(ns_tensor).max(1)[0].unsqueeze(1)
                    target = r_tensor + GAMMA * next_q * (1 - d_tensor)

                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # decay eps
        eps = max(EPS_END, eps * EPS_DECAY)

        # update target
        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # logging
        if ep % 10 == 0:
            print(f"Ep {ep}/{TRAIN_EPISODES} reward={episode_reward:.3f} eps={eps:.3f}")

        # save model
        if episode_reward > best_score:
            best_score = episode_reward
            torch.save(policy_net.state_dict(), "models/dqn_agent.pth")

    print("Training finished. Model saved to models/dqn_agent.pth")

if __name__ == "__main__":
    train()
