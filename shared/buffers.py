# Rollout Buffer Class for PPO Model
#
# Corresponds to transitions collected using the current policy
# Repo Reference for Code:
#   DLR-RM, github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py
#
# Game implementation adapted from:
#   Alan H. Yue, github.com/alanhyue/RL-2048-with-n-tuple-network


import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.filters = []

    def add(self, obs, action, reward, log_prob, value, done, mask):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.filters.append(mask)

    def reset(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.filters = []

    def get(self, gamma=0.99, gae_lambda=0.95):
        # Outputs the returns and GAE
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        last_adv = 0
        last_val = 0

        for t in reversed(range(n)):
            if self.dones[t]:
                last_adv = 0
                last_val = 0

            delta = self.rewards[t] + gamma * last_val * (1 - self.dones[t]) - self.values[t]
            advantages[t] = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_adv
            returns[t] = advantages[t] + self.values[t]

            last_adv = advantages[t]
            last_val = self.values[t]

        obs = torch.FloatTensor(np.array(self.obs)).to(device)
        actions = torch.LongTensor(self.actions).to(device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)
        fil = torch.FloatTensor(np.array(self.filters)).to(device)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return obs, actions, old_log_probs, advantages, returns, fil
