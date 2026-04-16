import random
import time
from collections import deque
from copy import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd



# action encoding
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTIONS = [UP, RIGHT, DOWN, LEFT]


# exceptions for illegal actions and game over conditions
class IllegalAction(Exception):
    pass


class GameOver(Exception):
    pass


# drop empty cells to simplify merge lofgic
def _compress(row):
    # remove empty cells so merge logic only processes tile exponents
    return [x for x in row if x != 0]

# merge a row to the left
def _merge_row(row):
    # remove empty cells
    row = _compress(row)

    # inialize reward
    reward = 0

    # initialize list to hold merged tiles
    merged = []

    # track last unmerged tile
    hold = None

    # iterate over non-empty tiles in row
    while row:
        # pop leftmost tile
        v = row.pop(0)

        # if we do not have a held tile, hold current tile
        if hold is None:
            hold = v

        # equal tiles merge
        elif hold == v:
            # add merge reward
            reward += 2 ** (hold + 1)

            # append merged tile
            merged.append(hold + 1)

            # clear hold
            hold = None

        # no merge
        else:
            # append held tile to merged list
            merged.append(hold)

            # hold current tile
            hold = v

    # append any remaining held tiles to merged list
    if hold is not None:
        merged.append(hold)

    # add zeros to end of merged list
    merged += [0] * (4 - len(merged))
    return reward, merged

# class for the board state
class Board:
    # inialize board
    def __init__(self):
        self.reset()

    # start a new game with two random tiles
    def reset(self):
        self.board = [0] * 16
        self.spawn_tile()
        self.spawn_tile()

    # place a random tile (2 or 4) in an empty cell
    def spawn_tile(self):
        # get list of empty cell indices
        empty = [i for i, v in enumerate(self.board) if v == 0]

        # if no empty cells, raise game over
        if not empty:
            raise GameOver()
        
        # randomly place tile with 90% chance of 2-tile (log2 = 1), 10% chance of 4-tile (log2 = 2)
        self.board[random.choice(empty)] = 2 if np.random.rand() < 0.1 else 1

    # get max tile
    def max_tile(self):
        return max(self.board)

    # convert max tile to actual value
    def max_tile_value(self):
        return 2 ** self.max_tile()

    # check if any moves are possible
    def can_move(self):
        # if any cell is empty, we can move
        if any(v == 0 for v in self.board):
            return True

        # check for adjacent horizontal merges
        for r in range(4):
            for c in range(3):
                i = r * 4 + c
                if self.board[i] == self.board[i + 1]:
                    return True

        # check for adjacent vertical merges
        for r in range(3):
            for c in range(4):
                i = r * 4 + c
                if self.board[i] == self.board[i + 4]:
                    return True

        return False

    # game is over if no legal moves remain
    def is_game_over(self):
        return not self.can_move()

    # rotate board (used for merging)
    def _rotate(self):
        b = np.array(self.board).reshape(4, 4)
        self.board = np.rot90(b, -1).flatten().tolist()

    # merge each row left
    def merge_left(self):
        # initialize reward and new board state
        reward = 0
        new_board = []

        # iterate through each row
        for r in range(4):
            # extract row
            row = self.board[r * 4:(r + 1) * 4]

            # merge row and get reward
            rwd, merged = _merge_row(row)

            # accumulate reward
            reward += rwd

            # append merged row to new board state
            new_board.extend(merged)

        # check if board state changed
        changed = new_board != self.board

        # update board state
        self.board = new_board
        return reward, changed

    # apply one action and return reward
    def act(self, action):

        # if left, merge left
        if action == LEFT:
            reward, changed = self.merge_left()

        # if right, rotate board 180, merge left, rotate back
        elif action == RIGHT:
            self._rotate(); self._rotate()
            reward, changed = self.merge_left()
            self._rotate(); self._rotate()

        # if up, rotate 90, merge, rotate back
        elif action == UP:
            self._rotate(); self._rotate(); self._rotate()
            reward, changed = self.merge_left()
            self._rotate()

        # if down, rotate 270, merge, rotate back
        elif action == DOWN:
            self._rotate()
            reward, changed = self.merge_left()
            self._rotate(); self._rotate(); self._rotate()

        # invalid action
        else:
            raise ValueError()

        # if move did not change board, raise illegal action
        if not changed:
            raise IllegalAction()

        return reward


# gym wrapper
class Game2048Env:
    # initialize environment
    def __init__(self):
        self.board = Board()

    # reset board an observation
    def reset(self):
        self.board.reset()
        return self._obs()

    # convert board state to 4x4 numpy array observation
    def _obs(self):
        return np.array(self.board.board, dtype=np.float32).reshape(4, 4)

    # reward shaping used by this project:
    # - merge score from board.act(action)
    # - +1 living/progress bonus on legal actions
    # - -0.5 penalty for illegal actions (move does not change board)
    def step(self, action):
        try:
            reward = self.board.act(action)
            self.board.spawn_tile()
            done = self.board.is_game_over()

            reward += 1.0

        except IllegalAction:
            reward = -0.5
            done = False

        return self._obs(), reward, done, {}

# injects learnable noise into layers for exploration
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # learnable mean and std for weights and biases
        # epsilon buffers are sampled noise variables
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        # get biases
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        # initialize hyperparameter for noise scale
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # initalize parameters for noise
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)

        # reduce noise for larger layers
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        # scale noise
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        # sample noise for weights and biases
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(eps_out, eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        # if training, inject noise
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon

        # otherwise, use mean params
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)


# implement prioritized experience replay
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, state_shape=(4, 4)):
        self.capacity = capacity
        self.alpha = alpha
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

        # initalize buffers for states, actions, rewards, next states, dones, and priorities
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def __len__(self):
        return self.size

    # add new transition to buffer with max priority
    def add(self, state, action, reward, next_state, done):
        # new transitions added with max priority to make sure sampled at least once
        i = self.pos
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.priorities[i] = self.max_priority

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # sample batch of transtions
    def sample(self, batch_size, beta=0.4):
        # check if buffer empty
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        # compute sampling probabilities from priorities
        prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        probs /= probs.sum()

        # sample indices according to probabilities
        idxs = np.random.choice(self.size, batch_size, p=probs)

        # compute importance sampling weights for bias correction
        weights = (self.size * probs[idxs]) ** (-beta)
        weights /= weights.max()

        # gather sampled batch
        batch = (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )
        return batch, idxs, weights.astype(np.float32)

    # update priorities of sampled transitions based on new loss magnitudes
    def update_priorities(self, idxs, priorities):
        priorities = np.asarray(priorities, dtype=np.float32)
        self.priorities[idxs] = priorities
        self.max_priority = max(self.max_priority, float(priorities.max()))


# implement rainbow dqn
class RainbowDQN(nn.Module):
    def __init__(self, num_actions=4, num_atoms=51):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms

        # convolutional layers to extract features from 4x4 board state
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 2), nn.ReLU(),
        )

        self.feature_dim = 64 * 2 * 2

        # dueling dqn needs v(s) and a(s, a)

        # estimates for v(s)
        self.value_fc1 = NoisyLinear(self.feature_dim, 128)
        self.value_fc2 = NoisyLinear(128, num_atoms)

        # estimates for a(s, a)
        self.adv_fc1 = NoisyLinear(self.feature_dim, 128)
        self.adv_fc2 = NoisyLinear(128, num_actions * num_atoms)

    # reset noise in noisy layers
    def reset_noise(self):
        # resample noise
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    # compute categorical distribution over returns for each action
    def dist(self, x):
        # pass through conv layers and flatten
        x = x.view(-1, 1, 4, 4)
        x = self.conv(x).flatten(1)

        # compute value and advantage streams
        value = torch.relu(self.value_fc1(x))
        value = self.value_fc2(value).view(-1, 1, self.num_atoms)

        advantage = torch.relu(self.adv_fc1(x))
        advantage = self.adv_fc2(advantage).view(-1, self.num_actions, self.num_atoms)

        # dueling aggregation done on logits before softmax.
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        return torch.softmax(q_atoms, dim=2)

    # define forward pass
    def forward(self, x, support):
        # convert dist to expected q-values
        dist = self.dist(x)
        return torch.sum(dist * support, dim=2)

# main agent class
class RainbowAgent:
    def __init__(self, device):
        # define hyperparameters and networks
        self.device = device
        self.num_actions = 4
        self.num_atoms = 51
        self.v_min = -10.0
        self.v_max = 5000.0


        self.gamma = 0.99
        self.n_step = 3
        self.batch_size = 128
        self.target_update = 1000
        self.per_eps = 1e-6

        # support for distributional rl
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms, device=device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        # online and target networks
        self.q = RainbowDQN(num_actions=self.num_actions, num_atoms=self.num_atoms).to(device)
        self.t = RainbowDQN(num_actions=self.num_actions, num_atoms=self.num_atoms).to(device)
        self.t.load_state_dict(self.q.state_dict())

        # define optimzer
        self.opt = optim.Adam(self.q.parameters(), lr=1e-4)

        # replay buffer
        self.replay = PrioritizedReplayBuffer(capacity=50000, alpha=0.6)

        # n-step buffer
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.learn_steps = 0

    # build n-step transition from local buffer, with early termination on done
    def _aggregate_n_step(self):
        reward = 0.0
        next_state = self.n_step_buffer[-1][3]
        done = self.n_step_buffer[-1][4]

        # iterate over n-step buffer and accumulate discounted reward, next state, and done flag
        for i, (_, _, r, s2, d) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** i) * r
            next_state = s2
            done = d
            if d:
                break

        state, action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        return state, action, reward, next_state, done

    # store transition and build n-step transition
    def store(self, state, action, reward, next_state, done):
        # append new transition to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # if enough transitions, aggregate
        if done:
            # if episode ended, aggregate all
            while self.n_step_buffer:
                self.replay.add(*self._aggregate_n_step())
                self.n_step_buffer.popleft()
            return

        # if buffer full, aggregate
        if len(self.n_step_buffer) == self.n_step:
            # Normal rolling behavior for non-terminal trajectories.
            self.replay.add(*self._aggregate_n_step())
            self.n_step_buffer.popleft()

    # epsilon-greedy action selection with exploration from noise
    def act(self, state, eps=0.0, train=True):
        # keep epsilon small because of noisy exploration
        if random.random() < eps:
            return random.randint(0, self.num_actions - 1)

        # select action with highest expected q-value from online network
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.q.train(mode=train)
        if train:
            # resample noise for action selection during training to ensure exploration
            self.q.reset_noise()

        # compute q-values and select action with highest expected return
        with torch.no_grad():
            q_values = self.q(s, self.support)
            return int(q_values.argmax(1).item())

    # main learning step
    def learn(self, beta=0.4):
        # sample batch of transitions
        (s, a, r, s2, d), idxs, weights = self.replay.sample(self.batch_size, beta)

        # convert batch to tensors
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)
        w = torch.tensor(weights, dtype=torch.float32, device=self.device)

        #resample noise
        self.q.reset_noise()
        self.t.reset_noise()

        # get predicted distributions for current states and chosen actions
        dist = self.q.dist(s)
        chosen_dist = dist[torch.arange(self.batch_size, device=self.device), a]
        log_p = torch.log(chosen_dist + 1e-8)

        # compute target distributions using double dqn
        with torch.no_grad():
            # select action from online network
            next_q = self.q(s2, self.support)
            next_action = next_q.argmax(1)

            # get distribution for next state and selected action from target network
            next_dist_all = self.t.dist(s2)
            next_dist = next_dist_all[torch.arange(self.batch_size, device=self.device), next_action]

            # n-step distributional bellman target
            gamma_n = self.gamma ** self.n_step
            tz = r.unsqueeze(1) + (1.0 - d.unsqueeze(1)) * gamma_n * self.support.unsqueeze(0)
            tz = tz.clamp(self.v_min, self.v_max)

            # compute project of target dist using C51 formula
            b = (tz - self.v_min) / self.delta_z
            l = b.floor().long().clamp(0, self.num_atoms - 1)
            u = b.ceil().long().clamp(0, self.num_atoms - 1)

            # handle edge case where tz is exactly on a support atom
            proj = torch.zeros_like(next_dist)
            for i in range(self.batch_size):
                for j in range(self.num_atoms):
                    lj = int(l[i, j].item())
                    uj = int(u[i, j].item())
                    p = next_dist[i, j]
                    if lj == uj:
                        proj[i, lj] += p
                    else:
                        proj[i, lj] += p * (float(uj) - b[i, j])
                        proj[i, uj] += p * (b[i, j] - float(lj))

        # compute cross-entropy loss between projected target distribution and predicted distribution
        per_sample_loss = -(proj * log_p).sum(dim=1)

        # apply importance sampling weights
        loss = (w * per_sample_loss).mean()

        # optimize online network
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        # update priorities in replay buffer based on new loss magnitudes
        new_priorities = per_sample_loss.detach().cpu().numpy() + self.per_eps
        self.replay.update_priorities(idxs, new_priorities)

        # periodically update target network with online network weights
        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            # Hard target sync (periodic copy).
            self.t.load_state_dict(self.q.state_dict())


# helper to visualize training progress
def plot_training_rewards(rewards, window=100):
    # no plot if there is nothing to show
    if len(rewards) == 0:
        print("No training rewards to plot.")
        return

    # moving average smooths high-variance episode returns
    rolling = pd.Series(rewards).rolling(window, min_periods=1).mean()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(rewards, alpha=0.25, label="Episode reward")
    ax.plot(rolling, linewidth=2, label=f"Moving average ({window})")
    ax.set_title("Rainbow DQN Training Rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    plt.tight_layout()
    plt.show()


# training loop
def run_train(episodes=2000, model_path="rainbow.pt", moving_avg_window=100, plot_training=True):
    # set up environment, agent, and load existing model if available to continue training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Game2048Env()
    agent = RainbowAgent(device)

    # try to load existing model weights to continue training, but handle cases where no model exists or architecture mismatch
    try:
        agent.q.load_state_dict(torch.load(model_path, map_location=device))
        agent.t.load_state_dict(agent.q.state_dict())
        print("Loaded existing model, continuing training")
    except FileNotFoundError:
        print("No model found, training from scratch")
    except RuntimeError:
        print("Checkpoint architecture mismatch, training from scratch")

    # beta anneals from beta_start to 1.0 for per bias correction
    frame_idx = 0
    beta_start = 0.4
    beta_frames = 200000

    # store per-episode rewards for training diagnostics
    rewards = []

    # iterate through episodes
    for ep in tqdm(range(episodes), desc="Training"):
        s = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            # with small epsilon for some random exploration
            a = agent.act(s, eps=0.01, train=True)
            s2, r, done, _ = env.step(a)

            # store transition and learn
            agent.store(s, a, r, s2, done)
            s = s2
            ep_reward += r
            frame_idx += 1

            # only learn if we have enough samples in replay buffer for a full batch
            if len(agent.replay) >= agent.batch_size:
                beta = min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
                agent.learn(beta=beta)

        # keep full reward history for moving-average plotting
        rewards.append(ep_reward)

    # save online network weights for later eval/continue training
    torch.save(agent.q.state_dict(), model_path)

    # report final summary and plot learning curve
    if len(rewards) > 0:
        recent_window = min(moving_avg_window, len(rewards))
        recent_avg = float(np.mean(rewards[-recent_window:]))
        print(f"Training complete. Last {recent_window}-episode avg reward: {recent_avg:.2f}")

    if plot_training:
        plot_training_rewards(rewards, window=moving_avg_window)

    return rewards


# evaluation loop
def run_eval(model_path="rainbow.pt", episodes=500):
    # set up environment and agent, and load trained model weights for evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Game2048Env()
    agent = RainbowAgent(device)

    # load trained model weights, but handle cases where model is missing or architecture mismatch
    try:
        agent.q.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Model not found at {model_path}. Train first.")
        return
    except RuntimeError:
        print("Checkpoint architecture mismatch. Re-train before evaluation.")
        return

    agent.q.eval()

    # episode-level metrics for summary + plotting
    rewards, lengths, tiles = [], [], []

    for _ in tqdm(range(episodes), desc="Evaluating"):
        s = env.reset()
        done = False
        total = 0
        steps = 0

        # hard step cap to avoid infinite loops
        while not done and steps < 5000:
            a = agent.act(s, eps=0.0, train=False)
            s, r, done, _ = env.step(a)

            total += r
            steps += 1

        rewards.append(total)
        lengths.append(steps)
        tiles.append(env.board.max_tile_value())

    print("\n===== EVAL RESULTS =====")
    print(f"Avg reward: {np.mean(rewards):.2f}")
    print(f"Avg length: {np.mean(lengths):.2f}")
    print(f"Best tile: {np.max(tiles)}")
    print(f"Avg tile: {np.mean(tiles):.2f}")

    # plot raw reward and smoothed trend, plus max tile histogram
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    rolling = pd.Series(rewards).rolling(50, min_periods=1).mean()
    ax[0].plot(rewards, alpha=0.3)
    ax[0].plot(rolling, linewidth=2)
    ax[0].set_title("Rewards")

    counts = pd.Series(tiles).value_counts().sort_index()
    ax[1].bar(counts.index.astype(str), counts.values)
    ax[1].set_title("Max Tile Distribution")

    plt.tight_layout()
    plt.show()



# uncomment this line to train / continue training
run_train(episodes=1000, model_path="rainbow.pt", moving_avg_window=100, plot_training=True)
# uncomment this line to run evaluation of trained model
#run_eval("rainbow.pt", episodes=500)