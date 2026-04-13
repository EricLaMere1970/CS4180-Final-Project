import random
import time
from collections import deque
from copy import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


# dqn implementation for 2048 using cnn.
# board values are stored in log2 form (0 empty, 1->tile2, 2->tile4, ...)


# define the action space
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTIONS = [UP, RIGHT, DOWN, LEFT]


# define exceptions for illegal actions and game over
class IllegalAction(Exception):
    pass


class GameOver(Exception):
    pass


# helper function to assit in merging
def _compress(row):
    # remove empty cells so merge logic only processes tile exponents
    return [x for x in row if x != 0]

# merge one row to the left and accumulate reward
def _merge_row(row):
    # remove empty cells
    row = _compress(row)

    # inialize reward
    reward = 0

    # initialize list to hold merged tiles
    merged = []

    # track last unmerged tile
    hold = None

    # iterate through non-empty tiles
    while row:
        # pop leftmost tile
        v = row.pop(0)

        # if we do not have a held tile, hold current tile
        if hold is None:
            hold = v

        # if we have a held tile, check if it can merge with current tile
        elif hold == v:
            # add merge reward
            reward += 2 ** (hold + 1)

            # append merged tile
            merged.append(hold + 1)

            # clear hold
            hold = None

        # have a held tile but can't merge
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

# define board class
class Board:
    # inialize board
    def __init__(self):
        self.reset()

    # initialize empty board and spawn two starting tiles
    def reset(self):
        self.board = [0] * 16
        self.spawn_tile()
        self.spawn_tile()

    # place a random new tile in an empty cell:
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

    # determine if any legal moves are available
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

    # define helper function to rotate board for directional merges
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

    # perform action
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


# define gym environment
class Game2048Env:
    # initialize environment
    def __init__(self):
        self.board = Board()

    # reset board an observation
    def reset(self):
        self.board.reset()
        return self._obs()

    # define observation as 4x4 matrix
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

# define dqn cnn model
class DQNCNN(nn.Module):
    # initalize cnn architecture
    def __init__(self):
        super().__init__()

        # define net
        self.net = nn.Sequential(
            # input is 4x4 board with 1 channel
            nn.Conv2d(1, 32, 2), nn.ReLU(),

            # output of conv1 is 32 channels of 3x3, so conv2 with kernel 2 has 32 channels of 2x2
            nn.Conv2d(32, 64, 2), nn.ReLU(),
        )

        # define fully connected layers
        self.fc = nn.Sequential(
            # flatten conv output
            nn.Linear(64 * 2 * 2, 128), nn.ReLU(),

            # output is 4 q-values for each action
            nn.Linear(128, 4)
        )

    # define forward pass
    def forward(self, x):
        # input can be single state or batch; reshape for Conv2d
        x = x.view(-1, 1, 4, 4)

        # pass through conv layers and flatten for fully connected layers
        x = self.net(x).flatten(1)

        # pass through fully connected layers to get q-values
        return self.fc(x)


# define dqn agent
class DQNAgent:
    def __init__(self, device):
        self.device = device
        # online network is updated every gradient step
        self.q = DQNCNN().to(device)
        # target network is updated periodically
        self.t = DQNCNN().to(device)

        # initially sync target network with online network
        self.t.load_state_dict(self.q.state_dict())

        # define optimizer for online network
        self.opt = optim.Adam(self.q.parameters(), lr=1e-4)

        # replay stores transitions: (state, action, reward, next_state, done)
        self.replay = deque(maxlen=50000)

    # epsilon-greedy exploration policy during training
    def act(self, s, eps):
        if random.random() < eps:
            return random.randint(0, 3)

        # convert state to tensor and add batch dimension
        s = torch.tensor(s).float().unsqueeze(0).to(self.device)
        return int(self.q(s).argmax(1).item())

    def train_step(self, batch):
        # unpack sampled minibatch
        s, a, r, s2, d = batch

        # convert to tensors and move to device
        s = torch.tensor(np.array(s)).float().to(self.device)
        s2 = torch.tensor(np.array(s2)).float().to(self.device)
        a = torch.tensor(a).long().to(self.device)
        r = torch.tensor(r).float().to(self.device)
        d = torch.tensor(d).float().to(self.device)

        # get q-values for actions taken in batch
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # compute target q-values
        with torch.no_grad():
            # one-step dqn target
            q2 = self.t(s2).max(1)[0]
            target = r + 0.99 * q2 * (1 - d)

        # calculate loss
        loss = nn.SmoothL1Loss()(q, target)

        # optimize online network
        self.opt.zero_grad()

        # backpropagate loss
        loss.backward()

        # gradient step
        self.opt.step()


# define training loop
def run_train(episodes=2000, model_path="dqn.pt"):
    # use gpu when available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Game2048Env()
    agent = DQNAgent(device)

    # load existing checkpoint if present
    try:
        agent.q.load_state_dict(torch.load(model_path, map_location=device))
        agent.t.load_state_dict(agent.q.state_dict())
        print("Loaded existing model, continuing training")
    except FileNotFoundError:
        print("No model found, training from scratch")

    # iterate over episdoes
    for ep in tqdm(range(episodes), desc="Training"):
        # reset environment
        s = env.reset()

        # track if episode is done
        done = False

        # initialize total reward
        total = 0

        # linear epsilon decay with floor at 0.05
        eps = max(0.05, 1 - ep / 1500)

        # while episode is not done, take actions and store transitions
        while not done:
            # epsilon-greedy action selection
            a = agent.act(s, eps)

            # take action and observe next state, reward, and done flag
            s2, r, done, _ = env.step(a)

            # store transition in replay buffer
            agent.replay.append((s, a, r, s2, done))
            s = s2

            # accumulate reward
            total += r

            # start updates once replay has enough samples for a full minibatch
            if len(agent.replay) > 128:
                batch = random.sample(agent.replay, 128)
                agent.train_step(zip(*batch))

        # periodically sync target network with online network
        if ep % 50 == 0:
            agent.t.load_state_dict(agent.q.state_dict())

    torch.save(agent.q.state_dict(), "dqn.pt")


# define evaluation loop
def run_eval(model_path="dqn.pt", episodes=500):
    # use gpu when available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Game2048Env()

    # load trained model
    agent = DQNAgent(device)
    agent.q.load_state_dict(torch.load(model_path, map_location=device))
    agent.q.eval()

    # track statistics for reporting and plots
    rewards, lengths, tiles, times = [], [], [], []

    # iterate over episodes and evaluate trained agent
    for ep in tqdm(range(episodes), desc="Evaluating"):
        # reset environment
        s = env.reset()

        # track if episode is done
        done = False

        # track reward
        total = 0

        # track steps
        steps = 0

        # while not done and under max steps, continue selecting actions
        while not done and steps < 5000:
            # select action with highest q-value from online network
            with torch.no_grad():
                st = torch.tensor(s).float().unsqueeze(0).to(device)
                a = int(agent.q(st).argmax(1).item())

            # take action and observe next state, reward, and done flag
            s, r, done, _ = env.step(a)

            # accumulate reward and increment steps
            total += r
            steps += 1

        # track statistics
        rewards.append(total)
        lengths.append(steps)
        tiles.append(env.board.max_tile_value())

    # print evaluation results
    print("\n===== EVAL RESULTS =====")
    print(f"Avg reward: {np.mean(rewards):.2f}")
    print(f"Avg length: {np.mean(lengths):.2f}")
    print(f"Best tile: {np.max(tiles)}")
    print(f"Avg tile: {np.mean(tiles):.2f}")

    # generate plots of rewards and max tile distribution
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
#run_train(episodes=40000)
# uncomment this line to run evaluation of trained model
run_eval("dqn.pt", episodes=500)