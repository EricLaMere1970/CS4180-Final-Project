# PPO Model for 2048
#
# Uses a CNN to process the board as a one-hot encoded tensor. The actor-critic architecture is used with a
# shared feature extractor.
#
# Paper References:
#   Schulman et al. (2017), "Proximal Policy Optimization Algorithms"
#
# Repo Reference for Code and Results:
#   Lucca Pineli, github.com/lucca11235/2048-PPO
#
# Game implementation adapted from:
#   Alan H. Yue, github.com/alanhyue/RL-2048-with-n-tuple-network

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from copy import copy
import game.game_2048 as game
from shared.buffers import RolloutBuffer

Board = game.Board
IllegalAction = game.IllegalAction
GameOver = game.GameOver
UP, RIGHT, DOWN, LEFT = game.UP, game.RIGHT, game.DOWN, game.LEFT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_board(board):
    # One Hot Encoding a board(16 log_2 values) into a 4x4x16 tensor
    t = np.zeros((16, 4, 4), dtype=np.float32)
    for i, val in enumerate(board):
        r, c = divmod(i, 4)
        if val < 16:
            t[val][r][c] = 1.0

    return t


def legal_moves_filter(board):
    # Return a filter of which moves are valid in the board configuration
    filtered_moves = np.zeros(4, dtype=np.float32)
    for a in [UP, RIGHT, DOWN, LEFT]:
        b = Board(copy(board))
        try:
            b.act(a)
        except IllegalAction:
            pass
        filtered_moves[a] = 1.0

    return filtered_moves


class ActorCritic(nn.Module):
    """Convolutional actor critic network for PPO."""

    def __init__(self):
        super().__init__()
        # The features will be made from two convolution layers
        self.features = nn.Sequential(
            nn.Conv2d(16, 128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        linear_dim = 128 * 2 * 2

        self.actor = nn.Sequential(
            nn.Linear(linear_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )
        self.critic = nn.Sequential(
            nn.Linear(linear_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        f = self.features(x)
        logits = self.actor(f)
        val = self.critic(f)
        return logits, val

    def get_action(self, obs, fil):
        # Sample an action after applying the legal moves filter
        with torch.no_grad():
            x = torch.FloatTensor(obs).unsqueeze(0).to(device)
            m = torch.FloatTensor(fil).unsqueeze(0).to(device)
            l, v = self.forward(x)

            # reduce illegal moves to large -ve numbers
            l = l - (1 - m) * 1e9
            probs = torch.softmax(l, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            return action.item(), dist.log_prob(action).item(), v.item()

    def evaluate(self, obs_batch, actions_batch, m):
        # Evaluate a batch of state-actions for the PPO update
        logits, values = self.forward(obs_batch)

        # reduce illegal moves to large -ve numbers
        logits = logits - (1 - m) * 1e9
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(actions_batch)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy


def play_single_game(model, buffer):
    # Play a single game and add the transitions to the buffer
    board = Board()
    total_score = 0
    steps = 0

    while True:
        obs = encode_board(board.board)
        filter = legal_moves_filter(board.board)

        # If the filtered moves
        if not filter.any():
            break

        action, log_prob, value = model.get_action(obs, filter)

        old_board = copy(board.board)
        b = Board(old_board)

        try:
            reward = b.act(action)
            board.board = b.copyboard()
            board.spawn_tile()
            total_score += reward
            done = False
        except IllegalAction:
            # NOTE : Defensive block to set negative rewards added after code was breaking here
            reward = -10
            done = True
        except GameOver:
            total_score += reward
            done = True

        # NOTE : we scale rewards to keep the gradients manageable for learning
        scaled_reward = reward / 1000.0
        buffer.add(obs, action, scaled_reward, log_prob, value, done, filter)
        steps += 1

        if done:
            break

    return total_score, board.max_tile_value(), steps


def ppo_update(model, optimizer, buffer, clip_eps=0.2, vl_coeff=0.5, el_coeff=0.01, epochs=5, batch_size=512, gamma=0.99, gae_lambda=0.95):
    # Perform PPO update on collected experience
    obs, actions, old_log_probs, advantages, returns, masks = buffer.get(gamma, gae_lambda)

    n = len(obs)
    total_loss_sum = 0
    n_updates = 0

    for _ in range(epochs):
        indices = torch.randperm(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            b_obs = obs[idx]
            b_actions = actions[idx]
            b_old_log_probs = old_log_probs[idx]
            b_advantages = advantages[idx]
            b_returns = returns[idx]
            b_masks = masks[idx]

            # evaluate current policy on these states
            new_log_probs, values, entropy = model.evaluate(b_obs, b_actions, b_masks)

            # policy loss clipping
            ratio = torch.exp(new_log_probs - b_old_log_probs)
            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # value loss
            value_loss = nn.functional.mse_loss(values, b_returns)

            # entropy bonus
            entropy_loss = -entropy.mean()

            loss = policy_loss + vl_coeff * value_loss + el_coeff * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss_sum += loss.item()
            n_updates += 1

    return total_loss_sum / max(n_updates, 1)


def train(n_episodes, lr, gamma, gae_lambda, clip_eps, epochs_per_update, batch_size, games_per_update, save_path, log_every=1000, save_every=10000):
    model = ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    buffer = RolloutBuffer()

    scores, tiles = [], []
    best_rate = 0.0
    log = {"episodes": [], "avg_scores": [], "max_scores": [], "rate_1024": [], "rate_2048": [], "rate_4096": []}

    print(f"Training PPO Begins")
    print(f"  {n_episodes} episodes, lr={lr}, gamma={gamma}")
    print(f"  {games_per_update} games per update, {epochs_per_update} epochs per update\n")

    ep = 0
    while ep < n_episodes:
        buffer.reset()
        for _ in range(games_per_update):
            score, max_tile, steps = play_single_game(model, buffer)
            scores.append(score)
            tiles.append(max_tile)
            ep += 1

            if ep >= n_episodes:
                break

        # Perform PPO update on collected batch
        avg_loss = ppo_update( model, optimizer, buffer, clip_eps=clip_eps, epochs=epochs_per_update, batch_size=batch_size, gamma=gamma, gae_lambda=gae_lambda, )

        if ep % log_every < games_per_update and ep >= log_every:
            rec_s = scores[-log_every:]
            rec_t = tiles[-log_every:]
            avg, mx = np.mean(rec_s), np.max(rec_s)
            r1024 = sum(t >= 1024 for t in rec_t) / len(rec_t) * 100
            r2048 = sum(t >= 2048 for t in rec_t) / len(rec_t) * 100
            r4096 = sum(t >= 4096 for t in rec_t) / len(rec_t) * 100

            print(f"ep {ep:>7d} | avg: {avg:>8.0f} | max: {mx:>8.0f} | 1024: {r1024:>5.1f}% | 2048: {r2048:>5.1f}% |  4096: {r4096:>5.1f}% | loss: {avg_loss:.4f}")

            log["episodes"].append(ep)
            log["avg_scores"].append(avg)
            log["max_scores"].append(mx)
            log["rate_1024"].append(r1024)
            log["rate_2048"].append(r2048)
            log["rate_4096"].append(r4096)

            if r2048 > best_rate:
                best_rate = r2048
                torch.save(model.state_dict(), save_path.replace(".pt", "_best.pt"))
                print(f"new best: {r2048:.1f}%")

        if ep % save_every < games_per_update and ep >= save_every:
            torch.save(model.state_dict(), save_path)
            base, _ = os.path.splitext(save_path)
            np.savez_compressed(base + "_log.npz",
                                **{k: np.array(v) for k, v in log.items()})
            print(f"saved to {save_path}")

    # final save
    torch.save(model.state_dict(), save_path)
    base, _ = os.path.splitext(save_path)
    np.savez_compressed(base + "_log.npz",
                        **{k: np.array(v) for k, v in log.items()})
    print(f"\nPPO Training Finished, saved to {save_path}")
    return model


if __name__ == "__main__":
    train(n_episodes=100000, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2, epochs_per_update=4, batch_size=512, games_per_update=10, save_path="model_ppo.pt", log_every=1000, save_every=10000,)