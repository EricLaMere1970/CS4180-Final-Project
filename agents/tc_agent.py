# Multi-Stage TD with Temporal Coherence (TC) Learning for 2048
#
# In this algorithm, instead of a single learning rate, each LUT gets its own adaptive rate.
# The learning rate increases if the weight keeps moving in the same direction or decreases if the weight moves in the opposite direction.
#
# References:
#   Jaskowski (2018), "Mastering 2048 with Delayed Temporal Coherence
#   Learning, Multi-Stage Weight Promotion, Redundant Encoding and
#   Carousel Shaping"
#
# Game implementation adapted from:
#   Alan H. Yue, github.com/alanhyue/RL-2048-with-n-tuple-network

import numpy as np
import json
import os
from copy import copy
import game.game_2048 as game
from shared.tuple_networks import TCNTupleNetwork

Board = game.Board
IllegalAction = game.IllegalAction
GameOver = game.GameOver
UP, RIGHT, DOWN, LEFT = game.UP, game.RIGHT, game.DOWN, game.LEFT

# 8 base 6-tuple patterns with 4 rectangular patterns and 4 L-shaped patterns.
# With rotations and reflections, this will expand to 64 patterns
PATTERNS = [
    (0, 1, 2, 3, 4, 5),
    (4, 5, 6, 7, 8, 9),
    (0, 1, 2, 4, 5, 6),
    (4, 5, 6, 8, 9, 10),
    (0, 1, 5, 6, 7, 11),
    (0, 1, 2, 5, 9, 13),
    (0, 1, 5, 9, 13, 14),
    (0, 1, 2, 3, 4, 8),
]

class MultiStageTCAgent:
    """Multi-stage TD agent with temporal coherence learning. Similar to MS-TD(0) agent but this uses an adaptive learning rate for each weight instead
    of a single learning rate."""

    def __init__(self, initial_alpha=0.0025, patterns=None, stage_thresholds=None,
                 alpha_inc=0.001, alpha_dec=0.01):
        self.patterns = patterns or PATTERNS
        self.thresholds = sorted(stage_thresholds or [12])
        self.n_stages = len(self.thresholds) + 1
        self.initial_alpha = initial_alpha
        self.alpha_increase = alpha_inc
        self.alpha_decrease = alpha_dec
        self.networks = [TCNTupleNetwork(self.patterns, initial_alpha) for _ in range(self.n_stages)]

    # Get the stage that this board is in
    def get_stage(self, board):
        mx = max(board)
        s = 0
        for th in self.thresholds:
            if mx >= th:
                s += 1
            else:
                break
        return s

    # Try action, return (post_state, reward, value, stage) or None if illegal
    def evaluate(self, board, action):
        b = Board(copy(board))
        try:
            reward = b.act(action)
        except IllegalAction:
            return None
        ps = b.copyboard()
        stg = self.get_stage(ps)
        val = reward + self.networks[stg].value(ps)
        return ps, reward, val, stg

    # Get the best actions in order
    def best_actions(self, board):
        best = (None, None, 0, 0, -float("inf"))
        for a in [UP, RIGHT, DOWN, LEFT]:
            res = self.evaluate(board, a)
            if res and res[2] > best[4]:
                best = (a, res[0], res[1], res[3], res[2])
        return best[0], best[1], best[2], best[3]

    # Traverse a single trajectory backwatds
    def learn(self, trajectory):
        if not trajectory:
            return

        for t in range(len(trajectory) - 1, -1, -1):
            ps, r, stg = trajectory[t]
            net = self.networks[stg]
            v = net.value(ps)

            # NOTE : at terminal or stage boundary the target should be 0
            if t == len(trajectory) - 1:
                target = 0.0
            else:
                ps_next, r_next, stg_next = trajectory[t + 1]
                if stg_next != stg:
                    target = 0.0
                else:
                    target = r_next + net.value(ps_next)

            net.tc_update(ps, target - v, alpha_inc=self.alpha_increase, alpha_dec=self.alpha_decrease)

    # Save the existing weights
    def save(self, path):
        base, ext = os.path.splitext(path)
        cfg = {"n_stages": self.n_stages, "thresholds": self.thresholds, "initial_alpha": self.initial_alpha, "alpha_increase": self.alpha_increase, "alpha_decrease": self.alpha_decrease,}
        with open(base + "_config.json", "w") as f:
            json.dump(cfg, f)
        for i, net in enumerate(self.networks):
            net.save(f"{base}_stage{i}{ext}")

    # Load the saved weights
    def load(self, path):
        base, ext = os.path.splitext(path)
        for i, net in enumerate(self.networks):
            sp = f"{base}_stage{i}{ext}"
            if os.path.exists(sp):
                net.load(sp)
            else:
                print(f"{sp} not found, stage {i} starting fresh")


def save_log(save_path, log):
    base, _ = os.path.splitext(save_path)
    np.savez_compressed(base + "_log.npz", **{k: np.array(v) for k, v in log.items()})


def train(n_episodes, initial_alpha, alpha_inc, alpha_dec, thresholds, save_path, resume_from=None, log_every=100, save_every=10000):

    agent = MultiStageTCAgent(initial_alpha=initial_alpha,stage_thresholds=thresholds,alpha_inc=alpha_inc,alpha_dec=alpha_dec,)

    scores, tiles = [], []
    best_rate = 0.0
    log = {"episodes": [], "avg_scores": [], "max_scores": [], "rate_1024": [], "rate_2048": [], "rate_4096": []}

    # NOTE : no valid actions breaks the loop
    for ep in range(1, n_episodes + 1):
        board = Board()
        traj = []
        score = 0

        while True:
            a, ps, r, stg = agent.best_actions(board.board)
            if a is None:
                break
            traj.append((copy(ps), r, stg))
            score += r
            board.board = copy(ps)
            try:
                board.spawn_tile()
            except GameOver:
                break

        agent.learn(traj)
        scores.append(score)
        tiles.append(board.max_tile_value())

        if ep % log_every == 0:
            rec_s = scores[-log_every:]
            rec_t = tiles[-log_every:]
            avg, mx = np.mean(rec_s), np.max(rec_s)
            r1024 = sum(t >= 1024 for t in rec_t) / log_every * 100
            r2048 = sum(t >= 2048 for t in rec_t) / log_every * 100
            r4096 = sum(t >= 4096 for t in rec_t) / log_every * 100

            print(f"ep {ep:>7d} | avg: {avg:>8.0f} | max: {mx:>8.0f} | 1024: {r1024:>5.1f}% | 2048: {r2048:>5.1f}% | 4096: {r4096:>5.1f}%")

            log["episodes"].append(ep)
            log["avg_scores"].append(avg)
            log["max_scores"].append(mx)
            log["rate_1024"].append(r1024)
            log["rate_2048"].append(r2048)
            log["rate_4096"].append(r4096)

            # NOTE : Changed to save the latest best 2048 rate model npz file
            if r2048 >= best_rate:
                best_rate = r2048
                agent.save(save_path.replace(".npz", "_best.npz"))
                print(f"new best: {r2048:.1f}%")

        if ep % save_every == 0:
            agent.save(save_path)
            save_log(save_path, log)

    agent.save(save_path)
    save_log(save_path, log)
    print(f"Training Complete, saved to {save_path}")
    return agent


if __name__ == "__main__":
    train( n_episodes=80000, initial_alpha=0.0025, alpha_inc=0.001, alpha_dec=0.01, thresholds=[12], save_path="model_tc.npz", resume_from=None,)
