# Multi-Stage TD(0) with N-Tuple Networks for 2048
#
# Two stages: stage 0 plays until a 4096 tile (2^12) appears,
# then stage 1 takes over with its own weights.
#
# Paper References:
#   Szubert & Jaskowski (2014), "Temporal Difference Learning of N-Tuple Networks
#   for the Game 2048", IEEE CIG 2014.
#   Yeh et al. (2016), "Multi-Stage Temporal Difference Learning for 2048-like
#   Games", IEEE Transactions on Computational Intelligence and AI in Games.
#
# Game implementation adapted from:
#   Alan H. Yue, github.com/alanhyue/RL-2048-with-n-tuple-network

import numpy as np
import json
import os
from copy import copy
import game_2048 as game

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


class NTupleNetwork:
    """Lookup-table based function approximator. Each tuple pattern indexes into its own weight table.
    Board value = sum of all lookups across all patterns."""

    def __init__(self, tuple_patterns, max_tile_power=15):
        self.base = max_tile_power + 1  # 16 possible cell values (0 thru 15)
        self.patterns = self.generate_pattern_variants(tuple_patterns)

        # one LUT per pattern, each with base^6 entries
        self.luts = [np.zeros(self.base ** len(p), dtype=np.float64)
                     for p in self.patterns]

    # Rotate board index 90deg clockwise.
    def rotate_clockwise(self, idx):
        r, c = divmod(idx, 4)
        return c * 4 + (3 - r)


    # Horizontal reflection.
    def flip(self, idx):
        r, c = divmod(idx, 4)
        return r * 4 + (3 - c)

    # Generate all patterns from the existing patterns using 4 reflections and 2 rotations.
    def generate_pattern_variants(self, patterns):
        result = []
        for pat in patterns:
            seen = set()
            cur = list(pat)
            for _ in range(4):
                t = tuple(cur)
                if t not in seen:
                    seen.add(t)
                    result.append(t)
                ref = tuple(self.flip(i) for i in cur)
                if ref not in seen:
                    seen.add(ref)
                    result.append(ref)
                cur = [self.rotate_clockwise(i) for i in cur]
        return result

    # Get LUT index from pattern
    def idx(self, board, pattern):
        i = 0
        for pos in pattern:
            i = i * self.base + board[pos]
        return i

    # Sum all LUT lookups for this board.
    def value(self, board):
        # NOTE : This is based on the paper referenced
        return sum(lut[self.idx(board, p)] for p, lut in zip(self.patterns, self.luts))

    # Update the board values by adding the difference = alpha * (target - estimate)
    def update(self, board, diff):
        for p, lut in zip(self.patterns, self.luts):
            lut[self.idx(board, p)] += diff

    # Save the
    def save(self, path):
        data = {f"lut_{i}": lut for i, lut in enumerate(self.luts)}
        np.savez_compressed(path, **data)

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        for i in range(len(self.luts)):
            k = f"lut_{i}"
            if k in data:
                self.luts[i] = data[k].astype(np.float64)


class MultiStageTDAgent:
    """Two-stage TD(0) agent. Stage 0 handles the game up to the 4096 tile,
    stage 1 handles everything after. Each stage has its own NTupleNetwork.

    When the stage changes mid-game, the old stage's last position gets
    a terminal update (target=0) to keep the value functions separate."""

    def __init__(self, alpha=0.0025, patterns=None, stage_thresholds=None):
        self.alpha = alpha
        self.initial_alpha = alpha
        self.patterns = patterns or PATTERNS
        self.thresholds = sorted(stage_thresholds or [12])
        self.n_stages = len(self.thresholds) + 1
        self.networks = [NTupleNetwork(self.patterns) for _ in range(self.n_stages)]

    # Get the stage that this board is in (stage 0 or stage 1 for two stages)
    def get_stage(self, board):
        mx = max(board)
        s = 0
        for th in self.thresholds:
            if mx >= th:
                s += 1
            else:
                break
        return s

    # Get the updated value of alpha
    def updated_alpha(self, ep, decay_rate=50000):
        return self.initial_alpha / (1 + ep / decay_rate)

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
    def learn(self, trajectory, alpha):
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

            net.update(ps, alpha * (target - v))

    # Save the existing weights
    def save(self, path):
        base, ext = os.path.splitext(path)
        cfg = {"n_stages": self.n_stages, "thresholds": self.thresholds, "alpha": self.initial_alpha}
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


def train(n_episodes, alpha, decay_rate, thresholds, save_path, resume_from=None,
          log_every=1000, save_every=10000):
    agent = MultiStageTDAgent(alpha=alpha, stage_thresholds=thresholds)
    if resume_from:
        try:
            agent.load(resume_from)
            print(f"resumed from {resume_from}")
        except Exception as e:
            print(f"unable to load {resume_from}, error: {e}")

    scores, tiles = [], []
    best_rate = 0.0
    log = {"episodes": [], "avg_scores": [], "max_scores": [], "rate_1024": [], "rate_2048": [], "rate_4096": [], "lr": []}

    for ep in range(1, n_episodes + 1):
        lr = agent.updated_alpha(ep, decay_rate)
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

        agent.learn(traj, lr)
        scores.append(score)
        tiles.append(board.max_tile_value())

        if ep % log_every == 0:
            rec_s = scores[-log_every:]
            rec_t = tiles[-log_every:]
            avg, mx = np.mean(rec_s), np.max(rec_s)
            r1024 = sum(t >= 1024 for t in rec_t) / log_every * 100
            r2048 = sum(t >= 2048 for t in rec_t) / log_every * 100
            r4096 = sum(t >= 4096 for t in rec_t) / log_every * 100

            print(f"ep {ep:>7d} | avg: {avg:>8.0f} | max: {mx:>8.0f} | "
                  f"1024: {r1024:>5.1f}% | 2048: {r2048:>5.1f}% | "
                  f"4096: {r4096:>5.1f}% | lr: {lr:.6f}")

            log["episodes"].append(ep)
            log["avg_scores"].append(avg)
            log["max_scores"].append(mx)
            log["rate_1024"].append(r1024)
            log["rate_2048"].append(r2048)
            log["rate_4096"].append(r4096)
            log["lr"].append(lr)

            if r2048 > best_rate:
                best_rate = r2048
                agent.save(save_path.replace(".npz", "_best.npz"))
                print(f"  new best: {r2048:.1f}%")

        if ep % save_every == 0:
            agent.save(save_path)
            save_log(save_path, log)

    agent.save(save_path)
    save_log(save_path, log)
    print(f"Training Complete, npz saved to {save_path}")
    return agent


if __name__ == "__main__":
    train(
        n_episodes=100000,
        alpha=0.0025,
        decay_rate=50000,
        thresholds=[12],
        save_path="model_mstd.npz",
        resume_from=None,
    )