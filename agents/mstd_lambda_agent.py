# Multi-Stage TD(Lambda) implementation with N-Tuple Networks & Eligibility Traces for 2048
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
import game.game_2048 as game
from shared.tuple_networks import NTupleNetwork

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


class MultiStageTDLambdaAgent:
    """Multi Stage (Lambda) agent with eligibility traces and two stages.
    Stage 0 handles the game up to the 4096 tile, stage 1 handles everything after. Each stage has its own NTupleNetwork.

    This is similar to the MS-TD(0) agent but instead of updating only the last post-state it tracks traces that
    propagate the TD error to earlier states as well. The traces decay by a factor of lambda each step.
    After a stage transition(from stage 0 to stage 1), the old traces are cleared to keep the traces for each stage independent of each other."""

    def __init__(self, alpha=0.0025, lam=0.5, patterns=None, stage_thresholds=None):
        self.alpha = alpha
        self.initial_alpha = alpha
        self.lam = lam
        self.patterns = patterns or PATTERNS
        self.thresholds = sorted(stage_thresholds or [12])
        self.n_stages = len(self.thresholds) + 1
        self.networks = [NTupleNetwork(self.patterns) for _ in range(self.n_stages)]

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

        traces = [{} for _ in range(self.n_stages)]
        old_stage = trajectory[0][2]

        for t in range(len(trajectory)):
            ps, r, current_stage = trajectory[t]
            net = self.networks[current_stage]

            # If a stage transition occurs clear the existing stage's trace
            if current_stage != old_stage:
                traces[old_stage] = {}
                old_stage = current_stage

            # compute TD error, at terminal stage the target will be 0
            v_t = net.value(ps)
            if t == len(trajectory) - 1:
                td_error = 0.0 - v_t
            else:
                ps_next, r_next, stg_next = trajectory[t + 1]
                if stg_next != current_stage:
                    td_error = 0.0 - v_t
                else:
                    v_next = net.value(ps_next)
                    td_error = r_next + v_next - v_t

            # decay existing traces for this stage
            trace = traces[current_stage]
            inactive_trace_keys = []
            for key in trace:
                trace[key] *= self.lam
                if abs(trace[key]) < 1e-6:
                    inactive_trace_keys.append(key)
            for key in inactive_trace_keys:
                del trace[key]

            # add current post-state to traces and replace old traces
            indices = net.get_indices(ps)
            for pi, lut_idx in enumerate(indices):
                trace[(pi, lut_idx)] = 1.0

            # update LUT entries
            for (pi, lut_idx), tr in trace.items():
                net.luts[pi][lut_idx] += alpha * td_error * tr

    # Save the existing weights
    def save(self, path):
        base, ext = os.path.splitext(path)
        cfg = {"n_stages": self.n_stages, "thresholds": self.thresholds, "alpha": self.initial_alpha, "lambda": self.lam,}
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


def train(n_episodes, alpha, lam, decay_rate, thresholds, save_path, log_every=100, save_every=10000):
    agent = MultiStageTDLambdaAgent(alpha=alpha, lam=lam, stage_thresholds=thresholds)

    scores = []
    tiles = []
    best_rate = 0.0
    log = {"episodes": [], "avg_scores": [], "max_scores": [], "rate_1024": [], "rate_2048": [], "rate_4096": [], "lr": []}

    print(f"Training begins MS-TD(Lambda = {lam})")
    print(f"  {agent.n_stages} stages... thresholds at: {agent.thresholds}")
    print(f"  {len(agent.networks[0].patterns)} patterns, rotating & reflecting")
    print(f"  Training for {n_episodes} episodes, alpha={alpha}, decay={decay_rate}\n")

    for ep in range(1, n_episodes + 1):
        lr = agent.updated_alpha(ep, decay_rate)
        board = Board()
        traj = []
        score = 0

        # NOTE : no valid actions breaks the loop
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
            avg_score = np.mean(rec_s)
            max_score = np.max(rec_s)
            r_1024 = sum(t >= 1024 for t in rec_t) / log_every * 100
            r_2048 = sum(t >= 2048 for t in rec_t) / log_every * 100
            r_4096 = sum(t >= 4096 for t in rec_t) / log_every * 100

            print(f"ep {ep:>7d} | avg: {avg_score:>8.0f} | max: {max_score:>8.0f} | 1024: {r_1024:>5.1f}% | 2048: {r_2048:>5.1f}% | 4096: {r_4096:>5.1f}% | lr: {lr:.6f}")

            log["episodes"].append(ep)
            log["avg_scores"].append(avg_score)
            log["max_scores"].append(max_score)
            log["rate_1024"].append(r_1024)
            log["rate_2048"].append(r_2048)
            log["rate_4096"].append(r_4096)
            log["lr"].append(lr)

            # NOTE : Changed to save the latest best 2048 rate model npz file
            if r_2048 >= best_rate:
                best_rate = r_2048
                agent.save(save_path.replace(".npz", "_best.npz"))
                print(f"new best: {r_2048:.1f}%")

        # Periodically save every 'save_every' episodes
        if ep % save_every == 0:
            agent.save(save_path)
            save_log(save_path, log)

    # Final saving after training
    agent.save(save_path)
    save_log(save_path, log)
    print(f"Training complete, model saved to {save_path}")
    return agent

if __name__ == "__main__":
    train(n_episodes=80000, alpha=0.0025, lam=0.3, decay_rate=50000, thresholds=[12], save_path="model_mstd_lambda_spot3.npz")