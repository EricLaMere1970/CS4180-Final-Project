# Game implementation adapted from:
# Alan H. Yue, "N-Tuple Networks for the Game 2048 in Python"
# https://github.com/alanhyue/RL-2048-with-n-tuple-network
#
# Modifications made for compatibility with Multi-Stage Temporal Difference
# Learning (Yeh et al., 2016) and Deep Q-Learning:
#   - Fixed board copy bug in act() (reference vs. shallow copy)
#   - Changed reset() to use random tile spawning (90/10 distribution)
#   - Made spawn_tile() always use random placement
#   - Added can_move() and is_game_over() for proper termination detection
#   - Added max_tile(), max_tile_value(), tile_values() for MS-TD stage transitions
#   - Changed if-chain to if/elif in act()

import random
import numpy as np
from copy import copy

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


def action_name(a):
    return "UP RIGHT DOWN LEFT".split()[a]


class IllegalAction(Exception):
    pass


class GameOver(Exception):
    pass


def compress(row):
    "remove 0s in the list"
    return [x for x in row if x != 0]


def merge(row):
    row = compress(row)
    reward = 0
    r = []
    hold = -1
    while len(row) > 0:
        v = row.pop(0)
        if hold != -1:
            if hold == v:
                reward = reward + (2 ** (hold + 1))
                r.append(hold + 1)
                hold = -1
            else:
                r.append(hold)
                hold = v
        else:
            hold = v
    if hold != -1:
        r.append(hold)
        hold = -1
    while len(r) < 4:
        r.append(0)
    return reward, r


class Board:
    def __init__(self, board=None):
        """board is a list of 16 integers (log2-encoded tile values)"""
        if board is not None:
            self.board = board
        else:
            self.reset()

    def reset(self):
        self.clear()
        self.spawn_tile()
        self.spawn_tile()

    def spawn_tile(self):
        """Spawn a random tile on an empty cell.
        90% chance of 2-tile (log2 = 1), 10% chance of 4-tile (log2 = 2).
        Raises GameOver if no empty cells are available.
        """
        empty = self.empty_tiles()
        if len(empty) == 0:
            raise GameOver("Board is full. Cannot spawn any tile.")
        k = 2 if np.random.rand() <= 0.1 else 1
        self.board[random.choice(empty)] = k

    def clear(self):
        self.board = [0] * 16

    def empty_tiles(self):
        return [i for (i, v) in enumerate(self.board) if v == 0]

    def max_tile(self):
        """Return the maximum log2-encoded tile value on the board."""
        return max(self.board)

    def max_tile_value(self):
        """Return the maximum tile value in base-10."""
        m = self.max_tile()
        return 2 ** m if m > 0 else 0

    def tile_values(self):
        """Return the set of distinct log2-encoded tile values on the board (excluding 0)."""
        return set(v for v in self.board if v > 0)

    def can_move(self):
        """Check if any move is possible (empty cells or adjacent equal tiles)."""
        # check for empty cells
        if any(v == 0 for v in self.board):
            return True
        # check horizontal neighbors
        for row in range(4):
            for col in range(3):
                idx = row * 4 + col
                if self.board[idx] == self.board[idx + 1]:
                    return True
        # check vertical neighbors
        for row in range(3):
            for col in range(4):
                idx = row * 4 + col
                if self.board[idx] == self.board[idx + 4]:
                    return True
        return False

    def is_game_over(self):
        """Return True if no legal moves remain."""
        return not self.can_move()

    def display(self):
        def format_row(lst):
            s = ""
            for l in lst:
                s += " {:5d}".format(l)
            return s
        for row in range(4):
            idx = row * 4
            print(format_row(self.base10_board[idx : idx + 4]))

    @property
    def base10_board(self):
        return [2 ** v if v > 0 else 0 for v in self.board]

    def act(self, a):
        """Perform action a on the board. Returns the merge reward.
        Raises IllegalAction if the move doesn't change the board.
        """
        original = copy(self.board)  # fixed: was `self.board` (no copy)
        if a == LEFT:
            r = self.merge_to_left()
        elif a == RIGHT:
            r = self.rotate().rotate().merge_to_left()
            self.rotate().rotate()
        elif a == UP:
            r = self.rotate().rotate().rotate().merge_to_left()
            self.rotate()
        elif a == DOWN:
            r = self.rotate().merge_to_left()
            self.rotate().rotate().rotate()
        else:
            raise ValueError(f"Invalid action: {a}")
        if original == self.board:
            raise IllegalAction("Action did not move any tile.")
        return r

    def rotate(self):
        "Rotate the board inplace 90 degrees clockwise."
        size = 4
        b = []
        for i in range(size):
            b.extend(self.board[i::4][::-1])
        self.board = b
        return self

    def merge_to_left(self):
        "Merge board to the left, returns the reward for merging tiles."
        r = []
        board_reward = 0
        for nrow in range(4):
            idx = nrow * 4
            row = self.board[idx : idx + 4]
            row_reward, row = merge(row)
            board_reward = board_reward + row_reward
            r.extend(row)
        self.board = r
        return board_reward

    def copyboard(self):
        return copy(self.board)