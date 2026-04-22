import numpy as np


class NTupleNetwork:
    """Lookup-table based function approximator. Each tuple pattern indexes into its own weight table.
    Board value = sum of all lookups across all patterns."""

    def __init__(self, tuple_patterns, max_tile_power=15):
        self.base = max_tile_power + 1
        self.patterns = self.generate_pattern_variants(tuple_patterns)
        self.luts = [np.zeros(self.base ** len(p), dtype=np.float64)
                     for p in self.patterns]

    def rotate_clockwise(self, idx):
        r, c = divmod(idx, 4)
        return c * 4 + (3 - r)

    def flip(self, idx):
        r, c = divmod(idx, 4)
        return r * 4 + (3 - c)

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

    def idx(self, board, pattern):
        i = 0
        for pos in pattern:
            i = i * self.base + board[pos]
        return i

    def value(self, board):
        return sum(lut[self.idx(board, p)] for p, lut in zip(self.patterns, self.luts))

    def update(self, board, diff):
        for p, lut in zip(self.patterns, self.luts):
            lut[self.idx(board, p)] += diff

    def get_indices(self, board):
        return [self.idx(board, p) for p in self.patterns]

    def save(self, path):
        data = {f"lut_{i}": lut for i, lut in enumerate(self.luts)}
        np.savez_compressed(path, **data)

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        for i in range(len(self.luts)):
            k = f"lut_{i}"
            if k in data:
                self.luts[i] = data[k].astype(np.float64)

