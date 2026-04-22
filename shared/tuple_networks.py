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


class TCNTupleNetwork:
    """N-tuple network with adaptive learning rates for temporal coherence. Each LUT entry will have a learning rate and previous direction 
    in addition to the value estimate."""

    def __init__(self, tuple_patterns, initial_alpha=0.0025, max_tile_power=15):
        self.base = max_tile_power + 1
        self.patterns = self.generate_pattern_variants(tuple_patterns)
        self.initial_alpha = initial_alpha

        self.luts = []
        self.alphas = []
        self.prev_deltas = []

        for p in self.patterns:
            size = self.base ** len(p)
            self.luts.append(np.zeros(size, dtype=np.float64))
            self.alphas.append(np.full(size, initial_alpha, dtype=np.float64))
            self.prev_deltas.append(np.zeros(size, dtype=np.float64))

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

    def get_indices(self, board):
        return [self.idx(board, p) for p in self.patterns]

    def tc_update(self, board, td_error, alpha_increase=0.001, alpha_decrease=0.01, alpha_min=1e-6, alpha_max=0.01):
        # If current td_error and previous delta have the same sign, increase that entry's learning rate (coherent updates)
        # If they have opposite signs, decrease it (incoherent)
        for idx, p in enumerate(self.patterns):
            lut_idx = self.idx(board, p)

            cur_delta = td_error
            prev = self.prev_deltas[idx][lut_idx]

            if prev * cur_delta > 0:
                # same direction, increase rate
                self.alphas[idx][lut_idx] += alpha_increase
            elif prev * cur_delta < 0:
                # opposite direction, decrease rate
                self.alphas[idx][lut_idx] *= (1.0 - alpha_decrease)

            # clamp
            a = self.alphas[idx][lut_idx]
            a = max(alpha_min, min(alpha_max, a))
            self.alphas[idx][lut_idx] = a

            # update the weight
            update = a * td_error
            self.luts[idx][lut_idx] += update
            self.prev_deltas[idx][lut_idx] = td_error

    def save(self, path):
        data = {}
        for i in range(len(self.luts)):
            data[f"lut_{i}"] = self.luts[i]
            data[f"alpha_{i}"] = self.alphas[i]
            data[f"prev_{i}"] = self.prev_deltas[i]
        np.savez_compressed(path, **data)

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        for i in range(len(self.luts)):
            if f"lut_{i}" in data:
                self.luts[i] = data[f"lut_{i}"].astype(np.float64)
            if f"alpha_{i}" in data:
                self.alphas[i] = data[f"alpha_{i}"].astype(np.float64)
            if f"prev_{i}" in data:
                self.prev_deltas[i] = data[f"prev_{i}"].astype(np.float64)
