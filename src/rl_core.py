"""Environment RL, pencarian acak, dan SA untuk S-box (F3, F4).
Semua metode memakai EvalCounter yang sama, jadi anggaran evaluasi adil.
Satu evaluasi = satu kali transformasi Walsh (biaya, NL, dan histogram state diambil dari situ)."""
import numpy as np
from metrics import walsh_spectrum, differential_uniformity, is_bijective


def default_X(nbits):
    return 2 ** (nbits // 2)          # 4x4 -> 4, 6x6 -> 8, 8x8 -> 16


def spectrum_stats(sbox, X, R):
    """Mengembalikan (biaya, NL, |W| tanpa baris b=0)."""
    W = np.abs(walsh_spectrum(sbox)[1:])
    c = float(np.sum(np.abs(W.astype(np.float64) - X) ** R))
    nl = len(sbox) // 2 - int(W.max()) // 2
    return c, nl, W


def cost(sbox, X=None, R=3):
    X = default_X(len(sbox).bit_length() - 1) if X is None else X
    return spectrum_stats(sbox, X, R)[0]


class EvalCounter:
    def __init__(self, nbits, X=None, R=3):
        self.n = 0
        self.X = default_X(nbits) if X is None else X
        self.R = R
        self.last_nl, self.last_W = None, None

    def __call__(self, sbox):
        c, nl, W = spectrum_stats(sbox, self.X, self.R)
        self.n += 1
        self.last_nl, self.last_W = nl, W
        return c


class BestTracker:
    """Pilih S-box terbaik: NL tertinggi, seri -> DU terendah, seri -> biaya terendah."""

    def __init__(self):
        self.s, self.nl, self.du, self.cost = None, -1, 10 ** 9, float("inf")

    def update(self, s, nl, c):
        if nl < self.nl:
            return False
        du = differential_uniformity(s)
        if self.s is None or (-nl, du, c) < (-self.nl, self.du, self.cost):
            self.s, self.nl, self.du, self.cost = s.copy(), nl, du, c
            return True
        return False

    def summary(self):
        return (self.nl, self.du, self.cost)


def random_search(nbits, budget, rng, X=None, R=3):
    N = 1 << nbits
    ev, tr = EvalCounter(nbits, X, R), BestTracker()
    while ev.n < budget:
        s = rng.permutation(N)
        c = ev(s)
        tr.update(s, ev.last_nl, c)
    return tr.s, tr.summary()


def simulated_annealing(nbits, budget, rng, T0=None, T_end_ratio=0.05, X=None, R=3):
    N = 1 << nbits
    ev, tr = EvalCounter(nbits, X, R), BestTracker()
    s = rng.permutation(N)
    c = ev(s)
    tr.update(s, ev.last_nl, c)
    T0 = T0 if T0 is not None else max(c * 0.02, 1.0)
    alpha = T_end_ratio ** (1.0 / max(budget - 1, 1))
    T = T0
    while ev.n < budget:
        i, j = rng.choice(N, 2, replace=False)
        s[i], s[j] = s[j], s[i]
        c2 = ev(s)
        if c2 <= c or rng.random() < np.exp((c - c2) / max(T, 1e-9)):
            c = c2
            tr.update(s, ev.last_nl, c)
        else:
            s[i], s[j] = s[j], s[i]
        T *= alpha
    return tr.s, tr.summary()


class SboxEnv:
    """Aksi: pilih 1 dari K kandidat tukar-dua-output. Selalu permutasi -> selalu bijektif."""
    STATE_DIM = 18  # 16 bin histogram + NL + sisa langkah

    def __init__(self, nbits=8, steps=200, n_cand=64, X=None, R=3, seed=0):
        self.nbits, self.N = nbits, 1 << nbits
        self.steps, self.K = steps, n_cand
        self.rng = np.random.default_rng(seed)
        self.ev = EvalCounter(nbits, X, R)
        self.tracker = BestTracker()

    def _state(self):
        h, _ = np.histogram(self.ev.last_W.ravel(), bins=16, range=(0, self.N))
        h = h / h.sum()
        nl = self.ev.last_nl / (self.N // 2)
        return np.concatenate([h, [nl, (self.steps - self.t) / self.steps]]).astype(np.float32)

    def _new_candidates(self):
        return np.array([self.rng.choice(self.N, 2, replace=False) for _ in range(self.K)])

    def cand_features(self, pairs):
        nb = self.nbits
        bits = lambda v: [(int(v) >> k) & 1 for k in range(nb)]
        return np.array([bits(i) + bits(j) + bits(self.s[i]) + bits(self.s[j]) for i, j in pairs],
                        dtype=np.float32)

    def reset(self):
        self.s = self.rng.permutation(self.N)
        self.t = 0
        self.c = self.ev(self.s)
        self.best_nl = self.ev.last_nl
        self.tracker.update(self.s, self.ev.last_nl, self.c)
        self.pairs = self._new_candidates()
        return self._state(), self.cand_features(self.pairs)

    def step(self, a):
        i, j = self.pairs[a]
        self.s[i], self.s[j] = self.s[j], self.s[i]
        c2 = self.ev(self.s)
        nl = self.ev.last_nl
        r = (self.c - c2) / max(self.c, 1.0)
        self.c = c2
        if nl > self.best_nl:
            r += 0.1 * (nl - self.best_nl)
            self.best_nl = nl
        self.tracker.update(self.s, nl, c2)
        self.t += 1
        done = self.t >= self.steps
        self.pairs = self._new_candidates()
        return (self._state(), self.cand_features(self.pairs)), r, done
