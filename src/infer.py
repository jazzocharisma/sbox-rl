"""Inferensi NumPy tanpa PyTorch (dipakai Streamlit, F10). Memuat dqn.npz."""
import numpy as np
from rl_core import SboxEnv
from metrics import is_bijective


def q_values(w, x):
    h = np.maximum(x @ w["W0"] + w["b0"], 0)
    h = np.maximum(h @ w["W1"] + w["b1"], 0)
    return (h @ w["W2"] + w["b2"]).ravel()


def generate(npz_path, seed, nbits=8, steps=200, n_cand=64):
    """Mulai dari permutasi acak (seed), pilih aksi greedy, kembalikan S-box terbaik episode."""
    w = np.load(npz_path)
    env = SboxEnv(nbits, steps, n_cand, seed=seed)
    state, cf = env.reset()
    for _ in range(steps):
        x = np.concatenate([np.repeat(state[None], len(cf), 0), cf], 1)
        (state, cf), _, done = env.step(int(np.argmax(q_values(w, x))))
        if done:
            break
    s = env.tracker.s
    assert is_bijective(s)
    return s
