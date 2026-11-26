"""Metrik S-box n x n (F1). Berlaku untuk n = 4, 6, 8."""
import numpy as np


def _parity_table(size):
    p = np.zeros(size, dtype=np.int8)
    for i in range(1, size):
        p[i] = p[i >> 1] ^ (i & 1)
    return p


def fwht_rows(M):
    """Fast Walsh-Hadamard transform sepanjang sumbu terakhir (salinan)."""
    M = M.astype(np.int32).copy()
    n = M.shape[-1]
    h = 1
    while h < n:
        M = M.reshape(M.shape[0], -1, 2, h)
        a, b = M[:, :, 0, :].copy(), M[:, :, 1, :].copy()
        M[:, :, 0, :], M[:, :, 1, :] = a + b, a - b
        M = M.reshape(M.shape[0], n)
        h *= 2
    return M


def walsh_spectrum(sbox):
    """W[b, a] = sum_x (-1)^{b.S(x) + a.x}, b = 0..N-1 (baris b=0 ikut dihitung)."""
    S = np.asarray(sbox)
    N = len(S)
    par = _parity_table(N)
    b = np.arange(N)[:, None]
    f = 1 - 2 * par[b & S[None, :]].astype(np.int32)  # (-1)^{b.S(x)}
    return fwht_rows(f)


def nonlinearity(sbox, W=None):
    N = len(sbox)
    W = walsh_spectrum(sbox) if W is None else W
    return N // 2 - int(np.abs(W[1:]).max()) // 2


def ddt(sbox):
    S = np.asarray(sbox)
    N = len(S)
    x = np.arange(N)
    T = np.zeros((N, N), dtype=np.int32)
    for dx in range(N):
        T[dx] = np.bincount(S ^ S[x ^ dx], minlength=N)
    return T


def differential_uniformity(sbox):
    return int(ddt(sbox)[1:].max())


def lat(sbox):
    """LAT[a, b] = #{x : a.x = b.S(x)} - N/2 (indeks a = mask input, b = mask output)."""
    W = walsh_spectrum(sbox)  # W[b, a]
    return (W.T // 2)


def lap(sbox):
    N = len(sbox)
    return int(np.abs(walsh_spectrum(sbox)[1:]).max()) / 2 / N


def dap(sbox):
    return differential_uniformity(sbox) / len(sbox)


def is_bijective(sbox):
    S = np.asarray(sbox)
    return len(S) > 0 and len(np.unique(S)) == len(S) and S.min() == 0 and S.max() == len(S) - 1


def invert(sbox):
    S = np.asarray(sbox)
    inv = np.zeros_like(S)
    inv[S] = np.arange(len(S))
    return inv


def _bits(v, n):
    return ((v[:, None] >> np.arange(n)) & 1).astype(np.int8)


def sac_matrix(sbox):
    S = np.asarray(sbox)
    N = len(S)
    n = N.bit_length() - 1
    x = np.arange(N)
    M = np.zeros((n, n))
    for i in range(n):
        d = _bits(S[x] ^ S[x ^ (1 << i)], n)
        M[i] = d.mean(axis=0)
    return M


def sac_summary(sbox):
    M = sac_matrix(sbox)
    return {"mean": float(M.mean()), "max_dev": float(np.abs(M - 0.5).max())}


def bic(sbox):
    """BIC-NL (minimum) dan BIC-SAC (rata-rata) atas semua pasangan bit output."""
    S = np.asarray(sbox)
    N = len(S)
    n = N.bit_length() - 1
    par = _parity_table(N)
    bits = _bits(S, n)
    nls, sacs = [], []
    x = np.arange(N)
    for j in range(n):
        for k in range(j + 1, n):
            f = (bits[:, j] ^ bits[:, k]).astype(np.int32)
            F = (1 - 2 * f)[None, :]
            W = fwht_rows(F)[0]
            nls.append(N // 2 - int(np.abs(W).max()) // 2)
            for i in range(n):
                sacs.append(np.mean(f ^ f[x ^ (1 << i)]))
    return {"bic_nl_min": int(min(nls)), "bic_sac_mean": float(np.mean(sacs))}


def fixed_points(sbox):
    S = np.asarray(sbox)
    x = np.arange(len(S))
    return int(np.sum(S == x)), int(np.sum(S == (x ^ (len(S) - 1))))


def evaluate(sbox):
    S = np.asarray(sbox)
    fp, ofp = fixed_points(S)
    out = {
        "bijective": bool(is_bijective(S)),
        "NL": nonlinearity(S),
        "DU": differential_uniformity(S),
        "LAP": lap(S),
        "DAP": dap(S),
        "SAC": sac_summary(S),
        "fixed_points": fp,
        "opposite_fixed_points": ofp,
    }
    out.update(bic(S))
    return out
