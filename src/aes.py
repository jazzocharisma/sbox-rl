"""AES-128 dengan S-box yang bisa diganti (F2). Bukan AES standar jika S-box diganti."""
import numpy as np


def _xtime(a):
    a <<= 1
    return (a ^ 0x11B) & 0xFF if a & 0x100 else a


def _gmul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a = _xtime(a)
        b >>= 1
    return r


def aes_sbox():
    inv = [0] * 256
    for a in range(1, 256):
        for b in range(1, 256):
            if _gmul(a, b) == 1:
                inv[a] = b
                break
    S = []
    for x in range(256):
        b = inv[x]
        s = b
        for i in range(1, 5):
            s ^= ((b << i) | (b >> (8 - i))) & 0xFF
        S.append(s ^ 0x63)
    return np.array(S, dtype=np.int64)


_MUL = {m: [_gmul(x, m) for x in range(256)] for m in (2, 3, 9, 11, 13, 14)}
RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]


class AES128:
    def __init__(self, key, sbox=None):
        self.sbox = [int(v) for v in (aes_sbox() if sbox is None else sbox)]
        self.inv = [0] * 256
        for i, v in enumerate(self.sbox):
            self.inv[v] = i
        self.rk = self._expand(bytes(key))

    def _expand(self, key):
        w = [list(key[4 * i:4 * i + 4]) for i in range(4)]
        for i in range(4, 44):
            t = list(w[i - 1])
            if i % 4 == 0:
                t = t[1:] + t[:1]
                t = [self.sbox[b] for b in t]
                t[0] ^= RCON[i // 4 - 1]
            w.append([w[i - 4][j] ^ t[j] for j in range(4)])
        return [sum((w[4 * r + c] for c in range(4)), []) for r in range(11)]

    @staticmethod
    def _add(s, k):
        return [a ^ b for a, b in zip(s, k)]

    def _sub(self, s, box):
        return [box[b] for b in s]

    @staticmethod
    def _shift(s, inv=False):
        out = [0] * 16
        for c in range(4):
            for r in range(4):
                out[4 * c + r] = s[4 * ((c + (-r if inv else r)) % 4) + r]
        return out

    @staticmethod
    def _mix(s, inv=False):
        m = (14, 11, 13, 9) if inv else (2, 3, 1, 1)
        out = []
        for c in range(4):
            col = s[4 * c:4 * c + 4]
            for r in range(4):
                v = 0
                for j in range(4):
                    coef = m[(j - r) % 4]
                    v ^= _MUL[coef][col[j]] if coef in _MUL else col[j]
                out.append(v)
        return out

    def encrypt_block(self, p, return_rounds=False):
        s = self._add(list(p), self.rk[0])
        states = [s]
        for r in range(1, 10):
            s = self._add(self._mix(self._shift(self._sub(s, self.sbox))), self.rk[r])
            states.append(s)
        s = self._add(self._shift(self._sub(s, self.sbox)), self.rk[10])
        states.append(s)
        return (bytes(s), states) if return_rounds else bytes(s)

    def decrypt_block(self, c):
        s = self._add(list(c), self.rk[10])
        for r in range(9, 0, -1):
            s = self._sub(self._shift(s, True), self.inv)
            s = self._mix(self._add(s, self.rk[r]), True)
        s = self._sub(self._shift(s, True), self.inv)
        return bytes(self._add(s, self.rk[0]))

    # --- mode CTR (untuk gambar, F9) ---
    def ctr(self, data, nonce):
        out = bytearray()
        for i in range(0, len(data), 16):
            ctr_block = bytes(nonce[:8]) + (i // 16).to_bytes(8, "big")
            ks = self.encrypt_block(ctr_block)
            out += bytes(a ^ b for a, b in zip(data[i:i + 16], ks))
        return bytes(out)
