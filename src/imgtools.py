"""Alat uji gambar untuk AES mode CTR (F9). Dipakai app.py; tidak bergantung pada Streamlit."""
import io, hashlib
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def fingerprint(sbox):
    """Sidik jari singkat S-box, ditanam di PNG agar dekripsi bisa mendeteksi S-box yang tidak cocok."""
    return hashlib.sha256(bytes(int(v) for v in sbox)).hexdigest()[:8]


def load_array(file_or_image, color="gray", max_side=256):
    im = file_or_image if isinstance(file_or_image, Image.Image) else Image.open(file_or_image)
    im = im.convert("L" if color == "gray" else "RGB")
    im.thumbnail((max_side, max_side))          # hanya memperkecil, rasio dipertahankan
    return np.array(im, dtype=np.uint8)


def ctr_array(cipher, arr, nonce):
    out = cipher.ctr(arr.tobytes(), nonce)
    return np.frombuffer(out, dtype=np.uint8).reshape(arr.shape).copy()


def to_png(arr, meta=None):
    info = PngInfo()
    for k, v in (meta or {}).items():
        info.add_text(k, str(v))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def read_png(data):
    """Mengembalikan (array uint8, metadata). Hanya PNG lossless mode L atau RGB."""
    im = Image.open(io.BytesIO(data))
    if im.format != "PNG":
        raise ValueError("Gambar terenkripsi harus berupa PNG hasil enkripsi (JPEG merusak ciphertext).")
    if im.mode not in ("L", "RGB"):
        raise ValueError(f"Mode gambar {im.mode} tidak didukung (hanya abu-abu atau RGB).")
    meta = dict(getattr(im, "text", {}) or {})
    return np.array(im, dtype=np.uint8), meta


def entropy(a):
    p = np.bincount(a.ravel(), minlength=256) / a.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def correlations(a):
    g = (a if a.ndim == 2 else a.mean(axis=2)).astype(np.float64)
    h = np.corrcoef(g[:, :-1].ravel(), g[:, 1:].ravel())[0, 1]
    v = np.corrcoef(g[:-1, :].ravel(), g[1:, :].ravel())[0, 1]
    d = np.corrcoef(g[:-1, :-1].ravel(), g[1:, 1:].ravel())[0, 1]
    return float(h), float(v), float(d)
