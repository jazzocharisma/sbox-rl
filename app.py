import os, sys, io, json
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from metrics import evaluate, ddt, lat, invert, is_bijective
from aes import AES128, aes_sbox
from infer import generate

MODEL = os.path.join(ROOT, "models", "dqn.npz")
FINAL = os.path.join(ROOT, "results", "sbox_final.csv")
AES_S = aes_sbox()

st.set_page_config(page_title="S-Box RL", layout="wide")
page = st.sidebar.radio("Halaman", ["Beranda", "Generator", "Analisis S-box", "Uji AES", "Eksperimen"])
st.sidebar.warning("AES dengan S-box baru BUKAN AES standar. Hanya untuk tugas kuliah, jangan dipakai untuk data nyata.")


def load_final():
    if os.path.exists(FINAL):
        return np.loadtxt(FINAL, dtype=int, delimiter=",")
    return None


def get_new_sbox():
    return st.session_state.get("sbox", load_final())


def sbox_csv(s):
    return "\n".join(str(int(v)) for v in s)


if page == "Beranda":
    st.title("S-Box 8x8 Baru dengan Reinforcement Learning")
    st.markdown("""
Alur: pelatihan DQN di Google Colab, ekspor bobot ke `dqn.npz`, lalu aplikasi ini membangkitkan S-box dengan NumPy.

**Batasan:** tidak ada klaim S-box ini lebih aman daripada AES. Kemiripan avalanche dan entropi
tidak membuktikan keamanan yang setara. Tidak untuk data nyata atau produksi.
""")

elif page == "Generator":
    st.header("Generator S-box")
    if not os.path.exists(MODEL):
        st.error("models/dqn.npz belum ada.")
    else:
        seed = st.number_input("Seed", 0, 10**6, 0)
        if st.button("Bangkitkan"):
            with st.spinner("Membangkitkan..."):
                s = generate(MODEL, int(seed))
            assert is_bijective(s)
            st.session_state["sbox"] = s
    s = get_new_sbox()
    if s is not None:
        st.success("S-box bijektif. NL dan DU: lihat halaman Analisis.")
        st.dataframe(np.array(s).reshape(16, 16), width="stretch")
        c1, c2 = st.columns(2)
        c1.download_button("Unduh S-box (CSV)", sbox_csv(s), "sbox.csv")
        c2.download_button("Unduh inverse (CSV)", sbox_csv(invert(s)), "inv_sbox.csv")
    fin = load_final()
    if fin is not None and st.button("Pakai S-box final dari Colab"):
        st.session_state["sbox"] = fin
        st.rerun()

elif page == "Analisis S-box":
    st.header("Analisis S-box")
    s = get_new_sbox()
    if s is None:
        st.info("Bangkitkan S-box di halaman Generator.")
    else:
        a, b = evaluate(AES_S), evaluate(s)
        rows = []
        for k in ["bijective", "NL", "DU", "LAP", "DAP", "fixed_points", "opposite_fixed_points", "bic_nl_min", "bic_sac_mean"]:
            rows.append({"Metrik": k, "AES": str(a[k]), "Baru": str(b[k])})
        rows.append({"Metrik": "SAC rata-rata", "AES": f'{a["SAC"]["mean"]:.4f}', "Baru": f'{b["SAC"]["mean"]:.4f}'})
        rows.append({"Metrik": "SAC simpangan maks", "AES": f'{a["SAC"]["max_dev"]:.4f}', "Baru": f'{b["SAC"]["max_dev"]:.4f}'})
        st.dataframe(rows, width="stretch")
        for name, fn in [("DDT", ddt), ("LAT", lat)]:
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.imshow(fn(AES_S)[:, :], title=f"{name} AES", aspect="auto"), width="stretch")
            c2.plotly_chart(px.imshow(fn(s)[:, :], title=f"{name} Baru", aspect="auto"), width="stretch")

elif page == "Uji AES":
    st.header("Uji AES-128: asli vs S-box baru")
    s = get_new_sbox()
    if s is None:
        st.info("Bangkitkan S-box di halaman Generator.")
    else:
        import time
        import imgtools as it
        k1, k2 = st.columns(2)
        key_txt = k1.text_input("Kunci (16 karakter)", "kuncirahasia1234")
        nonce_txt = k2.text_input("Nonce CTR (8 karakter, untuk gambar)", "nonce123")
        key = key_txt.ljust(16, "0").encode("utf-8")[:16]
        nonce = nonce_txt.ljust(8, "0").encode("utf-8")[:8]
        fp_new = it.fingerprint(s)
        ciphers = {"AES asli": (AES128(key, AES_S), it.fingerprint(AES_S)),
                   "AES S-box baru": (AES128(key, s), fp_new)}
        st.caption(f"Sidik jari S-box baru yang aktif: `{fp_new}`")
        tab1, tab_enc, tab_dec = st.tabs(["Teks", "Gambar: enkripsi", "Gambar: dekripsi"])

        with tab1:
            text = st.text_area("Teks", "Halo dunia, ini uji AES.")
            data = text.encode()
            data += b"\x00" * (-len(data) % 16)
            for label, (cipher, _) in ciphers.items():
                ct = b"".join(cipher.encrypt_block(data[i:i+16]) for i in range(0, len(data), 16))
                pt = b"".join(cipher.decrypt_block(ct[i:i+16]) for i in range(0, len(ct), 16))
                st.write(f"**{label}** ciphertext (hex): `{ct.hex()}`")
                st.write("Dekripsi sama dengan input:", pt == data)

        with tab_enc:
            up = st.file_uploader("Unggah gambar (mis. mandrill)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="up_enc")
            c1, c2 = st.columns(2)
            color = c1.radio("Mode warna", ["Abu-abu", "RGB"], horizontal=True)
            side = c2.selectbox("Sisi terpanjang (piksel)", [128, 256, 512], index=1)
            if up is not None:
                arr = it.load_array(up, "gray" if color == "Abu-abu" else "rgb", side)
                st.caption(f"Ukuran {arr.shape}, {arr.size} byte, sekitar {arr.size // 16} blok AES. "
                           "Python murni: 512 piksel dan RGB bisa memakan waktu lama.")
                if st.button("Enkripsi dan uji dekripsi"):
                    res = {}
                    with st.spinner("Mengenkripsi..."):
                        for name, (cipher, fp) in ciphers.items():
                            t0 = time.time(); ct_arr = it.ctr_array(cipher, arr, nonce); te = time.time() - t0
                            t0 = time.time(); back = it.ctr_array(cipher, ct_arr, nonce); td = time.time() - t0
                            res[name] = dict(cipher=ct_arr, ok=bool(np.array_equal(back, arr)), te=te, td=td,
                                             png=it.to_png(ct_arr, {"sbox_fp": fp, "versi": name, "mode": color}))
                    st.session_state["img_res"] = dict(arr=arr, res=res)
            R = st.session_state.get("img_res")
            if R is not None:
                cols = st.columns(3)
                items = [("Asli", R["arr"], None)] + [(f"Cipher {n}", r["cipher"], n) for n, r in R["res"].items()]
                for col, (label, a, n) in zip(cols, items):
                    col.image(a, caption=label)
                    h, v, d = it.correlations(a)
                    col.write(f"Entropi {it.entropy(a):.4f}")
                    col.write(f"Korelasi H {h:.4f} | V {v:.4f} | D {d:.4f}")
                    col.plotly_chart(go.Figure(go.Bar(y=np.bincount(a.ravel(), minlength=256))).update_layout(
                        title="Histogram", height=250, margin=dict(l=10, r=10, t=40, b=10)), width="stretch")
                    if n:
                        r = R["res"][n]
                        col.write(f"Dekripsi identik dengan gambar asli: **{r['ok']}**")
                        col.write(f"Waktu enkripsi {r['te']:.1f} dtk, dekripsi {r['td']:.1f} dtk")
                        col.download_button(f"Unduh cipher ({n}) PNG", r["png"],
                                            file_name=f"cipher_{'asli' if n == 'AES asli' else 'baru'}.png",
                                            mime="image/png", key=f"dl_{n}")

        with tab_dec:
            st.write("Unggah **PNG hasil enkripsi** dari tab sebelumnya. Kunci dan nonce harus sama dengan saat enkripsi.")
            up2 = st.file_uploader("Gambar terenkripsi (PNG)", type=["png"], key="up_dec")
            versi = st.radio("Versi AES untuk dekripsi", list(ciphers), horizontal=True)
            if up2 is not None:
                try:
                    c_arr, meta = it.read_png(up2.getvalue())
                except Exception as e:
                    st.error(str(e)); c_arr = None
                if c_arr is not None:
                    st.image(c_arr, caption="Gambar terenkripsi")
                    cipher, fp = ciphers[versi]
                    if meta.get("versi") and meta["versi"] != versi:
                        st.warning(f"Berkas ini dienkripsi dengan **{meta['versi']}**, tetapi Anda memilih **{versi}**. Hasilnya akan berupa noise.")
                    if meta.get("sbox_fp") and meta["sbox_fp"] != fp:
                        st.warning(f"S-box berkas (`{meta['sbox_fp']}`) berbeda dengan S-box aktif (`{fp}`). "
                                   "Pilih S-box yang sama di Generator (seed yang sama).")
                    if st.button("Dekripsi"):
                        with st.spinner("Mendekripsi..."):
                            out = it.ctr_array(cipher, c_arr, nonce)
                        st.image(out, caption=f"Hasil dekripsi ({versi})")
                        st.download_button("Unduh hasil dekripsi (PNG)", it.to_png(out), "hasil_dekripsi.png", "image/png")
                        st.caption("Jika hasilnya noise, kunci, nonce, versi AES, atau S-box tidak sama dengan saat enkripsi.")

elif page == "Eksperimen":
    st.header("Eksperimen")
    p = os.path.join(ROOT, "results", "experiment_runs.csv")
    if os.path.exists(p):
        import pandas as pd
        df = pd.read_csv(p)
        st.dataframe(df.groupby("method")[["NL", "DU"]].agg(["mean", "std", "max"]))
        st.plotly_chart(px.box(df, x="method", y="NL", points="all"), width="stretch")
    ph = os.path.join(ROOT, "results", "training_history.csv")
    if os.path.exists(ph):
        import pandas as pd
        h = pd.read_csv(ph)
        st.plotly_chart(px.line(h, x="ep", y=["best_nl_ep", "best_nl_all"], title="Training DQN 8x8"), width="stretch")
    if not (os.path.exists(p) or os.path.exists(ph)):
        st.info("Belum ada hasil eksperimen di results/.")
