# ─────────────────────────────────────────────────────────
#  pages/1_Detection.py — Halaman utama deteksi kendaraan
# ─────────────────────────────────────────────────────────
import io
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    APP_NAME,
    CLASS_COLORS,
    CLASS_NAMES,
    DEFAULT_CONF,
    DEFAULT_IOU,
)
from utils.model import run_inference
from utils.n8n_client import send_to_n8n

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title=f"Detection · {APP_NAME}",
    page_icon="🎯",
    layout="wide",
)

# ── CSS ─────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { 
        font-family: 'Plus Jakarta Sans', sans-serif; 
        color: #0F172A;
    }

    .result-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
    }
    .count-badge {
        background: linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 100%);
        border: 1px solid #E0E7FF;
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.06);
        transition: transform 0.2s ease;
    }
    .count-badge:hover {
        transform: translateY(-2px);
    }
    .count-number {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2563EB, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
    }
    .count-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.2rem;
    }
    .upload-zone {
        border: 2px dashed #818CF8;
        border-radius: 20px;
        padding: 2.2rem;
        text-align: center;
        background: linear-gradient(135deg, #EEF2FF 0%, #FAF5FF 100%);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.05);
    }
    [data-testid="metric-container"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
        border-right: 1px solid #E2E8F0;
    }
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ──────────────────────────────────────────────────
st.markdown(
    """
    <h1 style="
        background: linear-gradient(90deg, #2563EB, #7C3AED, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        margin-bottom: 0.2rem;
    ">🎯 Vehicle Detection</h1>
    <p style="color:#64748B; margin-bottom:1.5rem; font-weight: 500;">
        Upload gambar lalu biarkan model YOLO26 mendeteksi & menghitung kendaraan secara real-time.
    </p>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar — Parameters ────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parameter Model")
    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=DEFAULT_CONF,
        step=0.05,
        help="Semakin tinggi = hanya deteksi yang lebih yakin yang ditampilkan.",
    )
    iou_thresh = st.slider(
        "NMS IOU Threshold",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_IOU,
        step=0.05,
        help="Threshold untuk Non-Maximum Suppression (opsional pada YOLO26).",
    )

    st.markdown("---")
    st.markdown("### 📡 n8n Logging")
    auto_log = st.toggle(
        "Auto-log ke n8n",
        value=True,
        help="Setiap deteksi otomatis dikirim ke n8n webhook.",
    )
    st.markdown("---")
    st.markdown(
        "**Kelas:**\n"
        + "".join(
            f"\n- <span style='color:{CLASS_COLORS[c]};'>●</span> **{c.upper()}**"
            for c in CLASS_NAMES
        ),
        unsafe_allow_html=True,
    )

# ── Upload Section ──────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📁 Upload Gambar Kendaraan",
    type=["jpg", "jpeg", "png", "webp"],
    help="Format yang didukung: JPG, JPEG, PNG, WebP",
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="upload-zone">
            <p style="font-size:2.4rem; margin:0;">📷</p>
            <p style="color:#334155; font-size:1.05rem; font-weight:600; margin:0.6rem 0 0;">
                Drag & drop gambar di sini, atau klik tombol Browse
            </p>
            <p style="color:#64748B; font-size:0.85rem; margin:0.3rem 0 0;">
                Mendukung resolusi tinggi · Format JPG, PNG, WebP
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Inference ───────────────────────────────────────────────
image = Image.open(uploaded_file).convert("RGB")

col_img, col_result = st.columns([1.2, 1], gap="large")

with col_img:
    st.markdown("**📸 Gambar Input**")
    st.image(image, use_container_width=True, caption=uploaded_file.name)

with col_result:
    with st.spinner("🔍 Mendeteksi kendaraan dengan YOLO26..."):
        t0 = time.perf_counter()
        annotated_img, counts, detections = run_inference(image, conf_thresh, iou_thresh)
        elapsed = time.perf_counter() - t0

    st.markdown(f"**✅ Hasil Deteksi** &nbsp; <span style='color:#6366F1; font-weight:600; font-size:0.85rem;'>({elapsed*1000:.0f} ms)</span>", unsafe_allow_html=True)
    st.image(annotated_img, use_container_width=True, caption="Hasil Anotasi YOLO26")

    # Tombol download annotated image
    buf = io.BytesIO()
    annotated_img.save(buf, format="PNG")
    st.download_button(
        "⬇️ Download Hasil",
        data=buf.getvalue(),
        file_name=f"detected_{uploaded_file.name}",
        mime="image/png",
        use_container_width=True,
    )

# ── Count Summary ────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Rekap Kendaraan Terdeteksi")

total = sum(counts.values())
c_bus, c_car, c_van, c_total = st.columns(4)

for col, (cls, emoji) in zip(
    [c_car, c_van, c_bus],
    [("car", "🚗"), ("van", "🚐"), ("bus", "🚌")],
):
    color = CLASS_COLORS[cls]
    with col:
        st.markdown(
            f"""
            <div class="count-badge">
                <div style="font-size:2rem;">{emoji}</div>
                <div class="count-number" style="
                    background: linear-gradient(90deg, {color}, {color});
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                ">{counts[cls]}</div>
                <div class="count-label">{cls}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with c_total:
    st.markdown(
        f"""
        <div class="count-badge">
            <div style="font-size:2rem;">🚦</div>
            <div class="count-number" style="
                background: linear-gradient(90deg, #2563EB, #7C3AED, #EC4899);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            ">{total}</div>
            <div class="count-label">Total</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Summary Text ─────────────────────────────────────────────
if detections:
    summary_parts = [f"**{v} {k}**" for k, v in counts.items() if v > 0]
    summary_text  = ", ".join(summary_parts)
    st.success(f"🔎 Terdeteksi: {summary_text} — Total **{total} kendaraan**")
else:
    st.warning("⚠️ Tidak ada kendaraan yang terdeteksi. Coba sesuaikan Confidence Threshold.")

# ── Bar Chart ────────────────────────────────────────────────
if detections:
    df_counts = pd.DataFrame(
        [{"Kelas": k.upper(), "Jumlah": v, "Warna": CLASS_COLORS[k]} for k, v in counts.items()]
    )
    fig = px.bar(
        df_counts,
        x="Kelas",
        y="Jumlah",
        color="Kelas",
        color_discrete_map={k.upper(): v for k, v in CLASS_COLORS.items()},
        text="Jumlah",
        template="plotly_white",
        title="Distribusi Kendaraan Terdeteksi",
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color="#1E293B"),
        yaxis=dict(showgrid=True, gridcolor="rgba(15,23,42,0.06)"),
        height=280,
        margin=dict(t=40, b=20, l=0, r=0),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Detail Table ─────────────────────────────────────────────
if detections:
    with st.expander("📋 Detail Bounding Box Deteksi"):
        df_det = pd.DataFrame(
            [
                {
                    "#"          : i + 1,
                    "Kelas"      : d["class"].upper(),
                    "Confidence" : f"{d['confidence']:.1%}",
                    "BBox [x1,y1,x2,y2]": [round(v, 1) for v in d["bbox"]],
                }
                for i, d in enumerate(detections)
            ]
        )
        st.dataframe(df_det, use_container_width=True, hide_index=True)

# ── n8n Log ──────────────────────────────────────────────────
st.markdown("---")
col_log1, col_log2 = st.columns([3, 1])
with col_log1:
    st.markdown("### 📡 Log ke n8n")
    st.caption(
        "Kirim hasil deteksi ini ke n8n untuk di-log ke Google Sheets "
        "dan memicu alert jika traffic padat (>10 kendaraan)."
    )

with col_log2:
    manual_log = st.button(
        "📤 Kirim ke n8n",
        use_container_width=True,
        type="primary",
        disabled=(len(detections) == 0),
    )

if auto_log and detections:
    ok, msg = send_to_n8n(uploaded_file.name, counts, detections)
    if ok:
        st.toast(msg, icon="✅")
    else:
        st.toast(msg, icon="⚠️")

if manual_log:
    with st.spinner("Mengirim ke n8n..."):
        ok, msg = send_to_n8n(uploaded_file.name, counts, detections)
    if ok:
        st.success(msg)
    else:
        st.error(msg)
