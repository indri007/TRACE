# ─────────────────────────────────────────────────────────
#  main.py — Landing page VehicleScope AI (Colorful Light Theme)
# ─────────────────────────────────────────────────────────
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from config import APP_DESC, APP_NAME, APP_VERSION, CLASS_COLORS, CLASS_NAMES

# ── Page Setup ─────────────────────────────────────────────
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { 
        font-family: 'Plus Jakarta Sans', sans-serif; 
        color: #0F172A;
    }

    /* Hero gradient light & vibrant */
    .hero-container {
        background: linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 45%, #FDF2F8 100%);
        border: 1px solid #E0E7FF;
        border-radius: 24px;
        padding: 3.2rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 12px 35px -8px rgba(99, 102, 241, 0.15), 0 4px 12px -2px rgba(236, 72, 153, 0.08);
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -60%;
        left: -60%;
        width: 220%;
        height: 220%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.12) 0%, rgba(236, 72, 153, 0.06) 40%, transparent 70%);
        animation: pulse 6s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.7; }
        50%       { transform: scale(1.08); opacity: 1; }
    }
    .hero-title {
        font-size: 3.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2563EB 0%, #7C3AED 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.6rem;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        font-size: 1.15rem;
        color: #475569;
        font-weight: 500;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(90deg, rgba(37,99,235,0.1), rgba(236,72,153,0.1));
        border: 1px solid rgba(124,58,237,0.3);
        color: #6366F1;
        padding: 6px 16px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 1rem;
        box-shadow: 0 2px 6px rgba(99, 102, 241, 0.1);
    }

    /* Colorful Feature cards */
    .feature-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 1.6rem 1.4rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.05);
        position: relative;
        overflow: hidden;
    }
    .feature-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 16px 30px -4px rgba(99, 102, 241, 0.16);
        border-color: #C7D2FE;
    }
    .feature-icon { 
        font-size: 2.6rem; 
        margin-bottom: 0.8rem;
        display: inline-block;
        padding: 10px;
        border-radius: 16px;
        background: #F8FAFC;
    }
    .feature-title { 
        font-size: 1.05rem; 
        font-weight: 700; 
        color: #0F172A; 
        margin-bottom: 0.4rem; 
    }
    .feature-desc  { 
        font-size: 0.88rem; 
        color: #64748B; 
        line-height: 1.55; 
    }

    /* Class badges */
    .class-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 18px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.92rem;
        margin: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        transition: transform 0.2s ease;
    }
    .class-badge:hover {
        transform: scale(1.04);
    }
    .class-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }

    /* Metrics row */
    [data-testid="metric-container"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
        border-right: 1px solid #E2E8F0;
    }

    /* Hide default Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Hero Section ────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero-container">
        <div class="hero-badge">⚡ YOLO26 · NMS-free · Real-Time Edge AI</div>
        <div class="hero-title">🚗 {APP_NAME}</div>
        <div class="hero-sub">{APP_DESC}</div>
        <br>
        <div class="hero-sub" style="font-size:0.85rem; color:#64748B;">
            v{APP_VERSION} &nbsp;·&nbsp; Capstone Project Module 4
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Feature Cards ───────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
features = [
    ("🎯", "Deteksi Otomatis", "Upload gambar, model YOLO26 mendeteksi bus, car, dan van secara instan.", "#3B82F6"),
    ("📊", "Vehicle Counting", "Hitung jumlah tiap jenis kendaraan secara presisi dari satu gambar.", "#8B5CF6"),
    ("📡", "Live Logging via n8n", "Setiap deteksi di-log otomatis ke Google Sheets via n8n webhook.", "#10B981"),
    ("🔔", "Traffic Alert", "Notifikasi otomatis via Telegram jika kepadatan kendaraan melebihi threshold.", "#F59E0B"),
]
for col, (icon, title, desc, accent) in zip([c1, c2, c3, c4], features):
    with col:
        st.markdown(
            f"""
            <div class="feature-card" style="border-top: 4px solid {accent};">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# helper untuk rgba
def _hex_parts(h):
    h = h.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"

# ── Kelas yang Dideteksi ────────────────────────────────────
st.subheader("🏷️ Kelas Kendaraan yang Dideteksi")
badge_html = ""
for cls in CLASS_NAMES:
    color = CLASS_COLORS[cls]
    badge_html += (
        f'<span class="class-badge" style="background:rgba({_hex_parts(color)},0.12);'
        f'border:1px solid {color}; color:{color};">'
        f'<span class="class-dot" style="background:{color};"></span>'
        f'{cls.upper()}</span>'
    )

st.markdown(
    f'<div style="margin:1rem 0;">{badge_html}</div>',
    unsafe_allow_html=True,
)

# ── Quick Stats ─────────────────────────────────────────────
st.markdown("---")
st.subheader("⚡ Quick Info")
q1, q2, q3, q4 = st.columns(4)
q1.metric("Model", "YOLO26", "Ultralytics SOTA 2026")
q2.metric("Dataset", "Vehicle Detection", "3 kelas (9.725 imgs)")
q3.metric("Input Size", "640 × 640", "px")
q4.metric("Backend", "Apple MPS", "M4 Pro GPU")

# ── CTA ─────────────────────────────────────────────────────
st.markdown("---")
st.info(
    "👈 **Mulai dari sidebar** — pilih **Detection** untuk upload gambar dan deteksi kendaraan, "
    "atau **Dashboard** untuk melihat history log deteksi."
)
