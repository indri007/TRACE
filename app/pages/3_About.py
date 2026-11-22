# ─────────────────────────────────────────────────────────
#  pages/3_About.py — Info model & project (Colorful Light Theme)
# ─────────────────────────────────────────────────────────
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import APP_DESC, APP_NAME, APP_VERSION, CLASS_COLORS, CLASS_NAMES, MODEL_PATH

st.set_page_config(
    page_title=f"About · {APP_NAME}",
    page_icon="ℹ️",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { 
        font-family: 'Plus Jakarta Sans', sans-serif; 
        color: #0F172A;
    }
    .info-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 1.8rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
        margin-bottom: 1.2rem;
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

st.markdown(
    """
    <h1 style="
        background: linear-gradient(90deg, #2563EB, #7C3AED, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        margin-bottom: 0.5rem;
    ">ℹ️ About VehicleScope AI</h1>
    """,
    unsafe_allow_html=True,
)

# ── Project Info ────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div class="info-card" style="border-top: 4px solid #3B82F6;">
            <h3 style="margin-top:0;">🎯 Tentang Proyek</h3>
            <p><strong>{APP_NAME}</strong> adalah aplikasi deteksi kendaraan berbasis AI yang dibangun sebagai Capstone Project Module 4.</p>
            <p>Aplikasi ini menggabungkan:</p>
            <ul>
                <li>🤖 <strong>YOLO26</strong> (Ultralytics SOTA 2026) — model object detection end-to-end native NMS-free</li>
                <li>⚙️ <strong>n8n</strong> — workflow automation untuk pencatatan & notifikasi otomatis</li>
                <li>🎨 <strong>Streamlit</strong> — web frontend modern dan interaktif</li>
            </ul>
            <blockquote style="border-left: 3px solid #3B82F6; padding-left: 10px; color:#475569; margin: 10px 0;">
                <em>{APP_DESC}</em>
            </blockquote>
            <hr style="margin: 15px 0; border: none; border-top: 1px solid #E2E8F0;">
            <strong>🚀 Keunggulan YOLO26:</strong>
            <ul>
                <li><strong>Native NMS-Free:</strong> Inferensi instan tanpa post-processing bottleneck (0.08 ms).</li>
                <li><strong>MuSGD Optimizer:</strong> Konvergensi bobot model lebih stabil.</li>
                <li><strong>Tanpa Distribution Focal Loss (DFL):</strong> Arsitektur kepala deteksi lebih ramping dan cepat.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="info-card" style="border-top: 4px solid #8B5CF6;">
            <h3 style="margin-top:0;">📊 Dataset & Model Spec</h3>
            <table style="width:100%; border-collapse: collapse; font-size: 0.95rem;">
                <tr style="border-bottom: 1px solid #F1F5F9;">
                    <td style="padding: 8px 0; color: #64748B;">Nama Dataset</td>
                    <td style="padding: 8px 0; font-weight: 600;">Vehicle Detection (Roboflow Universe)</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9;">
                    <td style="padding: 8px 0; color: #64748B;">Lisensi</td>
                    <td style="padding: 8px 0; font-weight: 600;">Creative Commons CC BY 4.0</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9;">
                    <td style="padding: 8px 0; color: #64748B;">Jumlah Gambar</td>
                    <td style="padding: 8px 0; font-weight: 600;">9.725 Gambar (19.334 Anotasi)</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9;">
                    <td style="padding: 8px 0; color: #64748B;">Kelas Objek</td>
                    <td style="padding: 8px 0; font-weight: 600;">Bus, Car, Van</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9;">
                    <td style="padding: 8px 0; color: #64748B;">Input Resolution</td>
                    <td style="padding: 8px 0; font-weight: 600;">640 × 640 piksel</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9;">
                    <td style="padding: 8px 0; color: #64748B;">Ukuran Model</td>
                    <td style="padding: 8px 0; font-weight: 600;">21.8M Parameter (75.0 GFLOPs)</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9;">
                    <td style="padding: 8px 0; color: #64748B;">Status Model Lokal</td>
                    <td style="padding: 8px 0; font-weight: 600;">{'✅ Tersedia (best.pt)' if MODEL_PATH.exists() else '⚠️ Model belum dimuat'}</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Tech Stack ──────────────────────────────────────────────
st.markdown("### 🛠️ Teknologi yang Digunakan")
t1, t2, t3, t4 = st.columns(4)

stacks = [
    ("🧠", "Ultralytics YOLO26", "Detektor objek deep learning SOTA dengan inferensi NMS-free.", "#3B82F6"),
    ("⚡", "Streamlit", "Framework UI modern berbasis Python untuk deployment analitik data.", "#EC4899"),
    ("⚙️", "n8n Workflow", "Automation node menghubungkan inferensi model ke Google Sheets & Telegram.", "#10B981"),
    ("📈", "Plotly Charts", "Visualisasi interaktif untuk grafik bar, pie, dan tren data.", "#F59E0B"),
]

for col, (icon, title, desc, border_color) in zip([t1, t2, t3, t4], stacks):
    with col:
        st.markdown(
            f"""
            <div class="info-card" style="border-top: 4px solid {border_color}; text-align: center; height: 100%;">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">{icon}</div>
                <div style="font-weight: 700; color: #0F172A; margin-bottom: 0.3rem;">{title}</div>
                <div style="font-size: 0.85rem; color: #64748B;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
