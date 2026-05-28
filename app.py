# ============================================================
#  app.py  —  Streamlit UI (REDESIGN v6 — Figma Match)
#  Drowsiness Detection System (Swin Transformer + LSTM)
# ============================================================
# PERUBAHAN v6 (vs v5):
#   - Tampilan disesuaikan dengan desain Figma (5 PDF: Beranda,
#     Tentang, Demo, Fitur, Kontak).
#   - Sidebar: branding "Drowsy DETECTION · THESIS PROJECT 2026",
#     menu bahasa Indonesia (Beranda/Tentang/Demo/Fitur/Kontak)
#     dengan ikon, card COMPUTE di bawah.
#   - Setiap halaman punya eyebrow "BAB 0X · NAMA" dan footer
#     "Real-time Driver Drowsiness Detection · © 2026 · Thesis Edition".
#   - Beranda: hero dengan 2 CTA, 4-col stat strip kompak, 2 kartu
#     (Dataset + 3 tile angka, Metrik + β=2).
#   - Tentang: 5 step dengan badge tags kanan.
#   - Demo: status card 2x4 grid + drowsy bar 3-label + alarm READY.
#   - Fitur: 12 item bernomor format bersih.
#   - Kontak: single card + Instagram card.
#
#   FUNGSI TIDAK DIUBAH: VideoProcessor, AlarmManager (winsound),
#   shared state, CSV logger, auto-refresh, WebRTC. Semua metric
#   tetap di-track persis seperti v5.
#
# DEPENDENCY:
#   pip install streamlit streamlit-webrtc streamlit-autorefresh
# ============================================================


import os
import csv
import time
import threading
from datetime import datetime
import platform

import av
import numpy as np
import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
)


try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False


from inference import DrowsinessDetector, USE_CUDA, DetectionResult


import base64

def play_web_alarm_fast(tipe_alarm):
    """
    tipe_alarm: "WARNING" atau "DANGER"
    """
    # 1. Sesuaikan nama file dengan file audio MP3 ukuran kecil (< 50KB) yang Anda punya
    file_path = "alarm_pendek.mp3" if tipe_alarm == "WARNING" else "alarm_panjang.mp3"
    
    try:
        # Buka file audio dan ubah ke bentuk Teks (Base64)
        with open(file_path, "rb") as f:
            data_audio = f.read()
            b64_audio = base64.b64encode(data_audio).decode()
            
        # 2. Inject paksa ke browser agar Autoplay instan tanpa loading
        html_code = f"""
            <audio autoplay="true" style="display:none;">
                <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
            </audio>
        """
        st.markdown(html_code, unsafe_allow_html=True)
    except Exception as e:
        pass

# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title=" Drowsy Detection",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
#  STYLE  — Figma match (editorial, minimalist)
# ============================================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,300;1,9..144,400;1,9..144,500;1,9..144,600&family=Inter:wght@400;500;600;700&display=swap');


  :root {
    --bg-top:       #F0F9FF;
    --bg-bot:       #E0F2FE;
    --surface:      #FFFFFF;
    --surface-soft: rgba(255,255,255,0.92);
    --border:       #BAE6FD;
    --border-soft:  #E0F2FE;
    --ink:          #0C4A6E;
    --ink-2:        #1E293B;
    --muted:        #64748B;
    --muted-soft:   #94A3B8;
    --accent:       #0284C7;
    --accent-2:     #0369A1;
    --accent-soft:  #38BDF8;
    --ok:           #10B981;
    --warn:         #F59E0B;
    --danger:       #EF4444;
    --shadow:       0 10px 30px -10px rgba(2,132,199,0.14);
    --shadow-soft:  0 2px 8px rgba(2,132,199,0.06);
    --radius:       16px;
    --radius-sm:    12px;
  }


  /* ── Base ── */
  html, body, [data-testid="stAppViewContainer"],
  [data-testid="stAppViewContainer"] > section,
  .stApp {
    background:
      radial-gradient(1200px 600px at 85% -10%, #DBEAFE 0%, transparent 60%),
      linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bot) 100%) !important;
    background-attachment: fixed !important;
    color: var(--ink-2) !important;
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
  }


  .block-container {
    padding-top: 3.2rem !important;
    padding-bottom: 4rem;
    max-width: 1180px;
  }


  header[data-testid="stHeader"] {
    background: rgba(240, 249, 255, 0.75) !important;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--border-soft);
    height: 2.6rem;
  }
  header[data-testid="stHeader"] > div { height: 2.6rem; }


  /* ── FORCE LIGHT THEME (override dark mode) ── */
  [data-testid="stSidebar"],
  section[data-testid="stSidebar"] > div,
  [data-testid="stSidebarContent"] {
    background: #F5FBFF !important;
    border-right: 1px solid var(--border-soft);
  }
  [data-testid="stSidebar"] *,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] div { color: var(--ink) !important; }
  [data-testid="stSidebar"] [role="radiogroup"] label p,
  [data-testid="stSidebar"] [role="radiogroup"] label div {
    color: var(--ink) !important;
  }


  /* ── Typography ── */
  h1, h2, h3, h4, .hero-title, .sec-title, .chapter-title {
    font-family: 'Fraunces', 'Times New Roman', serif;
    color: var(--ink);
    letter-spacing: -0.02em;
    font-feature-settings: "ss01";
  }


  .bab-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1rem;
  }


  .hero-title {
    font-size: clamp(2.6rem, 5.6vw, 4.8rem);
    font-weight: 400;
    line-height: 1.02;
    margin: 0 0 1.4rem;
  }
  .hero-title em,
  .sec-title em,
  .chapter-title em {
    font-style: italic;
    font-weight: 400;
    color: var(--accent-2);
  }


  .hero-sub {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    font-weight: 400;
    color: var(--muted);
    line-height: 1.7;
    max-width: 680px;
    margin-bottom: 1.8rem;
  }


  .sec-title {
    font-size: 2.2rem;
    font-weight: 400;
    margin: 0 0 1rem;
    line-height: 1.15;
  }


  .chapter-title {
    font-size: clamp(2.2rem, 4.4vw, 3.2rem);
    font-weight: 400;
    line-height: 1.1;
    margin: 0 0 1.4rem;
  }


  .eyebrow-sm {
    font-family: 'Inter', sans-serif;
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 8px;
  }


  p, li, div[data-testid="stMarkdownContainer"] p {
    color: var(--ink-2);
    line-height: 1.7;
    font-size: 0.95rem;
  }

  .muted { color: var(--muted); font-size: 0.9rem; line-height: 1.65; }


  /* ── CTA buttons (hero) ── */
  .cta-row { display: flex; gap: 0.8rem; margin-top: 0.4rem; flex-wrap: wrap; }
  .cta {
    display: inline-flex;
    align-items: center;
    padding: 0.7rem 1.4rem;
    border-radius: 999px;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    font-weight: 500;
    text-decoration: none;
    transition: all .2s ease;
    border: 1px solid transparent;
    cursor: pointer;
  }
  .cta-primary {
    background: var(--accent);
    color: #FFFFFF !important;
    border-color: var(--accent);
  }
  .cta-primary:hover {
    background: var(--accent-2);
    border-color: var(--accent-2);
    transform: translateY(-1px);
  }
  .cta-ghost {
    background: transparent;
    color: var(--ink) !important;
    border: 1px solid var(--border);
  }
  .cta-ghost:hover {
    background: var(--surface);
    border-color: var(--accent);
  }


  /* ── Hero stats (4-col horizontal strip card) ── */
  .stat-strip {
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius);
    box-shadow: var(--shadow-soft);
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    margin: 2.4rem 0 1.8rem;
    overflow: hidden;
  }
  .stat-cell {
    padding: 1.4rem 1.6rem;
    border-right: 1px solid var(--border-soft);
  }
  .stat-cell:last-child { border-right: none; }
  .stat-cell .stat-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 10px;
  }
  .stat-cell .stat-value {
    font-family: 'Fraunces', serif;
    font-size: 1.45rem;
    font-weight: 500;
    color: var(--ink);
    line-height: 1.15;
    letter-spacing: -0.01em;
    margin-bottom: 6px;
  }
  .stat-cell .stat-meta {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.02em;
  }


  /* ── Generic card ── */
  .card {
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius);
    padding: 1.8rem 2rem;
    box-shadow: var(--shadow-soft);
    margin-bottom: 1.2rem;
    transition: border-color .25s ease, box-shadow .25s ease;
  }
  .card:hover {
    border-color: var(--border);
    box-shadow: var(--shadow);
  }
  .card-title {
    font-family: 'Fraunces', serif;
    font-size: 1.6rem;
    font-weight: 500;
    line-height: 1.2;
    color: var(--ink);
    letter-spacing: -0.015em;
    margin: 6px 0 0.8rem;
  }


  /* ── Dataset number tiles (3 across) ── */
  .num-tiles {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.7rem;
    margin-top: 1.2rem;
  }
  .num-tile {
    background: #FFFFFF;
    border: 1px solid var(--border-soft);
    border-radius: var(--radius-sm);
    padding: 0.95rem 0.8rem;
    text-align: center;
  }
  .num-tile .n-value {
    font-family: 'Fraunces', serif;
    font-size: 1.55rem;
    font-weight: 500;
    color: var(--ink);
    line-height: 1.1;
    margin-bottom: 4px;
  }
  .num-tile .n-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--muted);
    line-height: 1.3;
  }


  /* ── Metric kv row (Recall Drowsy / High Priority) ── */
  .kv-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.7rem 0;
    border-bottom: 1px solid var(--border-soft);
  }
  .kv-row:last-child { border-bottom: none; }
  .kv-key {
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    color: var(--muted);
    font-weight: 500;
  }
  .kv-value {
    font-family: 'Fraunces', serif;
    font-size: 1rem;
    font-weight: 500;
    color: var(--ink);
    font-style: italic;
  }
  .kv-value.plain { font-style: normal; }


  /* ── Pipeline step (About) ── */
  .step-row {
    display: grid;
    grid-template-columns: 90px 1fr auto;
    gap: 1.6rem;
    padding: 1.6rem 0;
    border-top: 1px solid var(--border-soft);
    align-items: start;
  }
  .step-row:first-of-type { border-top: none; padding-top: 0.8rem; }
  .step-num-lg {
    font-family: 'Fraunces', serif;
    font-style: italic;
    font-weight: 300;
    font-size: 2.4rem;
    color: var(--accent);
    line-height: 1;
    letter-spacing: -0.03em;
  }
  .step-content-title {
    font-family: 'Fraunces', serif;
    font-size: 1.4rem;
    font-weight: 500;
    color: var(--ink);
    letter-spacing: -0.015em;
    margin-bottom: 0.4rem;
    line-height: 1.2;
  }
  .step-content-body {
    font-family: 'Inter', sans-serif;
    color: var(--ink-2);
    font-size: 0.93rem;
    line-height: 1.65;
    max-width: 520px;
  }
  .tags-col {
    display: flex;
    flex-direction: column;
    gap: 8px;
    align-items: flex-end;
    flex-wrap: wrap;
  }
  .tag-pill {
    background: #F0F9FF;
    color: var(--accent-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 4px 12px;
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    white-space: nowrap;
  }


  /* ── Numbered feature item (Fitur) ── */
  .f-item {
    padding: 1.15rem 0;
    border-top: 1px solid var(--border-soft);
  }
  .f-item:first-of-type { border-top: none; padding-top: 0.6rem; }
  .f-item-head {
    display: flex;
    align-items: baseline;
    gap: 14px;
    margin-bottom: 6px;
  }
  .f-num {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.08em;
    min-width: 26px;
  }
  .f-title {
    font-family: 'Fraunces', serif;
    font-size: 1.12rem;
    font-weight: 500;
    color: var(--ink);
    letter-spacing: -0.01em;
  }
  .f-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    color: var(--muted);
    line-height: 1.65;
    margin-left: 40px;
    max-width: 720px;
  }


  /* ── Status pill badges ── */
  .pill {
    display: inline-flex;
    align-items: center;
    padding: 6px 14px;
    border-radius: 999px;
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    border: 1px solid transparent;
    gap: 6px;
  }
  .pill-normal  { background: #ECFDF5; color: #065F46; border-color: #A7F3D0; }
  .pill-warning { background: #FFFBEB; color: #92400E; border-color: #FDE68A; }
  .pill-danger  { background: #FEF2F2; color: #991B1B; border-color: #FECACA; }
  .pill-info    { background: #EFF6FF; color: #1E40AF; border-color: #BFDBFE; }
  .pill-ghost   { background: #FFFFFF; color: var(--ink); border-color: var(--border-soft); }


  /* ── Demo: Webcam idle panel ── */
  .webcam-idle {
    background: linear-gradient(180deg, #F0F9FF 0%, #E0F2FE 100%);
    border: 1px dashed var(--border);
    border-radius: var(--radius);
    padding: 3.2rem 1.4rem 1.4rem;
    text-align: center;
    margin-bottom: 1rem;
  }
  .webcam-idle .idle-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
  }
  .webcam-idle .idle-text {
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    color: var(--muted);
    margin-bottom: 1.6rem;
  }
  .webcam-idle .idle-meta {
    display: flex;
    justify-content: center;
    gap: 1.8rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    color: var(--muted-soft);
    text-transform: uppercase;
  }
  .webcam-idle .idle-meta b {
    color: var(--ink);
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: none;
    font-size: 0.84rem;
  }


  /* ── Demo: metric tile 2x4 grid ── */
  .mgrid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2px;
    background: var(--border-soft);
    border-radius: var(--radius-sm);
    overflow: hidden;
    border: 1px solid var(--border-soft);
  }
  .mtile {
    background: var(--surface);
    padding: 0.85rem 0.95rem;
  }
  .mtile .ml {
    font-family: 'Inter', sans-serif;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 4px;
    line-height: 1.3;
  }
  .mtile .mv {
    font-family: 'Fraunces', serif;
    font-size: 1.15rem;
    font-weight: 500;
    color: var(--ink);
    letter-spacing: -0.01em;
    line-height: 1.2;
  }


  /* ── Drowsy probability bar with 3-label strip ── */
  .dp-wrap { margin: 1.1rem 0 0.4rem; }
  .dp-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 8px;
  }
  .dp-head .dp-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .dp-head .dp-pct {
    font-family: 'Fraunces', serif;
    font-size: 1.05rem;
    font-weight: 500;
    color: var(--ink);
  }
  .dp-bar {
    background: #E2E8F0;
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
  }
  .dp-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.45s ease, background 0.3s ease;
  }
  .dp-legend {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    margin-top: 10px;
    font-family: 'Inter', sans-serif;
    font-size: 0.66rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .dp-legend > div { padding: 0; }
  .dp-legend .ll { text-align: left; }
  .dp-legend .lc { text-align: center; }
  .dp-legend .lr { text-align: right; }


  /* ── Alarm output card ── */
  .alarm-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.9rem 1rem;
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius-sm);
    margin-top: 1rem;
  }
  .alarm-row .al-left {
    display: flex;
    align-items: center;
    gap: 0.8rem;
  }
  .alarm-row .al-ico {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: #EFF6FF;
    color: var(--accent);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.95rem;
  }
  .alarm-row .al-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .alarm-row .al-body {
    font-family: 'Fraunces', serif;
    font-size: 0.98rem;
    font-weight: 500;
    color: var(--ink);
    letter-spacing: -0.005em;
  }


  /* ── Contact project card ── */
  .contact-card {
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius);
    padding: 2rem 2.2rem;
    box-shadow: var(--shadow-soft);
  }
  .ig-pill {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.2rem;
    background: #F8FBFF;
    border: 1px solid var(--border-soft);
    border-radius: var(--radius-sm);
    margin-top: 1.4rem;
    text-decoration: none;
    transition: all .2s ease;
  }
  .ig-pill:hover {
    border-color: var(--accent);
    transform: translateY(-1px);
  }
  .ig-pill .ig-l {
    display: flex;
    align-items: center;
    gap: 0.9rem;
  }
  .ig-pill .ig-ico {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    background: #FFFFFF;
    border: 1px solid var(--border-soft);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
  }
  .ig-pill .ig-lbl {
    font-family: 'Inter', sans-serif;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .ig-pill .ig-val {
    font-family: 'Fraunces', serif;
    font-size: 1rem;
    font-weight: 500;
    color: var(--ink);
  }
  .ig-pill .ig-arrow {
    color: var(--accent);
    font-size: 1rem;
  }


  /* ── Sidebar branding ── */
  .sb-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.5rem 0 0.4rem 0;
  }
  .sb-brand .sb-logo {
    width: 34px;
    height: 34px;
    border-radius: 9px;
    background: #FFFFFF;
    border: 1px solid var(--border);
    color: var(--ink);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: 'Fraunces', serif;
    font-weight: 500;
    font-size: 1.1rem;
    letter-spacing: -0.02em;
  }
  .sb-brand .sb-name {
    font-family: 'Fraunces', serif;
    font-style: italic;
    font-weight: 500;
    font-size: 1.15rem;
    color: var(--ink);
    letter-spacing: -0.01em;
    line-height: 1;
  }
  .sb-brand .sb-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 3px;
  }
  .sb-proj {
    font-family: 'Inter', sans-serif;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 0.6rem 0 1.2rem 0;
    border-bottom: 1px solid var(--border-soft);
    margin-bottom: 1rem;
  }
  .sb-section-lbl {
    font-family: 'Inter', sans-serif;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 0.4rem 0 0.5rem 0;
  }
  .sb-compute {
    margin-top: 1.2rem;
    padding: 0.8rem 0.9rem;
    background: #FFFFFF;
    border: 1px solid var(--border-soft);
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .sb-compute .sc-ico {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: #EFF6FF;
    color: var(--accent);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
  }
  .sb-compute .sc-body { line-height: 1.2; }
  .sb-compute .sc-lbl {
    font-family: 'Inter', sans-serif;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .sb-compute .sc-val {
    font-family: 'Fraunces', serif;
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--ink);
    letter-spacing: -0.005em;
  }


  /* Sidebar radio styling (menu items look like PDF) */
  div[data-testid="stSidebar"] [role="radiogroup"] { gap: 3px; }
  div[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 0.55rem 0.7rem;
    border-radius: 10px;
    transition: background .2s ease;
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
  }
  div[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(2,132,199,0.08);
  }


  /* ── Buttons (inline Streamlit) ── */
  .stButton > button,
  .stDownloadButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 999px !important;
    border: 1px solid var(--border) !important;
    background: #FFFFFF !important;
    color: var(--ink) !important;
    padding: 0.48rem 1.2rem !important;
    transition: all .2s ease !important;
    box-shadow: 0 1px 2px rgba(2,132,199,0.06) !important;
  }
  .stButton > button * { color: var(--ink) !important; }
  .stButton > button:hover {
    background: var(--ink) !important;
    color: #F0F9FF !important;
    border-color: var(--ink) !important;
  }
  .stButton > button:hover * { color: #F0F9FF !important; }


  [data-baseweb="select"] > div {
    border-radius: 10px !important;
    border-color: var(--border-soft) !important;
    background: #FFFFFF !important;
  }
  [data-baseweb="select"] > div * { color: var(--ink) !important; }


  /* ── Footer mark (shared) ── */
  .page-foot {
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border-soft);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
  }
  .page-foot .pf-l {
    font-family: 'Fraunces', serif;
    font-style: italic;
    font-size: 0.9rem;
    color: var(--muted);
  }
  .page-foot .pf-r {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
  }


  /* ── Entry animation (CSS-only, no JS dependency) ── */
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .reveal {
    opacity: 1;
    transform: translateY(0);
    animation: fadeUp .55s ease-out both;
  }
  .reveal:nth-of-type(2) { animation-delay: .06s; }
  .reveal:nth-of-type(3) { animation-delay: .12s; }
  .reveal:nth-of-type(4) { animation-delay: .18s; }
  .reveal:nth-of-type(5) { animation-delay: .24s; }
  .reveal:nth-of-type(6) { animation-delay: .30s; }


  hr { border: none; height: 1px; background: var(--border-soft); margin: 1rem 0; }
  code {
    background: #E0F2FE;
    color: var(--accent-2);
    padding: 0.08rem 0.32rem;
    border-radius: 6px;
    font-size: 0.85em;
  }


  @media (prefers-reduced-motion: reduce) {
    .reveal { animation: none; opacity: 1; transform: none; }
    .card, .stButton > button { transition: none; }
  }


  @media (max-width: 900px) {
    .stat-strip { grid-template-columns: repeat(2, 1fr); }
    .stat-cell { border-right: none; border-bottom: 1px solid var(--border-soft); }
    .stat-cell:nth-child(2) { border-right: none; }
    .step-row { grid-template-columns: 1fr; gap: 0.5rem; }
    .tags-col { flex-direction: row; justify-content: flex-start; align-items: flex-start; margin-top: 0.4rem; }
    .hero-title { font-size: 2.4rem; }
    .chapter-title { font-size: 2rem; }
    .mgrid { grid-template-columns: 1fr; }
    .dp-legend { font-size: 0.58rem; }
  }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  ALARM CONSTANTS
# ============================================================
WARNING_DURATION = 3.0   # detik
DANGER_DURATION  = 6.0   # detik


# ============================================================
#  ALARM MANAGER  — winsound-only
# ============================================================
class AlarmManager:
    """
    Single-path alarm via Windows winsound.Beep.
    WARNING → 3 detik beep 880 Hz interval 250 ms
    DANGER  → 6 detik beep 1320/1100 Hz alternating interval 250 ms
    NORMAL  → stop alarm
    """


    def __init__(self):
        self._lock        = threading.Lock()
        self._stop_event  = threading.Event()
        self._thread      = None
        self._current_lvl = "NORMAL"


    def trigger(self, level: str):
        with self._lock:
            if level == self._current_lvl:
                return
            self._current_lvl = level

        if level == "NORMAL":
            self._stop_playback()
            return

        duration = WARNING_DURATION if level == "WARNING" else DANGER_DURATION
        self._stop_playback()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._beep_worker, args=(level, duration), daemon=True
        )
        self._thread.start()

    def _stop_playback(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.8)

    def _beep_worker(self, level: str, duration: float):
        import platform
        import os
        
        sistem_operasi = platform.system()
        t_end = time.time() + duration
        
        try:
            if sistem_operasi == "Windows":
                # Teknologi Windows: winsound
                import winsound
                freq_a = 880  if level == "WARNING" else 1320
                freq_b = None if level == "WARNING" else 1100
                while time.time() < t_end and not self._stop_event.is_set():
                    winsound.Beep(freq_a, 250)
                    if freq_b and not self._stop_event.is_set():
                        winsound.Beep(freq_b, 250)
                        
            elif sistem_operasi == "Darwin":
                # Teknologi Apple (Mac): afplay
                while time.time() < t_end and not self._stop_event.is_set():
                    # Memutar suara bawaan Mac (Ping) - akan berulang sesuai durasi
                    os.system("afplay /System/Library/Sounds/Ping.aiff")
                    time.sleep(0.1) # Jeda sedikit agar tidak bentrok
                    
            else:
                # Linux / Raspberry Pi: aplay
                while time.time() < t_end and not self._stop_event.is_set():
                    # Jika ada file suara eksternal misalnya "alarm.wav"
                    # os.system("aplay alarm.wav")
                    print("\007") # Fallback Linux: Cetak ASCII Bell (Terminal beep)
                    time.sleep(0.5)
                    
        except Exception as e:
            print(f"[ALARM] Hardware audio gagal dieksekusi: {e}")


# ============================================================
#  SHARED STATE  (thread-safe singleton)
# ============================================================
@st.cache_resource
def _get_shared():
    lock  = threading.Lock()
    stats = {
        "face_count": 0, "pred_label": "-", "confidence": 0.0,
        "status_level": "NORMAL", "run_state": "MATI",
        "min_ear": 0.0, "processing_fps": 0.0,
        "buffer_fill": 0, "infer_ms": 0.0, "prob_drowsy": 0.0,
    }
    alarm = AlarmManager()
    return {"lock": lock, "stats": stats, "alarm": alarm}


@st.cache_resource
def _get_log_state():
    return {
        "initialized": False,
        "file": os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "logs",
            f"drowsy_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        ),
    }


def _update_stats(shared: dict, new_vals: dict):
    with shared["lock"]:
        shared["stats"].update(new_vals)


def _read_stats(shared: dict) -> dict:
    with shared["lock"]:
        return dict(shared["stats"])


def _reset_stats(shared: dict):
    with shared["lock"]:
        shared["stats"].update({
            "face_count": 0, "pred_label": "-", "confidence": 0.0,
            "status_level": "NORMAL", "run_state": "MATI",
            "min_ear": 0.0, "processing_fps": 0.0,
            "buffer_fill": 0, "infer_ms": 0.0, "prob_drowsy": 0.0,
        })


# ============================================================
#  SESSION STATE
# ============================================================
def _init_session():
    defaults = {"cam_key": 0}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()


# ============================================================
#  CSV LOGGER
# ============================================================
_LOG_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
_LOG_HEADER = ["timestamp", "pred_label", "confidence_pct",
               "prob_drowsy_pct", "status_level", "min_ear", "run_state"]


def _write_log_row(row: dict, enabled: bool):
    if not enabled:
        return
    log_state = _get_log_state()
    os.makedirs(_LOG_DIR, exist_ok=True)
    file_path = log_state["file"]
    write_header = not log_state["initialized"]
    try:
        with open(file_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_LOG_HEADER)
            if write_header:
                w.writeheader()
                log_state["initialized"] = True
            w.writerow(row)
    except Exception as e:
        print(f"[LOG] gagal tulis: {e}")


# ============================================================
#  VIDEO PROCESSOR (WebRTC thread)
# ============================================================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self._shared     = _get_shared()
        n_frames         = st.session_state.get("cfg_infer_n", 1)
        self.log_enabled = st.session_state.get("cfg_log", False)
        self.draw_mesh   = st.session_state.get("cfg_mesh", True)
        self.detector    = DrowsinessDetector(infer_every_n=n_frames)


    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        
        # Tambahkan 2 baris ini untuk membalik gambar (menghilangkan efek mirror)
        import cv2
        img_bgr = cv2.flip(img_bgr, 1)

        annotated, result = self.detector.process_frame(
            img_bgr, draw_mesh=self.draw_mesh
        )


        _update_stats(self._shared, {
            "face_count":     result.face_count,
            "pred_label":     result.pred_label,
            "confidence":     result.confidence,
            "status_level":   result.status_level,
            "run_state":      result.run_state,
            "min_ear":        result.min_ear,
            "processing_fps": result.processing_fps,
            "buffer_fill":    result.buffer_fill,
            "infer_ms":       result.infer_ms,
            "prob_drowsy":    result.prob_drowsy,
        })


        self._shared["alarm"].trigger(result.status_level)


        if self.log_enabled and result.run_state == "PROSES":
            _write_log_row({
                "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "pred_label":      result.pred_label,
                "confidence_pct":  f"{result.confidence*100:.2f}",
                "prob_drowsy_pct": f"{result.prob_drowsy*100:.2f}",
                "status_level":    result.status_level,
                "min_ear":         f"{result.min_ear:.4f}",
                "run_state":       result.run_state,
            }, enabled=True)


        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "stun:stun1.l.google.com:19302"},
        {
            "urls": "turn:openrelay.metered.ca:80",
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": "turn:openrelay.metered.ca:443?transport=tcp",
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]
})


# ============================================================
#  SIDEBAR — Figma match (branding + menu Bahasa + compute card)
# ============================================================
st.sidebar.markdown(
    '<div class="sb-brand">'
    '<div class="sb-logo">D</div>'
    '<div>'
    '<div class="sb-name">Drowsy</div>'
    '<div class="sb-sub">Detection</div>'
    '</div>'
    '</div>'
    '<div class="sb-proj">· 2026 · </div>'
    '<div class="sb-section-lbl">Navigasi</div>',
    unsafe_allow_html=True,
)


# Menu Bahasa Indonesia sesuai Figma
_MENU_ITEMS = {
    # "🏠  Beranda":  "Beranda",
    # "ⓘ  Tentang":  "Tentang",
    "📷  Demo":     "Demo",
    # "✦  Fitur":    "Fitur",
    # "✉  Kontak":   "Kontak",
}
_menu_display = st.sidebar.radio(
    "Navigasi",
    list(_MENU_ITEMS.keys()),
    index=0,
    label_visibility="collapsed",
)
menu = _MENU_ITEMS[_menu_display]


# Compute card bawah sidebar
gpu_text = "GPU · CUDA" if USE_CUDA else "CPU · Fallback"
st.sidebar.markdown(
    f'<div class="sb-compute">'
    f'<div class="sc-ico">⚡</div>'
    f'<div class="sc-body">'
    f'<div class="sc-lbl">Compute</div>'
    f'<div class="sc-val">{gpu_text}</div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)


# ============================================================
#  HELPER — footer mark (shared across pages)
# ============================================================
def _page_footer():
    st.markdown(
        '<div class="page-foot">'
        '<div class="pf-l">Real-time Driver Drowsiness Detection</div>'
        '<div class="pf-r">© 2026 · Edition</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
#  BERANDA (Home)
# ============================================================
if menu == "Beranda":
    # Hero
    st.markdown(
        '<div class="reveal">'
        '<div class="bab-eyebrow">Computer Vision · Tesis 2026</div>'
        '<div class="hero-title">Deteksi <em>Kantuk</em><br>'
        'Pengemudi <em>Real-time</em>.</div>'
        '<div class="hero-sub">Sistem analisis kelelahan mata pengemudi '
        'melalui webcam yang memadukan Swin Transformer sebagai ekstraktor '
        'fitur visual dan LSTM dengan mekanisme atensi sebagai pemodel '
        'temporal. Memberi peringatan tiga tingkat secara langsung untuk '
        'mengurangi risiko kecelakaan akibat microsleep.</div>'
        '</div>',
        unsafe_allow_html=True,
    )


    # CTA row — Streamlit buttons untuk navigasi halaman
    cta_l, cta_r, _ = st.columns([2, 2, 6], gap="small")
    with cta_l:
        if st.button("Coba Demo Langsung", key="cta_demo", use_container_width=True):
            st.info("Buka menu **Demo** di sidebar untuk memulai webcam.")
    with cta_r:
        if st.button("Baca Pipeline", key="cta_pipe", use_container_width=True):
            st.info("Buka menu **Tentang** di sidebar untuk membaca pipeline.")


    # 4-col stat strip
    st.markdown(
        '<div class="stat-strip reveal">'
        '<div class="stat-cell">'
        '<div class="stat-label">Feature Model</div>'
        '<div class="stat-value">Swin Transformer</div>'
        '<div class="stat-meta">tiny · pretrained</div>'
        '</div>'
        '<div class="stat-cell">'
        '<div class="stat-label">Temporal Model</div>'
        '<div class="stat-value">LSTM + Attention</div>'
        '<div class="stat-meta">additive scoring</div>'
        '</div>'
        '<div class="stat-cell">'
        '<div class="stat-label">Sequence Length</div>'
        '<div class="stat-value">30 Frames</div>'
        '<div class="stat-meta">≈ 1 detik @ 30 fps</div>'
        '</div>'
        '<div class="stat-cell">'
        '<div class="stat-label">Alert Levels</div>'
        '<div class="stat-value">3 Tingkat</div>'
        '<div class="stat-meta">Normal · Warning · Danger</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


    # Dataset & Metrik cards
    c_ds, c_mt = st.columns(2, gap="large")
    with c_ds:
        st.markdown(
            '<div class="card reveal" style="height:100%;">'
            '<div class="eyebrow-sm">📊 Dataset</div>'
            '<div class="card-title">NTHU-DDD2 + MRL Eye</div>'
            '<p class="muted">Korpus gabungan untuk pelatihan dan evaluasi '
            'yang menyatukan citra mata real-world dan video kantuk pengemudi.</p>'
            '<div class="num-tiles">'
            '<div class="num-tile">'
            '<div class="n-value">1302</div>'
            '<div class="n-label">Test Clips</div>'
            '</div>'
            '<div class="num-tile">'
            '<div class="n-value">548</div>'
            '<div class="n-label">Drowsy</div>'
            '</div>'
            '<div class="num-tile">'
            '<div class="n-value">754</div>'
            '<div class="n-label">Not-<br>Drowsy</div>'
            '</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c_mt:
        st.markdown(
            '<div class="card reveal" style="height:100%;">'
            '<div class="eyebrow-sm">◎ Metrik Utama</div>'
            '<div class="card-title">Recall Drowsy <em>&amp; F2-Score</em></div>'
            '<p class="muted">Memprioritaskan minimnya kasus kantuk yang '
            'terlewat. F2-Score memberi bobot dua kali lipat pada recall '
            'dibanding precision.</p>'
            '<div class="kv-row">'
            '<div class="kv-key">Recall Drowsy</div>'
            '<div class="kv-value plain">High Priority</div>'
            '</div>'
            '<div class="kv-row">'
            '<div class="kv-key">F2-Score Weight</div>'
            '<div class="kv-value">β = 2</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )


    _page_footer()


# ============================================================
#  TENTANG (About)
# ============================================================
elif menu == "Tentang":
    st.markdown(
        '<div class="reveal">'
        '<div class="bab-eyebrow">Bab 02 · Metodologi</div>'
        '<div class="chapter-title">Pipeline <em>5 langkah</em>.</div>'
        '<p class="muted" style="max-width:720px;">Setiap frame webcam '
        'menempuh lima tahap pemrosesan — dari akuisisi citra hingga '
        'keluaran peringatan — yang dirancang agar latensi tetap rendah '
        'tanpa mengorbankan recall pada kondisi kantuk.</p>'
        '</div>',
        unsafe_allow_html=True,
    )


    steps = [
        ("01", "Frame Acquisition",
         "Pengambilan citra langsung dari webcam pengguna menggunakan "
         "WebRTC dengan resolusi tinggi dan laju frame stabil.",
         ["WebRTC", "1280×720", "30 FPS"]),
        ("02", "Face Mesh + EAR",
         "Deteksi titik landmark wajah lalu menghitung Eye Aspect Ratio "
         "dari 16 titik di sekitar kelopak mata.",
         ["MediaPipe Face Mesh", "16-Landmark EAR"]),
        ("03", "Eye Crop & Resize",
         "Memotong area mata dengan margin 18% lalu mengubah ukurannya "
         "secara halus ke standar input transformer.",
         ["Margin 18%", "LANCZOS", "224×224"]),
        ("04", "Swin Feature Extraction",
         "Kedua mata diekstraksi secara terpisah lalu fitur 512 dimensi "
         "dirata-ratakan sebagai representasi kantuk.",
         ["Swin Transformer", "Dual-Eye AVG", "512-dim"]),
        ("05", "LSTM + Attention + Alert",
         "Sequence 30 frame diproses oleh LSTM dengan additive attention. "
         "Output dipetakan ke tiga tingkat peringatan.",
         ["LSTM", "Additive Attention", "3-Level Alert"]),
    ]


    for num, title, body, tags in steps:
        tags_html = "".join(f'<div class="tag-pill">{t}</div>' for t in tags)
        st.markdown(
            f'<div class="step-row reveal">'
            f'<div class="step-num-lg">{num}</div>'
            f'<div>'
            f'<div class="step-content-title">{title}</div>'
            f'<div class="step-content-body">{body}</div>'
            f'</div>'
            f'<div class="tags-col">{tags_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


    _page_footer()


# ============================================================
#  DEMO — webcam + status panel (2-column)
# ============================================================
elif menu == "Demo":
    if _HAS_AUTOREFRESH:
        st_autorefresh(interval=300, key="live_panel_refresh")
    else:
        st.warning(
            "⚠️ Install `streamlit-autorefresh` agar panel kanan update otomatis: "
            "`pip install streamlit-autorefresh`"
        )


    shared = _get_shared()


    # Header
    st.markdown(
        '<div class="reveal">'
        '<div class="chapter-title">Demo <em>Webcam</em> Langsung.</div>'
        '<p class="muted" style="max-width:720px;">Pratinjau real-time '
        'inferensi model. Tekan MULAI untuk memulai pengambilan frame '
        'webcam dan analisis kantuk.</p>'
        '</div>',
        unsafe_allow_html=True,
    )


    # ── Sidebar: pengaturan demo (tetap ada, fungsi tidak diubah) ─
    with st.sidebar:
        st.markdown('<div class="sb-section-lbl" style="margin-top:1.4rem;">'
                    'Pengaturan Demo</div>', unsafe_allow_html=True)


        cam_choice = st.selectbox(
            "Jenis kamera",
            ["Kamera bawaan laptop", "Webcam eksternal"],
            index=0,
        )


        vid_w, vid_h, fps_choice = 1280, 720, 30


        infer_n = st.selectbox(
            "Frame Skip (N)",
            [1],  # [1, 2, 3],
            index=0,
            help=(
                "N=1 → proses setiap frame (akurasi maks)"
                # "N=2 → beban GPU turun 50%, buffer penuh ~2 detik\n"
                # "N=3 → paling ringan, buffer penuh ~3 detik"
            ),
        )

        draw_mesh = st.toggle("Tampilkan Mesh wajah", value=True)
        # log_csv   = st.toggle("CSV Logger", value=False,
        #                       help=f"Log disimpan ke: {_LOG_DIR}")

        st.session_state["cfg_infer_n"] = infer_n
        st.session_state["cfg_mesh"]    = draw_mesh
        # st.session_state["cfg_log"]     = log_csv


    # ── Two-column layout ─────────────────────────────────────
    left_col, right_col = st.columns([6, 4], gap="large")


    # LEFT: Webcam panel
    with left_col:
        st.markdown('<div class="card reveal" style="padding:1.4rem 1.4rem;">',
                    unsafe_allow_html=True)


        # Idle preview placeholder (above actual WebRTC)
        st.markdown(
            '<div class="webcam-idle">'
            '<div class="idle-eyebrow">Webcam Idle</div>'
            '<div class="idle-text">Tekan MULAI untuk mengaktifkan kamera</div>'
            '<div class="idle-meta">'
            f'<span><b>Res</b> 1280×720</span>'
            f'<span><b>FPS</b> —</span>'
            f'<span><b>Target</b> 30</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )


        # WebRTC streamer + tombol reset
        col_info, col_reset = st.columns([3, 2])
        with col_info:
            st.markdown(
                f'<div style="display:flex;gap:14px;align-items:center;'
                f'font-family:Inter,sans-serif;font-size:0.78rem;'
                f'letter-spacing:0.14em;text-transform:uppercase;'
                f'color:#64748B;padding:4px 0;">'
                f'<span>1280×720 · 30 FPS</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_reset:
            if st.button("🔄 Reset Camera", use_container_width=True):
                _reset_stats(shared)
                shared["alarm"].trigger("NORMAL")
                st.session_state["cam_key"] += 1
                st.rerun()


        webrtc_ctx = webrtc_streamer(
            key                      = f"drowsiness-v{st.session_state['cam_key']}",
            mode                     = WebRtcMode.SENDRECV,
            rtc_configuration        = RTC_CONFIG,
            media_stream_constraints = {
                "video": {
                    "width":      {"ideal": vid_w},
                    "height":     {"ideal": vid_h},
                    "frameRate":  {"ideal": fps_choice},
                    "facingMode": "user",
                },
                "audio": False,
            },
            video_processor_factory  = VideoProcessor,
            async_processing         = True,
            desired_playing_state    = None,
        )


        if webrtc_ctx and not webrtc_ctx.state.playing:
            shared["alarm"].trigger("NORMAL")


        st.markdown('</div>', unsafe_allow_html=True)


        # Tips operasional
        st.markdown(
            '<div class="card reveal">'
            '<div class="eyebrow-sm">Tips Operasional</div>'
            '<ul style="margin:0.4rem 0 0 1rem;padding:0;color:#64748B;'
            'font-size:0.9rem;line-height:1.8;">'
            '<li>Pastikan pencahayaan merata di area wajah untuk akurasi '
            'landmark optimal.</li>'
            '<li>Lepas kacamata reflektif jika confidence terus rendah '
            'di bawah 0.50.</li>'
            '<li>Gunakan Frame Skip 2 atau 3 pada perangkat tanpa GPU '
            'untuk menjaga FPS.</li>'
            '<li>Aktifkan CSV Logger ketika menjalankan sesi rekam evaluasi.</li>'
            '</ul>'
            '</div>',
            unsafe_allow_html=True,
        )


    # RIGHT: Status card
    with right_col:
        stats = _read_stats(shared)
        sl    = stats["status_level"]


        pill_class = {"NORMAL": "pill-normal",
                      "WARNING": "pill-warning",
                      "DANGER":  "pill-danger"}.get(sl, "pill-info")
        run_class  = {"PROSES": "pill-normal",
                      "MULAI":  "pill-info",
                      "MATI":   "pill-ghost"}.get(stats["run_state"], "pill-ghost")


        st.markdown('<div class="card reveal" style="padding:1.5rem 1.4rem;">',
                    unsafe_allow_html=True)


        # Header: STATUS pill + RUN STATE pill
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;margin-bottom:1.1rem;gap:8px;">'
            f'<span class="pill {pill_class}">Status · {sl}</span>'
            f'<span class="pill {run_class}">{stats["run_state"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


        # Helper untuk format "—" jika kosong
        def _fmt(v, unit="", fmt=None, dash_on_zero=False):
            if dash_on_zero and (v == 0 or v is None):
                return "—"
            if isinstance(v, str) and v in ("-", ""):
                return "—"
            if fmt:
                return fmt.format(v) + unit
            return f"{v}{unit}"


        is_active = stats["run_state"] == "PROSES"
        faces_v = stats["face_count"]   if is_active else "—"
        ear_v   = f'{stats["min_ear"]:.3f}' if is_active and stats["min_ear"] > 0 else "—"
        pred_v  = stats["pred_label"]   if is_active else "—"
        conf_v  = f'{stats["confidence"]*100:.1f}%' if is_active else "—"
        prob_v  = f'{stats["prob_drowsy"]*100:.1f}%' if is_active else "—"
        buf_v   = f'{stats["buffer_fill"]} / 30'
        fps_v   = f'{stats["processing_fps"]:.1f}'   if is_active else "—"
        lat_v   = f'{stats["infer_ms"]:.0f} ms'      if is_active else "—"


        # 2x4 metric grid
        st.markdown(
            f'<div class="mgrid">'
            f'<div class="mtile"><div class="ml">Faces Detected</div>'
            f'<div class="mv">{faces_v}</div></div>'
            f'<div class="mtile"><div class="ml">Min EAR</div>'
            f'<div class="mv">{ear_v}</div></div>'
            f'<div class="mtile"><div class="ml">Prediction</div>'
            f'<div class="mv" style="font-size:0.95rem;">{pred_v}</div></div>'
            f'<div class="mtile"><div class="ml">Confidence</div>'
            f'<div class="mv">{conf_v}</div></div>'
            f'<div class="mtile"><div class="ml">Drowsy Prob.</div>'
            f'<div class="mv">{prob_v}</div></div>'
            f'<div class="mtile"><div class="ml">Buffer Fill</div>'
            f'<div class="mv">{buf_v}</div></div>'
            f'<div class="mtile"><div class="ml">Processing FPS</div>'
            f'<div class="mv">{fps_v}</div></div>'
            f'<div class="mtile"><div class="ml">Inference Latency</div>'
            f'<div class="mv">{lat_v}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )


        # Drowsy probability bar dengan 3-label legend
        drowsy_ratio = min(stats["prob_drowsy"], 1.0) if is_active else 0.0
        bar_color    = ("#10b981" if sl == "NORMAL" else
                        "#f59e0b" if sl == "WARNING" else "#ef4444")
        pct_text     = f'{drowsy_ratio*100:.0f}%' if is_active else '0%'
        st.markdown(
            f'<div class="dp-wrap">'
            f'<div class="dp-head">'
            f'<div class="dp-title">Drowsy Probability</div>'
            f'<div class="dp-pct">{pct_text}</div>'
            f'</div>'
            f'<div class="dp-bar">'
            f'<div class="dp-bar-fill" style="width:{drowsy_ratio*100:.0f}%;'
            f'background:{bar_color};"></div>'
            f'</div>'
            f'<div class="dp-legend">'
            f'<div class="ll">0% · Normal</div>'
            f'<div class="lc">50% · Warning</div>'
            f'<div class="lr">Danger · 100%</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


        # Alarm output card
        import base64
        import os

        audio_file_path = os.path.join(os.path.dirname(__file__), "alarm_beep.mp3")
        try:
            with open(audio_file_path, "rb") as f:
                audio_bytes = f.read()
                audio_b64 = base64.b64encode(audio_bytes).decode()
        except Exception:
            audio_b64 = ""

        # Logic Audio: Hanya tampilkan (dan autoplay) HTML Audio jika status BUKAN Normal
        if sl in ["WARNING", "DANGER"] and audio_b64:
            audio_html = f"""
                <audio autoplay loop>
                    <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
            
        alarm_state = "READY" if sl == "NORMAL" else "ACTIVE"
        alarm_pill  = "pill-normal" if sl == "NORMAL" else "pill-warning"
        st.markdown(
            f'<div class="alarm-row">'
            f'<div class="al-left">'
            f'<div class="al-ico">🔊</div>'
            f'<div>'
            f'<div class="al-title">Alarm Output</div>'
            f'<div class="al-body">Web Browser Audio Player</div>'
            f'</div>'
            f'</div>'
            f'<span class="pill {alarm_pill}">{alarm_state}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


        st.markdown('</div>', unsafe_allow_html=True)


    _page_footer()


# ============================================================
#  FITUR (Features)
# ============================================================
elif menu == "Fitur":
    st.markdown(
        '<div class="reveal">'
        '<div class="bab-eyebrow">Bab 04 · Kapabilitas</div>'
        '<div class="chapter-title">Dua belas <em>fitur inti</em>.</div>'
        '<p class="muted" style="max-width:720px;">Setiap kapabilitas di '
        'bawah ini diturunkan langsung dari kebutuhan eksperimen tesis: '
        'stabilitas inferensi, kontrol false alarm, serta kemudahan '
        'reproduksi hasil.</p>'
        '</div>',
        unsafe_allow_html=True,
    )


    features = [
        ("01", "Eye Crop (Bukan Face Crop)",
         "Memfokuskan input model hanya pada area mata dengan margin 18% "
         "sehingga sinyal kantuk lebih bersih."),
        ("02", "Dual-Eye Averaged Feature",
         "Fitur dari mata kiri dan kanan dirata-ratakan. Mode dapat "
         "dikonfigurasi melalui env var EYE_INFERENCE_MODE."),
        ("03", "16-Landmark EAR + Min-EAR Selection",
         "Menggunakan 16 landmark per mata dan memilih nilai EAR terkecil "
         "sebagai indikator paling konservatif."),
        ("04", "EAR Confidence Gate",
         "Gerbang ambang: EAR ≤ 0.15 memaksa label drowsy, sedangkan "
         "≤ 0.20 menjadi sinyal pendukung."),
        ("05", "Rapid-Clear DANGER → NORMAL",
         "Pemulihan cepat menggunakan lookback 3 frame dengan threshold "
         "probabilitas 0.35 untuk menghindari false alarm berkepanjangan."),
        ("06", "Face Loss Tolerance",
         "Sistem tetap stabil ketika wajah hilang sementara hingga 10 "
         "frame berturut-turut tanpa reset state."),
        ("07", "LANCZOS Resize + MRL Normalization",
         "Resize halus 224×224 dilanjutkan normalisasi mengikuti "
         "distribusi MRL Eye (mean 0.3772, std 0.1544)."),
        ("08", "Swin + LSTM 30-Frame Sequence",
         "Fitur Swin diumpankan ke LSTM dalam jendela sequence 30 frame "
         "untuk memodelkan dinamika temporal."),
        ("09", "Three-Level Alert · Rolling Window 10",
         "10 prediksi terakhir ditimbang: Normal 0–2 drowsy, Warning 3–6 "
         "(alarm 3 detik), Danger 7–10 (alarm 6 detik)."),
        ("10", "Winsound Beep — Stabil Tanpa Download",
         "Alarm memakai winsound.Beep langsung. Tidak ada unduh audio, "
         "tidak ada dependency sounddevice."),
        ("11", "Frame-Skip 1/2/3 Performance Toggle",
         "Pengaturan infer_every_n mengatur frekuensi panggilan GPU. N=1 "
         "default akurasi maksimal, N=2/3 untuk menurunkan beban."),
        ("12", "CSV Logger Otomatis",
         "Setiap prediksi disimpan ke logs/ (timestamp, label, confidence, "
         "prob drowsy, status, EAR, run_state) sebagai lampiran skripsi."),
    ]


    for num, title, desc in features:
        st.markdown(
            f'<div class="f-item reveal">'
            f'<div class="f-item-head">'
            f'<div class="f-num">{num}</div>'
            f'<div class="f-title">{title}</div>'
            f'</div>'
            f'<div class="f-desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


    _page_footer()


# ============================================================
#  KONTAK (Contact)
# ============================================================
elif menu == "Kontak":
    st.markdown(
        '<div class="reveal">'
        '<div class="bab-eyebrow">Bab 05 · Kontak</div>'
        '<div class="chapter-title">Hubungi <em>peneliti</em>.</div>'
        '</div>',
        unsafe_allow_html=True,
    )


    st.markdown(
        '<div class="contact-card reveal">'
        '<div class="eyebrow-sm">Project</div>'
        '<div class="card-title" style="margin-top:2px;">'
        'Real-time Driver Drowsiness Detection</div>'
        '<p class="muted">Penelitian tesis Computer Vision yang '
        'menggabungkan Swin Transformer dan LSTM dengan attention untuk '
        'mendeteksi kelelahan mata pengemudi secara real-time melalui '
        'webcam. Untuk diskusi, kolaborasi, atau pertanyaan teknis '
        'silakan hubungi melalui kanal di bawah.</p>'
        '<a class="ig-pill" href="https://www.instagram.com/ddapppa/" '
        'target="_blank">'
        '<div class="ig-l">'
        '<div class="ig-ico">📸</div>'
        '<div>'
        '<div class="ig-lbl">Instagram</div>'
        '<div class="ig-val">@ddapppa</div>'
        '</div>'
        '</div>'
        '<div class="ig-arrow">↗</div>'
        '</a>'
        '</div>',
        unsafe_allow_html=True,
    )


    _page_footer()
