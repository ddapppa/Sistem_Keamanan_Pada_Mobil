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
import gc
import csv
import time
import threading
from datetime import datetime
import platform

from streamlit.runtime.scriptrunner import get_script_run_ctx
import uuid

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
import training_pipeline as tp
import torch
import cv2

import base64

def get_session_id() -> str:
    ctx = get_script_run_ctx()
    if ctx is None:
        return f"local_fallback_{uuid.uuid4().hex}"
    return ctx.session_id

def play_web_alarm_fast(tipe_alarm: str, session_id: str):
    """
    Alarm browser berbasis Web Audio API.
    Stabil untuk rerun Streamlit karena state disimpan per-session di window global JS.
    """
    js_code = f"""
    <script>
    (function() {{
        const SID = "{session_id}";
        window.__drowsyAudio = window.__drowsyAudio || {{}};
        const store = window.__drowsyAudio;

        if (!store[SID]) {{
            store[SID] = {{
                ctx: null,
                unlocked: false,
                currentMode: "NORMAL",
                timerIds: []
            }};
        }}

        const state = store[SID];

        function clearTimers() {{
            if (state.timerIds && state.timerIds.length) {{
                state.timerIds.forEach(id => clearTimeout(id));
            }}
            state.timerIds = [];
        }}

        function ensureCtx() {{
            if (!state.ctx || state.ctx.state === "closed") {{
                state.ctx = new (window.AudioContext || window.webkitAudioContext)();
            }}
            return state.ctx;
        }}

        function stopAll() {{
            clearTimers();
            state.currentMode = "NORMAL";
        }}

        function beep(freq, startOffset, duration, type="sine", gainValue=0.08) {{
            const ctx = ensureCtx();
            const now = ctx.currentTime + startOffset;
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();

            osc.type = type;
            osc.frequency.setValueAtTime(freq, now);

            gain.gain.setValueAtTime(0.0001, now);
            gain.gain.exponentialRampToValueAtTime(gainValue, now + 0.02);
            gain.gain.exponentialRampToValueAtTime(0.0001, now + duration);

            osc.connect(gain);
            gain.connect(ctx.destination);

            osc.start(now);
            osc.stop(now + duration + 0.02);
        }}

        function playWarningPattern() {{
            stopAll();
            state.currentMode = "WARNING";
            beep(880, 0.00, 0.18, "sine", 0.06);
            beep(880, 0.28, 0.18, "sine", 0.06);
        }}

        function playDangerPattern() {{
            stopAll();
            state.currentMode = "DANGER";

            const pattern = [1320, 1100, 1320, 1100, 1320, 1100];
            pattern.forEach((f, i) => {{
                beep(f, i * 0.22, 0.16, "square", 0.07);
            }});

            // ulangi jika status tetap DANGER
            const id = setTimeout(() => {{
                if (state.currentMode === "DANGER") {{
                    playDangerPattern();
                }}
            }}, 1700);
            state.timerIds.push(id);
        }}

        window.unlockDrowsyAudio = async function(sessionKey) {{
            const k = sessionKey || SID;
            window.__drowsyAudio = window.__drowsyAudio || {{}};
            if (!window.__drowsyAudio[k]) {{
                window.__drowsyAudio[k] = {{
                    ctx: null,
                    unlocked: false,
                    currentMode: "NORMAL",
                    timerIds: []
                }};
            }}
            const s = window.__drowsyAudio[k];
            if (!s.ctx || s.ctx.state === "closed") {{
                s.ctx = new (window.AudioContext || window.webkitAudioContext)();
            }}
            try {{
                await s.ctx.resume();
                s.unlocked = true;

                // bunyi test singkat agar user tahu audio aktif
                const osc = s.ctx.createOscillator();
                const gain = s.ctx.createGain();
                osc.type = "sine";
                osc.frequency.value = 660;
                gain.gain.value = 0.03;
                osc.connect(gain);
                gain.connect(s.ctx.destination);
                osc.start();
                osc.stop(s.ctx.currentTime + 0.08);

                console.log("Drowsy audio unlocked:", k);
            }} catch (e) {{
                console.error("Audio unlock failed:", e);
            }}
        }};

        const mode = "{tipe_alarm}";

        if (!state.unlocked) {{
            console.log("Audio not unlocked yet for session:", SID);
            stopAll();
            return;
        }}

        if (mode === "NORMAL") {{
            stopAll();
        }} else if (mode === "WARNING") {{
            if (state.currentMode !== "WARNING") {{
                playWarningPattern();
            }}
        }} else if (mode === "DANGER") {{
            if (state.currentMode !== "DANGER") {{
                playDangerPattern();
            }}
        }}
    }})();
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)

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


  p, li, div[data-testid="stMarkdownContainer"] p, div[data-testid="stText"] {
    color: var(--ink-2) !important;
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

  /* ── Mengubah warna drop-down list (Popover Menu) menjadi Putih ── */
  [data-baseweb="popover"],
  [data-baseweb="popover"] > div,
  div[data-testid="stSelectbox"] [data-baseweb="popover"] div,
  [data-baseweb="menu"],
  ul[role="listbox"],
  div[role="listbox"] {
    background-color: #FFFFFF !important;
    border-radius: 10px !important;
  }
  
  /* Teks di dalam pilihan dropdown menjadi Hitam/Gelap */
  ul[role="listbox"] li,
  div[role="listbox"] li,
  [data-baseweb="popover"] li,
  [data-baseweb="popover"] li span,
  [data-baseweb="menu"] li {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    font-weight: 500;
  }

  /* Warna saat salah satu pilihan di-hover */
  ul[role="listbox"] li:hover,
  div[role="listbox"] li:hover,
  [data-baseweb="popover"] li:hover,
  [data-baseweb="popover"] li:hover span {
    background-color: #F0F9FF !important;
    color: #0284C7 !important;
  }

  /* ── Input box biasa (seperti Number Input st.number_input) ── */
  [data-baseweb="input"] > div {
    border-radius: 10px !important;
    border-color: var(--border-soft) !important;
    background: #FFFFFF !important;
  }
  [data-baseweb="input"] > div * { color: var(--ink) !important; }


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

  /* ── File Uploader Styling ── */
  [data-testid="stFileUploaderDropzone"],
  [data-testid="stFileUploadDropzone"],
  section[data-testid="stFileUploaderDropzone"] {
      background-color: #FFFFFF !important;
      border: 2px dashed #0C4A6E !important;
      border-radius: var(--radius) !important;
  }
  
  [data-testid="stFileUploaderDropzone"] *,
  [data-testid="stFileUploadDropzone"] * {
      color: #0C4A6E !important;
  }
  
  [data-testid="stFileUploaderDropzone"]:hover,
  [data-testid="stFileUploadDropzone"]:hover {
      background-color: #F0F9FF !important;
      border-color: #0284C7 !important;
  }
  
  [data-testid="stFileUploaderDropzone"]:hover *,
  [data-testid="stFileUploadDropzone"]:hover * {
      color: #0C4A6E !important;
  }
  
  [data-testid="stFileUploaderDropzone"] button,
  [data-testid="stFileUploadDropzone"] button {
      background-color: #0C4A6E !important;
      color: #FFFFFF !important;
      border: none !important;
      border-radius: 999px !important;
      padding: 0.4rem 1rem !important;
  }
  
  [data-testid="stFileUploaderDropzone"] button *,
  [data-testid="stFileUploadDropzone"] button *,
  [data-testid="stFileUploaderDropzone"] button span,
  [data-testid="stFileUploadDropzone"] button span {
      color: #FFFFFF !important;
  }
  
  /* ── Samakan warna st.info() ── */
  [data-testid="stAlert"],
  [data-testid="stInfo"],
  div[data-testid="stAlert"] {
      background-color: #FFFFFF !important;
      border: 2px dashed #0C4A6E !important;
      border-radius: var(--radius) !important;
  }
  
  [data-testid="stAlert"] *,
  [data-testid="stInfo"] * {
      color: #0C4A6E !important;
  }
  
  [data-testid="stAlert"] svg,
  [data-testid="stInfo"] svg {
      fill: #0C4A6E !important;
      color: #0C4A6E !important;
  }
  
  /* ── Fix warna teks Spinner ── */
  [data-testid="stSpinner"] * {
      color: #1E293B !important;
  }
  [data-testid="stSpinner"] i {
      border-top-color: #0284C7 !important;
      border-right-color: #0284C7 !important;
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
def _get_all_sessions():
    return {}

def get_shared(session_id: str):
    all_sessions = _get_all_sessions()

    if session_id not in all_sessions:
        all_sessions[session_id] = {
            "lock": threading.Lock(),
            "stats": {
                "face_count": 0,
                "pred_label": "-",
                "confidence": 0.0,
                "status_level": "NORMAL",
                "run_state": "MATI",
                "min_ear": 0.0,
                "processing_fps": 0.0,
                "buffer_fill": 0,
                "infer_ms": 0.0,
                "prob_drowsy": 0.0,
            },
            "alarm": AlarmManager()
        }

    return all_sessions[session_id]


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
    defaults = {
        "cam_key": 0,
        "train_step": 1,
        "train_ds_info": None,
        "train_features": None,
        "train_preproc_cfg": None,
        "train_cfg": None,
        "train_seq_len": 30,
        "train_stride": 15,
        "train_splits": None,
        "train_norm_mean": None,
        "train_norm_std": None,
        "train_model": None,
        "train_hist": None,
        "train_metrics": None,
        "train_best_state": None,
        "export_paths": None,
        "custom_model_pth": None,
        "custom_model_npz": None,
        "custom_model_label": None,
        "active_model_mode": "bawaan",
        "train_config_hash": None,
        "nav_page": "Beranda",
    }
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
        pass

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        img_bgr = cv2.flip(img_bgr, 1)

        annotated, result = self.detector.process_frame(
            img_bgr, draw_mesh=self.draw_mesh
            , flip_frame=False
        )

        my_session_id = self.session_id or "local_fallback"
        my_shared = get_shared(my_session_id)

        _update_stats(my_shared, {
            "face_count": result.face_count,
            "pred_label": result.pred_label,
            "confidence": result.confidence,
            "status_level": result.status_level,
            "run_state": result.run_state,
            "min_ear": result.min_ear,
            "processing_fps": result.processing_fps,
            "buffer_fill": result.buffer_fill,
            "infer_ms": result.infer_ms,
            "prob_drowsy": result.prob_drowsy,
        })

        my_shared["alarm"].trigger(result.status_level)

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
    "  Beranda":  "Beranda",
    "  Tentang":  "Tentang",
    "  Demo":     "Demo",
    "  Pelatihan":"Pelatihan",
    "  Fitur":    "Fitur",
    "  Kontak":   "Kontak",
}
# BARU
_keys = list(_MENU_ITEMS.keys())
_active_index = next(
    (i for i, k in enumerate(_keys) if _MENU_ITEMS[k] == st.session_state.nav_page),
    0
)
_menu_display = st.sidebar.radio(
    "Navigasi",
    _keys,
    index=_active_index,
    label_visibility="collapsed",
)
menu = _MENU_ITEMS[_menu_display]
st.session_state.nav_page = menu   # sinkronisasi saat klik manual


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


def get_cfg_hash(cfg) -> str:
    """Buat fingerprint dari config training untuk deteksi perubahan."""
    import hashlib
    cfg_str = f"{cfg.seq_len}_{cfg.stride}_{cfg.hidden_dim}_{cfg.num_layers}_{cfg.epochs}_{cfg.learning_rate}"
    return hashlib.md5(cfg_str.encode()).hexdigest()[:8]

# ============================================================
#  BERANDA (Home)
# ============================================================
if menu == "Beranda":
    # Hero
    st.markdown(
        '<div class="reveal">'
        '<div class="bab-eyebrow">Computer Vision </div>'
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
            st.session_state.nav_page = "Demo"
            st.rerun()
    with cta_r:
        if st.button("Baca Pipeline", key="cta_pipe", use_container_width=True):
            st.session_state.nav_page = "Tentang"
            st.rerun()

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
            '<div class="eyebrow-sm"> Dataset</div>'
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
        '<div class="chapter-title">Pipeline <em>5 langkah</em>.</div>'
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
    MY_SESSION_ID = get_session_id()
    shared = get_shared(MY_SESSION_ID)
    if _HAS_AUTOREFRESH:
        st_autorefresh(interval=300, key="live_panel_refresh")
    else:
        st.warning(
            " Install `streamlit-autorefresh` agar panel kanan update otomatis: "
            "`pip install streamlit-autorefresh`"
        )

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

    # ============================================================
    # PATCH 1: Model Selector di halaman Demo
    # ============================================================
    if "active_model_mode" not in st.session_state:
        st.session_state["active_model_mode"] = "bawaan"

    with st.container():
        st.markdown("""
        <div style="background:#f8fafc;border:1.5px solid #e2e8f0;border-radius:10px;
                    padding:14px 20px;margin-bottom:16px;">
            <span style="font-size:11px;font-weight:700;letter-spacing:.08em;
                         color:#64748b;text-transform:uppercase;">MODEL AKTIF</span>
        </div>
        """, unsafe_allow_html=True)
        col_sel, col_badge = st.columns([3, 1])

        with col_sel:
            model_choices = ["Model Bawaan (SWIN_LSTM_EXP_K_BEST)"]
            custom_pth = st.session_state.get("custom_model_pth")
            custom_npz = st.session_state.get("custom_model_npz")
            custom_label = st.session_state.get("custom_model_label", "Custom Model")

            if custom_pth and custom_npz:
                model_choices.append(f"⚡ {custom_label} (Custom)")

            selected_model = st.selectbox(
                "Pilih Model untuk Inferensi",
                options=model_choices,
                key="demo_model_selector",
                help="Pilih model bawaan (terlatih peneliti) atau model custom hasil pelatihan."
            )

            st.session_state["active_model_mode"] = (
                "custom" if "Custom" in selected_model else "bawaan"
            )

        with col_badge:
            if st.session_state["active_model_mode"] == "custom":
                st.markdown("""
                <div style="background:#dcfce7;color:#166534;border-radius:20px;
                            padding:6px 14px;font-size:12px;font-weight:700;
                            text-align:center;margin-top:28px;">
                    ✓  CUSTOM AKTIF
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:#dbeafe;color:#1e40af;border-radius:20px;
                            padding:6px 14px;font-size:12px;font-weight:700;
                            text-align:center;margin-top:28px;">
                    ◈  BAWAAN AKTIF
                </div>""", unsafe_allow_html=True)
    # ============================================================


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
            if st.button(" Reset Camera", use_container_width=True):
                _reset_stats(shared)
                shared["alarm"].trigger("NORMAL")
                st.session_state["cam_key"] += 1
                st.rerun()


        # ─── SNAPSHOT session_state di main thread ───────────────────
        _active_mode = st.session_state.get("active_model_mode", "bawaan")
        _custom_pth  = st.session_state.get("custom_model_pth")
        _custom_npz  = st.session_state.get("custom_model_npz")
        _infer_n     = st.session_state.get("cfg_infer_n", 1)
        _draw_mesh   = st.session_state.get("cfg_mesh", True)
        _log_enabled = st.session_state.get("cfg_log", False)

        def make_processor():
            vp = VideoProcessor.__new__(VideoProcessor)  # bypass __init__
            vp.session_id  = None
            vp.log_enabled = _log_enabled
            vp.draw_mesh   = _draw_mesh
            vp.detector    = DrowsinessDetector(infer_every_n=_infer_n)
            if _active_mode == "custom" and _custom_pth and _custom_npz:
                try:
                    vp.detector.load_custom_lstm(_custom_pth, _custom_npz)
                    print(f"[INFO] VideoProcessor: memakai model CUSTOM — {_custom_pth}")
                except Exception as e:
                    print(f"[ERROR] Gagal load custom model: {e}")
                    print("[INFO] VideoProcessor: fallback ke model BAWAAN")
            else:
                print("[INFO] VideoProcessor: memakai model BAWAAN")
            return vp

        webrtc_ctx = webrtc_streamer(
            key=f"drowsiness-v{st.session_state['cam_key']}",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": vid_w},
                    "height": {"ideal": vid_h},
                    "frameRate": {"ideal": fps_choice},
                    "facingMode": "user",
                },
                "audio": False,
            },
            video_processor_factory=make_processor,
            async_processing=True,
            desired_playing_state=None,
        )

        if webrtc_ctx and webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.session_id = MY_SESSION_ID 


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
        # NOTE: base64 dan os sudah diimport di atas (top-level)

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
            f'<div class="al-ico"></div>'
            f'<div>'
            f'<div class="al-title">Alarm Output</div>'
            f'<div class="al-body">Web Browser Audio Player</div>'
            f'</div>'
            f'</div>'
            f'<span class="pill {alarm_pill}">{alarm_state}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


    _page_footer()


# ============================================================
#  PELATIHAN (Manual Training)
# ============================================================
elif menu == "Pelatihan":
    st.markdown(
        '<div class="reveal">'
        '<div class="bab-eyebrow">Bab 03 · Pelatihan Manual</div>'
        '<div class="chapter-title">Latih <em>model kustom</em>.</div>'
        '<p class="muted" style="max-width:720px;">Sesuaikan hyperparameter '
        'dan latih model LSTM Anda sendiri langsung dari antarmuka ini. '
        'Sistem akan mengekstrak fitur secara otomatis dengan Swin Transformer.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Step Indicator
    steps = ["1. Dataset", "2. statistik", "3. Preprocessing", "4. Ekstraksi Fitur", "5. Training", "6. Evaluasi", "7. Ekspor"]
    current_step = st.session_state.train_step

    st.markdown('<div style="display:flex; justify-content:space-between; margin-bottom: 2rem; border-bottom: 1px solid var(--border-soft); padding-bottom: 1rem;">', unsafe_allow_html=True)
    for i, step_name in enumerate(steps, 1):
        color = "var(--accent)" if i == current_step else "var(--ink)" if i < current_step else "var(--muted-soft)"
        fw = "bold" if i == current_step else "normal"
        st.markdown(f'<span style="color: {color}; font-weight: {fw};">{step_name}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if current_step == 1:
        st.markdown('<div class="card"><div class="card-title">1. Upload Dataset</div>', unsafe_allow_html=True)
        ds_mode = st.radio("Jenis Input", ["Frame Images"], horizontal=True)
        
        uploaded_zip = st.file_uploader(
            "Upload Dataset (.zip)",
            type=["zip"],
            help="Ukuran maksimal ditentukan oleh konfigurasi server. Untuk dataset besar, gunakan mode Frame Images."
        )
        st.info(" **Format File:** Harus berupa file `.zip` yang di dalamnya terdapat minimal 2 sub-folder seperti `Drowsy/` dan `NotDrowsy/`.")
        
        if uploaded_zip is not None:
            file_size_mb = uploaded_zip.size / (1024 * 1024)
            st.caption(f" Ukuran file: {file_size_mb:.1f} MB")
            
            if file_size_mb > 500:
                st.warning(
                    " File besar terdeteksi. Proses ekstraksi mungkin memakan waktu lebih lama. "
                    "Pertimbangkan menggunakan mode **Frame Images** untuk dataset > 500MB."
                )
        
        if st.button("Jalankan", type="primary"):
            if uploaded_zip is None:
                st.error("Silakan unggah file dataset (.zip) terlebih dahulu.")
            else:
                with st.spinner("Mengekstrak dan memvalidasi dataset..."):
                    import zipfile
                    import shutil
                    
                    # Buat folder sementara di dalam direktori proyek
                    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_dataset")
                    
                    # Bersihkan folder jika sudah ada
                    if os.path.exists(temp_dir):
                        try:
                            shutil.rmtree(temp_dir)
                        except Exception as e:
                            st.warning(f"Gagal membersihkan folder lama: {e}")
                            
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    try:
                        progress_text = st.empty()
                        prog_bar = st.progress(0)
                        
                        def prep_cb(msg, curr, total):
                            safe_val = min(curr / total, 1.0) if total > 0 else 0.0
                            prog_bar.progress(safe_val)
                            progress_text.markdown(f"<span style='color:#1E293B; font-weight:600;'>{msg}</span>", unsafe_allow_html=True)
                            
                        # Preprocess otomatis: ekstrak, resize, hapus corrupt, dedup
                        stats = tp.preprocess_dataset_zip(
                            uploaded_zip.getvalue(),
                            temp_dir,
                            target_size=224,
                            progress_cb=prep_cb
                        )
                        
                        if stats["errors"]:
                            for err in stats["errors"]:
                                st.error(err)
                        else:
                            st.success(
                                f"Preprocess awal selesai! Input: {stats['total_input']} img, "
                                f"Output: {stats['total_output']} img "
                                f"(Dihapus: {stats['removed_corrupt']} corrupt, {stats['removed_duplicate']} duplikat)."
                            )
                            
                            info = tp.validate_dataset_folder(stats["output_dir"])
                            
                            if info.is_valid:
                                st.session_state.train_ds_info = info
                                st.session_state.train_step = 2
                                st.rerun()
                            else:
                                for err in info.errors:
                                    st.error(err)
                    except zipfile.BadZipFile:
                        st.error("File yang diunggah rusak atau bukan file ZIP yang valid.")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memproses file: {e}")
                        
        st.markdown('</div>', unsafe_allow_html=True)

    elif current_step == 2:
        st.markdown('<div class="card"><div class="card-title">2. Statistik Dataset</div>', unsafe_allow_html=True)
        info = st.session_state.train_ds_info
        
        label_total = "Total Gambar" if info.mode == "frame_flat" else "Total Clips"
        st.write(f" **{label_total}:** {info.total_clips}")

        cols = st.columns(len(info.clips_per_class))
        for idx, (label, count) in enumerate(info.clips_per_class.items()):
            label_name = "Not Drowsy (1)" if label == 1 else "Drowsy (0)"
            cols[idx].metric(label_name, count)
            
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Kembali"):
                st.session_state.train_step = 1
                st.rerun()
        with c2:
            if st.button("Lanjut ke Preprocessing", type="primary"):
                st.session_state.train_step = 3
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    elif current_step == 3:
        st.markdown('<div class="card"><div class="card-title">3. Preprocessing</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            resize = st.selectbox("Resize Target", [224, 256, 384], index=0)
            interp = st.selectbox("Interpolasi", ["LANCZOS", "BILINEAR", "BICUBIC"], index=0)
            margin = st.slider("Eye Crop Margin", 0.0, 0.3, 0.18, 0.01)
            is_pre = st.checkbox("Data Sudah Di-Crop (Bypass MediaPipe)", value=False)
        with c2:
            aug_hflip = st.checkbox("Horizontal Flip", value=True)
            aug_rot = st.checkbox("Random Rotation", value=False)
            aug_rot_deg = st.slider("Rotasi (Derajat)", 5, 30, 15) if aug_rot else 15
        
        st.session_state.train_preproc_cfg = tp.PreprocessConfig(
            resize=resize, interpolation=interp, eye_crop_margin=margin,
            is_pre_cropped=is_pre,
            aug_hflip=aug_hflip, aug_rotation=aug_rot, aug_rotation_deg=aug_rot_deg
        )
        
        if not is_pre:
            if st.button("👁️ Preview Hasil Eye Crop"):
                with st.spinner("Memproses preview..."):
                    previews = tp.preview_eye_crops(st.session_state.train_ds_info, margin=margin, max_previews=3)
                    if previews:
                        cols = st.columns(len(previews))
                        for idx, p in enumerate(previews):
                            with cols[idx]:
                                st.image([p['left_bgr'], p['right_bgr']], caption=[f"L ({p['label']})", f"R ({p['label']})"], channels="BGR")
                    else:
                        st.warning("Gagal membuat preview. Pastikan ada wajah terdeteksi.")
        
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Kembali"):
                st.session_state.train_step = 2
                st.rerun()
        with b2:
            if st.button("Jalankan Preprocessing & Ekstraksi Fitur", type="primary"):
                st.session_state.train_step = 4
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    elif current_step == 4:
        st.markdown('<div class="card"><div class="card-title">4. Ekstraksi Fitur Swin</div>', unsafe_allow_html=True)
        if st.session_state.train_features is None:
            st.info("Mengekstrak fitur menggunakan Swin Transformer (pretrained). Ini mungkin membutuhkan waktu beberapa menit.")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def pcb(msg, curr, total):
                safe_val = min(curr / total, 1.0) if total > 0 else 0.0
                progress_bar.progress(safe_val)
                status_text.markdown(f"<span style='color:#1E293B; font-weight:600;'>{msg}</span>", unsafe_allow_html=True)
                
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                if st.button("Kembali ke Preprocessing"):
                    st.session_state.train_step = 3
                    st.rerun()
            with b_col2:
                if st.button("Mulai Ekstraksi", type="primary"):
                    with st.spinner("Memuat model Swin..."):
                        swin_model = tp.load_swin_model()
                    
                    with st.spinner("Mengekstrak fitur..."):
                        res = tp.process_full_dataset(
                            st.session_state.train_ds_info,
                            st.session_state.train_preproc_cfg,
                            swin_model,
                            progress_cb=pcb,
                            seq_len=st.session_state.get("train_seq_len", 30)
                        )
                        st.session_state.train_features = res

                        # MEMORY FIX — INI AKAR MASALAH UTAMA:
                        # swin_model di atas dimuat ke VRAM/RAM HANYA untuk
                        # ekstraksi fitur dan TIDAK PERNAH dibersihkan.
                        # st.rerun() di bawah melempar exception untuk
                        # menghentikan script saat ini, tapi itu TIDAK
                        # menjamin tensor CUDA milik swin_model langsung
                        # dilepas dari VRAM — variabel lokal ini bisa
                        # menggantung di memori GPU.
                        # Akibatnya: saat nanti halaman Demo membuat
                        # DrowsinessDetector() baru (yang memuat Swin model
                        # KEDUA + LSTM bawaan, ditambah LSTM custom jika
                        # mode custom aktif), VRAM sudah terbebani sisa
                        # model ekstraksi ini — beresiko gagal alokasi
                        # CUDA dan mematikan proses Python tanpa traceback
                        # yang rapi (persis seperti yang dialami).
                        del swin_model
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        st.success(f"Ekstraksi selesai! {res['valid_frames']} frame valid diproses.")
                        st.rerun()
        else:
            res = st.session_state.train_features
            st.success(f"Ekstraksi selesai! {res['valid_frames']} frame valid diproses dari total {res['total_frames']} frame.")
            
            st.subheader("Konfigurasi Sequence")
            
            if "train_seq_len" not in st.session_state:
                st.session_state.train_seq_len = 30
            if "train_stride" not in st.session_state:
                st.session_state.train_stride = 15
                
            c1, c2 = st.columns(2)
            with c1:
                seq_opts = [15, 30, 45, 60]
                idx = seq_opts.index(st.session_state.train_seq_len) if st.session_state.train_seq_len in seq_opts else 1
                seq_len = st.selectbox("Sequence Length", seq_opts, index=idx, key="widget_seqlen")
            with c2: 
                stride = st.number_input("Stride", min_value=1, max_value=60, value=st.session_state.train_stride, key="widget_stride")
                
            st.session_state.train_seq_len = seq_len
            st.session_state.train_stride = stride
            
            b1, b2 = st.columns(2)
            with b1:
                if st.button("Ekstrak Ulang"):
                    st.session_state.train_features = None
                    st.rerun()
            with b2:
                if st.button("Lanjut ke Training", type="primary"):
                    st.session_state.train_step = 5
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    elif current_step == 5:
        st.markdown('<div class="card"><div class="card-title">5. Hyperparameter Tuning</div>', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            hidden_dim = st.selectbox("Hidden Dim", [128, 256, 512], index=1)
            num_layers = st.selectbox("Num Layers", [1, 2, 3], index=1)
            bidir = st.checkbox("Bidirectional", value=False)
            attn = st.checkbox("Use Attention", value=True)
        with c2:
            lr = st.selectbox("Learning Rate", [1e-2, 1e-3, 5e-4, 1e-4], index=1)
            batch = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
            epochs = st.slider("Epochs", 5, 100, 50)
            optim_ = st.selectbox("Optimizer", ["AdamW", "Adam", "SGD"], index=0)
        with c3:
            lstm_do = st.slider("LSTM Dropout", 0.0, 0.7, 0.3)
            fc_do = st.slider("FC Dropout", 0.0, 0.7, 0.4)
            fc_act = st.selectbox("FC Activation", ["gelu", "relu", "silu", "elu"], index=0)
            sched = st.selectbox("Scheduler", ["OneCycleLR", "StepLR", "None"], index=0)
            
        cfg = tp.TrainingConfig(
            hidden_dim=hidden_dim, num_layers=num_layers, bidirectional=bidir,
            use_attention=attn, fc_activation=fc_act, lstm_dropout=lstm_do,
            fc_dropout=fc_do, learning_rate=lr, batch_size=batch, epochs=epochs,
            optimizer=optim_, scheduler=sched, seq_len=st.session_state.train_seq_len,
            stride=st.session_state.train_stride
        )
        
        st.subheader("Validasi Konfigurasi")
        ds_size = st.session_state.train_ds_info.total_clips
        msgs = tp.validate_hyperparameters(cfg, ds_size)
        has_error = False
        for m in msgs:
            if m.level == "error":
                st.error(m.message)
                has_error = True
            elif m.level == "warning":
                st.warning(m.message)
            else:
                st.info(" " + m.message)
        
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Kembali"):
                st.session_state.train_step = 4
                st.rerun()
        with b2:
            if not has_error:
                if st.button("Mulai Training", type="primary"):
                    new_hash = get_cfg_hash(cfg)
                    
                    if st.session_state.get('train_config_hash') != new_hash:
                        st.session_state['train_metrics'] = None
                        st.session_state['train_hist'] = None
                        st.session_state['train_model'] = None
                        st.session_state['export_paths'] = None
                        st.session_state['train_config_hash'] = new_hash
                        
                    st.session_state.train_cfg = cfg
                    with st.spinner("Membangun sequences..."):
                        splits = tp.build_splits_with_sequences(
                            st.session_state.train_features["clip_features"],
                            st.session_state.train_features["clip_labels"],
                            seq_len=cfg.seq_len, stride=cfg.stride,
                            train_ratio=cfg.split_train, val_ratio=cfg.split_val
                        )
                        norm_mean, norm_std = tp.compute_norm_stats(splits["train"][0])
                        
                        st.session_state.train_splits = splits
                        st.session_state.train_norm_mean = norm_mean
                        st.session_state.train_norm_std = norm_std

                    progress_text = st.empty()
                    prog_bar = st.progress(0)
                    
                    def train_cb(ep, tot, mdict):
                        prog_bar.progress((ep+1)/tot)
                        progress_text.markdown(f"<span style='color:#1E293B; font-weight:600;'>Epoch {ep+1}/{tot} - Loss: {mdict['train_loss']:.4f} - Val F2: {mdict['val_f2']:.4f}</span>", unsafe_allow_html=True)
                        
                    with st.spinner("Training sedang berlangsung..."):
                        model, hist, best_state = tp.train_lstm_model(
                            cfg, splits, norm_mean, norm_std, progress_cb=train_cb
                        )
                        
                        eval_metrics = tp.evaluate_model(model, splits["test"][0], splits["test"][1], norm_mean, norm_std)

                        # MEMORY FIX: 'model' adalah objek PyTorch hidup
                        # (parameter + graf, masih nempel di GPU/CPU).
                        # Objek ini TIDAK pernah dibaca lagi di app.py —
                        # export_model() hanya memakai 'best_state' (cpu
                        # clone, jauh lebih ringan). Sebelumnya seluruh
                        # objek model ikut disimpan ke session_state dan
                        # menggantung sepanjang step 6+7, menambah beban
                        # memori yang tidak perlu di atas train_features
                        # yang juga masih ada. Jangan simpan ke session_state;
                        # lepas referensinya & bersihkan cache di sini.
                        del model
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        st.session_state.train_hist = hist
                        st.session_state.train_norm_mean = norm_mean
                        st.session_state.train_norm_std = norm_std
                        st.session_state.train_metrics = eval_metrics
                        st.session_state.train_best_state = best_state
                        
                    st.session_state.train_step = 6
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    elif current_step == 6:
        st.markdown('<div class="card"><div class="card-title">6. Evaluasi Model</div>', unsafe_allow_html=True)
        metrics = st.session_state.train_metrics
        hist = st.session_state.train_hist
        
        splits = st.session_state.get("train_splits")
        if splits is None:
            st.warning("⚠️ Data training tidak ditemukan. Silakan ulangi dari Step 5.")
            if st.button("Kembali ke Training"):
                st.session_state.train_step = 5
                st.rerun()
            st.stop()
            
        if 'train_cfg' in st.session_state:
            cfg_used = st.session_state.train_cfg
            splits_display = splits
            train_seq = len(splits_display['train'][0])
            val_seq   = len(splits_display['val'][0])
            test_seq  = len(splits_display['test'][0])
            total_seq = train_seq + val_seq + test_seq
            
            pct = lambda x: f"{x/total_seq*100:.0f}" if total_seq > 0 else "0"
            
            st.info(
                f"📊 **Statistik Sequence (seqlen={cfg_used.seq_len}, stride={cfg_used.stride}):**  \n"
                f"Total: **{total_seq}** sequences dari {st.session_state.train_ds_info.total_clips} frame  \n"
                f"Train: **{train_seq}** ({pct(train_seq)}%) · "
                f"Val: **{val_seq}** ({pct(val_seq)}%) · "
                f"Test (Confusion Matrix): **{test_seq}** ({pct(test_seq)}%)  \n"
                f"⚠️ Confusion Matrix dihitung dari **test set saja** ({test_seq} sequences) — ini benar secara metodologi."
            )
            
            if test_seq < 30:
                st.warning("⚠️ Perhatian: Test set kurang dari 30 sequence. Metrik evaluasi mungkin kurang stabil/reliabel.")

        st.markdown(f"### F2-Score: {metrics.f2:.4f}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{metrics.accuracy:.4f}")
        c2.metric("Precision", f"{metrics.precision:.4f}")
        c3.metric("Recall", f"{metrics.recall:.4f}")
        
        if tp._PLT_AVAILABLE:
            st.pyplot(tp.plot_training_curves(hist))
            st.pyplot(tp.plot_confusion_matrix(metrics))
            
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Ulangi Training (Hyperparameter Baru)"):
                st.session_state.train_step = 5
                st.rerun()
        with b2:
            if st.button("Ubah Preprocessing"):
                st.session_state.train_step = 3
                st.rerun()
        with b3:
            if st.button("Lanjut Simpan Model", type="primary"):
                st.session_state.train_step = 7
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    elif current_step == 7:
        st.markdown("""<div class="card"><div class="card-title">7. Ekspor Model</div>""",
                    unsafe_allow_html=True)
        st.info("Simpan model Anda untuk digunakan nanti atau langsung gunakan di halaman Demo.")

        # ── Jalankan ekspor jika belum ada ──────────────────────────────────────
        if 'export_paths' not in st.session_state or st.session_state['export_paths'] is None:
            try:
                # FIX: path absolut berbasis lokasi app.py, bukan relatif
                # terhadap current working directory (CWD). Path relatif
                # "models/custom" sebelumnya bisa menulis ke folder yang
                # salah tergantung dari mana Streamlit dijalankan (lokal
                # vs SSH vs service), yang berisiko bikin file "tidak
                # ditemukan" di langkah berikutnya walau penulisan sukses.
                _models_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "models", "custom"
                )
                os.makedirs(_models_dir, exist_ok=True)
                paths = tp.export_model(
                    st.session_state.train_best_state,
                    st.session_state.train_cfg,
                    st.session_state.train_norm_mean,
                    st.session_state.train_norm_std,
                    st.session_state.train_metrics,
                    st.session_state.train_hist,
                    _models_dir
                )
                if paths is None:
                    st.error("❌ export_model() mengembalikan None. Cek fungsi export_model di training_pipeline.py.")
                    st.stop()
                # Verifikasi semua file benar-benar ada di disk
                missing = [k for k, v in paths.items() if not os.path.exists(v)]
                if missing:
                    st.error(f"❌ File tidak terbuat: {missing}. Cek izin folder models/custom/")
                    st.stop()
                st.session_state['export_paths'] = paths

                # MEMORY FIX: setelah file .pth/.npz/.json berhasil ditulis
                # ke disk, semua informasi penting sudah aman tersimpan.
                # Bersihkan memori di titik ini supaya beban RAM/VRAM turun
                # sebelum lanjut render tombol download di bawah.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                st.error(f"❌ Gagal ekspor model: {e}")
                st.stop()

        paths = st.session_state['export_paths']

        # ── Tombol download ──────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            with open(paths["pth"], "rb") as f:
                st.download_button("Download .pth", f,
                                   file_name=os.path.basename(paths["pth"]))
        with c2:
            with open(paths["npz"], "rb") as f:
                st.download_button("Download .npz", f,
                                   file_name=os.path.basename(paths["npz"]))
        with c3:
            with open(paths["json"], "rb") as f:
                st.download_button("Download .json", f,
                                   file_name=os.path.basename(paths["json"]))

        zip_data = tp.create_zip_bundle(paths)
        st.download_button("Download Semua (ZIP)", zip_data,
                           file_name="drowsy_model_custom.zip",
                           mime="application/zip", type="primary",
                           use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        col_retrain, col_new = st.columns(2)
        with col_retrain:
            if st.button("🔄 Pelatihan Ulang (Parameter Baru)", use_container_width=True):
                # Reset hanya hasil training, pertahankan dataset & features
                for k in ['train_metrics', 'train_hist', 'train_model',
                          'train_best_state', 'export_paths', 'train_config_hash']:
                    st.session_state[k] = None
                st.session_state['train_step'] = 5
                st.rerun()
        with col_new:
            if st.button("📂 Upload Dataset Baru", use_container_width=True):
                # BUG-11 FIX: Bersihkan temp_dataset folder
                import shutil
                temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_dataset")
                if os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass
                # Full reset semua state
                for k in ['train_metrics', 'train_hist', 'train_model', 'train_best_state',
                          'export_paths', 'train_features', 'train_ds_info',
                          'train_cfg', 'train_config_hash']:
                    st.session_state[k] = None
                st.session_state['train_step'] = 1
                st.rerun()

        if st.button("▶ Gunakan di Demo", use_container_width=True):
            st.session_state['custom_model_pth'] = paths["pth"]
            st.session_state['custom_model_npz'] = paths["npz"]
            st.session_state['custom_model_label'] = f"Custom_{os.path.basename(paths['pth']).replace('.pth','')}"
            st.success("✅ Model custom siap! Silakan buka halaman Demo.")

        st.markdown("</div>", unsafe_allow_html=True)
    
    _page_footer()


# ============================================================
#  FITUR (Features)
# ============================================================
elif menu == "Fitur":
    st.markdown(
        '<div class="reveal">'
        '<div class="chapter-title">Dua belas <em>fitur inti</em>.</div>'
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
         "Fitur Swin diumpankan ke LSTM dalam jendela sequence  30 frame "
         "untuk memodelkan dinamika temporal."),
        ("09", "Three-Level Alert · Rolling Window 10",
         "10 prediksi terakhir ditimbang: Normal 0–22 drowsy, Warning 23-29 "
         "(alarm 3 detik), Danger 30 (alarm 6 detik)."),
        ("10", "Alarm Pemberitahuan Real-Time Berbasis Browser",
         "Alarm dipakai langsung melalui audio browser tanpa perangkat keras tambahan"),
        ("11", "Pelatihan Manual dengan Hyperparameter Lengkap",
         "Latih model LSTM kustom Anda dengan kontrol penuh atas arsitektur, "
         "hyperparameter, dan dataset."),
        ("12", "Penerapan Model Mandiri di Demo",
         "Ekspor model yang dilatih dan gunakan langsung di halaman Demo untuk deteksi kantuk real-time."),
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
        '<div class="chapter-title">Hubungi <em>peneliti</em>.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(      # ← tambah 4 spasi indentasi
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
        '<div class="ig-ico">'
        '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#E1306C" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="2" y="2" width="20" height="20" rx="5" ry="5"></rect>'
        '<path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"></path>'
        '<line x1="17.5" y1="6.5" x2="17.51" y2="6.5"></line>'
        '</svg>'
        '</div>'
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
