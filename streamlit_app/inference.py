# ============================================================
#  inference.py  —  Pipeline inferensi real-time (FIXED v4)
#  Drowsiness Detection System (Swin + LSTM)
# ============================================================
# ALUR CODE:
# frame_bgr → optional flip non-mirror → MediaPipe (CPU) → landmark + multi-EAR (16-pt)
#   → face loss tolerance check (FACE_LOSS_TOLERANCE_FRAMES)
#   → cek face coverage (skip jika terlalu dekat)
#   → CROP KEDUA MATA (16 landmark + margin 18%) sesuai training
#     06_Mediapipeline_Dataset_NTHUD → transform MRL mean/std (LANCZOS)
#     → Swin GPU forward L+R dalam 1 batch → RATA-RATA fitur 512-dim
#   → deque buffer (maxlen=30) → stack sequence → norm Z-score
#   → DrowsinessLSTM + AdditiveAttention GPU → softmax
#   → EAR confidence gate (16-pt min-EAR) → rolling history
#   → rapid-clear check → DetectionResult → draw overlay → return frame + result
#
# PERUBAHAN v4 vs v3 (BUG KRITIS):
#   [BUG-C] Model Swin dilatih pakai CROP MATA (1 mata per gambar, struktur
#           file ..._L.jpg / ..._R.jpg dari 06_Mediapipeline_Dataset_NTHUD).
#           v3 masih crop WAJAH PENUH → distribusi input mismatch parah.
#           v4 crop 2 mata terpisah, forward batch, rata-rata fitur →
#           input distribution match training.
#           Override via env var EYE_INFERENCE_MODE:
#             "both" (default) : crop L+R, batch, rata-rata 512-dim
#             "min"            : pilih mata dengan EAR terkecil (pesimistis)
#             "left" / "right" : fix ke satu sisi
#
# PERUBAHAN v3 vs v2:
#   [BUG-A] cfg key mismatch di _load_lstm — fixed (snake_case)
#   [BUG-B] LSTM path: update ke SWIN_LSTM_EXP_K_BEST.pth (Percobaan 2)
#   [FIX-1] infer_every_n default=1 — Swin tiap frame → buffer 30f = 1 detik nyata
#           (RTX 3060 sanggup ~4ms/frame Swin-tiny → 30fps achievable)
#   [FIX-2] Buffer RESET saat wajah hilang/terlalu dekat berturut-turut
#           > FACE_LOSS_TOLERANCE_FRAMES (10 frame ≈ 0.33s)
#   [FIX-3] Resize LANCZOS (sesuai PIL training preprocessing)
#   [FIX-4] Rapid-clear mechanism:
#           Jika RAPID_CLEAR_LOOKBACK (3) prediksi terakhir SEMUA jelas not-drowsy
#           (prob_drowsy < RAPID_CLEAR_PROB_THRESH = 0.35):
#           → step-down 1 level (DANGER→WARNING atau WARNING→NORMAL)
#   [EAR]   16-point landmarks matching 06_Mediapipeline_Dataset_NTHUD:
#             LEFT_EYE_IDX  : [33, 7, 163, 144, 145, 153, 154, 155,
#                             133, 173, 157, 158, 159, 160, 161, 246]
#             RIGHT_EYE_IDX : [362, 382, 381, 380, 374, 373, 390, 249,
#                             263, 466, 388, 387, 386, 385, 384, 398]
#           Multi-pair EAR (rata-rata 4 pasang vertikal) → lebih akurat
#           Min-EAR selection: ambil mata dengan EAR terkecil
#           EAR confidence gate:
#             EAR < 0.15 → paksa drowsy (override prob_drowsy ke 0.90)
#             EAR < 0.20 → boost prob_drowsy ke min 0.70
# ============================================================
# rumus:  eye 16 point
# (Atas: 9, 10, 11, 12, 13, 14, 15)
#    [0]---------------------------------[8]  <-- H (Horizontal)
#         (Bawah: 1, 2, 3, 4, 5, 6, 7)

#    Pasangan vertikal yang kamu gunakan di kode:
#    V1 = |pt13 - pt3| (Mid-Lateral)
#    V2 = |pt12 - pt4| (Mid)
#    V3 = |pt11 - pt5| (Mid-Medial)
#    V4 = |pt10 - pt6| (Inner)

import os
import gc
import time
import platform
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple


import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T


# ── MediaPipe ────────────────────────────────────────────────
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    print("[ERROR] MediaPipe tidak ditemukan. Jalankan: pip install mediapipe")


# ── timm (Swin backbone) ─────────────────────────────────────
try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False
    print("[ERROR] timm tidak ditemukan. Jalankan: pip install timm")


# ============================================================
#  DEVICE SETUP
# ============================================================
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"
USE_AMP  = USE_CUDA


if USE_CUDA:
    torch.backends.cudnn.benchmark = True
    _gpu_name = torch.cuda.get_device_name(0)
    _vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[INFO] GPU: {_gpu_name}  |  VRAM: {_vram_gb:.1f} GB")
    print(f"[INFO] AMP float16 aktif — inferensi lebih cepat & hemat VRAM")
else:
    print("[WARNING] CUDA tidak tersedia, inferensi berjalan di CPU — akan lebih lambat.")


# ============================================================
#  PATH MODEL
# ============================================================
_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_BASE_DIR, "models")


SWIN_MODEL_PATH = os.getenv(
    "SWIN_MODEL_PATH",
    os.path.join(_MODEL_DIR, "SWIN_BEST.pth"),
)


# [BUG-B FIX] LSTM model path — sekarang menunjuk ke SWIN_LSTM_EXP_K Percobaan 2.
# Override via env var jika lokasi berbeda.
LSTM_MODEL_PATH = os.getenv(
    "LSTM_MODEL_PATH",
    os.path.join(_MODEL_DIR, "SWIN_LSTM_EXP_K_BEST.pth"),
)


NORM_STATS_PATH = os.getenv(
    "NORM_STATS_PATH",
    os.path.join(_MODEL_DIR, "norm_stats.npz"),
)


# ============================================================
#  KONSTANTA
# ============================================================
# Normalisasi IMAGE — dari dataset MRL, notebook 03_Model_MRL
# WAJIB sama dengan training. Jangan ganti ke ImageNet mean/std!
MRL_IMG_MEAN = [0.3772, 0.3772, 0.3772]
MRL_IMG_STD  = [0.1544, 0.1544, 0.1544]


SEQ_LEN        = 30    # frame per sekuens LSTM (= 1 detik @ 30fps)
HISTORY_SIZE   = 30    # rolling window prediksi alert
WARNING_THRESH = 23     
DANGER_THRESH  = 30  


# [FIX-4] Confidence-gated rolling window
MIN_DROWSY_CONFIDENCE = 0.60   # prob_drowsy < 0.60 → dianggap not-drowsy di history
MIN_HISTORY_SIZE      = 5      # butuh >= 5 prediksi sebelum alert aktif


# [FIX-4] Rapid-clear mechanism — N prediksi terakhir berturut-turut
#         jelas not-drowsy → step-down 1 level langsung (responsive recovery)
RAPID_CLEAR_LOOKBACK    = 3      # jumlah prediksi terakhir yang dicek
RAPID_CLEAR_PROB_THRESH = 0.35   # prob_drowsy < ini → "jelas not-drowsy"


# [FIX-2] Face loss tolerance — reset buffer hanya setelah hilang sustained
FACE_LOSS_TOLERANCE_FRAMES = 10   # ≈ 0.33 detik @ 30fps


# Face coverage threshold (out-of-distribution detector)
MAX_FACE_WIDTH_RATIO  = 0.65   # face_w / frame_w > 0.65 → skip inferensi


# [EAR] EAR-based confidence gate
EAR_CLOSED_GATE = 0.15   # min_ear < ini → mata pasti tertutup, paksa drowsy
EAR_DROWSY_GATE = 0.20   # min_ear < ini → mata hampir tertutup, boost prob


# [BUG-C] Eye crop margin — match training preprocessing
# 06_Mediapipeline_Dataset_NTHUD: square bounding box per mata dengan padding.
# 0.18 = 18% margin pada lebar/tinggi mata (menampung alis + kelopak).
EYE_CROP_MARGIN = float(os.getenv("EYE_CROP_MARGIN", "0.18")) #memberikan ruang tambahan (padding) sebesar 18% ke arah luar dari ukuran mata tersebut.


# [BUG-C] Eye inference mode — cara gabung L/R eye ke feature vector Swin:
#   "both"   : forward L+R dalam 1 batch, rata-rata fitur 512-dim (default)
#              → match training distribution (uniform L & R)
#   "min"    : pilih mata dengan EAR terkecil → 1 forward pass (hemat compute)
#              → bias pesimistis, bagus untuk worst-case sensitivity
#   "left"   : selalu mata kiri (debug only)
#   "right"  : selalu mata kanan (debug only)
EYE_INFERENCE_MODE = os.getenv("EYE_INFERENCE_MODE", "both").lower().strip()
if EYE_INFERENCE_MODE not in ("both", "min", "left", "right"):
    print(f"[WARNING] EYE_INFERENCE_MODE='{EYE_INFERENCE_MODE}' tidak dikenali. "
          f"Fallback ke 'both'.")
    EYE_INFERENCE_MODE = "both"


# [PERF] Periodic VRAM cleanup
VRAM_CLEANUP_INTERVAL = 500    # setiap 500 inferensi


CLASS_NAMES = {0: "Drowsy", 1: "Not Drowsy"}


# ============================================================
#  EAR — 16-POINT LANDMARKS (sesuai 06_Mediapipeline_Dataset_NTHUD)
# ============================================================
# Landmark positions in eye_idx:
#   pos 0 = outer corner (lateral)
#   pos 8 = inner corner (medial)
#   pos 1..7  = lower lid (perimeter)
#   pos 9..15 = upper lid (perimeter)
#
# Multi-pair EAR memakai 4 pasang vertikal (lebih robust dari 6-pt klasik):
#   - Pair (13, 3): mid-lateral
#   - Pair (12, 4): mid
#   - Pair (11, 5): mid-medial
#   - Pair (10, 6): inner
# Horizontal: pos 0 ↔ pos 8
# ============================================================
LEFT_EYE_IDX  = [33,  7,   163, 144, 145, 153, 154, 155,
                 133, 173, 157, 158, 159, 160, 161, 246]


RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249,
                 263, 466, 388, 387, 386, 385, 384, 398]


# Pairs (sama untuk kedua mata karena layout simetris di eye_idx)
_EAR_H_OUTER = 0   # outer corner position dalam eye_idx
_EAR_H_INNER = 8   # inner corner position dalam eye_idx
_EAR_V_PAIRS = [(13, 3), (12, 4), (11, 5), (10, 6)]


# ============================================================
#  ARSITEKTUR LSTM  —  identik dengan 09_B_TEST_TUNING
# ============================================================
class AdditiveAttention(nn.Module):
    """Bahdanau-style attention. Input: [B, T, H] → Output: [B, H]."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v    = nn.Linear(hidden_dim, 1, bias=False)


    def forward(self, lstm_out: torch.Tensor):
        energy  = torch.tanh(self.attn(lstm_out))               # [B, T, H]
        scores  = self.v(energy).squeeze(-1)                    # [B, T]
        weights = torch.softmax(scores, dim=-1)                 # [B, T]
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return context, weights


class DrowsinessLSTM(nn.Module):
    """
    Arsitektur LSTM sesuai notebook 09_B_TEST_TUNING.
    cfg dict dari checkpoint akan override default ini secara otomatis.
    """
    def __init__(
        self,
        input_dim:    int   = 512,
        hidden_dim:   int   = 256,
        num_layers:   int   = 2,
        num_classes:  int   = 2,
        bidirectional: bool = False,
        use_attention: bool = True,
        lstm_dropout: float = 0.3,
        fc_dropout:   float = 0.4,
        fc_activation: str  = "gelu",
    ):
        super().__init__()
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        num_dir       = 2 if bidirectional else 1
        lstm_out_dim  = hidden_dim * num_dir


        self.input_norm = nn.LayerNorm(input_dim)


        self.lstm = nn.LSTM(
            input_size    = input_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = bidirectional,
            dropout       = lstm_dropout if num_layers > 1 else 0.0,
        )


        self.attention = AdditiveAttention(lstm_out_dim) if use_attention else None


        _act_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "elu":  nn.ELU(),
        }
        act = _act_map.get(fc_activation.lower(), nn.GELU())


        self.classifier = nn.Sequential(
            nn.Dropout(fc_dropout),
            nn.Linear(lstm_out_dim, 128),
            act,
            nn.Dropout(fc_dropout * 0.5),
            nn.Linear(128, num_classes),
        )


    def forward(self, x: torch.Tensor):
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)
        if self.use_attention and self.attention is not None:
            context, attn_w = self.attention(lstm_out)
        else:
            context = lstm_out[:, -1, :]
            attn_w  = None
        logits = self.classifier(context)
        return logits, attn_w


# ============================================================
#  SWIN FEATURE EXTRACTOR (auto-detect head)
# ============================================================
class SwinFeatureExtractor(nn.Module):
    """
    Swin backbone dengan auto-detection arsitektur head.

    Mendukung 2 skema training:
      A. Training pakai timm `num_classes=512` (head bawaan timm)
         → forward() returns [B, 512] langsung dari head
      B. Training pakai backbone-only + custom proj 768→512
         → forward() returns [B, 512] dari proj layer

    Auto-detect via parameter `head_out_dim`:
      - jika diset (misal 512): pakai skema A
      - jika None: pakai skema B (custom proj)
    """
    def __init__(self, model_name: str = "swin_tiny_patch4_window7_224",
                 head_out_dim: Optional[int] = None):
        super().__init__()
        if not _TIMM_AVAILABLE:
            raise ImportError("timm wajib diinstall: pip install timm")


        if head_out_dim is not None and head_out_dim > 2:
            # SKEMA A: pakai head timm bawaan sebagai feature projector
            self.backbone = timm.create_model(
                model_name, pretrained=False, num_classes=head_out_dim,
            )
            self._use_proj      = False
            self._feature_dim   = head_out_dim
            print(f"[INFO] Swin: skema A (timm head, {head_out_dim}-dim feature)")
        else:
            # SKEMA B: backbone only + custom proj
            self.backbone = timm.create_model(
                model_name, pretrained=False, num_classes=0,
            )
            self.proj = nn.Linear(768, 512, bias=True)
            self._use_proj    = True
            self._feature_dim = 512
            print(f"[INFO] Swin: skema B (backbone+proj, 512-dim feature)")


        # Verifikasi output dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out   = self.backbone(dummy)
            self._backbone_out_dim = out.shape[-1]
        print(f"[INFO] Swin backbone output: {self._backbone_out_dim}-dim, "
              f"final feature: {self._feature_dim}-dim")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if self._use_proj:
            feat = self.proj(feat)
        return feat


# ============================================================
#  DATA RESULT
# ============================================================
@dataclass
class DetectionResult:
    face_count:     int
    pred_class:     Optional[int]
    pred_label:     str
    confidence:     float
    status_level:   str      # NORMAL / WARNING / DANGER
    run_state:      str      # MULAI / PROSES / MATI / TOO_CLOSE
    min_ear:        float
    prob_drowsy:    float
    processing_fps: float
    buffer_fill:    int      # 0–30
    infer_ms:       float


# ============================================================
#  UTILITAS GAMBAR
# ============================================================
def _draw_text_bg(img, text, org, fg=(255, 255, 255), bg=(20, 35, 60),
                  scale=0.55, thick=2):
    x, y = org
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 10, y + bl - 2), bg, -1)
    cv2.putText(img, text, (x + 5, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)


def _crop_eye(
    frame_bgr: np.ndarray,
    pts,
    eye_idx,
    margin: float = EYE_CROP_MARGIN,
) -> np.ndarray:
    """
    [BUG-C] Crop ROI 1 mata dari 16 landmark eye_idx dengan margin.

    Match training preprocessing (06_Mediapipeline_Dataset_NTHUD):
      - Ambil bounding box dari 16 titik landmark mata
      - Tambah padding margin (default 18%) agar kelopak/alis tidak terpotong
      - Clamp ke dimensi frame

    Args:
      frame_bgr : frame input (H, W, 3) BGR
      pts       : list (x, y) 468 landmark MediaPipe dalam koordinat pixel
      eye_idx   : list 16 indeks landmark mata (LEFT_EYE_IDX / RIGHT_EYE_IDX)
      margin    : fraksi margin (0.18 = 18%)

    Returns:
      crop (H', W', 3) BGR. Shape kosong (0,0,0) jika crop invalid.
    """
    h, w = frame_bgr.shape[:2]
    eye_pts = [pts[i] for i in eye_idx]
    xs = [p[0] for p in eye_pts]
    ys = [p[1] for p in eye_pts]


    eye_w_raw = max(xs) - min(xs)
    eye_h_raw = max(ys) - min(ys)
    mx = int(eye_w_raw * margin)
    my = int(eye_h_raw * margin)


    x1 = max(0, min(xs) - mx)
    y1 = max(0, min(ys) - my)
    x2 = min(w, max(xs) + mx)
    y2 = min(h, max(ys) + my)


    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=frame_bgr.dtype)
    return frame_bgr[y1:y2, x1:x2]


def _compute_ear_16pt(pts, eye_idx):
    """
    Hitung Eye Aspect Ratio dari 16 titik landmark
    (sesuai 06_Mediapipeline_Dataset_NTHUD).

    Multi-pair EAR (rata-rata 4 pasang vertikal):
        EAR = mean(||v_up_i - v_low_i||) / ||p_outer - p_inner||

    Lebih robust dari 6-point klasik karena merata-ratakan beberapa
    pengukuran vertikal di sepanjang kelopak mata.

    Args:
      pts     : list semua landmark (x, y) dalam koordinat pixel
      eye_idx : list 16 indeks landmark (LEFT_EYE_IDX atau RIGHT_EYE_IDX)

    Returns:
      EAR (float). Nilai 0.0 jika gagal hitung.
    """
    try:
        p_outer = np.asarray(pts[eye_idx[_EAR_H_OUTER]], dtype=np.float32)
        p_inner = np.asarray(pts[eye_idx[_EAR_H_INNER]], dtype=np.float32)
        h = float(np.linalg.norm(p_outer - p_inner))
        if h < 1e-6:
            return 0.0


        verticals = []
        for (ui, li) in _EAR_V_PAIRS:
            p_up  = np.asarray(pts[eye_idx[ui]], dtype=np.float32)
            p_low = np.asarray(pts[eye_idx[li]], dtype=np.float32)
            verticals.append(float(np.linalg.norm(p_up - p_low)))


        return float(np.mean(verticals)) / h
    except Exception:
        return 0.0


# ============================================================
#  KELAS DETEKTOR UTAMA
# ============================================================
class DrowsinessDetector:
    """
    Pipeline inferensi real-time:
      CPU : MediaPipe Face Mesh, OpenCV overlay, logika alert
      GPU : Swin feature extractor, LSTM classifier


    Parameter publik:
      infer_every_n          : proses tiap N frame.
                               DEFAULT = 1 (rekomendasi: jangan ubah).
                               N>1 akan merusak temporal density LSTM —
                               buffer 30 fitur tidak lagi = 1 detik nyata.
      min_drowsy_confidence  : threshold prob drowsy untuk masuk rolling window
                               (0.60 default — bisa di-tune dari app.py)
    """


    def __init__(
        self,
        infer_every_n: int = 1,                # [FIX-1] default=1
        min_drowsy_confidence: float = MIN_DROWSY_CONFIDENCE,
    ):
        self.device                 = DEVICE
        self.infer_n                = max(1, int(infer_every_n))
        self.min_drowsy_confidence  = float(min_drowsy_confidence)


        if self.infer_n > 1:
            print(f"[WARNING] infer_every_n={self.infer_n} > 1 — "
                  f"temporal integrity buffer LSTM TERGANGGU. "
                  f"Buffer 30 fitur ≠ 1 detik nyata lagi. "
                  f"Disarankan tetap di 1 untuk akurasi terbaik.")


        print(f"[INFO] Eye inference mode: '{EYE_INFERENCE_MODE}'  |  "
              f"crop margin: {EYE_CROP_MARGIN:.2f}")


        # ── MediaPipe (CPU) ───────────────────────────────────
        if not _MP_AVAILABLE:
            raise ImportError("MediaPipe tidak ditemukan.")
        _mp = mp.solutions.face_mesh
        self.mp_drawing        = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh_cls  = _mp
        self.face_mesh = _mp.FaceMesh(
            static_image_mode        = False,
            max_num_faces            = 1,
            refine_landmarks         = True,
            min_detection_confidence = 0.5,
            min_tracking_confidence  = 0.5,
        )


        # ── Transform image (MRL stats, LANCZOS sesuai training) ──
        # [FIX-3] Interpolasi LANCZOS — match PIL training preprocessing.
        # Default torchvision adalah BILINEAR yang menyebabkan domain shift halus.
        self.img_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224), interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
            T.Normalize(mean=MRL_IMG_MEAN, std=MRL_IMG_STD),
        ])


        # ── Load model (GPU) ──────────────────────────────────
        self.swin_model               = self._load_swin()
        self.lstm_model               = self._load_lstm()
        self.norm_mean, self.norm_std = self._load_norm_stats()


        # ── Buffer & State ────────────────────────────────────
        self.feature_buf  = deque(maxlen=SEQ_LEN)
        self.pred_history = deque(maxlen=HISTORY_SIZE)
        self._recent_probs = deque(maxlen=RAPID_CLEAR_LOOKBACK)  # [FIX-4]


        self._last_pred       = None
        self._last_conf       = 0.0
        self._last_prob_d     = 0.0
        self._last_status     = "NORMAL"
        self._frame_idx       = 0
        self._infer_count     = 0       # untuk VRAM cleanup periodik
        self._too_close_warn  = False
        self._face_loss_counter = 0     # [FIX-2] counter wajah hilang


    # ── Loader Swin (auto-detect head) ────────────────────────
    def _load_swin(self) -> nn.Module:
        if not os.path.exists(SWIN_MODEL_PATH):
            raise FileNotFoundError(
                f"SWIN model tidak ditemukan: {SWIN_MODEL_PATH}\n"
                f"Pastikan file ada di folder models/"
            )
        print(f"[INFO] Loading Swin: {SWIN_MODEL_PATH}")
        ckpt = torch.load(SWIN_MODEL_PATH, map_location=self.device,
                          weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt


        # Auto-detect: cari key 'head.weight' untuk tahu output dim
        head_out_dim = None
        for key in ["head.weight", "head.fc.weight", "model.head.weight"]:
            if key in state:
                head_out_dim = int(state[key].shape[0])
                print(f"[INFO] Swin head terdeteksi: {key} → {head_out_dim}-dim")
                break


        # Build model sesuai hasil deteksi
        model = SwinFeatureExtractor(head_out_dim=head_out_dim).to(self.device)
        missing, unexpected = model.load_state_dict(state, strict=False)


        if missing:
            print(f"[WARNING] Swin — keys missing: {len(missing)}. "
                  f"Contoh: {missing[:3]}")
        if unexpected:
            print(f"[WARNING] Swin — keys unexpected: {len(unexpected)}. "
                  f"Contoh: {unexpected[:3]}")


        model.eval()
        return model


    # ── Loader LSTM ──────────────────────────────────────────
    def _load_lstm(self) -> nn.Module:
        if not os.path.exists(LSTM_MODEL_PATH):
            raise FileNotFoundError(
                f"LSTM model tidak ditemukan: {LSTM_MODEL_PATH}\n"
                f"Set env var LSTM_MODEL_PATH atau update default di kode."
            )
        print(f"[INFO] Loading LSTM: {LSTM_MODEL_PATH}")
        ckpt = torch.load(LSTM_MODEL_PATH, map_location=self.device,
                          weights_only=False)


        cfg = {}
        if isinstance(ckpt, dict):
            cfg = ckpt.get("cfg", {})


        # [BUG-A FIX] cfg key dengan UNDERSCORE (snake_case) — sesuai
        # cara training menyimpan di 09_B_TEST_TUNING.
        # Sebelumnya: "hiddendim", "numlayers", dll → SELALU pakai default.
        model = DrowsinessLSTM(
            input_dim     = 512,
            hidden_dim    = cfg.get("hidden_dim",    256),
            num_layers    = cfg.get("num_layers",    2),
            num_classes   = 2,
            bidirectional = cfg.get("bidirectional", False),
            use_attention = cfg.get("use_attention", True),
            lstm_dropout  = cfg.get("lstm_dropout",  0.3),
            fc_dropout    = cfg.get("fc_dropout",    0.4),
            fc_activation = cfg.get("fc_activation", "gelu"),
        ).to(self.device)


        state = (
            ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
            if isinstance(ckpt, dict) else ckpt
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARNING] LSTM — keys missing: {missing[:3]}")
        model.eval()
        print(
            f"[INFO] LSTM loaded — cfg: hidden={cfg.get('hidden_dim',256)}, "
            f"layers={cfg.get('num_layers',2)}, "
            f"act={cfg.get('fc_activation','gelu')}, "
            f"bidir={cfg.get('bidirectional',False)}"
        )
        return model


    # ── Loader Norm Stats ────────────────────────────────────
    def _load_norm_stats(self):
        """Z-score stats untuk sequence fitur. npz → checkpoint → fallback."""
        if os.path.exists(NORM_STATS_PATH):
            data = np.load(NORM_STATS_PATH)
            mean = torch.tensor(data["mean"], dtype=torch.float32, device=self.device)
            std  = torch.tensor(data["std"],  dtype=torch.float32, device=self.device)
            std  = std.clamp(min=1e-8)
            print(f"[INFO] norm_stats.npz loaded — mean.mean={mean.mean():.4f}")
            return mean, std


        try:
            ckpt = torch.load(LSTM_MODEL_PATH, map_location=self.device,
                              weights_only=False)
            if isinstance(ckpt, dict) and "norm_mean" in ckpt:
                mean_v = ckpt["norm_mean"]
                std_v  = ckpt["norm_std"]
                if not torch.is_tensor(mean_v):
                    mean_v = torch.tensor(mean_v, dtype=torch.float32)
                    std_v  = torch.tensor(std_v,  dtype=torch.float32)
                mean = mean_v.to(self.device).float()
                std  = std_v.to(self.device).float().clamp(min=1e-8)
                print("[INFO] Norm stats diambil dari checkpoint LSTM.")
                return mean, std
        except Exception:
            pass


        raise RuntimeError(
            "[ERROR] norm_stats tidak ditemukan baik di npz maupun di checkpoint LSTM!\n"
            "Sistem menolak fallback ke mean=0, std=1 karena dapat menyebabkan\n"
            "distribusi input mismatch yang fatal. Pastikan norm_stats tersedia!"
        )


    # ── Reset sesi (full) ────────────────────────────────────
    def reset(self):
        """Full reset (tombol Reset Camera di app.py)."""
        self._reset_temporal_state()
        self._frame_idx       = 0
        self._infer_count     = 0
        gc.collect()
        if USE_CUDA:
            torch.cuda.empty_cache()


    # ── Load Custom LSTM (dari hasil training user) ──────────
    def load_custom_lstm(self, pth_path: str, npz_path: str):
        """
        Ganti LSTM model dan norm stats dengan model custom.
        Dipanggil dari app.py saat user memilih model custom di Demo.

        Args:
            pth_path: path ke file .pth checkpoint LSTM custom
            npz_path: path ke file .npz norm stats custom
        """
        if not os.path.exists(pth_path):
            raise FileNotFoundError(f"Custom LSTM model tidak ditemukan: {pth_path}")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Custom norm stats tidak ditemukan: {npz_path}")

        print(f"[INFO] Loading custom LSTM: {pth_path}")
        ckpt = torch.load(pth_path, map_location=self.device, weights_only=False)

        cfg = {}
        if isinstance(ckpt, dict):
            cfg = ckpt.get("cfg", {})

        # Bangun model LSTM baru dengan config dari checkpoint custom
        new_lstm = DrowsinessLSTM(
            input_dim     = 512,
            hidden_dim    = cfg.get("hidden_dim",    256),
            num_layers    = cfg.get("num_layers",    2),
            num_classes   = 2,
            bidirectional = cfg.get("bidirectional", False),
            use_attention = cfg.get("use_attention", True),
            lstm_dropout  = cfg.get("lstm_dropout",  0.3),
            fc_dropout    = cfg.get("fc_dropout",    0.4),
            fc_activation = cfg.get("fc_activation", "gelu"),
        ).to(self.device)

        state = (
            ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
            if isinstance(ckpt, dict) else ckpt
        )
        missing, unexpected = new_lstm.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARNING] Custom LSTM — keys missing: {missing[:3]}")
        new_lstm.eval()

        print(
            f"[INFO] Custom LSTM loaded — cfg: hidden={cfg.get('hidden_dim', 256)}, "
            f"layers={cfg.get('num_layers', 2)}, "
            f"act={cfg.get('fc_activation', 'gelu')}, "
            f"bidir={cfg.get('bidirectional', False)}"
        )

        # Load custom norm stats
        print(f"[INFO] Loading custom norm stats: {npz_path}")
        data = np.load(npz_path)
        new_mean = torch.tensor(data["mean"], dtype=torch.float32, device=self.device)
        new_std  = torch.tensor(data["std"],  dtype=torch.float32, device=self.device)
        new_std  = new_std.clamp(min=1e-8)
        print(f"[INFO] Custom norm_stats loaded — mean.mean={new_mean.mean():.4f}")

        # Ganti model dan stats yang aktif
        self.lstm_model = new_lstm
        self.norm_mean  = new_mean
        self.norm_std   = new_std

        # Reset buffer temporal agar tidak ada state dari model lama
        self._reset_temporal_state()
        print("[INFO] Custom LSTM swap complete — buffer direset")


    # ── Reset state temporal saja (auto, saat wajah hilang) ──
    def _reset_temporal_state(self):
        """
        [FIX-2] Dipanggil otomatis saat wajah hilang/terlalu dekat
        sustained > FACE_LOSS_TOLERANCE_FRAMES frames.
        Mengosongkan buffer fitur + history prediksi tanpa reset
        counter frame keseluruhan (UI counter tetap jalan).
        """
        self.feature_buf.clear()
        self.pred_history.clear()
        self._recent_probs.clear()
        self._last_pred       = None
        self._last_conf       = 0.0
        self._last_prob_d     = 0.0
        self._last_status     = "NORMAL"


    # ── Evaluasi alert (FIXED v3) ────────────────────────────
    def _eval_alert(
        self,
        pred_class: Optional[int],
        prob_drowsy: float,
        new_inference: bool,
    ) -> str:
        """
        Logika tier alert berbasis rolling window + rapid-clear.

        [v3 FIXED]
          - History HANYA update saat ada inferensi BARU
          - Confidence-gating: prob_drowsy < threshold → treat as not-drowsy
          - MIN_HISTORY_SIZE: tidak alert sebelum cukup data
          - RAPID-CLEAR: 3 prediksi terakhir jelas not-drowsy → step down 1 level
        """
        # Update history HANYA saat ada inferensi baru
        if new_inference and pred_class is not None:
            # Confidence-gated: drowsy marginal tidak masuk hitungan
            effective = pred_class
            if pred_class == 0 and prob_drowsy < self.min_drowsy_confidence:
                effective = 1   # demote ke not-drowsy
            self.pred_history.append(effective)


            # Track recent probs untuk rapid-clear
            self._recent_probs.append((effective, float(prob_drowsy)))


        # Belum cukup data → NORMAL (cegah false alarm awal)
        if len(self.pred_history) < MIN_HISTORY_SIZE:
            self._last_status = "NORMAL"
            return "NORMAL"


        # Base alert level dari rolling window
        drowsy_n = sum(1 for p in self.pred_history if p == 0)


        if drowsy_n >= DANGER_THRESH:
            status = "DANGER"
        elif drowsy_n >= WARNING_THRESH:
            status = "WARNING"
        else:
            status = "NORMAL"


        # [FIX-4] RAPID-CLEAR: kalau 3 prediksi terakhir SEMUA not-drowsy
        # dan prob_drowsy-nya rendah → step down 1 level langsung.
        # Ini menghindari "DANGER latch" saat user bangun cepat.
        if (len(self._recent_probs) == RAPID_CLEAR_LOOKBACK and
                all(pc == 1 and pd < RAPID_CLEAR_PROB_THRESH
                    for pc, pd in self._recent_probs)):
            if status == "DANGER":
                status = "WARNING"
            elif status == "WARNING":
                status = "NORMAL"


        # NOTE: Alarm dipicu di app.py via AlarmManager (bukan di sini)
        self._last_status = status
        return status


    # ── EAR confidence gate ──────────────────────────────────
    def _apply_ear_gate(self, pred_class, prob_drowsy, min_ear):
        """
        [EAR] Hard/soft gate berbasis EAR.

        EAR < 0.15 (CLOSED_GATE):
            Mata pasti tertutup → paksa kelas drowsy + prob 0.90.
            Justifikasi: visual model bisa miss subtle eye-closure tapi
            geometric EAR tidak bohong saat kelopak benar-benar menutup.

        EAR < 0.20 (DROWSY_GATE):
            Mata hampir tertutup → boost prob_drowsy ke min 0.70.
            Tidak override pred_class — biar model tetap berperan, tapi
            confidence-gating di _eval_alert akan terlewat (≥ 0.60).

        Returns (effective_pred, effective_prob).
        """
        if pred_class is None:
            return pred_class, prob_drowsy


        if min_ear < EAR_CLOSED_GATE:
            return 0, max(float(prob_drowsy), 0.90)
        elif min_ear < EAR_DROWSY_GATE:
            return pred_class, max(float(prob_drowsy), 0.70)
        else:
            return pred_class, float(prob_drowsy)


    # ── Main process_frame ────────────────────────────────────
    def process_frame(
        self,
        frame_bgr: np.ndarray,
        draw_mesh: bool = True,
        flip_frame: bool = True,
    ) -> Tuple[np.ndarray, DetectionResult]:


        t_start    = time.perf_counter()
        if flip_frame:
            frame_bgr = cv2.flip(frame_bgr, 1)
        h, w       = frame_bgr.shape[:2]


        # ── MediaPipe (CPU) ───────────────────────────────────
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_res = self.face_mesh.process(rgb)


        # ── Tidak ada wajah ───────────────────────────────────
        if not mp_res.multi_face_landmarks:
            self._frame_idx += 1
            # [FIX-2] Increment face loss counter, reset state jika sustained
            self._face_loss_counter += 1
            if self._face_loss_counter >= FACE_LOSS_TOLERANCE_FRAMES:
                self._reset_temporal_state()


            elapsed = (time.perf_counter() - t_start) * 1000.0
            return frame_bgr, DetectionResult(
                face_count=0, pred_class=None, pred_label="No Face",
                confidence=0.0, status_level="NORMAL", run_state="MATI",
                min_ear=0.0, prob_drowsy=self._last_prob_d,
                processing_fps=1000.0 / max(elapsed, 1e-6),
                buffer_fill=len(self.feature_buf), infer_ms=elapsed,
            )


        face_lm = mp_res.multi_face_landmarks[0]


        # Landmark → pixel
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_lm.landmark]


        # ── EAR 16-point (sesuai training preprocessing) ──────
        ear_l   = _compute_ear_16pt(pts, LEFT_EYE_IDX)
        ear_r   = _compute_ear_16pt(pts, RIGHT_EYE_IDX)
        # [EAR] Min-EAR: ambil mata yang lebih tertutup sebagai sinyal
        min_ear = min(ear_l, ear_r)


        # [BUG-C] Face bbox HANYA untuk cek coverage (too_close),
        # BUKAN dipakai sebagai input model. Input model = eye crops.
        xs_face = [p[0] for p in pts]
        face_w_raw = max(xs_face) - min(xs_face)


        # [BUG-C] Crop KEDUA mata dari 16 landmark + margin 18%
        # Match training distribution (model dilatih per-eye, bukan face).
        eye_crop_l = _crop_eye(frame_bgr, pts, LEFT_EYE_IDX,  EYE_CROP_MARGIN)
        eye_crop_r = _crop_eye(frame_bgr, pts, RIGHT_EYE_IDX, EYE_CROP_MARGIN)


        # Cek face coverage — terlalu dekat = OOD dari training
        face_coverage = face_w_raw / max(w, 1)
        too_close     = face_coverage > MAX_FACE_WIDTH_RATIO


        # [FIX-2] Counter face-loss: face hilang ATAU terlalu dekat
        if too_close:
            self._too_close_warn = True
            self._face_loss_counter += 1
            if self._face_loss_counter >= FACE_LOSS_TOLERANCE_FRAMES:
                self._reset_temporal_state()
            _draw_text_bg(
                frame_bgr,
                "TERLALU DEKAT — mundur dari kamera",
                (10, 178), bg=(0, 100, 200),  # orange-ish (BGR)
            )
        else:
            self._too_close_warn = False
            self._face_loss_counter = 0      # wajah valid → reset counter


        # [BUG-C] Validasi eye crops — jika salah satu kosong → crop error.
        # Untuk mode "both" butuh dua-duanya valid; mode lain butuh sisi target-nya.
        if EYE_INFERENCE_MODE == "both":
            eye_invalid = eye_crop_l.size == 0 or eye_crop_r.size == 0
        elif EYE_INFERENCE_MODE == "left":
            eye_invalid = eye_crop_l.size == 0
        elif EYE_INFERENCE_MODE == "right":
            eye_invalid = eye_crop_r.size == 0
        else:  # "min"
            target = eye_crop_l if ear_l < ear_r else eye_crop_r
            eye_invalid = target.size == 0


        if eye_invalid:
            elapsed = (time.perf_counter() - t_start) * 1000.0
            return frame_bgr, DetectionResult(
                face_count=1, pred_class=None, pred_label="Crop Error",
                confidence=0.0, status_level="NORMAL", run_state="MATI",
                min_ear=min_ear, prob_drowsy=self._last_prob_d,
                processing_fps=1000.0 / max(elapsed, 1e-6),
                buffer_fill=len(self.feature_buf), infer_ms=elapsed,
            )


        # ── Inferensi GPU ─────────────────────────────────────
        self._frame_idx += 1
        do_infer       = (self._frame_idx % self.infer_n == 0) and not too_close
        new_infer_made = False


        if do_infer:
            # [BUG-C] Swin forward — satu atau dua mata, tergantung mode.
            if EYE_INFERENCE_MODE == "both":
                # L + R dalam 1 batch → 2× efisien dari 2 forward sekuensial
                t_l = self.img_transform(
                    cv2.cvtColor(eye_crop_l, cv2.COLOR_BGR2RGB)
                )
                t_r = self.img_transform(
                    cv2.cvtColor(eye_crop_r, cv2.COLOR_BGR2RGB)
                )
                tensor_img = torch.stack([t_l, t_r], dim=0).to(
                    self.device, non_blocking=True
                )
            else:
                # Mode min / left / right → 1 forward pass
                if EYE_INFERENCE_MODE == "left":
                    eye_crop = eye_crop_l
                elif EYE_INFERENCE_MODE == "right":
                    eye_crop = eye_crop_r
                else:  # "min" — pesimistis, ambil mata lebih tertutup
                    eye_crop = eye_crop_l if ear_l < ear_r else eye_crop_r
                tensor_img = self.img_transform(
                    cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
                ).unsqueeze(0).to(self.device, non_blocking=True)


            with torch.no_grad():
                if USE_AMP:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        feats = self.swin_model(tensor_img)
                else:
                    feats = self.swin_model(tensor_img)


            # [BUG-C] Rata-rata fitur L+R jika mode=both; jika single ambil langsung.
            if EYE_INFERENCE_MODE == "both":
                feat = feats.mean(dim=0, keepdim=True)   # [2,512] → [1,512]
            else:
                feat = feats                             # sudah [1,512]


            self.feature_buf.append(feat.squeeze(0).float().detach())


            if len(self.feature_buf) >= SEQ_LEN:
                # Sequence → LSTM (GPU)
                seq = torch.stack(list(self.feature_buf), dim=0).unsqueeze(0).to(self.device)
                seq = (seq - self.norm_mean) / self.norm_std


                with torch.no_grad():
                    if USE_AMP:
                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                            logits, _ = self.lstm_model(seq)
                    else:
                        logits, _ = self.lstm_model(seq)


                probs       = torch.softmax(logits.float(), dim=1)
                pred_class  = int(torch.argmax(probs, dim=1).item())
                confidence  = float(probs[0, pred_class].item())
                prob_drowsy = float(probs[0, 0].item())


                self._last_pred   = pred_class
                self._last_conf   = confidence
                self._last_prob_d = prob_drowsy
                new_infer_made    = True


            self._infer_count += 1


            # VRAM cleanup periodik (cegah fragmentasi di sesi panjang)
            if USE_CUDA and self._infer_count % VRAM_CLEANUP_INTERVAL == 0:
                torch.cuda.empty_cache()


        pred_class  = self._last_pred
        confidence  = self._last_conf
        prob_drowsy = self._last_prob_d
        buf_fill    = len(self.feature_buf)


        # ── State & Alert ─────────────────────────────────────
        if buf_fill < SEQ_LEN:
            run_state    = "MULAI"
            status_level = "NORMAL"
            pred_label   = "Buffering..."
            ui_prob_drowsy = prob_drowsy
        else:
            run_state    = "PROSES"


            # [EAR] EAR confidence gate diterapkan SEBELUM rolling history
            # Hanya pengaruhi history saat ada inferensi baru
            if new_infer_made:
                effective_pred, effective_prob = self._apply_ear_gate(
                    pred_class, prob_drowsy, min_ear
                )
            else:
                effective_pred = pred_class
                effective_prob = prob_drowsy


            status_level = self._eval_alert(
                effective_pred, effective_prob,
                new_inference=new_infer_made,
            )
            pred_label   = CLASS_NAMES.get(pred_class, "-")
            ui_prob_drowsy = (
                effective_prob if new_infer_made and effective_prob != prob_drowsy
                else prob_drowsy
            )


        # [PENTING] Gambar mesh jaring (Tesselation) diletakkan di SINI,
        # SETELAH sistem AI selesai memotong mata dan memprosesnya,
        # agar tidak merusak tekstur gambar mata yang dianalisis Swin.
        if draw_mesh:
            self.mp_drawing.draw_landmarks(
                image                   = frame_bgr,
                landmark_list           = face_lm,
                connections             = self.mp_face_mesh_cls.FACEMESH_TESSELATION,
                landmark_drawing_spec   = None,
                connection_drawing_spec =
                    self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )


        # ── Overlay text di frame ─────────────────────────────
        # NOTE: format warna di sini dalam BGR (sesuai standar OpenCV) —
        # tidak diubah agar tampilan video tetap konsisten dengan versi lama.
        _color_map = {
            "NORMAL":  (34, 197, 94),
            "WARNING": (251, 191, 36),
            "DANGER":  (239, 68, 68),
        }
        overlay_color = _color_map.get(status_level, (255, 255, 255))


        _draw_text_bg(frame_bgr, f"Status : {run_state}",     (10, 28),  bg=(15, 30, 60))
        _draw_text_bg(frame_bgr, f"Alert  : {status_level}",  (10, 58),  bg=(15, 30, 60),
                      fg=overlay_color)
        _draw_text_bg(frame_bgr, f"EAR    : {min_ear:.3f}",   (10, 88),  bg=(15, 30, 60))
        _draw_text_bg(frame_bgr, f"Buffer : {buf_fill}/30",   (10, 118), bg=(15, 30, 60))


        # Selalu tampilkan probabilitas drowsy (lebih informatif)
        if pred_class is not None:
            _bg = (22, 101, 52) if pred_class == 1 else (127, 29, 29)
            _draw_text_bg(
                frame_bgr,
                f"Drowsy : {ui_prob_drowsy*100:.1f}%  (pred: {pred_label})",
                (10, 148), bg=_bg,
            )


        elapsed = (time.perf_counter() - t_start) * 1000.0
        fps     = 1000.0 / max(elapsed, 1e-6)


        # Override run_state jika face terlalu dekat
        if too_close:
            run_state = "TOO_CLOSE"


        result = DetectionResult(
            face_count     = 1,
            pred_class     = pred_class,
            pred_label     = pred_label,
            confidence     = confidence,
            status_level   = status_level,
            run_state      = run_state,
            min_ear        = min_ear,
            prob_drowsy    = ui_prob_drowsy,
            processing_fps = fps,
            buffer_fill    = buf_fill,
            infer_ms       = elapsed,
        )
        return frame_bgr, result
