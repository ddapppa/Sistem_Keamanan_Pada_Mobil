# ============================================================
# training_pipeline.py — Backend Pelatihan Manual
# Drowsiness Detection System (Swin Transformer + LSTM)
# ============================================================
# DESKRIPSI:
# Module ini berisi semua logika backend untuk fitur
# "Pelatihan Manual" di Streamlit UI. Mencakup:
# - Validasi & loading dataset (video / frame)
# - Preprocessing dataset otomatis (resize, convert, dedup, corrupt check)
# - Ekstraksi frame dari video
# - MediaPipe eye cropping (reuse dari inference.py)
# - Preview Eye Crop dari dataset
# - Swin feature extraction (batch, pretrained fix)
# - Sequence building untuk LSTM
# - Validasi hyperparameter
# - Training loop LSTM
# - Evaluasi model (F2-score, confusion matrix)
# - Export model (.pth, .npz, .json, .zip)
#
# CATATAN:
# Swin Transformer TIDAK di-training ulang.
# Hanya LSTM yang di-training oleh pengguna.
#
# FIX v2:
# - BUGFIX: Data leakage — split kini dilakukan di level CLIP,
#   bukan di level sequence. Sequence dibangun SETELAH split clip.
# - BARU: preprocess_dataset_zip() — pipeline preprocessing otomatis
#   (resize, convert ke JPG, hapus corrupt, hapus duplikat via hash)
# - BARU: preview_eye_crops() — preview 1–5 crop mata dari dataset
# ============================================================

import os
import gc
import io
import json
import time
import math
import random
import hashlib
import zipfile
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Callable, Any
from collections import Counter, OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset as TorchDataset, DataLoader
import torchvision.transforms as T

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PLT_AVAILABLE = True
except ImportError:
    _PLT_AVAILABLE = False

from inference import (
    SwinFeatureExtractor,
    DrowsinessLSTM,
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    _crop_eye,
    MRL_IMG_MEAN,
    MRL_IMG_STD,
    DEVICE,
    USE_CUDA,
    USE_AMP,
    SWIN_MODEL_PATH,
)

# ============================================================
# KONSTANTA
# ============================================================
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# Folder label mapping — case-insensitive
# KONVENSI: drowsy=0 (kelas positif), notdrowsy=1
# Harus KONSISTEN dengan CLASS_NAMES di inference.py:
#   {0: "Drowsy", 1: "Not Drowsy"}
_LABEL_MAP = {
    "drowsy": 0,
    "mengantuk": 0,
    "closed": 0,
    "notdrowsy": 1,
    "not_drowsy": 1,
    "not drowsy": 1,
    "normal": 1,
    "open": 1,
    "tidakmengantuk": 1,
    "tidak_mengantuk": 1,
}

# ============================================================
# DATACLASSES — Konfigurasi & Hasil
# ============================================================
@dataclass
class DatasetInfo:
    """Informasi dataset setelah validasi."""
    path: str = ""
    mode: str = "video"  # "video" atau "frame"
    class_folders: Dict[str, int] = field(default_factory=dict)
    clips_per_class: Dict[int, int] = field(default_factory=dict)
    total_clips: int = 0
    sample_paths: Dict[int, List[str]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    is_valid: bool = False

@dataclass
class PreprocessConfig:
    """Konfigurasi preprocessing."""
    resize: int = 224
    interpolation: str = "LANCZOS"  # LANCZOS / BILINEAR / BICUBIC
    eye_crop_margin: float = 0.18
    norm_mode: str = "auto"  # "auto" / "mrl"
    is_pre_cropped: bool = False
    # Augmentasi
    aug_hflip: bool = True
    aug_rotation: bool = False
    aug_rotation_deg: int = 15
    aug_brightness: bool = False
    aug_brightness_val: float = 0.2
    aug_blur: bool = False
    # Ekstraksi video
    extract_fps: int = 30  # FPS target saat ekstraksi frame

@dataclass
class TrainingConfig:
    """Konfigurasi hyperparameter LSTM."""
    hidden_dim: int = 256
    num_layers: int = 2
    bidirectional: bool = False
    use_attention: bool = True
    fc_activation: str = "gelu"
    lstm_dropout: float = 0.3
    fc_dropout: float = 0.4
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 50
    optimizer: str = "AdamW"  # Adam / AdamW / SGD
    scheduler: str = "OneCycleLR"  # OneCycleLR / StepLR / None / CosineAnnealing (legacy)
    seq_len: int = 30
    stride: int = 15 # jarak geser sliding window; overlap = seq_len - stride
    split_train: float = 0.70
    split_val: float = 0.10
    split_test: float = 0.20
    early_stop_patience: int = 10
    weight_decay: float = 1e-4
    
@dataclass
class HPValidationMsg:
    """Pesan validasi hyperparameter."""
    level: str = "tip"  # "error" / "warning" / "tip"
    message: str = ""

@dataclass
class TrainingHistory:
    """Riwayat per-epoch selama training."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    val_f2: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_f2: float = 0.0

@dataclass
class EvalMetrics:
    """Hasil evaluasi model."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    f2: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    confusion_matrix: List[List[int]] = field(default_factory=lambda: [[0, 0], [0, 0]])


# ============================================================
# SECTION 0: PREPROCESSING DATASET OTOMATIS (BARU)
# ============================================================

def _image_hash(img_bgr: np.ndarray) -> str:
    """Hitung perceptual hash sederhana dari gambar untuk deteksi duplikat."""
    small = cv2.resize(img_bgr, (16, 16), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return hashlib.md5(gray.tobytes()).hexdigest()


def preprocess_dataset_zip(
    zip_bytes: bytes,
    output_dir: str,
    target_size: int = 224,
    target_quality: int = 90,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Preprocessing pipeline otomatis untuk dataset yang di-upload sebagai ZIP.

    Pipeline:
    1. Ekstrak ZIP ke folder sementara
    2. Validasi struktur folder (Drowsy/ & NotDrowsy/)
    3. Per gambar:
       a. Cek corrupt (cv2.imread gagal → buang)
       b. Resize ke target_size x target_size dengan aspect-ratio preserved (letterbox)
       c. Convert semua ke JPG (hemat ukuran)
       d. Cek duplikat via MD5 hash → buang duplikat
    4. Simpan hasilnya ke output_dir dengan struktur yang sama

    Returns:
        {
            "output_dir": str,
            "total_input": int,
            "total_output": int,
            "removed_corrupt": int,
            "removed_duplicate": int,
            "class_counts": {class_name: int},
            "errors": [str],
        }
    """
    import tempfile
    import shutil

    errors = []
    stats = {
        "output_dir": output_dir,
        "total_input": 0,
        "total_output": 0,
        "removed_corrupt": 0,
        "removed_duplicate": 0,
        "class_counts": {},
        "errors": errors,
    }

    # Step 1: Ekstrak ZIP ke tempdir
    tmp_extract = tempfile.mkdtemp(prefix="ds_raw_")
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(tmp_extract)
    except Exception as e:
        errors.append(f"Gagal ekstrak ZIP: {e}")
        shutil.rmtree(tmp_extract, ignore_errors=True)
        return stats

    # Step 2: Cari root folder dataset (skip folder wrapper jika ada satu layer)
    root = tmp_extract
    sub_items = [d for d in os.listdir(root) if not d.startswith(".")]
    if len(sub_items) == 1 and os.path.isdir(os.path.join(root, sub_items[0])):
        root = os.path.join(root, sub_items[0])

    # Validasi struktur
    class_dirs = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    ]
    if len(class_dirs) < 2:
        errors.append(
            "ZIP harus berisi minimal 2 folder kelas (Drowsy/ dan NotDrowsy/)."
        )
        shutil.rmtree(tmp_extract, ignore_errors=True)
        return stats

    os.makedirs(output_dir, exist_ok=True)

    # Hitung total gambar untuk progress
    all_files = []
    for cls_dir in class_dirs:
        cls_path = os.path.join(root, cls_dir)
        imgs = [
            f for f in os.listdir(cls_path)
            if os.path.isfile(os.path.join(cls_path, f))
            and os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTS
        ]
        for f in imgs:
            all_files.append((cls_dir, os.path.join(cls_path, f)))

    stats["total_input"] = len(all_files)
    seen_hashes = set()

    for idx, (cls_dir, src_path) in enumerate(all_files):
        if progress_cb and idx % 20 == 0:
            progress_cb(
                f"Preprocessing: {cls_dir}/{os.path.basename(src_path)}",
                idx, len(all_files)
            )

        # a. Cek corrupt
        img = cv2.imread(src_path)
        if img is None or img.size == 0:
            stats["removed_corrupt"] += 1
            continue

        # b. Resize dengan letterbox (tidak crop, tidak distorsi)
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Pad ke target_size x target_size (hitam)
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        pad_top  = (target_size - new_h) // 2
        pad_left = (target_size - new_w) // 2
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        # c. Cek duplikat
        img_hash = _image_hash(canvas)
        if img_hash in seen_hashes:
            stats["removed_duplicate"] += 1
            continue
        seen_hashes.add(img_hash)

        # d. Simpan sebagai JPG
        out_cls_dir = os.path.join(output_dir, cls_dir)
        os.makedirs(out_cls_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(src_path))[0]
        out_path  = os.path.join(out_cls_dir, f"{base_name}.jpg")
        cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, target_quality])

        stats["total_output"] += 1
        stats["class_counts"][cls_dir] = stats["class_counts"].get(cls_dir, 0) + 1

    shutil.rmtree(tmp_extract, ignore_errors=True)
    return stats


# ============================================================
# SECTION 1: DATASET LOADING & VALIDASI
# ============================================================
def validate_dataset_folder(path: str) -> DatasetInfo:
    """
    Validasi folder dataset.
    Struktur yang diterima:
    path/
      Drowsy/    → video clips atau sub-folder frame
      NotDrowsy/ → video clips atau sub-folder frame
    """
    info = DatasetInfo(path=path)

    if not path or not os.path.isdir(path):
        info.errors.append(f"Folder tidak ditemukan: {path}")
        return info

    sub_dirs = [
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and not d.startswith(".")
    ]

    if len(sub_dirs) < 2:
        info.errors.append(
            "Dataset harus memiliki minimal 2 sub-folder kelas "
            "(contoh: Drowsy/ dan NotDrowsy/)"
        )
        return info

    for folder_name in sub_dirs:
        key = folder_name.lower().replace(" ", "").replace("-", "").replace("_", "")
        label = _LABEL_MAP.get(key)
        if label is None:
            for known, lbl in _LABEL_MAP.items():
                if known in key or key in known:
                    label = lbl
                    break
        if label is None:
            info.errors.append(
                f"Folder '{folder_name}' tidak dikenali. "
                f"Gunakan nama: Drowsy, NotDrowsy, Open, Closed, dst."
            )
            continue
        info.class_folders[folder_name] = label

    if len(info.class_folders) < 2:
        info.errors.append("Minimal 2 kelas terdeteksi.")
        return info

    for folder_name, label in info.class_folders.items():
        folder_path = os.path.join(path, folder_name)
        contents    = os.listdir(folder_path)

        videos = [
            f for f in contents
            if os.path.isfile(os.path.join(folder_path, f))
            and os.path.splitext(f)[1].lower() in SUPPORTED_VIDEO_EXTS
        ]
        clip_dirs = [
            d for d in contents
            if os.path.isdir(os.path.join(folder_path, d))
        ]

        if videos:
            info.mode   = "video"
            clip_count  = len(videos)
            samples     = [os.path.join(folder_path, v) for v in videos[:5]]
        elif clip_dirs:
            info.mode  = "frame"
            clip_count = len(clip_dirs)
            samples    = []
            for cd in clip_dirs[:5]:
                cd_path = os.path.join(folder_path, cd)
                frames  = sorted([
                    f for f in os.listdir(cd_path)
                    if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTS
                ])
                if frames:
                    samples.append(os.path.join(cd_path, frames[0]))
        else:
            images = [
                f for f in contents
                if os.path.isfile(os.path.join(folder_path, f))
                and os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTS
            ]
            if images:
                info.mode  = "frame_flat"
                clip_count = len(images)
                samples    = [os.path.join(folder_path, img) for img in images[:5]]
            else:
                info.errors.append(
                    f"Folder '{folder_name}' kosong — tidak ada video atau gambar."
                )
                continue

        info.clips_per_class[label] = info.clips_per_class.get(label, 0) + clip_count
        info.total_clips            += clip_count
        info.sample_paths[label]    = info.sample_paths.get(label, []) + samples

    if info.mode == "frame_flat":
        for label, count in info.clips_per_class.items():
            label_name = "Drowsy" if label == 0 else "NotDrowsy"
            if count < 30:
                info.errors.append(
                    f"Folder {label_name} hanya memiliki {count} gambar. "
                    f"Minimal 30 gambar per kelas diperlukan untuk membangun 1 sequence."
                )
    elif info.total_clips < 4:
        info.errors.append(
            f"Dataset terlalu kecil ({info.total_clips} clips terdeteksi). "
            f"Minimal 4 clips diperlukan (2 per kelas). "
            f"Pastikan struktur folder benar: setiap subfolder kelas berisi "
            f"minimal 2 file video atau 2 subfolder frame."
        )

    if not info.errors:
        info.is_valid = True

    return info


# ============================================================
# SECTION 2: FRAME EXTRACTION DARI VIDEO
# ============================================================
def extract_video_frames(
    video_path: str,
    target_fps: int = 30,
    max_frames: int = 0,
) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step    = max(1, round(src_fps / target_fps))
    frames  = []
    idx     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame)
            if 0 < max_frames <= len(frames):
                break
        idx += 1

    cap.release()
    return frames


def load_clip_frames(
    clip_path: str,
    mode: str,
    target_fps: int = 30,
) -> List[np.ndarray]:
    if mode == "video":
        return extract_video_frames(clip_path, target_fps)
    else:
        if os.path.isdir(clip_path):
            img_files = sorted([
                f for f in os.listdir(clip_path)
                if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTS
            ])
            frames = []
            for f in img_files:
                img = cv2.imread(os.path.join(clip_path, f))
                if img is not None:
                    frames.append(img)
            return frames
        elif os.path.isfile(clip_path):
            img = cv2.imread(clip_path)
            return [img] if img is not None else []
        return []


# ============================================================
# SECTION 3: MEDIAPIPE EYE CROPPING
# ============================================================
def process_frames_to_eye_crops(
    frames: List[np.ndarray],
    margin: float = 0.18,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> List[Optional[Any]]:
    """
    Proses frame melalui MediaPipe → crop mata (rata-rata L+R via Swin).
    Mengembalikan list (eye_left_bgr, eye_right_bgr) atau None jika gagal.
    """
    if not _MP_AVAILABLE:
        raise ImportError("MediaPipe tidak tersedia.")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    results = []
    for i, frame in enumerate(frames):
        if progress_cb and i % 10 == 0:
            progress_cb(i, len(frames))

        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_res = face_mesh.process(rgb)

        if not mp_res.multi_face_landmarks:
            results.append(None)
            continue

        pts = [
            (int(lm.x * w), int(lm.y * h))
            for lm in mp_res.multi_face_landmarks[0].landmark
        ]

        crop_l = _crop_eye(frame, pts, LEFT_EYE_IDX, margin)
        crop_r = _crop_eye(frame, pts, RIGHT_EYE_IDX, margin)

        if crop_l.size == 0 or crop_r.size == 0:
            results.append(None)
            continue

        results.append((crop_l, crop_r))

    face_mesh.close()
    return results


def preview_eye_crops(
    dataset_info: DatasetInfo,
    margin: float = 0.18,
    max_previews: int = 5,
) -> List[Dict[str, Any]]:
    """
    Ambil beberapa contoh crop mata dari dataset untuk preview di UI.

    Returns:
        List of dict: [
            {
                "label":     "Drowsy" | "NotDrowsy",
                "label_int": 0 | 1,
                "left_bgr":  np.ndarray,  # crop mata kiri BGR
                "right_bgr": np.ndarray,  # crop mata kanan BGR
                "source":    str,         # nama file sumber
            },
            ...
        ]
    """
    if not _MP_AVAILABLE:
        return []

    results = []

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    for folder_name, label in dataset_info.class_folders.items():
        folder_path = os.path.join(dataset_info.path, folder_name)
        label_name  = folder_name

        candidate_frames = []

        if dataset_info.mode == "video":
            clips = sorted([
                f for f in os.listdir(folder_path)
                if os.path.splitext(f)[1].lower() in SUPPORTED_VIDEO_EXTS
            ])
            for clip in clips[:3]:
                frames = extract_video_frames(
                    os.path.join(folder_path, clip), target_fps=5, max_frames=10
                )
                for fr in frames:
                    candidate_frames.append((fr, clip))
                if len(candidate_frames) >= max_previews * 3:
                    break

        elif dataset_info.mode == "frame":
            clip_dirs = sorted([
                d for d in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, d))
            ])
            for cd in clip_dirs[:3]:
                cd_path   = os.path.join(folder_path, cd)
                img_files = sorted([
                    f for f in os.listdir(cd_path)
                    if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTS
                ])
                for f in img_files[:5]:
                    img = cv2.imread(os.path.join(cd_path, f))
                    if img is not None:
                        candidate_frames.append((img, f"{cd}/{f}"))
                if len(candidate_frames) >= max_previews * 3:
                    break

        else:  # frame_flat
            img_files = sorted([
                f for f in os.listdir(folder_path)
                if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTS
            ])
            for f in img_files[:max_previews * 3]:
                img = cv2.imread(os.path.join(folder_path, f))
                if img is not None:
                    candidate_frames.append((img, f))

        found = 0
        for frame, source_name in candidate_frames:
            if found >= max_previews:
                break

            h, w   = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_res = face_mesh.process(rgb)

            if not mp_res.multi_face_landmarks:
                continue

            pts = [
                (int(lm.x * w), int(lm.y * h))
                for lm in mp_res.multi_face_landmarks[0].landmark
            ]

            crop_l = _crop_eye(frame, pts, LEFT_EYE_IDX, margin)
            crop_r = _crop_eye(frame, pts, RIGHT_EYE_IDX, margin)

            if crop_l.size == 0 or crop_r.size == 0:
                continue

            results.append({
                "label":     label_name,
                "label_int": label,
                "left_bgr":  crop_l.copy(),
                "right_bgr": crop_r.copy(),
                "source":    source_name,
            })
            found += 1

    face_mesh.close()
    return results


# ============================================================
# SECTION 4: SWIN FEATURE EXTRACTION
# ============================================================
def load_swin_model(model_path: str = None) -> nn.Module:
    """Load pretrained Swin model untuk feature extraction."""
    path = model_path or SWIN_MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Swin model tidak ditemukan: {path}")

    ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    head_out_dim = None
    for key in ["head.weight", "head.fc.weight", "model.head.weight"]:
        if key in state:
            head_out_dim = int(state[key].shape[0])
            break

    model = SwinFeatureExtractor(head_out_dim=head_out_dim).to(DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def extract_features_from_crops(
    swin_model: nn.Module,
    eye_crops: List[Optional[Tuple[np.ndarray, np.ndarray]]],
    preproc_cfg: PreprocessConfig,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> List[Optional[torch.Tensor]]:
    """
    Ekstraksi fitur Swin dari eye crops.
    Untuk setiap pasangan (left, right):
    1. Transform kedua mata (resize, normalize)
    2. Forward Swin batch [2, 3, 224, 224]
    3. Rata-rata fitur → [512]
    """
    interp_map = {
        "LANCZOS": T.InterpolationMode.LANCZOS,
        "BILINEAR": T.InterpolationMode.BILINEAR,
        "BICUBIC": T.InterpolationMode.BICUBIC,
    }
    interp = interp_map.get(preproc_cfg.interpolation, T.InterpolationMode.LANCZOS)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((preproc_cfg.resize, preproc_cfg.resize), interpolation=interp),
        T.ToTensor(),
        T.Normalize(mean=MRL_IMG_MEAN, std=MRL_IMG_STD),
    ])

    features = []

    with torch.no_grad():
        for i, crops in enumerate(eye_crops):
            if progress_cb and i % 10 == 0:
                progress_cb(i, len(eye_crops))

            if crops is None:
                features.append(None)
                continue

            crop_l, crop_r = crops
            try:
                t_l   = transform(cv2.cvtColor(crop_l, cv2.COLOR_BGR2RGB))
                t_r   = transform(cv2.cvtColor(crop_r, cv2.COLOR_BGR2RGB))
                batch = torch.stack([t_l, t_r], dim=0).to(DEVICE, non_blocking=True)

                if USE_AMP:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        feats = swin_model(batch)
                else:
                    feats = swin_model(batch)

                feat = feats.mean(dim=0).float().cpu()
                features.append(feat)
            except Exception:
                features.append(None)

    return features


def extract_features_direct(
    swin_model: nn.Module,
    frames: List[np.ndarray],
    preproc_cfg: PreprocessConfig,
    progress_cb=None,
) -> List[Optional[torch.Tensor]]:
    """
    Bypass MediaPipe — langsung forward gambar ke Swin.
    Dipakai untuk dataset yang sudah berupa crop mata
    (contoh: NTHU-DDD frame_flat, MRL single-eye images).
    """
    interp_map = {
        "LANCZOS": T.InterpolationMode.LANCZOS,
        "BILINEAR": T.InterpolationMode.BILINEAR,
        "BICUBIC": T.InterpolationMode.BICUBIC,
    }
    interp = interp_map.get(preproc_cfg.interpolation, T.InterpolationMode.LANCZOS)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((preproc_cfg.resize, preproc_cfg.resize), interpolation=interp),
        T.ToTensor(),
        T.Normalize(mean=MRL_IMG_MEAN, std=MRL_IMG_STD),    
    ])

    features = []
    with torch.no_grad():
        for i, frame in enumerate(frames):
            if progress_cb and i % 20 == 0:
                progress_cb(i, len(frames))
            try:
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                t    = transform(rgb).unsqueeze(0).to(DEVICE, non_blocking=True)

                if USE_AMP:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        feat = swin_model(t)
                else:
                    feat = swin_model(t)

                features.append(feat.squeeze(0).float().cpu())
            except Exception:
                features.append(None)

    return features


# ============================================================
# SECTION 5: FULL PIPELINE — Dataset → Features (per-clip)
# ============================================================
def process_full_dataset(
    dataset_info: DatasetInfo,
    preproc_cfg: PreprocessConfig,
    swin_model: nn.Module,
    progress_cb = None,
    seq_len: int = 30,    # ← parameter baru untuk hitung CHUNK_SIZE
) -> Dict[str, Any]:
    """
    Pipeline lengkap: dataset folder → fitur per clip.

    Returns:
        {
            "clip_features":  {clip_name: [feat_0, feat_1, ...]},
            "clip_labels":    {clip_name: label},
            "total_frames":   int,
            "valid_frames":   int,
            "skipped_frames": int,
        }
    """
    clip_features = OrderedDict()
    clip_labels   = {}
    total_frames  = 0
    valid_frames  = 0
    clip_idx      = 0

    # [FIX-3] CHUNK_SIZE ADAPTIF — berlaku untuk semua ukuran dataset
    if dataset_info.mode == "frame_flat" and dataset_info.clips_per_class:
        min_imgs  = min(dataset_info.clips_per_class.values())
        computed  = max(1, min_imgs // 15)        # target 15 clips/kelas
        CHUNK_SIZE = max(seq_len, min(computed, 200))
        # Warning jika data terlalu kecil
        for lbl, cnt in dataset_info.clips_per_class.items():
            est_clips = max(1, cnt // CHUNK_SIZE)
            if est_clips < 5:
                lbl_name = "Drowsy" if lbl == 0 else "NotDrowsy"
                print(f"[WARNING] Kelas {lbl_name} hanya ~{est_clips} clip virtual.")
    else:
        CHUNK_SIZE = 250   # video/frame mode tidak terpengaruh
    if dataset_info.mode == "frame_flat":
        total_chunk_count = sum(
            math.ceil(len([
                f for f in os.listdir(os.path.join(dataset_info.path, fn))
                if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTS
            ]) / CHUNK_SIZE)
            for fn in dataset_info.class_folders.keys()
        )
    else:
        total_chunk_count = dataset_info.total_clips

    for folder_name, label in dataset_info.class_folders.items():
        folder_path = os.path.join(dataset_info.path, folder_name)

        if dataset_info.mode == "video":
            clips = sorted([
                f for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
                and os.path.splitext(f)[1].lower() in SUPPORTED_VIDEO_EXTS
            ])
            clip_paths = [os.path.join(folder_path, c) for c in clips]

        elif dataset_info.mode == "frame":
            clips = sorted([
                d for d in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, d))
            ])
            clip_paths = [os.path.join(folder_path, c) for c in clips]

        else:  # frame_flat
            all_images = sorted([
                f for f in os.listdir(folder_path)
                if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTS
            ])
            clips              = []
            clip_paths_chunked = []
            for chunk_i in range(0, len(all_images), CHUNK_SIZE):
                chunk          = all_images[chunk_i:chunk_i + CHUNK_SIZE]
                clip_name_chunk = f"{folder_name}_chunk{chunk_i // CHUNK_SIZE:03d}"
                clips.append(clip_name_chunk)
                clip_paths_chunked.append((folder_path, chunk))
            clip_paths = clip_paths_chunked

        for ci, (clip_name, clip_path) in enumerate(zip(clips, clip_paths)):
            clip_idx += 1

            if progress_cb:
                progress_cb(
                    f"Memproses clip {clip_idx}/{total_chunk_count}: {clip_name}",
                    clip_idx, total_chunk_count,
                )

            if dataset_info.mode == "video":
                frames = extract_video_frames(clip_path, target_fps=preproc_cfg.extract_fps)
            elif dataset_info.mode == "frame":
                frames = load_clip_frames(clip_path, "frame")
            else:
                folder_path_chunk, img_filenames = clip_path
                frames = []
                for fname in img_filenames:
                    img = cv2.imread(os.path.join(folder_path_chunk, fname))
                    if img is not None:
                        frames.append(img)

            if not frames:
                continue

            total_frames += len(frames)

            if preproc_cfg.is_pre_cropped:
                feats = extract_features_direct(swin_model, frames, preproc_cfg)
            else:
                eye_crops = process_frames_to_eye_crops(
                    frames, margin=preproc_cfg.eye_crop_margin
                )
                feats = extract_features_from_crops(swin_model, eye_crops, preproc_cfg)

            valid_feats  = [f for f in feats if f is not None]
            valid_frames += len(valid_feats)

            if valid_feats:
                unique_name               = f"{folder_name}/{clip_name}"
                clip_features[unique_name] = valid_feats
                clip_labels[unique_name]   = label

    return {
        "clip_features":  clip_features,
        "clip_labels":    clip_labels,
        "total_frames":   total_frames,
        "valid_frames":   valid_frames,
        "skipped_frames": total_frames - valid_frames,
    }


# ============================================================
# SECTION 6: SPLIT CLIP → SEQUENCE BUILDING
# ============================================================
# FIX: Split kini dilakukan di level CLIP agar tidak ada data leakage.
# Sequence dibangun SETELAH split, sehingga frame dari clip yang sama
# tidak mungkin tersebar di train dan test sekaligus.

def split_clips(
    clip_features: Dict[str, List[torch.Tensor]],
    clip_labels: Dict[str, int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Split pada level CLIP (bukan sequence) untuk mencegah data leakage.
    Stratified berdasarkan label.

    Returns:
        {
            "train": {"clip_features": {...}, "clip_labels": {...}},
            "val":   {"clip_features": {...}, "clip_labels": {...}},
            "test":  {"clip_features": {...}, "clip_labels": {...}},
        }
    """
    random.seed(seed)

    by_label: Dict[int, List[str]] = {}
    for clip_name, label in clip_labels.items():
        by_label.setdefault(label, []).append(clip_name)

    splits = {
        "train": {"clip_features": {}, "clip_labels": {}},
        "val":   {"clip_features": {}, "clip_labels": {}},
        "test":  {"clip_features": {}, "clip_labels": {}},
    }

    for lbl, clip_names in by_label.items():
        random.shuffle(clip_names)
        n       = len(clip_names)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))

        train_clips = clip_names[:n_train]
        val_clips   = clip_names[n_train:n_train + n_val]
        test_clips  = clip_names[n_train + n_val:]

        if not test_clips and len(val_clips) > 1:
            test_clips = [val_clips.pop()]

        for split_name, clip_list in [
            ("train", train_clips),
            ("val",   val_clips),
            ("test",  test_clips),
        ]:
            for cn in clip_list:
                splits[split_name]["clip_features"][cn] = clip_features[cn]
                splits[split_name]["clip_labels"][cn]   = clip_labels[cn]

    return splits


def build_sequences(
    clip_features: Dict[str, List[torch.Tensor]],
    clip_labels: Dict[str, int],
    seq_len: int = 30,
    stride: int = 15,
) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Bangun sequences sliding-window dari fitur per-clip.
    """
    sequences = []
    labels    = []

    for clip_name, feats in clip_features.items():
        label = clip_labels[clip_name]

        if len(feats) < seq_len:
            if len(feats) >= seq_len // 2:
                padded = feats.copy()
                while len(padded) < seq_len:
                    padded.append(padded[-1])
                seq = torch.stack(padded[:seq_len], dim=0)
                sequences.append(seq)
                labels.append(label)
            continue

        for start in range(0, len(feats) - seq_len + 1, stride):
            seq = torch.stack(feats[start:start + seq_len], dim=0)
            sequences.append(seq)
            labels.append(label)

    return sequences, labels


def build_splits_with_sequences(
    clip_features: Dict[str, List[torch.Tensor]],
    clip_labels: Dict[str, int],
    seq_len: int = 30,
    stride: int = 15,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, Tuple[List[torch.Tensor], List[int]]]:
    """
    Fungsi utama: split clips dulu, lalu bangun sequences per split.
    MENGGANTIKAN pola lama (build_sequences → split_sequences) untuk
    mencegah data leakage.

    Returns:
        {"train": (seqs, labels), "val": (seqs, labels), "test": (seqs, labels)}
    """
    clip_splits = split_clips(
        clip_features, clip_labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    result = {}
    for split_name, split_data in clip_splits.items():
        seqs, lbls = build_sequences(
            split_data["clip_features"],
            split_data["clip_labels"],
            seq_len=seq_len,
            stride=stride,
        )
        result[split_name] = (seqs, lbls)

    return result


def compute_norm_stats(
    sequences: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hitung mean dan std dari sequences TRAINING ONLY untuk Z-score normalisasi.
    """
    all_feats = torch.cat([s.view(-1, s.shape[-1]) for s in sequences], dim=0)
    mean = all_feats.mean(dim=0)
    std  = all_feats.std(dim=0).clamp(min=1e-8)
    return mean, std


def split_sequences(
    sequences: List[torch.Tensor],
    labels: List[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, Tuple[List[torch.Tensor], List[int]]]:
    """
    [DEPRECATED — gunakan build_splits_with_sequences() untuk menghindari
    data leakage. Fungsi ini dipertahankan untuk backward-compatibility.]
    """
    random.seed(seed)
    by_label = {}
    for seq, lbl in zip(sequences, labels):
        by_label.setdefault(lbl, []).append(seq)

    splits = {"train": ([], []), "val": ([], []), "test": ([], [])}

    for lbl, seqs in by_label.items():
        random.shuffle(seqs)
        n       = len(seqs)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))

        train_seqs = seqs[:n_train]
        val_seqs   = seqs[n_train:n_train + n_val]
        test_seqs  = seqs[n_train + n_val:]

        if not test_seqs and len(val_seqs) > 1:
            test_seqs = [val_seqs.pop()]

        splits["train"][0].extend(train_seqs)
        splits["train"][1].extend([lbl] * len(train_seqs))
        splits["val"][0].extend(val_seqs)
        splits["val"][1].extend([lbl] * len(val_seqs))
        splits["test"][0].extend(test_seqs)
        splits["test"][1].extend([lbl] * len(test_seqs))

    return splits


# ============================================================
# SECTION 7: PYTORCH DATASET
# ============================================================
class SequenceDataset(TorchDataset):
    """Dataset PyTorch untuk sequences + labels."""

    def __init__(
        self,
        sequences: List[torch.Tensor],
        labels: List[int],
        norm_mean: Optional[torch.Tensor] = None,
        norm_std: Optional[torch.Tensor] = None,
        augment: bool = False,
    ):
        self.sequences = sequences
        self.labels    = labels
        self.norm_mean = norm_mean
        self.norm_std  = norm_std
        self.augment   = augment

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].clone()

        if self.norm_mean is not None and self.norm_std is not None:
            seq = (seq - self.norm_mean) / self.norm_std

        if self.augment and random.random() < 0.3:
            noise = torch.randn_like(seq) * 0.01
            seq   = seq + noise

        label = self.labels[idx]
        return seq, torch.tensor(label, dtype=torch.long)


# ============================================================
# SECTION 8: HYPERPARAMETER VALIDATION
# ============================================================
def validate_hyperparameters(
    config: TrainingConfig,
    dataset_size: int = 100,
) -> List[HPValidationMsg]:
    msgs = []

    if config.learning_rate > 0.1:
        msgs.append(HPValidationMsg("error", "Learning rate > 0.1 terlalu besar, training akan diverge."))
    if config.learning_rate <= 0:
        msgs.append(HPValidationMsg("error", "Learning rate harus positif."))
    if config.epochs < 1:
        msgs.append(HPValidationMsg("error", "Jumlah epoch harus minimal 1."))
    if config.batch_size < 1:
        msgs.append(HPValidationMsg("error", "Batch size harus minimal 1."))
    if config.seq_len < 5:
        msgs.append(HPValidationMsg("error", "Sequence length terlalu pendek (minimal 5)."))
    if config.split_train + config.split_val + config.split_test > 1.01:
        msgs.append(HPValidationMsg("error", "Total split ratio melebihi 1.0."))

    if config.lstm_dropout > 0.5:
        msgs.append(HPValidationMsg("warning", f"LSTM Dropout={config.lstm_dropout:.2f} terlalu tinggi, model mungkin underfitting."))
    if config.fc_dropout > 0.5:
        msgs.append(HPValidationMsg("warning", f"FC Dropout={config.fc_dropout:.2f} terlalu tinggi."))
    if config.epochs > 100:
        msgs.append(HPValidationMsg("warning", f"Epochs={config.epochs} — training akan memakan waktu lama."))
    if config.batch_size > dataset_size // 2:
        msgs.append(HPValidationMsg("warning", f"Batch size ({config.batch_size}) > setengah dataset ({dataset_size}). Mungkin gradient tidak stabil."))
    if config.hidden_dim >= 512 and config.bidirectional and config.num_layers >= 3:
        msgs.append(HPValidationMsg("warning", "Kombinasi hidden_dim ≥ 512 + bidirectional + 3 layers membutuhkan VRAM besar."))

    if dataset_size < 500 and config.hidden_dim > 256:
        msgs.append(HPValidationMsg("tip", f"Dataset kecil ({dataset_size} sample). Coba hidden_dim=128 atau 256 untuk menghindari overfitting."))
    if config.bidirectional:
        msgs.append(HPValidationMsg("tip", "Bidirectional LSTM ~2x lebih lambat tapi biasanya lebih akurat."))
    if not config.use_attention:
        msgs.append(HPValidationMsg("tip", "Attention biasanya meningkatkan performa. Pertimbangkan untuk mengaktifkannya."))

    return msgs


# ============================================================
# SECTION 9: TRAINING LOOP
# ============================================================
def _compute_f2(y_true, y_pred) -> float:
    """Hitung F2-score secara manual (tanpa sklearn).
    Kelas positif = 0 (Drowsy), sesuai _LABEL_MAP dan CLASS_NAMES.
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    beta  = 2.0
    denom = (beta**2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / denom


def train_lstm_model(
    config: TrainingConfig,
    splits: Dict[str, Tuple[List[torch.Tensor], List[int]]],
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    progress_cb: Optional[Callable[[int, int, Dict], None]] = None,
) -> Tuple[nn.Module, TrainingHistory, Dict]:
    train_ds = SequenceDataset(
        splits["train"][0], splits["train"][1],
        norm_mean, norm_std, augment=True,
    )
    val_ds = SequenceDataset(
        splits["val"][0], splits["val"][1],
        norm_mean, norm_std, augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        shuffle=True, drop_last=False, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False, drop_last=False, num_workers=0,
    )

    model = DrowsinessLSTM(
        input_dim=512,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_classes=2,
        bidirectional=config.bidirectional,
        use_attention=config.use_attention,
        lstm_dropout=config.lstm_dropout,
        fc_dropout=config.fc_dropout,
        fc_activation=config.fc_activation,
    ).to(DEVICE)

    train_labels  = splits["train"][1]
    label_counts  = Counter(train_labels)
    total         = len(train_labels)
    if len(label_counts) == 2 and all(c > 0 for c in label_counts.values()):
        weight_0      = total / (2.0 * label_counts[0])
        weight_1      = total / (2.0 * label_counts[1])
        class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(DEVICE)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if config.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    if config.scheduler == "OneCycleLR":
        total_steps = max(1, config.epochs * max(len(train_loader), 1))
        scheduler   = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.learning_rate, total_steps=total_steps,
            pct_start=0.3, anneal_strategy="cos", cycle_momentum=True,
            base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=1e4,
        )
    elif config.scheduler == "CosineAnnealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    elif config.scheduler == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, config.epochs // 3), gamma=0.1)
    else:
        scheduler = None

    history          = TrainingHistory()
    best_state       = None
    best_val_f2      = -1.0
    patience_counter = 0

    for epoch in range(config.epochs):
        # ── Train ──
        model.train()
        running_loss  = 0.0
        correct       = 0
        total_samples = 0

        for batch_seqs, batch_labels in train_loader:
            batch_seqs   = batch_seqs.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            optimizer.zero_grad()

            if USE_AMP:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits, _ = model(batch_seqs)
                    loss      = criterion(logits.float(), batch_labels)
            else:
                logits, _ = model(batch_seqs)
                loss      = criterion(logits, batch_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler and config.scheduler == "OneCycleLR":
                scheduler.step()

            running_loss  += loss.item() * batch_seqs.size(0)
            preds          = torch.argmax(logits, dim=1)
            correct       += (preds == batch_labels).sum().item()
            total_samples += batch_seqs.size(0)

        train_loss = running_loss / max(total_samples, 1)
        train_acc  = correct     / max(total_samples, 1)

        # ── Validate ──
        model.eval()
        val_loss_sum = 0.0
        val_correct  = 0
        val_total    = 0
        all_true     = []
        all_pred     = []

        with torch.no_grad():
            for batch_seqs, batch_labels in val_loader:
                batch_seqs   = batch_seqs.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)

                if USE_AMP:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        logits, _ = model(batch_seqs)
                        loss      = criterion(logits.float(), batch_labels)
                else:
                    logits, _ = model(batch_seqs)
                    loss      = criterion(logits, batch_labels)

                val_loss_sum += loss.item() * batch_seqs.size(0)
                preds         = torch.argmax(logits, dim=1)
                val_correct  += (preds == batch_labels).sum().item()
                val_total    += batch_seqs.size(0)
                all_true.extend(batch_labels.cpu().tolist())
                all_pred.extend(preds.cpu().tolist())

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc  = val_correct  / max(val_total, 1)
        val_f2   = _compute_f2(all_true, all_pred)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_acc.append(train_acc)
        history.val_acc.append(val_acc)
        history.val_f2.append(val_f2)

        if val_f2 > best_val_f2:
            best_val_f2         = val_f2
            history.best_epoch  = epoch
            history.best_val_f2 = val_f2
            best_state          = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter    = 0
        else:
            patience_counter += 1

        if scheduler and config.scheduler != "OneCycleLR":
            scheduler.step()

        if progress_cb:
            progress_cb(epoch, config.epochs, {
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "train_acc":  train_acc,
                "val_acc":    val_acc,
                "val_f2":     val_f2,
                "best_f2":    best_val_f2,
                "best_epoch": history.best_epoch,
                "lr":         optimizer.param_groups[0]["lr"],
            })

        if patience_counter >= config.early_stop_patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    if USE_CUDA:
        torch.cuda.empty_cache()

    return model, history, best_state


# ============================================================
# SECTION 10: EVALUASI MODEL
# ============================================================
def evaluate_model(
    model: nn.Module,
    sequences: List[torch.Tensor],
    labels: List[int],
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    batch_size: int = 32,
) -> EvalMetrics:
    """Evaluasi model pada test set."""
    ds     = SequenceDataset(sequences, labels, norm_mean, norm_std, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch_seqs, batch_labels in loader:
            batch_seqs = batch_seqs.to(DEVICE)
            if USE_AMP:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits, _ = model(batch_seqs)
            else:
                logits, _ = model(batch_seqs)
            preds = torch.argmax(logits, dim=1)
            all_true.extend(batch_labels.tolist())
            all_pred.extend(preds.cpu().tolist())

    # Kelas positif = 0 (Drowsy), sesuai _LABEL_MAP dan CLASS_NAMES
    tp = sum(1 for t, p in zip(all_true, all_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(all_true, all_pred) if t == 1 and p == 0)
    fn = sum(1 for t, p in zip(all_true, all_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(all_true, all_pred) if t == 1 and p == 1)

    total     = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    f2        = _compute_f2(all_true, all_pred)
    accuracy  = (tp + tn) / total if total > 0 else 0.0

    # Confusion matrix layout standar sklearn:
    #               Predicted
    #               Drowsy(0)  NotDrowsy(1)
    # Actual Drowsy(0)   [ TP        FN ]
    # Actual NotDrowsy(1) [ FP        TN ]
    return EvalMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        f2=f2,
        tp=tp, fp=fp, fn=fn, tn=tn,
        confusion_matrix=[[tp, fn], [fp, tn]],
    )


def plot_confusion_matrix(metrics: EvalMetrics):
    """Buat matplotlib figure untuk confusion matrix."""
    if not _PLT_AVAILABLE:
        return None

    fig, ax = plt.subplots(figsize=(5, 4))
    cm = np.array(metrics.confusion_matrix)
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Drowsy (0)", "Not Drowsy (1)"], fontsize=10)
    ax.set_yticklabels(["Drowsy (0)", "Not Drowsy (1)"], fontsize=10)
    ax.set_xlabel("Prediksi", fontsize=11, fontweight="bold")
    ax.set_ylabel("Aktual",   fontsize=11, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=12)

    for i in range(2):
        for j in range(2):
            val   = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=16, fontweight="bold", color=color)

    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_training_curves(history: TrainingHistory):
    """Buat matplotlib figure untuk loss dan accuracy curves."""
    if not _PLT_AVAILABLE:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, len(history.train_loss) + 1)

    ax1.plot(epochs, history.train_loss, "o-", color="#0284C7",
             label="Train Loss", markersize=3, linewidth=1.5)
    ax1.plot(epochs, history.val_loss,   "s-", color="#EF4444",
             label="Val Loss",   markersize=3, linewidth=1.5)
    ax1.axvline(history.best_epoch + 1, color="#10B981",
                linestyle="--", alpha=0.6, label=f"Best Epoch ({history.best_epoch+1})")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss",  fontsize=10)
    ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history.train_acc, "o-", color="#0284C7",
             label="Train Acc", markersize=3, linewidth=1.5)
    ax2.plot(epochs, history.val_acc,   "s-", color="#F59E0B",
             label="Val Acc",   markersize=3, linewidth=1.5)
    ax2.plot(epochs, history.val_f2,    "D-", color="#10B981",
             label="Val F2",    markersize=3, linewidth=1.5)
    ax2.axvline(history.best_epoch + 1, color="#10B981",
                linestyle="--", alpha=0.6, label=f"Best Epoch ({history.best_epoch+1})")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Score", fontsize=10)
    ax2.set_title("Accuracy & F2-Score", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ============================================================
# SECTION 11: EXPORT MODEL
# ============================================================
def export_model(
    model_state: Dict,
    config: TrainingConfig,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    eval_metrics: EvalMetrics,
    history: TrainingHistory,
    save_dir: str,
) -> Dict[str, str]:
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    pth_path   = os.path.join(save_dir, f"lstm_custom_{timestamp}.pth")
    checkpoint = {
        "model_state": model_state,
        "cfg": {
            "hidden_dim":    config.hidden_dim,
            "num_layers":    config.num_layers,
            "bidirectional": config.bidirectional,
            "use_attention": config.use_attention,
            "fc_activation": config.fc_activation,
            "lstm_dropout":  config.lstm_dropout,
            "fc_dropout":    config.fc_dropout,
        },
        "norm_mean": norm_mean.cpu(),
        "norm_std":  norm_std.cpu(),
    }
    torch.save(checkpoint, pth_path)

    npz_path = os.path.join(save_dir, f"norm_stats_{timestamp}.npz")
    np.savez(npz_path, mean=norm_mean.cpu().numpy(), std=norm_std.cpu().numpy())

    json_path = os.path.join(save_dir, f"training_report_{timestamp}.json")
    report    = {
        "timestamp": timestamp,
        "hyperparameters": {
            "hidden_dim":    config.hidden_dim,
            "num_layers":    config.num_layers,
            "bidirectional": config.bidirectional,
            "use_attention": config.use_attention,
            "fc_activation": config.fc_activation,
            "lstm_dropout":  config.lstm_dropout,
            "fc_dropout":    config.fc_dropout,
            "learning_rate": config.learning_rate,
            "batch_size":    config.batch_size,
            "epochs":        config.epochs,
            "optimizer":     config.optimizer,
            "scheduler":     config.scheduler,
            "seq_len":       config.seq_len,
            "stride":        config.stride,
        },
        "metrics": {
            "accuracy":         round(eval_metrics.accuracy,  4),
            "precision":        round(eval_metrics.precision, 4),
            "recall":           round(eval_metrics.recall,    4),
            "f1_score":         round(eval_metrics.f1,        4),
            "f2_score":         round(eval_metrics.f2,        4),
            "confusion_matrix": eval_metrics.confusion_matrix,
        },
        "training": {
            "best_epoch":       history.best_epoch + 1,
            "best_val_f2":      round(history.best_val_f2, 4),
            "total_epochs_run": len(history.train_loss),
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return {"pth": pth_path, "npz": npz_path, "json": json_path}


def create_zip_bundle(file_paths: Dict[str, str]) -> bytes:
    """Buat ZIP bundle dari file-file model."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for key, fpath in file_paths.items():
            if os.path.isfile(fpath):
                zf.write(fpath, os.path.basename(fpath))
    return buf.getvalue()