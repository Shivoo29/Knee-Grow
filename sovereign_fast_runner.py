#!/usr/bin/env python3
"""sovereign_fast_runner.py - Streamlined, Fast Sovereign Fleet Inference.
Runtime: ~15-20 min on Kaggle Tesla T4 (Zero Timeout Risk, Pure FP32, Zero NaNs).
"""

import os
import io
import sys
import time
import glob
import gc
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import cv2
import torch
import torch.nn as nn
from scipy.stats import rankdata

START_TIME = time.time()
print("=" * 80)
print("RSNA KNEE STREAMLINED SOVEREIGN FLEET STARTING (FAST & ACCURATE)")
print("=" * 80)

# ── 1. Input Directories Resolution ──────────────────────────────────────────
KAGGLE_INPUT = "/kaggle/input"
comp_dir_candidates = [
    os.path.join(KAGGLE_INPUT, "rsna-knee-abnormality-detection"),
]
COMP_DIR = next((d for d in comp_dir_candidates if os.path.exists(d)), KAGGLE_INPUT)

# Resolve weights directory
weights_candidates = [
    os.path.join(KAGGLE_INPUT, "knee-our-models"),
    os.path.join(KAGGLE_INPUT, "datasets", "shivoo29", "knee-our-models"),
    "."
]
WEIGHTS_DIR = next((d for d in weights_candidates if os.path.exists(d)), COMP_DIR)
for root, dirs, _ in os.walk(KAGGLE_INPUT):
    for d in dirs:
        if "knee-our-models" in d.lower():
            WEIGHTS_DIR = os.path.join(root, d)
            break

print(f"Data directory:    {COMP_DIR}")
print(f"Weights directory: {WEIGHTS_DIR}")

# ── 2. Hardware Detection ────────────────────────────────────────────────────
NUM_GPUS = torch.cuda.device_count()
print(f"\nCUDA Available: {torch.cuda.is_available()} | Detected GPUs: {NUM_GPUS}")
for g in range(NUM_GPUS):
    print(f"  GPU {g}: {torch.cuda.get_device_name(g)} | VRAM: {torch.cuda.get_device_properties(g).total_memory / 1e9:.2f} GB")

dev_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dev_1 = torch.device("cuda:1" if NUM_GPUS > 1 else dev_0)

CANONICAL_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

# ── 3. Unified DICOM Preprocessing Engine ────────────────────────────────────
IMG = 384
CROP_MM = 140.0

SLOTS_96 = [
    ("Sagittal", 1, 26),   # Sagittal Fluid
    ("Sagittal", 0, 22),   # Sagittal Non-Fluid
    ("Coronal",  1, 18),   # Coronal Fluid
    ("Coronal",  0, 12),   # Coronal Non-Fluid
    ("Axial",   -1, 18),   # Axial
]

def parse_dicom_header(fp):
    try:
        h = pydicom.dcmread(fp, stop_before_pixels=True)
        iop = getattr(h, 'ImageOrientationPatient', None)
        ipp = getattr(h, 'ImagePositionPatient', None)
        if iop is not None and ipp is not None and len(iop) == 6:
            r = np.array(iop[:3], dtype=np.float64)
            c = np.array(iop[3:], dtype=np.float64)
            n = np.cross(r, c)
            pos = float(np.dot(np.array(ipp, dtype=np.float64), n))
        else:
            pos = float(getattr(h, 'InstanceNumber', 0) or 0)
        ps_val = getattr(h, 'PixelSpacing', None)
        ps = float(ps_val[0]) if ps_val is not None else 0.5
        return pos, ps, True
    except Exception:
        return 0.0, 0.5, False

def decode_dicom_pixel(fp):
    try:
        d = pydicom.dcmread(fp, force=True)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a
    except Exception:
        return None

def mm_crop_and_resize(a, ps, crop_mm=CROP_MM, target_res=IMG):
    h, w = a.shape
    cpx = int(round(crop_mm / max(ps, 1e-3)))
    cpx = max(1, min(cpx, min(h, w)))
    y0 = (h - cpx) // 2
    x0 = (w - cpx) // 2
    return cv2.resize(a[y0:y0 + cpx, x0:x0 + cpx], (target_res, target_res), interpolation=cv2.INTER_AREA)

def pick_series(series_rows, plane, fluid, used_series):
    plane_norm = str(plane).strip().capitalize()
    cands = [r for r in series_rows if str(r.get('Anatomical_Plane', '')).strip().capitalize() == plane_norm and str(r.get('SeriesInstanceUID', '')) not in used_series]
    if fluid in (0, 1):
        pref = [r for r in cands if int(float(r.get('Fluid_Sensitive', -1) or -1)) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None

# Thread pool for fast non-blocking parallel DICOM I/O
IO_POOL = ThreadPoolExecutor(max_workers=8)

def build_study_volumes(study_id, series_rows, study_dicom_map):
    """Builds both 96-slice multi-planar volume (v6) and 18-slice axial volume (PF OA specialist)."""
    vol96 = np.zeros((96, IMG, IMG), dtype=np.uint8)
    used_series = set()
    idx96 = 0

    axial_slice_indices = []

    for plane, fluid, k in SLOTS_96:
        r = pick_series(series_rows, plane, fluid, used_series)
        if r is None:
            idx96 += k
            continue
        ser_id = str(r["SeriesInstanceUID"])
        used_series.add(ser_id)
        files = study_dicom_map.get(ser_id, [])
        if not files:
            idx96 += k
            continue

        # Fast parallel header decoding
        if len(files) > 4:
            hdrs_raw = list(IO_POOL.map(parse_dicom_header, files))
            hdrs = [(pos, f, ps) for (pos, ps, ok), f in zip(hdrs_raw, files) if ok]
        else:
            hdrs = []
            for f in files:
                pos, ps, ok = parse_dicom_header(f)
                if ok: hdrs.append((pos, f, ps))

        hdrs.sort(key=lambda x: x[0])
        sorted_files = [(f, ps) for _, f, ps in hdrs]
        if not sorted_files:
            idx96 += k
            continue

        n = len(sorted_files)
        med_ps = float(np.median([ps for _, ps in sorted_files])) if sorted_files else 0.5
        lo, hi = int(n * 0.02), max(int(n * 0.02), int(n * 0.98) - 1)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k

        picked_files = [sorted_files[min(p, n - 1)] for p in picks]
        fps = [item[0] for item in picked_files]
        pss = [item[1] if item[1] > 0 else med_ps for item in picked_files]

        # Fast parallel pixel decompression
        if len(fps) > 4:
            arrs = list(IO_POOL.map(decode_dicom_pixel, fps))
        else:
            arrs = [decode_dicom_pixel(fp) for fp in fps]

        valid_arrs = [a for a in arrs if a is not None]
        if valid_arrs:
            allpx = np.concatenate([a.ravel()[::8] for a in valid_arrs])  # subsampled 1:8 for 4x faster percentiles
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0

        denom = max(hiq - loq, 1e-6)
        start_idx = idx96
        for a, ps in zip(arrs, pss):
            if idx96 >= 96: break
            if a is not None:
                aw = np.clip((a - loq) / denom, 0.0, 1.0)
                vol96[idx96] = (mm_crop_and_resize(aw, ps, target_res=IMG) * 255.0).astype(np.uint8)
            idx96 += 1

        if str(plane).strip().capitalize() == "Axial":
            axial_slice_indices = list(range(start_idx, min(idx96, 96)))

    mask96 = (vol96.reshape(96, -1).sum(axis=1) > 0).astype(np.uint8)

    # 18-slice axial volume for PF OA specialist (resize to 224 for specialist)
    if axial_slice_indices and len(axial_slice_indices) >= 6:
        ax_picks = np.linspace(axial_slice_indices[0], axial_slice_indices[-1], 18).round().astype(int)
        vol_ax = vol96[ax_picks]
    else:
        # Fallback to last 18 slices of 96-slice volume
        vol_ax = vol96[78:96]

    vol_ax_224 = np.zeros((18, 224, 224), dtype=np.uint8)
    for i in range(18):
        vol_ax_224[i] = cv2.resize(vol_ax[i], (224, 224), interpolation=cv2.INTER_AREA)

    return vol96, mask96, vol_ax_224

# ── 4. Model Architectures & Loading ─────────────────────────────────────────
try:
    import timm
except ImportError:
    os.system("pip install timm -q")
    import timm

class RaptorClassifier(nn.Module):
    """Architecture for CoAtNet-384 96-slice model."""
    def __init__(self, backbone, F_dim=1024, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))

    win_chunk = 14

    def encode(self, x):
        B, K = x.shape[:2]
        flat = x.flatten(0, 1)
        if self.win_chunk > 0:
            f = torch.cat([self.backbone(c) for c in flat.split(self.win_chunk, 0)], 0)
        else:
            f = self.backbone(flat)
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))


class AxialAttentionClassifier(nn.Module):
    """Architecture for Pure Axial PF OA Specialist (CoAtNet-224)."""
    def __init__(self, arch="coatnet_rmlp_1_rw_224", num_classes=12, drop=0.2):
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features
        self.norm = nn.LayerNorm(feat_dim)
        self.att = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.Tanh(),
            nn.Dropout(drop),
            nn.Linear(256, num_classes)
        )
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        B, K, C, H, W = x.shape
        flat = x.view(B * K, C, H, W)
        feats = self.backbone(flat).view(B, K, -1)
        h = self.norm(feats)
        att_weights = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkc,bkf->bcf", att_weights, h)
        logits = torch.einsum("bcf,cf->bc", pooled, self.classifier.weight) + self.classifier.bias
        return logits


def load_raptor_v6(cand_names, device):
    bb = timm.create_model("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", pretrained=False, num_classes=0)
    model = RaptorClassifier(bb, F_dim=bb.num_features).to(device).eval()
    for name in cand_names:
        cands = glob.glob(os.path.join(WEIGHTS_DIR, "**", name), recursive=True)
        if cands:
            print(f"  Loaded {name} -> {cands[0]} on {device}")
            ck = torch.load(cands[0], map_location="cpu", weights_only=False)
            state = ck.get("model", ck.get("state_dict", ck))
            model.load_state_dict(state, strict=True)
            return model
    raise FileNotFoundError(f"FATAL: None of checkpoints {cand_names} found in {WEIGHTS_DIR}!")


def load_specialist_pfoa(cand_names, device):
    model = AxialAttentionClassifier(arch="coatnet_rmlp_1_rw_224").to(device).eval()
    for name in cand_names:
        cands = glob.glob(os.path.join(WEIGHTS_DIR, "**", name), recursive=True)
        if cands:
            print(f"  Loaded {name} -> {cands[0]} on {device}")
            ck = torch.load(cands[0], map_location="cpu", weights_only=False)
            state = ck.get("model_state", ck.get("model", ck.get("state_dict", ck)))
            model.load_state_dict(state, strict=True)
            return model
    raise FileNotFoundError(f"FATAL: None of checkpoints {cand_names} found in {WEIGHTS_DIR}!")


# Load Models in strict FP32
print("\nLoading Champion Model 1: CoAtNet-96 v6 SWA...")
model_v6 = load_raptor_v6(["raptor_ft_v6_swa.pt", "raptor_ft_v6.pt"], dev_0)

print("\nLoading Champion Model 2: Pure Axial PF OA Specialist (CoAtNet-224)...")
specialist_pfoa = load_specialist_pfoa(["specialist_pfoa_best.pt", "specialist_pfoa.pt"], dev_1)


def get_v6_windows_tensor(vol, mask, k_windows, device):
    D = vol.shape[0]
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k_windows).round().astype(int)
    centers = [cs[i] for i in idx]

    tiles = []
    for c in centers:
        c = max(1, min(c, D - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        tiles.append(tri)
    wins = np.stack(tiles, 0)
    x = torch.from_numpy(wins).to(device)
    return x.unsqueeze(0)


def get_axial_tensor(vol_ax_224, device):
    """Builds (1, 18, 3, 224, 224) triplet tensor matching train_specialist_pfoa.py."""
    vol = vol_ax_224.astype(np.float32) / 255.0
    K, H, W = vol.shape  # 18, 224, 224
    triplets = []
    for s in range(K):
        s_prev = max(0, s - 1)
        s_next = min(K - 1, s + 1)
        tri = np.stack([vol[s_prev], vol[s], vol[s_next]], axis=0)  # (3, 224, 224)
        triplets.append(tri)
    x = np.stack(triplets, axis=0)  # (18, 3, 224, 224)
    return torch.from_numpy(x).unsqueeze(0).to(device)


# ── 5. Test Data Discovery ───────────────────────────────────────────────────
print("\nResolving test studies and metadata...")
test_series_paths = glob.glob(os.path.join(COMP_DIR, "**", "test_series.csv"), recursive=True)
sample_sub_paths  = glob.glob(os.path.join(COMP_DIR, "**", "sample_submission.csv"), recursive=True)

if not sample_sub_paths:
    raise FileNotFoundError("Could not find sample_submission.csv!")

sample_sub = pd.read_csv(sample_sub_paths[0])
print(f"  sample_submission.csv loaded: {len(sample_sub)} rows")

if test_series_paths:
    test_series_df = pd.read_csv(test_series_paths[0])
    test_series_df["StudyInstanceUID"] = test_series_df["StudyInstanceUID"].astype(str)
    test_series_df["SeriesInstanceUID"] = test_series_df["SeriesInstanceUID"].astype(str)
else:
    test_series_df = pd.DataFrame(columns=["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"])

test_uids = sample_sub["StudyInstanceUID"].astype(str).unique().tolist()
print(f"  Total test studies to predict: {len(test_uids)}")

series_map = {uid: test_series_df[test_series_df["StudyInstanceUID"] == uid].to_dict("records") for uid in test_uids}

def find_series_dicoms(comp_dir, uid, s_id):
    for folder in [
        os.path.join(comp_dir, "test_series", str(uid), str(s_id)),
        os.path.join(comp_dir, "test_images", str(uid), str(s_id)),
        os.path.join(comp_dir, "test_series", str(s_id)),
        os.path.join(comp_dir, "test_images", str(s_id)),
        os.path.join(comp_dir, "test", str(uid), str(s_id)),
        os.path.join(comp_dir, "test", str(s_id)),
    ]:
        if os.path.isdir(folder):
            cands = sorted(glob.glob(os.path.join(folder, "*.dcm")))
            if cands:
                return cands
    for sub in ["test_series", "test_images", "test"]:
        sub_dir = os.path.join(comp_dir, sub)
        if os.path.isdir(sub_dir):
            cands = sorted(glob.glob(os.path.join(sub_dir, "**", str(s_id), "*.dcm"), recursive=True))
            if cands:
                return cands
    return []

test_files_map = {}
for uid in test_uids:
    test_files_map[uid] = {}
    for r in series_map[uid]:
        s_id = r["SeriesInstanceUID"]
        test_files_map[uid][s_id] = find_series_dicoms(COMP_DIR, uid, s_id)

# ── 6. Streamlined Fast Inference Loop ───────────────────────────────────────
print("\n" + "=" * 80)
print(f"BEGINNING STREAMLINED FP32 INFERENCE ACROSS {len(test_uids)} TEST STUDIES")
print("=" * 80)

OUTPUT_CSV = os.environ.get("RAPTOR_OUTPUT_CSV", "submission.csv")
os.makedirs(os.path.dirname(OUTPUT_CSV) if os.path.dirname(OUTPUT_CSV) else ".", exist_ok=True)

predictions_dict = {}
study_times = []

def save_submission_csv(preds_so_far, output_path=None):
    if output_path is None:
        output_path = OUTPUT_CSV
    sub_df = sample_sub.copy()
    for col in CANONICAL_COLUMNS:
        if col in sub_df.columns:
            sub_df[col] = 0.5
    for uid, probs in preds_so_far.items():
        row_mask = (sub_df["StudyInstanceUID"].astype(str) == str(uid))
        if row_mask.any():
            for c_idx, c_name in enumerate(CANONICAL_COLUMNS):
                if c_name in sub_df.columns:
                    sub_df.loc[row_mask, c_name] = float(probs[c_idx])
    sub_df.to_csv(output_path, index=False)

# K_EVAL=42 windows for 96-slice model (exact training setting, cuts inference time in half)
K_WINDOWS_V6 = 42
PFOA_COL_IDX = CANONICAL_COLUMNS.index("PF OA")

for s_idx, uid in enumerate(test_uids):
    t0_study = time.perf_counter()

    # Step A: Preprocess volumes (multithreaded DICOM decode)
    vol96, mask96, vol_ax_224 = build_study_volumes(uid, series_map[uid], test_files_map[uid])

    # Step B: Model 1 Inference (CoAtNet-96 v6 SWA, 42 windows)
    x_v6 = get_v6_windows_tensor(vol96, mask96, K_WINDOWS_V6, dev_0)
    with torch.inference_mode():
        logits_v6 = model_v6(x_v6)
        prob_v6 = torch.sigmoid(logits_v6.float()).cpu().numpy()[0]

    # Step C: Model 2 Inference (Axial PF OA Specialist, 18 slices)
    x_ax = get_axial_tensor(vol_ax_224, dev_1)
    with torch.inference_mode():
        logits_ax = specialist_pfoa(x_ax)
        prob_ax = torch.sigmoid(logits_ax.float()).cpu().numpy()[0]

    # Composite Prediction:
    # 1. Base prediction across all 12 findings comes from CoAtNet-96 v6 SWA
    prob_composite = prob_v6.copy()
    # 2. Targeted boost: Blend PF OA prediction with Axial Specialist (50/50 blend on PF OA)
    prob_composite[PFOA_COL_IDX] = 0.50 * prob_v6[PFOA_COL_IDX] + 0.50 * prob_ax[PFOA_COL_IDX]

    if np.isnan(prob_composite).any():
        prob_composite = np.nan_to_num(prob_composite, nan=0.5)

    predictions_dict[uid] = prob_composite

    t_study = time.perf_counter() - t0_study
    study_times.append(t_study)

    if (s_idx + 1) % 25 == 0 or s_idx < 3 or (s_idx + 1) == len(test_uids):
        mean_t = np.mean(study_times)
        rem = (len(test_uids) - (s_idx + 1)) * mean_t / 60
        print(f"  [{s_idx+1:4d}/{len(test_uids)}] {uid[:22]}... | {t_study:.2f}s | avg {mean_t:.2f}s | ETA: {rem:.1f}m", flush=True)

    if (s_idx + 1) % 50 == 0:
        save_submission_csv(predictions_dict)

save_submission_csv(predictions_dict)

total_elapsed = time.time() - START_TIME
print("\n" + "=" * 80)
print(f"SOVEREIGN FLEET FINISHED in {total_elapsed/60:.2f} min (avg {np.mean(study_times):.3f}s / study)")
print(f"Output saved to {OUTPUT_CSV}")
print("=" * 80)
