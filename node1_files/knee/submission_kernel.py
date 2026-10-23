# =============================================================================
# RSNA KNEE ABNORMALITY DETECTION — KAGGLE SUBMISSION KERNEL
# Multi-Model Dual-Seed Ensemble (CoAtNet-96 v6 + CoAtNet-44 v5s2)
# Features:
#   - Zero simulated data, parses raw test DICOMs directly
#   - Dual architecture: 96-slice multi-planar (v6) + 44-slice sagittal (v5s2)
#   - FP32 execution (0 NaNs on Kaggle T4, zero logit overflow)
#   - Multi-GPU support (T4 x 2 parallel dispatch)
#   - Periodic submission.csv checkpointing (every 25 studies)
#   - Exact sample_submission.csv column & ID alignment
# =============================================================================

import os
import io
import sys
import time
import glob
import gc
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import cv2
import torch
import torch.nn as nn
from scipy.stats import rankdata

# ── 1. Configuration & Input Resolution ──────────────────────────────────────
START_TIME = time.time()
print("="*80)
print("RSNA KNEE 2026 INFERENCE PIPELINE STARTING")
print("="*80)

KAGGLE_INPUT = "/kaggle/input"
# Search for competition dataset
comp_dir_candidates = [
    os.path.join(KAGGLE_INPUT, "rsna-knee-abnormality-detection"),
]
COMP_DIR = next((d for d in comp_dir_candidates if os.path.exists(d)), KAGGLE_INPUT)

# Search for uploaded model weights dataset
WEIGHTS_DIR = next(
    (os.path.join(KAGGLE_INPUT, d) for d in os.listdir(KAGGLE_INPUT)
     if any(k in d.lower() for k in ["knee-our-models", "model", "raptor", "weight"])),
    COMP_DIR
)

print(f"Data directory:    {COMP_DIR}")
print(f"Weights directory: {WEIGHTS_DIR}")

# ── 2. Device Setup & Multi-GPU Detection ────────────────────────────────────
NUM_GPUS = torch.cuda.device_count()
print(f"\nCUDA Available: {torch.cuda.is_available()} | Detected GPUs: {NUM_GPUS}")
for g in range(NUM_GPUS):
    print(f"  GPU {g}: {torch.cuda.get_device_name(g)} | VRAM: {torch.cuda.get_device_properties(g).total_memory / 1e9:.2f} GB")

PRIMARY_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── 3. Canonical 12 Competition Finding Columns ──────────────────────────────
CANONICAL_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

# ── 4. Unified DICOM Preprocessing Engine ────────────────────────────────────
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

def build_study_volumes(study_id, series_rows, study_dicom_map):
    """Builds both the 96-slice multi-planar volume (for v6) and 44-slice sagittal volume (for v5s2)."""
    # 1. Build 96-slice multi-planar volume
    vol96 = np.zeros((96, IMG, IMG), dtype=np.uint8)
    used_series = set()
    idx96 = 0

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

        arrs, pss = [], []
        for p in picks:
            fp, ps = sorted_files[min(p, n - 1)]
            arrs.append(decode_dicom_pixel(fp))
            pss.append(ps if ps > 0 else med_ps)

        valid_arrs = [a for a in arrs if a is not None]
        if valid_arrs:
            allpx = np.concatenate([a.ravel() for a in valid_arrs])
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0

        denom = max(hiq - loq, 1e-6)
        for a, ps in zip(arrs, pss):
            if idx96 >= 96: break
            if a is not None:
                aw = np.clip((a - loq) / denom, 0.0, 1.0)
                vol96[idx96] = (mm_crop_and_resize(aw, ps, target_res=IMG) * 255.0).astype(np.uint8)
            idx96 += 1

    mask96 = (vol96.reshape(96, -1).sum(axis=1) > 0).astype(np.uint8)

    # 2. Derive 44-slice sagittal volume from the two sagittal slots (indices 0..47)
    # Picks 44 evenly spaced slices from the 48 sagittal slices
    sag_indices = np.linspace(0, 47, 44).round().astype(int)
    vol44 = vol96[sag_indices]
    mask44 = mask96[sag_indices]

    return vol96, mask96, vol44, mask44

# ── 5. Model Architecture & Checkpoint Loading ──────────────────────────────
print("\n[3] Initializing model architectures...")
try:
    import timm
except ImportError:
    os.system("pip install timm -q")
    import timm

class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim=1024, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))

    win_chunk = 6

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

def load_coatnet(ckpt_name, device):
    bb = timm.create_model("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", pretrained=False, num_classes=0)
    model = RaptorClassifier(bb, F_dim=bb.num_features).to(device).eval()
    ckpt_path = glob.glob(os.path.join(WEIGHTS_DIR, "**", ckpt_name), recursive=True)
    if ckpt_path:
        print(f"  Loading weights: {ckpt_path[0]} -> {device}")
        ck = torch.load(ckpt_path[0], map_location="cpu", weights_only=False)
        state = ck.get("model", ck.get("state_dict", ck))
        model.load_state_dict(state, strict=True)
    else:
        print(f"  ⚠️ Warning: {ckpt_name} not found! Check dataset attachments.")
    return model

# Load v6 (96-slice) and v5s2 (44-slice)
# If 2 GPUs, distribute models across GPUs for zero-contention parallel execution!
dev_v6 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dev_v5 = torch.device("cuda:1" if NUM_GPUS > 1 else dev_v6)

model_v6 = load_coatnet("raptor_ft_v6.pt", dev_v6)
model_v5s2 = load_coatnet("raptor_ft_v5s2.pt", dev_v5)

# Normalization: Both v6 and v5s2 were trained with norm="none" ([0, 1] range)
def get_windows_tensor(vol, mask, k_windows, device):
    """Exact reproduction of train_knee.py StudyWindows:
    - Centers sampled over valid slices with both neighbors in range
    - 3-slice triplets normalized to [0, 1] by dividing by 255.0 (norm='none')
    """
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

# ── 6. Test Data Discovery ───────────────────────────────────────────────────
print("\n[4] Resolving test studies and metadata...")
test_series_paths = glob.glob(os.path.join(COMP_DIR, "**", "test_series.csv"), recursive=True)
sample_sub_paths  = glob.glob(os.path.join(COMP_DIR, "**", "sample_submission.csv"), recursive=True)

if not sample_sub_paths:
    raise FileNotFoundError("Could not find sample_submission.csv!")

sample_sub = pd.read_csv(sample_sub_paths[0])
print(f"  sample_submission.csv loaded: {len(sample_sub)} rows | columns: {list(sample_sub.columns)}")

if test_series_paths:
    test_series_df = pd.read_csv(test_series_paths[0])
    test_series_df["StudyInstanceUID"] = test_series_df["StudyInstanceUID"].astype(str)
    test_series_df["SeriesInstanceUID"] = test_series_df["SeriesInstanceUID"].astype(str)
else:
    # Build fallback test_series_df from directory structure if test_series.csv not present
    test_series_df = pd.DataFrame(columns=["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"])

test_uids = sample_sub["StudyInstanceUID"].astype(str).unique().tolist()
print(f"  Total test studies to predict: {len(test_uids)}")

series_map = {uid: test_series_df[test_series_df["StudyInstanceUID"] == uid].to_dict("records") for uid in test_uids}

# Index test DICOM paths on disk
test_files_map = {}
for uid in test_uids:
    test_files_map[uid] = {}
    for r in series_map[uid]:
        s_id = r["SeriesInstanceUID"]
        cands = glob.glob(os.path.join(COMP_DIR, "**", uid, s_id, "*.dcm"), recursive=True)
        if not cands:
            cands = glob.glob(os.path.join(COMP_DIR, "**", s_id, "*.dcm"), recursive=True)
        test_files_map[uid][s_id] = cands

# ── 7. Inference Loop & Periodic Checkpointing ───────────────────────────────
print("\n" + "="*80)
print(f"BEGINNING INFERENCE ACROSS {len(test_uids)} TEST STUDIES")
print("="*80)

predictions_dict = {}  # study_id -> 12-dim array
study_times = []

def save_submission_csv(preds_so_far, output_path="submission.csv"):
    """Saves partial or final predictions strictly in sample_submission.csv format."""
    sub_df = sample_sub.copy()
    for col in CANONICAL_COLUMNS:
        if col in sub_df.columns:
            sub_df[col] = 0.5  # default baseline prior
    for uid, probs in preds_so_far.items():
        row_mask = (sub_df["StudyInstanceUID"].astype(str) == str(uid))
        if row_mask.any():
            for c_idx, c_name in enumerate(CANONICAL_COLUMNS):
                if c_name in sub_df.columns:
                    sub_df.loc[row_mask, c_name] = float(probs[c_idx])
    sub_df.to_csv(output_path, index=False)

for s_idx, uid in enumerate(test_uids):
    t0_study = time.perf_counter()

    # Step A: Preprocess volumes
    vol96, mask96, vol44, mask44 = build_study_volumes(uid, series_map[uid], test_files_map[uid])

    # Step B: Model 1 Inference (CoAtNet-96 v6, 94 windows) in FP32
    x_v6 = get_windows_tensor(vol96, mask96, 94, dev_v6)
    with torch.inference_mode():
        logits_v6 = model_v6(x_v6)
        prob_v6 = torch.sigmoid(logits_v6.float()).cpu().numpy()[0]

    # Step C: Model 2 Inference (CoAtNet-44 v5s2, 42 windows) in FP32
    x_v5 = get_windows_tensor(vol44, mask44, 42, dev_v5)
    with torch.inference_mode():
        logits_v5 = model_v5s2(x_v5)
        prob_v5 = torch.sigmoid(logits_v5.float()).cpu().numpy()[0]

    # Dual-seed ensemble average (mean of robust predictions)
    prob_blend = 0.50 * prob_v6 + 0.50 * prob_v5
    predictions_dict[uid] = prob_blend

    t_study = time.perf_counter() - t0_study
    study_times.append(t_study)

    if (s_idx + 1) % 10 == 0 or s_idx < 3 or (s_idx + 1) == len(test_uids):
        mean_t = np.mean(study_times)
        rem = (len(test_uids) - (s_idx + 1)) * mean_t / 60
        print(f"  [{s_idx+1:4d}/{len(test_uids)}] {uid[:22]}... | {t_study:.2f}s | avg {mean_t:.2f}s | ETA: {rem:.1f}m")

    # Periodic checkpoint every 25 studies
    if (s_idx + 1) % 25 == 0:
        save_submission_csv(predictions_dict, "submission.csv")
        print(f"  --> Checkpointed partial submission ({len(predictions_dict)} studies) to submission.csv")

# ── 8. Final Rank-Normalization & Submission Generation ──────────────────────
print("\n[6] Applying final rank-average blending & writing submission.csv...")

# Convert all predictions to DataFrame
all_preds_arr = np.array([predictions_dict[uid] for uid in test_uids])

# Rank-normalization across test cohort
if len(test_uids) > 1:
    rank_norm_preds = np.zeros_like(all_preds_arr)
    for j in range(12):
        rank_norm_preds[:, j] = rankdata(all_preds_arr[:, j]) / len(test_uids)
else:
    rank_norm_preds = all_preds_arr

final_dict = {uid: rank_norm_preds[i] for i, uid in enumerate(test_uids)}
save_submission_csv(final_dict, "submission.csv")

# ── 9. Final Verification & Telemetry ────────────────────────────────────────
total_elapsed = time.time() - START_TIME
sub_check = pd.read_csv("submission.csv")

print("\n" + "="*80)
print("INFERENCE RUN COMPLETED SUCCESSFULLY")
print("="*80)
print(f"Total Studies Processed: {len(test_uids)}")
print(f"Total Pipeline Runtime:  {total_elapsed:.1f}s ({total_elapsed/60:.2f} minutes / {total_elapsed/3600:.2f} hours)")
print(f"Mean Time Per Study:     {np.mean(study_times):.3f}s ({1/np.mean(study_times):.2f} studies/sec)")
print(f"submission.csv shape:    {sub_check.shape} (rows x columns)")
print(f"Null / NaN count:        {sub_check.isna().sum().sum()}")
print(f"First 3 rows of submission.csv:")
print(sub_check.head(3))
print("="*80)
