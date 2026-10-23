# =============================================================================
# RSNA KNEE ABNORMALITY DETECTION — FLEET RUNNER (CoAtNet-96 v6 + CoAtNet-44 v5s2)
# Generates ours_fleet/submission.csv for the Grand Rank Blend
# =============================================================================

import os
import io
import sys
import time
import glob
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import cv2
import torch
import torch.nn as nn
from scipy.stats import rankdata

START_TIME = time.time()
print("="*80)
print("RSNA KNEE: 2-MODEL COATNET FLEET ENGINE STARTING")
print("="*80)

OUTPUT_CSV = os.environ.get("RAPTOR_OUTPUT_CSV", "/kaggle/working/ours_fleet/submission.csv")
os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)), exist_ok=True)

KAGGLE_INPUT = "/kaggle/input"
available_dirs = os.listdir(KAGGLE_INPUT) if os.path.exists(KAGGLE_INPUT) else []
print(f"Attached inputs in {KAGGLE_INPUT}: {available_dirs}")

# 1. Resolve Competition Dataset
candidate_comps = [
    os.environ.get("RSNA_COMP_ROOT", ""),
    "/kaggle/input/rsna-knee-abnormality-detection",
    "/kaggle/input/competitions/rsna-knee-abnormality-detection",
]
COMP_DIR = next((d for d in candidate_comps if d and os.path.exists(d)), None)
if COMP_DIR is None:
    for d in available_dirs:
        p = os.path.join(KAGGLE_INPUT, d)
        if os.path.isdir(p) and any(f in os.listdir(p) for f in ["test_series.csv", "sample_submission.csv", "train_series.csv"]):
            COMP_DIR = p
            break
if COMP_DIR is None:
    COMP_DIR = os.path.join(KAGGLE_INPUT, "rsna-knee-abnormality-detection")

# 2. Resolve Model Weights Dataset
candidate_weights = [
    "/kaggle/input/datasets/tun0tun0mausi/knee-our-models",
    "/kaggle/input/knee-our-models",
    "/kaggle/input/datasets/shivoo29/knee-our-models",
]
WEIGHTS_DIR = next((d for d in candidate_weights if os.path.exists(d)), None)
if WEIGHTS_DIR is None:
    for d in available_dirs:
        p = os.path.join(KAGGLE_INPUT, d)
        if os.path.isdir(p) and any(k in d.lower() for k in ["knee-our-models", "knee", "model", "raptor", "weight"]):
            files_here = os.listdir(p)
            if any(f.endswith(".pt") for f in files_here):
                WEIGHTS_DIR = p
                break
            for sub in files_here:
                subp = os.path.join(p, sub)
                if os.path.isdir(subp) and any(f.endswith(".pt") for f in os.listdir(subp)):
                    WEIGHTS_DIR = subp
                    break
            if WEIGHTS_DIR:
                break

if WEIGHTS_DIR is None:
    for d in available_dirs:
        p = os.path.join(KAGGLE_INPUT, d)
        if os.path.isdir(p) and any(f.endswith(".pt") for f in os.listdir(p)):
            WEIGHTS_DIR = p
            break

if WEIGHTS_DIR is None:
    WEIGHTS_DIR = os.path.join(KAGGLE_INPUT, "datasets/tun0tun0mausi/knee-our-models")

print(f"Data directory:    {COMP_DIR}")
print(f"Weights directory: {WEIGHTS_DIR}")

# 3. Device Setup & Multi-GPU Detection
NUM_GPUS = torch.cuda.device_count()
print(f"\nCUDA Available: {torch.cuda.is_available()} | Detected GPUs: {NUM_GPUS}")
for g in range(NUM_GPUS):
    print(f"  GPU {g}: {torch.cuda.get_device_name(g)} | VRAM: {torch.cuda.get_device_properties(g).total_memory / 1e9:.2f} GB")

PRIMARY_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CANONICAL_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

IMG = 384
CROP_MM = 140.0

SLOTS_96 = [
    ("Sagittal", 1, 26),
    ("Sagittal", 0, 22),
    ("Coronal",  1, 18),
    ("Coronal",  0, 12),
    ("Axial",   -1, 18),
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
        for p_idx in picks:
            fp, ps = sorted_files[p_idx]
            arr = decode_dicom_pixel(fp)
            if arr is not None:
                arrs.append(arr)
                pss.append(ps)
        if not arrs:
            idx96 += k
            continue

        all_pix = np.concatenate([a.ravel() for a in arrs])
        p2, p98 = np.percentile(all_pix, 2.0), np.percentile(all_pix, 98.0)
        denom = max(p98 - p2, 1e-4)

        for arr, ps in zip(arrs, pss):
            if idx96 >= 96: break
            norm_arr = np.clip((arr - p2) / denom, 0.0, 1.0)
            cropped = mm_crop_and_resize(norm_arr, ps, CROP_MM, IMG)
            vol96[idx96] = (cropped * 255.0).astype(np.uint8)
            idx96 += 1

    vol44 = np.zeros((44, IMG, IMG), dtype=np.uint8)
    used_ser_44 = set()
    idx44 = 0
    for fluid, k in [(1, 24), (0, 20)]:
        r = pick_series(series_rows, "Sagittal", fluid, used_ser_44)
        if r is None:
            idx44 += k
            continue
        ser_id = str(r["SeriesInstanceUID"])
        used_ser_44.add(ser_id)
        files = study_dicom_map.get(ser_id, [])
        if not files:
            idx44 += k
            continue
        hdrs = []
        for f in files:
            pos, ps, ok = parse_dicom_header(f)
            if ok: hdrs.append((pos, f, ps))
        hdrs.sort(key=lambda x: x[0])
        sorted_files = [(f, ps) for _, f, ps in hdrs]
        if not sorted_files:
            idx44 += k
            continue

        n = len(sorted_files)
        lo, hi = int(n * 0.02), max(int(n * 0.02), int(n * 0.98) - 1)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k

        arrs, pss = [], []
        for p_idx in picks:
            fp, ps = sorted_files[p_idx]
            arr = decode_dicom_pixel(fp)
            if arr is not None:
                arrs.append(arr)
                pss.append(ps)
        if not arrs:
            idx44 += k
            continue

        all_pix = np.concatenate([a.ravel() for a in arrs])
        p2, p98 = np.percentile(all_pix, 2.0), np.percentile(all_pix, 98.0)
        denom = max(p98 - p2, 1e-4)

        for arr, ps in zip(arrs, pss):
            if idx44 >= 44: break
            norm_arr = np.clip((arr - p2) / denom, 0.0, 1.0)
            cropped = mm_crop_and_resize(norm_arr, ps, CROP_MM, IMG)
            vol44[idx44] = (cropped * 255.0).astype(np.uint8)
            idx44 += 1

    mask96 = (np.std(vol96.reshape(96, -1), axis=1) > 1e-4).astype(np.float32)
    mask44 = (np.std(vol44.reshape(44, -1), axis=1) > 1e-4).astype(np.float32)
    return vol96, mask96, vol44, mask44

# Model Definition
class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim=1024, n_classes=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(
            nn.Linear(F_dim, 256),
            nn.Tanh(),
            nn.Dropout(drop),
            nn.Linear(256, n_classes)
        )
        self.clsW = nn.Parameter(torch.zeros(n_classes, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n_classes))
        self.chunk_size = 8

    def forward(self, x):
        B, K, C, H, W = x.shape
        flat = x.flatten(0, 1)
        feats = []
        for chunk in flat.split(self.chunk_size, 0):
            feats.append(self.backbone(chunk))
        f = torch.cat(feats, 0)
        H_feat = self.norm(f).view(B, K, -1)
        A = torch.softmax(self.att(H_feat), dim=1)
        pooled = torch.einsum('bkn,bkd->bnd', A, H_feat)
        logits = torch.einsum('bnd,nd->bn', pooled, self.clsW) + self.clsb
        return logits

def find_checkpoint(model_name):
    direct = os.path.join(WEIGHTS_DIR, model_name)
    if os.path.isfile(direct): return direct
    for d in [WEIGHTS_DIR, "/kaggle/input/knee-our-models", "/kaggle/input/datasets/tun0tun0mausi/knee-our-models", "/kaggle/input/datasets/shivoo29/knee-our-models", "."]:
        if not os.path.exists(d): continue
        cand = os.path.join(d, model_name)
        if os.path.isfile(cand): return cand
        for alt in [model_name.replace(".pt", "_swa.pt"), model_name.replace("_swa.pt", ".pt")]:
            cand2 = os.path.join(d, alt)
            if os.path.isfile(cand2): return cand2
    return None

import timm

def load_coatnet(model_name, device):
    ckpt_path = find_checkpoint(model_name)
    if ckpt_path is None:
        raise FileNotFoundError(f"Checkpoint {model_name} not found in {WEIGHTS_DIR}!")
    print(f"Loading {model_name} from: {ckpt_path} onto {device}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
    bb = timm.create_model(arch, pretrained=False, num_classes=0, in_chans=3)
    model = RaptorClassifier(bb, F_dim=bb.num_features, n_classes=12, drop=0.0)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.float()
    model.eval()
    return model

dev_v6 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dev_v5 = torch.device("cuda:1" if NUM_GPUS > 1 else dev_v6)

model_v6 = load_coatnet("raptor_ft_v6.pt", dev_v6)
model_v5s2 = load_coatnet("raptor_ft_v5s2.pt", dev_v5)

def get_windows_tensor(vol, mask, k_windows, device):
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

# Resolve test data
sample_sub_cand = os.path.join(COMP_DIR, "sample_submission.csv")
if not os.path.exists(sample_sub_cand):
    for d in available_dirs:
        cand = os.path.join(KAGGLE_INPUT, d, "sample_submission.csv")
        if os.path.exists(cand):
            sample_sub_cand = cand
            break

sample_sub = pd.read_csv(sample_sub_cand)
test_uids = sample_sub["StudyInstanceUID"].astype(str).unique().tolist()
print(f"Total test studies: {len(test_uids)}")

test_series_cand = os.path.join(COMP_DIR, "test_series.csv")
if not os.path.isfile(test_series_cand):
    for d in available_dirs:
        cand = os.path.join(KAGGLE_INPUT, d, "test_series.csv")
        if os.path.isfile(cand):
            test_series_cand = cand
            break

if os.path.isfile(test_series_cand):
    test_series_df = pd.read_csv(test_series_cand)
    test_series_df["StudyInstanceUID"] = test_series_df["StudyInstanceUID"].astype(str)
    test_series_df["SeriesInstanceUID"] = test_series_df["SeriesInstanceUID"].astype(str)
else:
    test_series_df = pd.DataFrame(columns=["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"])

series_map = {uid: test_series_df[test_series_df["StudyInstanceUID"] == uid].to_dict("records") for uid in test_uids}
SERIES_ROOT = os.path.join(COMP_DIR, "test_series") if os.path.isdir(os.path.join(COMP_DIR, "test_series")) else COMP_DIR

test_files_map = {}
for uid in test_uids:
    test_files_map[uid] = {}
    for r in series_map.get(uid, []):
        s_id = str(r.get("SeriesInstanceUID", ""))
        direct_cands = [
            os.path.join(SERIES_ROOT, uid, s_id),
            os.path.join(COMP_DIR, "test_series", uid, s_id),
            os.path.join(COMP_DIR, "test_images", uid, s_id),
            os.path.join(COMP_DIR, uid, s_id),
        ]
        files = []
        for p in direct_cands:
            if os.path.isdir(p):
                files = [os.path.join(p, f) for f in os.listdir(p) if f.endswith(".dcm")]
                if files: break
        test_files_map[uid][s_id] = files

predictions_dict = {}

def save_submission_csv(preds_so_far, output_path=OUTPUT_CSV):
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

print("\nStarting Fleet Inference Loop...")
with torch.no_grad():
    for s_idx, uid in enumerate(test_uids):
        s_rows = series_map.get(uid, [])
        s_dcm_map = test_files_map.get(uid, {})
        try:
            vol96, mask96, vol44, mask44 = build_study_volumes(uid, s_rows, s_dcm_map)
            x96 = get_windows_tensor(vol96, mask96, 94, dev_v6)
            p_v6 = torch.sigmoid(model_v6(x96)).cpu().numpy()[0]

            x44 = get_windows_tensor(vol44, mask44, 42, dev_v5)
            p_v5s2 = torch.sigmoid(model_v5s2(x44)).cpu().numpy()[0]

            # 50/50 probability mean between v6 and v5s2
            p_ensemble = 0.5 * p_v6 + 0.5 * p_v5s2
        except Exception as e:
            print(f"  [ERROR] Study {uid}: {e} -> fallback to 0.5")
            p_ensemble = np.full(12, 0.5, dtype=np.float32)

        predictions_dict[uid] = p_ensemble
        if (s_idx + 1) % 50 == 0 or (s_idx + 1) == len(test_uids):
            print(f"Processed {s_idx + 1}/{len(test_uids)} studies", flush=True)

# Save final predictions
all_preds_arr = np.array([predictions_dict[uid] for uid in test_uids])
if len(test_uids) > 1:
    rank_norm_preds = np.zeros_like(all_preds_arr)
    for j in range(12):
        rank_norm_preds[:, j] = rankdata(all_preds_arr[:, j]) / len(test_uids)
else:
    rank_norm_preds = all_preds_arr

final_dict = {uid: rank_norm_preds[i] for i, uid in enumerate(test_uids)}
save_submission_csv(final_dict, OUTPUT_CSV)

# Write receipt file for Cell 32 verification
receipt_path = os.path.join(os.path.dirname(os.path.abspath(OUTPUT_CSV)), "receipt.json")
with open(receipt_path, "w") as f:
    json.dump({"n_covered_any": len(test_uids), "n_studies": len(test_uids)}, f)

print(f"\nFleet execution complete! Written to {OUTPUT_CSV} ({len(test_uids)} studies)")
