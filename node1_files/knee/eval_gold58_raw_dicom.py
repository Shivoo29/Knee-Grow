import os
import sys
import io
import time
import zipfile
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import cv2
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

KNEE = os.path.expanduser("~/projects/knee")
sys.path.insert(0, KNEE)

import rsna_preproc
import train_knee

print("="*85)
print("TASK 3: EVALUATE ALL 58 GOLD STUDIES THROUGH RAW-DICOM PREPROCESSING PATH")
print("="*85)

ZIP_PATH = os.path.join(KNEE, "raw", "rsna-knee-abnormality-detection.zip")
SERIES_CSV = os.path.join(KNEE, "ds", "comp", "train_series.csv")
CKPT_PATH = os.path.join(KNEE, "raptor_ft_v6.pt")
GOLD_NPZ = os.path.join(KNEE, "raptor_gold_v6.npz")
CACHE_58 = os.path.join(KNEE, "preproc_gold_58.npz")

FINDINGS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

# 1. Load series metadata
print("[1] Loading series metadata...")
series_df = pd.read_csv(SERIES_CSV)
series_df["StudyInstanceUID"] = series_df["StudyInstanceUID"].astype(str)
series_df["SeriesInstanceUID"] = series_df["SeriesInstanceUID"].astype(str)
series_map = {uid: grp.to_dict("records") for uid, grp in series_df.groupby("StudyInstanceUID")}

# 2. Identify all 58 Gold studies
gold_data = np.load(GOLD_NPZ, allow_pickle=True)
gold_ids = gold_data["ids"].astype(str)
gold_truth = gold_data["truth"].astype(float)
gold_offline_preds = gold_data["pred"].astype(float)
N = len(gold_ids)
print(f"[2] Loaded {N} gold studies from {GOLD_NPZ}")

# 3. Preprocess or load from cache
if os.path.exists(CACHE_58):
    print(f"[3] Loading cached 58 preprocessed volumes from {CACHE_58}...")
    cached = np.load(CACHE_58)
    vols = cached["vols"]
    masks = cached["masks"]
    print("  Loaded from cache successfully.")
else:
    print(f"[3] Preprocessing all 58 gold studies from raw DICOM zip...")
    sample_series_uids = set()
    for uid in gold_ids:
        for r in series_map.get(uid, []):
            sample_series_uids.add(r["SeriesInstanceUID"])

    file_map = {}
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        for name in z.namelist():
            if not name.endswith(".dcm"): continue
            parts = name.split("/")
            if len(parts) >= 3 and parts[-2] in sample_series_uids:
                file_map.setdefault(parts[-2], []).append(name)

    vols = []
    masks = []
    t0 = time.time()
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        for i, study_id in enumerate(gold_ids):
            t_s = time.time()
            s_rows = series_map.get(study_id, [])
            vol = np.zeros((rsna_preproc.MAXS, rsna_preproc.IMG, rsna_preproc.IMG), dtype=np.uint8)
            used_series = set()
            idx = 0

            for plane, fluid, k in rsna_preproc.SLOTS:
                r = rsna_preproc.pick_series_for_slot(s_rows, plane, fluid, used_series)
                if r is None:
                    idx += k
                    continue
                ser_id = str(r["SeriesInstanceUID"])
                used_series.add(ser_id)
                files = file_map.get(ser_id, [])
                if not files:
                    idx += k
                    continue

                hdrs = []
                for f in files:
                    pos, ps, ok = rsna_preproc.parse_dicom_header(z.read(f))
                    if ok: hdrs.append((pos, f, ps))
                hdrs.sort(key=lambda x: x[0])
                sorted_files = [(f, ps) for _, f, ps in hdrs]
                if not sorted_files:
                    idx += k
                    continue

                n = len(sorted_files)
                med_ps = float(np.median([ps for _, ps in sorted_files])) if sorted_files else 0.5
                picks = rsna_preproc.pick_slice_indices(n, k)

                arrs, pss = [], []
                for p in picks:
                    fp, ps = sorted_files[min(p, n - 1)]
                    try:
                        arrs.append(rsna_preproc.decode_dicom_pixel(z.read(fp)))
                        pss.append(ps if ps > 0 else med_ps)
                    except Exception:
                        arrs.append(None)
                        pss.append(med_ps)

                slot_pack = rsna_preproc.normalize_and_pack_slot(arrs, pss, target_res=rsna_preproc.IMG)
                vol[idx:idx + k] = slot_pack
                idx += k

            mask = (vol.reshape(rsna_preproc.MAXS, -1).sum(axis=1) > 0).astype(np.uint8)
            vols.append(vol)
            masks.append(mask)
            if (i + 1) % 10 == 0 or i < 3:
                print(f"  Study {i+1:2d}/58 [{study_id[:20]}...]: {time.time()-t_s:.2f}s | filled = {mask.sum()}/96")

    vols = np.array(vols)
    masks = np.array(masks)
    np.savez(CACHE_58, vols=vols, masks=masks)
    print(f"  All 58 studies preprocessed in {time.time()-t0:.2f}s and cached to {CACHE_58}")

# 4. Load CoAtNet-96 model
print("\n[4] Loading CoAtNet-96 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bb = train_knee.build_backbone("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", pretrained=False)
model = train_knee.RaptorClassifier(bb, F_dim=bb.num_features).to(device)
model.win_chunk = 6
ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"], strict=True)
model = model.eval()

# 5. Run inference in FP32 & BF16
mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std_t  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

def get_study_windows_tensor(vol, mask):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3: valid = np.arange(3)
    cs = np.linspace(valid.min(), valid.max(), 94).round().astype(int)
    tiles = []
    for c in cs:
        c = max(1, min(c, 94))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        tiles.append(tri)
    wins = np.stack(tiles, 0)
    x = torch.from_numpy(wins).to(device)
    x = (x - mean_t) / std_t
    return x.unsqueeze(0)

print("\n[5] Running inference across all 58 gold studies...")
raw_preds = []
t0 = time.time()
for i in range(N):
    x = get_study_windows_tensor(vols[i], masks[i])
    with torch.inference_mode():
        logits = model(x)
        prob = torch.sigmoid(logits.float()).cpu().numpy()[0]
    raw_preds.append(prob)
    if (i + 1) % 15 == 0:
        print(f"  Processed {i+1}/58 studies in {time.time()-t0:.1f}s...")

raw_preds = np.array(raw_preds)
print(f"  Inference complete in {time.time()-t0:.1f}s ({((time.time()-t0)/N)*1000:.1f} ms/study)")

# 6. Compute per-finding AUC and Macro AUC
auc_raw = []
auc_offline = []
for j in range(12):
    auc_raw.append(roc_auc_score(gold_truth[:, j], raw_preds[:, j]))
    auc_offline.append(roc_auc_score(gold_truth[:, j], gold_offline_preds[:, j]))

macro_raw = np.mean(auc_raw)
macro_offline = np.mean(auc_offline)

print("\n" + "="*85)
print(f"{'#':<3} {'Finding':<22} {'Raw-DICOM AUC':>18} {'Offline AUC':>18} {'Delta (Δ)':>15}")
print("="*85)
for j in range(12):
    diff = auc_raw[j] - auc_offline[j]
    print(f"{j:<3} {FINDINGS[j]:<22} {auc_raw[j]:18.4f} {auc_offline[j]:18.4f} {diff:+15.4f}")

print("-"*85)
print(f"{'':<3} {'MACRO AUC (12 cols)':<22} {macro_raw:18.4f} {macro_offline:18.4f} {macro_raw-macro_offline:+15.4f}")
print("="*85)

corr = np.mean([pearsonr(raw_preds[:, j], gold_offline_preds[:, j])[0] for j in range(12)])
print(f"\nMean Pearson correlation across all 58 studies: r = {corr:.6f}")
