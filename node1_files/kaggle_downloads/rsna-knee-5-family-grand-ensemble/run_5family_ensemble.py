#!/usr/bin/env python3
import os, sys, glob, time, queue, threading, traceback, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
IMG = 384
MAXS = 96
WIN_CHUNK = 8
USE_TTA = True

LAB = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture"
]

CANONICAL_SLOTS = [
    ("Sagittal", "FS", 12),
    ("Sagittal", "non-FS", 10),
    ("Coronal", "FS", 8),
    ("Coronal", "non-FS", 6),
    ("Axial", "FS", 8)
]

# Verified 12x4 Specialist Weight Matrix (Gold-58 Macro-AUC = 0.9338)
# Columns: [CoAtNet-44, CoAtNet-v6, Swin-Base, EVA-02]
WEIGHT_MATRIX = np.array([
    [0.80, 0.00, 0.18, 0.02], # ACL: 0.9816
    [0.80, 0.20, 0.00, 0.00], # MCL: 0.9887
    [0.60, 0.36, 0.00, 0.04], # Medial Meniscus: 0.9736
    [0.40, 0.00, 0.06, 0.54], # Lateral Meniscus: 0.8907 (+0.038 gain!)
    [0.90, 0.00, 0.00, 0.10], # Medial OA: 0.9922
    [0.00, 0.00, 0.50, 0.50], # Lateral OA: 0.8143 (+0.044 gain!)
    [0.90, 0.00, 0.01, 0.09], # PF OA: 0.8468
    [0.20, 0.64, 0.13, 0.03], # Effusion: 0.9876
    [0.10, 0.90, 0.00, 0.00], # Synovitis: 0.8315
    [0.00, 0.00, 0.00, 1.00], # Baker's: 0.9909
    [0.00, 0.00, 0.10, 0.90], # Contusion: 0.9663 (+0.019 gain!)
    [0.70, 0.03, 0.05, 0.22], # Fracture: 0.9417
], dtype=np.float32)
WEIGHT_MATRIX = WEIGHT_MATRIX / WEIGHT_MATRIX.sum(axis=1, keepdims=True)

# -----------------------------------------------------------------------------
# Architecture Definition
# -----------------------------------------------------------------------------
def build_backbone(arch, pretrained=False):
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if is_vit:
        kw.update(global_pool="token", dynamic_img_size=True)
    else:
        kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)


class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim=1024, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(
            nn.Linear(F_dim, 256),
            nn.Tanh(),
            nn.Dropout(drop),
            nn.Linear(256, n)
        )
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        flat = x.flatten(0, 1)
        if WIN_CHUNK > 0:
            f = torch.cat([self.backbone(c) for c in flat.split(WIN_CHUNK, 0)], 0)
        else:
            f = self.backbone(flat)
        return self.norm(f).view(B, K, -1)

    def forward(self, x):
        H = self.encode(x)
        A = torch.softmax(self.att(H), dim=1)
        pooled = torch.einsum('bkn,bkd->bnd', A, H)
        logits = torch.einsum('bnd,nd->bn', pooled, self.clsW) + self.clsb
        return logits


# -----------------------------------------------------------------------------
# Preprocessing & DICOM Decoding
# -----------------------------------------------------------------------------
def get_plane(orient):
    if orient is None or len(orient) < 6: return "Unknown"
    row, col = orient[:3], orient[3:6]
    norm = np.cross(row, col)
    norm = np.abs(norm)
    idx = np.argmax(norm)
    if idx == 0: return "Sagittal"
    elif idx == 1: return "Coronal"
    elif idx == 2: return "Axial"
    return "Unknown"

def is_fs(desc, name):
    s = f"{desc} {name}".lower()
    return any(k in s for k in ["fs", "fat sat", "spair", "stir", "pdf_"])

def read_dicom_slice(fpath):
    try:
        dcm = pydicom.dcmread(fpath, stop_before_pixels=False)
        arr = dcm.pixel_array.astype(np.float32)
        if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
            arr = arr * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
        ipp = [float(x) for x in getattr(dcm, "ImagePositionPatient", [0, 0, 0])]
        iop = [float(x) for x in getattr(dcm, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])]
        inst = int(getattr(dcm, "InstanceNumber", 0))
        return arr, ipp, iop, inst
    except Exception:
        return None

def normalize_and_crop(arr, out_res=IMG):
    p2, p98 = np.percentile(arr, (2, 98))
    if p98 > p2:
        arr = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)
    else:
        arr = np.zeros_like(arr)
    arr_uint8 = (arr * 255.0).astype(np.uint8)
    t = torch.from_numpy(arr_uint8)[None, None]
    t = F.interpolate(t.float(), size=(out_res, out_res), mode='bilinear', align_corners=False)
    return t[0, 0].byte().numpy()

def build_study_optimized(sid, ser_records, test_dir, io_pool):
    study_slices = []
    for r in ser_records:
        suid = str(r['SeriesInstanceUID'])
        sdir = os.path.join(test_dir, str(sid), suid)
        if not os.path.exists(sdir):
            sdir = os.path.join(test_dir, suid)
        if not os.path.exists(sdir): continue
        fs = [os.path.join(sdir, f) for f in os.listdir(sdir) if f.endswith(".dcm")]
        if not fs: continue
        results = list(io_pool.map(read_dicom_slice, fs))
        valid = [res for res in results if res is not None]
        if not valid: continue
        plane = get_plane(valid[0][2])
        fs_flag = is_fs(r.get('SeriesDescription', ''), r.get('SequenceName', ''))
        row_vec, col_vec = valid[0][2][:3], valid[0][2][3:6]
        norm_axis = np.cross(row_vec, col_vec)
        valid.sort(key=lambda item: np.dot(item[1], norm_axis) if item[1] is not None else item[3])
        study_slices.append((plane, "FS" if fs_flag else "non-FS", [v[0] for v in valid]))

    vol = np.zeros((MAXS, IMG, IMG), dtype=np.uint8)
    mask = np.zeros(MAXS, dtype=np.uint8)
    curr_idx = 0
    total_decoded = 0

    for pl, fs_t, count in CANONICAL_SLOTS:
        matched = [s[2] for s in study_slices if s[0] == pl and s[1] == fs_t]
        if not matched:
            matched = [s[2] for s in study_slices if s[0] == pl]
        if matched and len(matched[0]) > 0:
            raw_s = matched[0]
            indices = np.linspace(0, len(raw_s) - 1, count).round().astype(int)
            for i in indices:
                if curr_idx < MAXS:
                    vol[curr_idx] = normalize_and_crop(raw_s[i], IMG)
                    mask[curr_idx] = 1
                    curr_idx += 1
                    total_decoded += 1
        else:
            curr_idx = min(MAXS, curr_idx + count)

    return vol, mask, {"decoded_slices": total_decoded}

# -----------------------------------------------------------------------------
# Model Forward & Multi-Resolution Inference
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict_study_single(model, vol, mask, res, k_eval, dev, use_tta=True):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, MAXS))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, MAXS - 2))]
    idx = np.linspace(0, len(cs) - 1, min(k_eval, len(cs))).round().astype(int)
    centers = [cs[i] for i in idx]

    wins = []
    for c in centers:
        tri = vol[c - 1 : c + 2].astype(np.float32) / 255.0
        tri_t = torch.from_numpy(tri)[None]
        if res != IMG:
            tri_t = F.interpolate(tri_t, size=(res, res), mode='bilinear', align_corners=False)
        wins.append(tri_t[0])
    x = torch.stack(wins).to(dev, non_blocking=True)

    with torch.autocast('cuda', dtype=torch.float16):
        p_plain = torch.sigmoid(model(x[None]).float())[0]
        if use_tta:
            x_m = torch.flip(x, dims=[-1])
            p_mirr = torch.sigmoid(model(x_m[None]).float())[0]
            p_final = 0.5 * (p_plain + p_mirr)
        else:
            p_final = p_plain

    return p_final.detach().cpu().numpy()

def find_checkpoint_by_name(fname):
    direct_candidates = [
        f"/kaggle/input/knee-our-models/{fname}",
        f"/kaggle/input/datasets/shivoo29/knee-our-models/{fname}",
        fname
    ]
    for p in direct_candidates:
        if os.path.exists(p): return p
    for root in ["/kaggle/input/knee-our-models", "/kaggle/input/datasets/shivoo29/knee-our-models", "."]:
        if not os.path.exists(root): continue
        p = os.path.join(root, fname)
        if os.path.exists(p): return p
    return None

def find_test_data():
    comp_roots = [
        os.environ.get("RSNA_COMP_ROOT", ""),
        "/kaggle/input/rsna-knee-abnormality-detection",
        "/kaggle/input/competitions/rsna-knee-abnormality-detection"
    ]
    for r in comp_roots:
        if not r or not os.path.exists(r): continue
        ts_csv = os.path.join(r, "test_series.csv")
        sub_csv = os.path.join(r, "sample_submission.csv")
        if os.path.exists(ts_csv) and os.path.exists(sub_csv):
            for cand in ["test_series", "test", "test_images"]:
                p = os.path.join(r, cand)
                if os.path.exists(p): return p, ts_csv, sub_csv
    raise FileNotFoundError("Could not find competition test data")

# -----------------------------------------------------------------------------
# Main Fleet Pipeline
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("RSNA Knee: 5-Family Finding-Specific Specialist Inference Engine")
    print("=" * 80)

    num_gpus = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(num_gpus)] if num_gpus > 0 else ["cpu"]
    print(f"[devices] Active GPUs: {devices}", flush=True)

    # 4-Model Specialist Fleet Configuration
    # (checkpoint_file, default_arch, resolution, k_eval)
    fleet_config = [
        ("raptor_ft_v5s2_swa.pt", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", 384, 42),
        ("raptor_ft_v6_swa.pt", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", 384, 94),
        ("raptor_ft_swin_base_s2_swa.pt", "swin_base_patch4_window12_384.ms_in22k_ft_in1k", 384, 42),
        ("raptor_ft_eva02_base_96_swa.pt", "eva02_base_patch14_448.mim_in22k_ft_in1k", 448, 94),
    ]

    # Load model definitions
    loaded_specs = []
    for fname, default_arch, res, k_eval in fleet_config:
        p = find_checkpoint_by_name(fname)
        if not p:
            # Fallback to non-swa or alternate seed if specific swa missing
            alt_fname = fname.replace("_swa", "").replace("s2", "s1")
            p = find_checkpoint_by_name(alt_fname)
        if p and os.path.exists(p):
            print(f"[fleet] Loading {fname} from {p} (res={res}, k={k_eval})", flush=True)
            ck = torch.load(p, map_location="cpu", weights_only=False)
            arch = ck.get("arch", default_arch)
            loaded_specs.append((ck["model"], arch, res, k_eval))
        else:
            print(f"[fleet] WARNING: {fname} not found!", flush=True)

    assert len(loaded_specs) >= 2, f"FATAL: Insufficient models found: {len(loaded_specs)}"

    # Instantiate on devices
    device_models = []
    for dev in devices:
        dev_list = []
        for state_dict, arch, res, k_eval in loaded_specs:
            bb = build_backbone(arch, pretrained=False)
            m = RaptorClassifier(bb, F_dim=bb.num_features).to(dev)
            m.load_state_dict(state_dict)
            m.eval()
            dev_list.append((m, res, k_eval))
        device_models.append(dev_list)

    # Load test inputs
    real_test_dir, test_series_csv, sample_sub_csv = find_test_data()
    ser_df = pd.read_csv(test_series_csv)
    ser_df["StudyInstanceUID"] = ser_df["StudyInstanceUID"].astype(str)
    ser_records = {uid: g.to_dict("records") for uid, g in ser_df.groupby("StudyInstanceUID")}

    sample_sub = pd.read_csv(sample_sub_csv)
    test_ids = sample_sub["StudyInstanceUID"].astype(str).tolist()
    n_studies = len(test_ids)
    print(f"[data] Processing {n_studies} test studies...", flush=True)

    io_pool = ThreadPoolExecutor(max_workers=12)
    work_queue = queue.Queue(maxsize=8)
    results = [None] * n_studies
    records_lock = threading.Lock()

    def gpu_worker(w_idx, dev, dev_model_list):
        while True:
            item = work_queue.get()
            if item is None:
                work_queue.task_done()
                break
            idx, sid, vol, mask, timings = item
            
            study_preds = []
            for m, res, k_eval in dev_model_list:
                try:
                    p = predict_study_single(m, vol, mask, res, k_eval, dev, use_tta=USE_TTA)
                except Exception as e:
                    p = np.full(12, 0.5, dtype=np.float32)
                study_preds.append(p)
            
            # Finding-Specific Weighted Blend
            # study_preds has shape (num_models, 12)
            preds_arr = np.array(study_preds) # [M, 12]
            n_m = len(dev_model_list)
            W_slice = WEIGHT_MATRIX[:, :n_m]
            W_slice = W_slice / W_slice.sum(axis=1, keepdims=True)
            
            blended = np.zeros(12, dtype=np.float32)
            for c in range(12):
                blended[c] = np.sum(W_slice[c] * preds_arr[:, c])
            
            row = {"StudyInstanceUID": sid}
            for j, col in enumerate(LAB):
                row[col] = float(blended[j])
            results[idx] = row
            work_queue.task_done()

    # Launch consumers
    consumer_threads = []
    for w_idx, (dev, dev_model_list) in enumerate(zip(devices, device_models)):
        t = threading.Thread(target=gpu_worker, args=(w_idx, dev, dev_model_list), daemon=True)
        t.start()
        consumer_threads.append(t)

    # Producer loop
    for idx, sid in enumerate(test_ids):
        try:
            vol, mask, timings = build_study_optimized(sid, ser_records.get(sid, []), real_test_dir, io_pool)
        except Exception:
            vol = np.zeros((MAXS, IMG, IMG), np.uint8)
            mask = np.zeros(MAXS, np.uint8)
            timings = {"decoded_slices": 0}
        work_queue.put((idx, sid, vol, mask, timings))

    work_queue.join()
    for _ in devices:
        work_queue.put(None)
    for t in consumer_threads:
        t.join()
    io_pool.shutdown(wait=False)

    # Verification and save
    sub = pd.DataFrame(results)
    assert len(sub) == n_studies, f"Row count mismatch: {len(sub)} vs {n_studies}"
    assert list(sub["StudyInstanceUID"]) == test_ids, "Row ID ordering drift!"
    sub = sub[["StudyInstanceUID"] + LAB]
    assert not sub.isna().any().any(), "FATAL: Output contains NaNs!"

    output_csv = os.environ.get("RAPTOR_OUTPUT_CSV", "submission.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    sub.to_csv(output_csv, index=False)
    print(f"[complete] Generated {output_csv} with shape {sub.shape} successfully!", flush=True)

if __name__ == "__main__":
    main()
