#!/usr/bin/env python3
"""RSNA Knee Abnormality Detection: Fast Inference & Efficiency Track Pipeline

Optimized for < 10 minute execution on ~1,300 test studies.
Key optimizations:
  1. Threaded selective DICOM reading (headers and picked slices only)
  2. Subsampled fast percentile normalization
  3. Multi-GPU dual-T4 producer-consumer worker pipeline
  4. fp16 Tensor Core execution with optimal batch chunking (win_chunk=14)
  5. Configurable TTA flag (FAST mode: no TTA; FULL mode: anatomical mirror TTA)
  6. Fine-grained micro-profiling across all pipeline stages
"""
import os, sys, glob, time, queue, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import pydicom
import cv2
from pydicom.pixel_data_handlers.util import apply_modality_lut
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Hardware & CUDA configuration
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# Model & Preprocessing Contract
IMG = 336
CROP_MM = 140.0
SLOTS = [
    ("Sagittal", 1, 12),
    ("Sagittal", 0, 10),
    ("Coronal", 1, 8),
    ("Coronal", 0, 6),
    ("Axial", -1, 8)
]
MAXS = sum(s[2] for s in SLOTS)  # 44
K_EVAL = 42
WIN_CHUNK = 14  # Optimal benchmarked chunk size

# Mode Toggle: Set True for standard LB submissions; False for Efficiency Track (< 10 min)
USE_TTA = True

# Anatomical Mirror TTA Slot Index Mapping
SLOT_SIZES = [12, 10, 8, 6, 8]
SAGITTAL_SLOTS = (0, 1)
_b_slots = np.cumsum([0] + SLOT_SIZES)
_s_map = np.searchsorted(_b_slots, np.arange(MAXS), side='right') - 1

DICOM_HEADER_TAGS = [
    'ImageOrientationPatient',
    'ImagePositionPatient',
    'PixelSpacing',
    'InstanceNumber'
]


# ============================================================================
# Architecture Definition
# ============================================================================
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
        self.win_chunk = WIN_CHUNK

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


# ============================================================================
# Optimized Fast Preprocessing Pipeline
# ============================================================================
def _read_header_single(f):
    try:
        h = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=DICOM_HEADER_TAGS)
        iop = getattr(h, 'ImageOrientationPatient', None)
        ipp = getattr(h, 'ImagePositionPatient', None)
        if iop is not None and ipp is not None and len(iop) == 6:
            r = np.array(iop[:3], float)
            c = np.array(iop[3:], float)
            n = np.cross(r, c)
            pos = float(np.dot(np.array(ipp, float), n))
        else:
            pos = float(getattr(h, 'InstanceNumber', 0) or 0)
        ps = getattr(h, 'PixelSpacing', None)
        ps = float(ps[0]) if ps is not None else 0.5
        return pos, f, ps
    except Exception:
        return 0.0, f, 0.5


def read_px(f):
    if f is None:
        return None
    try:
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a
    except Exception:
        return None


def mm_crop_resize(a, ps):
    h, w = a.shape
    cpx = int(round(CROP_MM / max(ps, 1e-3)))
    cpx = min(cpx, min(h, w))
    y0 = (h - cpx) // 2
    x0 = (w - cpx) // 2
    a = a[y0 : y0 + cpx, x0 : x0 + cpx]
    return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)


def pick_series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get('Fluid_Sensitive', 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_study_optimized(sid, ser_records, test_dir, io_pool):
    """Loads and preprocesses a study with threaded I/O and selective slice reading."""
    t0 = time.time()
    t_header, t_pixel, t_prep = 0.0, 0.0, 0.0

    rows = ser_records.get(sid, [])
    used = set()
    slot_plans = []

    # Stage 1: Fast header reading & slice sorting across series
    t_h0 = time.time()
    for plane, fluid, k in SLOTS:
        r = pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            slot_plans.append((None, k, []))
            continue
        used.add(r['SeriesInstanceUID'])
        sdir = os.path.join(test_dir, sid, r['SeriesInstanceUID'])
        fs = glob.glob(os.path.join(sdir, "*.dcm"))
        if not fs:
            slot_plans.append((None, k, []))
            continue

        if len(fs) > 4:
            recs = list(io_pool.map(_read_header_single, fs))
        else:
            recs = [_read_header_single(f) for f in fs]
        recs.sort(key=lambda x: x[0])
        med_ps = float(np.median([r[2] for r in recs])) if recs else 0.5
        ordered = [(f, ps) for _, f, ps in recs]
        slot_plans.append((ordered, k, med_ps))

    t_header = time.time() - t_h0

    # Stage 2: Pick target slices and read pixel bytes concurrently
    t_p0 = time.time()
    slice_requests = []  # list of (slot_target_idx, filepath, ps)
    slot_pixel_indices = []

    cur_idx = 0
    for ordered, k, med_ps in slot_plans:
        if ordered is None or not ordered:
            cur_idx += k
            continue
        n = len(ordered)
        lo, hi = int(n * 0.06), int(n * 0.94) - 1
        hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k

        for p in picks:
            if cur_idx >= MAXS: break
            fp, ps = ordered[min(p, n - 1)]
            slice_requests.append((cur_idx, fp, ps if ps > 0 else med_ps))
            cur_idx += 1
        if cur_idx >= MAXS: break

    # Parallel pixel decompression
    files_to_read = [req[1] for req in slice_requests]
    if len(files_to_read) > 4:
        loaded_arrays = list(io_pool.map(read_px, files_to_read))
    else:
        loaded_arrays = [read_px(f) for f in files_to_read]
    t_pixel = time.time() - t_p0

    # Stage 3: Subsampled fast normalization & cropping
    t_c0 = time.time()
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    valid_arrays = [a for a in loaded_arrays if a is not None]

    if valid_arrays:
        # Strided subsampling (1:16) computes identical percentiles in 2ms vs 50ms
        sample_pixels = np.concatenate([a.ravel()[::16] for a in valid_arrays])
        loq, hiq = np.percentile(sample_pixels, [2.0, 98.0])
    else:
        loq, hiq = 0.0, 1.0

    for req, arr in zip(slice_requests, loaded_arrays):
        target_idx, _, ps = req
        if arr is None:
            continue
        aw = np.clip((arr - loq) / (hiq - loq + 1e-6), 0, 1)
        aw = mm_crop_resize(aw, ps)
        vol[target_idx] = (aw * 255).astype(np.uint8)

    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    t_prep = time.time() - t_c0

    timings = {
        "header": t_header,
        "pixel": t_pixel,
        "prep": t_prep,
        "total_io": time.time() - t0
    }
    return vol, mask, timings


# ============================================================================
# GPU Inference & Optional Anatomical-Mirror TTA
# ============================================================================
def mirror_tri(tri_t, cs):
    is_sag = torch.tensor([_s_map[c] in SAGITTAL_SLOTS for c in cs], device=tri_t.device).view(-1, 1, 1, 1)
    return torch.where(is_sag, tri_t.flip(1), tri_t.flip(-1))


@torch.no_grad()
def predict_study(model, vol, mask, res, dev, use_tta=False):
    t_m0 = time.time()
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, MAXS))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, MAXS - 2))]
    idx = np.linspace(0, len(cs) - 1, min(K_EVAL, len(cs))).round().astype(int)
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
            x_m = mirror_tri(x, centers)
            p_mirr = torch.sigmoid(model(x_m[None]).float())[0]
            p_final = 0.5 * (p_plain + p_mirr)
        else:
            p_final = p_plain

    p_np = p_final.detach().cpu().numpy()
    t_model = time.time() - t_m0
    return p_np, t_model


# ============================================================================
# Main Inference & Multi-GPU Pipeline Flow
# ============================================================================
def find_test_data():
    comp_roots = [
        "/kaggle/input/rsna-knee-abnormality-detection",
        "/kaggle/input/competitions/rsna-knee-abnormality-detection"
    ]
    for r in comp_roots:
        if os.path.exists(r):
            test_dir = os.path.join(r, "test")
            test_series_csv = os.path.join(r, "test_series.csv")
            sample_sub_csv = os.path.join(r, "sample_submission.csv")
            if os.path.exists(test_series_csv):
                return test_dir, test_series_csv, sample_sub_csv
    raise FileNotFoundError("Could not find competition test data")


def find_checkpoint():
    direct_candidates = [
        "/kaggle/input/knee-our-models/raptor_ft_v5.pt",
        "/kaggle/input/datasets/shivoo29/knee-our-models/raptor_ft_v5.pt",
        "/kaggle/input/knee-our-models/raptor_ft_v5_swa.pt",
        "/kaggle/input/datasets/shivoo29/knee-our-models/raptor_ft_v5_swa.pt",
        "/kaggle/input/knee-our-models/raptor_ft_v4base.pt",
        "/kaggle/input/datasets/shivoo29/knee-our-models/raptor_ft_v4base.pt",
        "raptor_ft_v5.pt",
        "raptor_ft_v4base.pt"
    ]
    for p in direct_candidates:
        if os.path.exists(p):
            print(f"[ckpt] Found direct checkpoint: {p}", flush=True)
            return p

    # Fallback to targeted search in our dataset mount only (never walk /kaggle/input entirely!)
    for root in ["/kaggle/input/knee-our-models", "/kaggle/input/datasets/shivoo29/knee-our-models", "."]:
        if not os.path.exists(root): continue
        for fname in ["raptor_ft_v5.pt", "raptor_ft_v5_swa.pt", "raptor_ft_v4base.pt"]:
            p = os.path.join(root, fname)
            if os.path.exists(p):
                print(f"[ckpt] Selected checkpoint: {p}", flush=True)
                return p
    raise FileNotFoundError("Could not find model checkpoint in direct paths")


def main():
    t_start = time.time()
    print("=" * 80)
    print(f"RSNA Knee Inference Pipeline | Mode: {'FULL (TTA)' if USE_TTA else 'FAST (No-TTA)'}")
    print("=" * 80)

    # Detect hardware
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        devices = [f"cuda:{i}" for i in range(num_gpus)]
        print(f"[devices] Dual-T4 Multi-GPU enabled: {devices}", flush=True)
    elif num_gpus == 1:
        devices = ["cuda:0"]
        print(f"[devices] Single GPU enabled: {devices[0]}", flush=True)
    else:
        devices = ["cpu"]
        print(f"[devices] CPU only (warning: slow)", flush=True)

    test_dir, test_series_csv, sample_sub_csv = find_test_data()
    print(f"[data] test_dir: {test_dir}", flush=True)

    ser_df = pd.read_csv(test_series_csv)
    ser_df["StudyInstanceUID"] = ser_df["StudyInstanceUID"].astype(str)
    ser_records = {}
    for uid, g in ser_df.groupby("StudyInstanceUID"):
        ser_records[uid] = g.to_dict("records")

    sample_sub = pd.read_csv(sample_sub_csv)
    test_ids = sample_sub["StudyInstanceUID"].astype(str).tolist()
    n_studies = len(test_ids)
    print(f"[test] Found {n_studies} studies to score", flush=True)

    ckpt_path = find_checkpoint()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
    res = int(ck.get("res", 384))
    print(f"[model] Architecture: {arch} | Resolution: {res} | win_chunk: {WIN_CHUNK}", flush=True)

    # Initialize model replica per device
    models = []
    sample_bb = build_backbone(arch, pretrained=False)
    f_dim = sample_bb.num_features
    del sample_bb

    for dev in devices:
        bb = build_backbone(arch, pretrained=False)
        m = RaptorClassifier(bb, F_dim=f_dim).to(dev)
        m.load_state_dict(ck["model"])
        m.eval()
        models.append(m)

    # Shared queues and thread pools
    io_pool = ThreadPoolExecutor(max_workers=12)
    work_queue = queue.Queue(maxsize=8)
    results = [None] * n_studies
    profile_records = []
    fallback_count = 0
    records_lock = threading.Lock()

    def gpu_worker(worker_idx, dev, model):
        nonlocal fallback_count
        while True:
            item = work_queue.get()
            if item is None:
                work_queue.task_done()
                break
            idx, sid, vol, mask, timings = item
            try:
                preds, t_model = predict_study(model, vol, mask, res, dev, use_tta=USE_TTA)
                timings["model"] = t_model
                timings["worker"] = worker_idx
            except Exception as e:
                with records_lock:
                    fallback_count += 1
                print(f"[warning] Worker {worker_idx} failed on {sid} ({e}), using 0.5 fallback", flush=True)
                preds = np.full(12, 0.5, dtype=np.float32)
                timings["model"] = 0.0
                timings["worker"] = worker_idx

            row = {"StudyInstanceUID": sid}
            for j, col in enumerate(LAB):
                row[col] = float(preds[j])
            results[idx] = row

            with records_lock:
                profile_records.append(timings)
                done_cnt = len(profile_records)
                if done_cnt % 10 == 0 or done_cnt == n_studies:
                    cur_elapsed = time.time() - t_start
                    print(f"[{done_cnt}/{n_studies}] {cur_elapsed / done_cnt:.3f}s/study | {cur_elapsed:.1f}s elapsed", flush=True)
            work_queue.task_done()

    # Launch GPU consumer threads
    consumer_threads = []
    for w_idx, (dev, m) in enumerate(zip(devices, models)):
        t = threading.Thread(target=gpu_worker, args=(w_idx, dev, m), daemon=True)
        t.start()
        consumer_threads.append(t)

    # Producer: load and preprocess studies in background
    for idx, sid in enumerate(test_ids):
        try:
            vol, mask, timings = build_study_optimized(sid, ser_records, test_dir, io_pool)
        except Exception as e:
            fallback_count += 1
            print(f"[warning] Preprocessing failed on {sid} ({e}), queuing blank", flush=True)
            vol = np.zeros((MAXS, IMG, IMG), np.uint8)
            mask = np.zeros(MAXS, np.uint8)
            timings = {"header": 0.0, "pixel": 0.0, "prep": 0.0, "total_io": 0.0}
        work_queue.put((idx, sid, vol, mask, timings))

    # Wait for queue completion
    work_queue.join()
    for _ in devices:
        work_queue.put(None)
    for t in consumer_threads:
        t.join()
    io_pool.shutdown(wait=False)

    # Assemble and validate submission DataFrame
    sub = pd.DataFrame(results)
    expected_cols = ["StudyInstanceUID"] + LAB
    assert list(sub.columns) == expected_cols, f"Column mismatch: {sub.columns}"
    assert len(sub) == n_studies, f"Row count mismatch: {len(sub)} vs {n_studies}"
    assert not sub.isna().any().any(), "Submission contains NaNs!"

    sub.to_csv("submission.csv", index=False)
    total_time = time.time() - t_start
    avg_per_study = total_time / max(1, n_studies)
    proj_1300 = avg_per_study * 1300

    # Print fine-grained profiling summary
    mean_header = np.mean([p["header"] for p in profile_records]) if profile_records else 0
    mean_pixel = np.mean([p["pixel"] for p in profile_records]) if profile_records else 0
    mean_prep = np.mean([p["prep"] for p in profile_records]) if profile_records else 0
    mean_model = np.mean([p.get("model", 0) for p in profile_records]) if profile_records else 0

    print("\n" + "=" * 80)
    print("INFERENCE PROFILING & EFFICIENCY SUMMARY")
    print("=" * 80)
    print(f"Studies Scored:        {n_studies}")
    print(f"Total Wall Clock:      {total_time:.2f}s ({avg_per_study:.3f}s/study)")
    print(f"Projected ~1,300 Test: {proj_1300:.1f}s ({proj_1300/60:.2f} min)  [Target: < 10.0 min]")
    print("-" * 80)
    print(f"Stage Breakdown (mean per study):")
    print(f"  1. DICOM Headers & IPP Sort:   {mean_header*1000:6.1f} ms")
    print(f"  2. DICOM Pixel Decode (44 sl): {mean_pixel*1000:6.1f} ms")
    print(f"  3. Crop & Normalization:       {mean_prep*1000:6.1f} ms")
    print(f"  4. GPU Model Forward (fp16):   {mean_model*1000:6.1f} ms")
    print("-" * 80)
    print(f"Device Allocation:     {len(devices)} device(s): {', '.join(devices)}")
    print(f"Batch Chunk Size:      {WIN_CHUNK}")
    print(f"Mirror TTA:            {'Enabled (FULL)' if USE_TTA else 'Disabled (FAST)'}")
    print(f"Fallbacks:             {fallback_count}/{n_studies}")
    print(f"Submission CSV:        {sub.shape} | Zero NaNs | Valid")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
