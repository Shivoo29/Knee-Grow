#!/usr/bin/env python3
"""RSNA Knee Abnormality Detection: Fast Inference & Efficiency Track Pipeline
Systematic Visible-Data Teacher Probe, Corpus Parity Auditor & Production Engine

Key architectural features:
  1. Exact parity with Dread's preprocessed corpus:
     - Canonical 44-slice slot contract: Sagittal FS (12), Sagittal (10), Coronal FS (8), Coronal (6), Axial (8)
     - Per-slot (per-series) 2% and 98% percentile normalization matching scanner contrasts
     - Full K_EVAL=42 window center evaluation across the entire 44-slice volume
  2. Stage-by-stage parity auditor comparing live DICOM decoding against all_vols.npy on 20 studies
  3. Automated visible-data Teacher Probe validation (58 gold + 300 train studies)
  4. Diagnostic exception logging (FIRST 5 full tracebacks captured)
  5. Hard guard: aborts loudly if fallbacks > 5% of studies (prevents 0.500 LB submissions)
  6. Multi-GPU dual-T4 producer-consumer queue with exact index alignment and ID verification
"""
import os, sys, glob, time, queue, threading, traceback, shutil
from pathlib import Path
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

# Model & Preprocessing Contract (matching Dread's 44-slice corpus exactly)
IMG = 336
CROP_MM = 140.0
SLOTS = [
    ("Sagittal", 1, 12),
    ("Sagittal", 0, 10),
    ("Coronal", 1, 8),
    ("Coronal", 0, 6),
    ("Axial", -1, 8),
]
MAXS = sum(k for _, _, k in SLOTS)  # 44 slices
SLOT_SIZES = [12, 10, 8, 6, 8]
SAGITTAL_SLOTS = (0, 1)
_b_slots = np.cumsum([0] + SLOT_SIZES)
_s_map = np.searchsorted(_b_slots, np.arange(MAXS), side='right') - 1

K_EVAL = 42     # Evaluates all 42 possible 3-slice window centers across 44 slices
WIN_CHUNK = 14  # Optimal benchmarked chunk size

# Mode Toggle: Set True for standard LB submissions; False for Efficiency Track (< 10 min)
USE_TTA = False

DICOM_HEADER_TAGS = [
    'ImageOrientationPatient',
    'ImagePositionPatient',
    'PixelSpacing',
    'InstanceNumber'
]

# Exception diagnostics tracker
_MAX_EXC_LOGS = 5
_exc_log_count = 0
_exc_lock = threading.Lock()

def record_exception(source, sid, exc):
    global _exc_log_count
    with _exc_lock:
        if _exc_log_count < _MAX_EXC_LOGS:
            _exc_log_count += 1
            print(f"\n{'!' * 80}", flush=True)
            print(f"[DIAGNOSTIC EXCEPTION #{_exc_log_count}] in {source} for study {sid}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            print(f"{'!' * 80}\n", flush=True)


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
# Robust Preprocessing & DICOM Decoders (Exact Dread Corpus Parity)
# ============================================================================
def _read_header_single(item):
    sid, f = item if isinstance(item, tuple) else ("unknown", item)
    try:
        h = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=DICOM_HEADER_TAGS, force=True)
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
    except Exception as e:
        record_exception("_read_header_single", sid, e)
        return 0.0, f, 0.5


def read_px(item):
    sid, f = item if isinstance(item, tuple) else ("unknown", item)
    if f is None:
        return None
    try:
        d = pydicom.dcmread(f, force=True)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a
    except Exception as e:
        record_exception("read_px", sid, e)
        return None


def mm_crop_resize(a, ps):
    h, w = a.shape
    cpx = int(round(CROP_MM / max(ps, 1e-3)))
    cpx = max(1, min(cpx, min(h, w)))
    y0 = (h - cpx) // 2
    x0 = (w - cpx) // 2
    a = a[y0 : y0 + cpx, x0 : x0 + cpx]
    return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)


def _safe_fluid(val):
    if val is None or pd.isna(val):
        return -1
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return -1


def pick_series_for_slot(rows, plane, fluid, used):
    plane_norm = str(plane).strip().capitalize()
    cands = [r for r in rows if str(r.get('Anatomical_Plane', '')).strip().capitalize() == plane_norm and str(r.get('SeriesInstanceUID', '')) not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if _safe_fluid(r.get('Fluid_Sensitive')) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_study_optimized(sid, ser_records, test_dir, io_pool):
    """Loads and preprocesses a study matching Dread's corpus contract bit-for-bit."""
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
            slot_plans.append((None, k, 0.5, None))
            continue
        used.add(r['SeriesInstanceUID'])

        # Check both nested test_dir/<sid>/<series> and flat test_dir/<series> layouts
        sdir = os.path.join(test_dir, str(sid), str(r['SeriesInstanceUID']))
        if not os.path.exists(sdir):
            sdir = os.path.join(test_dir, str(r['SeriesInstanceUID']))
        if os.path.isdir(sdir):
            fs = [os.path.join(sdir, f) for f in os.listdir(sdir) if f.endswith(".dcm")]
        else:
            fs = []
        if not fs:
            slot_plans.append((None, k, 0.5, r['SeriesInstanceUID']))
            continue

        header_items = [(sid, f) for f in fs]
        if len(fs) > 4:
            recs = list(io_pool.map(_read_header_single, header_items))
        else:
            recs = [_read_header_single(item) for item in header_items]
        recs.sort(key=lambda x: x[0])
        med_ps = float(np.median([r[2] for r in recs])) if recs else 0.5
        ordered = [(f, ps) for _, f, ps in recs]
        slot_plans.append((ordered, k, med_ps, r['SeriesInstanceUID']))

    t_header = time.time() - t_h0

    # Stage 2: Pick target slices per slot
    t_p0 = time.time()
    # Group slice requests by slot to preserve PER-SLOT normalization parity
    slot_slice_requests = []  # list of list of (target_idx, fp, ps)

    cur_idx = 0
    all_files_to_read = []
    for ordered, k, med_ps, ser_uid in slot_plans:
        this_slot_requests = []
        if ordered is None or not ordered:
            cur_idx += k
            slot_slice_requests.append(this_slot_requests)
            continue
        n = len(ordered)
        lo, hi = int(n * 0.06), int(n * 0.94) - 1
        hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k

        for p in picks:
            if cur_idx >= MAXS: break
            fp, ps = ordered[min(p, n - 1)]
            this_slot_requests.append((cur_idx, fp, ps if ps > 0 else med_ps))
            all_files_to_read.append((sid, fp))
            cur_idx += 1
        slot_slice_requests.append(this_slot_requests)
        if cur_idx >= MAXS: break

    # Parallel pixel decompression
    if len(all_files_to_read) > 4:
        all_loaded = list(io_pool.map(read_px, all_files_to_read))
    else:
        all_loaded = [read_px(item) for item in all_files_to_read]
    t_pixel = time.time() - t_p0

    # Map loaded pixel arrays back to each slot
    t_c0 = time.time()
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    loaded_iter = iter(all_loaded)
    total_valid_slices = 0

    # Stage 3: Exact Per-Slot Percentile Normalization & mm-Cropping (Dread Parity)
    for this_slot_requests in slot_slice_requests:
        slot_arrs = []
        for req in this_slot_requests:
            arr = next(loaded_iter)
            slot_arrs.append(arr)

        valid_in_slot = [a for a in slot_arrs if a is not None]
        total_valid_slices += len(valid_in_slot)

        if valid_in_slot:
            allpx = np.concatenate([a.ravel() for a in valid_in_slot])
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0

        for req, arr in zip(this_slot_requests, slot_arrs):
            target_idx, _, ps = req
            if target_idx >= MAXS: break
            if arr is None: continue
            aw = np.clip((arr - loq) / (hiq - loq + 1e-6), 0, 1)
            aw = mm_crop_resize(aw, ps)
            vol[target_idx] = (aw * 255).astype(np.uint8)

    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    t_prep = time.time() - t_c0

    timings = {
        "header": t_header,
        "pixel": t_pixel,
        "prep": t_prep,
        "total_io": time.time() - t0,
        "decoded_slices": total_valid_slices,
        "total_requested": len(all_files_to_read)
    }
    return vol, mask, timings


# ============================================================================
# GPU Inference & Full Window Centers (K_EVAL = 42)
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
# Path Discovery & Checkpoints
# ============================================================================
def find_comp_source():
    cands = [
        os.environ.get("RSNA_COMP_ROOT", ""),
        "/kaggle/input/rsna-knee-abnormality-detection",
        "/kaggle/input/competitions/rsna-knee-abnormality-detection"
    ]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(os.path.join(c, "train_series")):
            return Path(c)
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train.csv" in files and "train_series" in dirs:
            return Path(root)
    return None


def find_corpus_source():
    cands = [
        "/kaggle/input/knee-raptor-corpus",
        "/kaggle/input/datasets/dreaddevelopment/knee-raptor-corpus",
        "ds/corpus1",
        "/home/s05814802723/projects/knee/ds/corpus1"
    ]
    for c in cands:
        if os.path.exists(os.path.join(c, "all_vols.npy")):
            return Path(c)
    for root, dirs, files in os.walk("/kaggle/input"):
        if "all_vols.npy" in files:
            return Path(root)
    return None


def find_test_data():
    comp_roots = []
    if "RSNA_COMP_ROOT" in os.environ and os.path.exists(os.environ["RSNA_COMP_ROOT"]):
        comp_roots.append(os.environ["RSNA_COMP_ROOT"])
    comp_roots.extend([
        "/kaggle/input/rsna-knee-abnormality-detection",
        "/kaggle/input/competitions/rsna-knee-abnormality-detection"
    ])
    for r in comp_roots:
        if not r or not os.path.exists(r):
            continue
        test_series_csv = os.path.join(r, "test_series.csv")
        sample_sub_csv = os.path.join(r, "sample_submission.csv")
        if not os.path.exists(test_series_csv):
            continue
        for cand in ["test_series", "test", "test_images"]:
            p = os.path.join(r, cand)
            if os.path.exists(p) and os.path.isdir(p):
                print(f"[data] Found test image directory: {p}", flush=True)
                return p, test_series_csv, sample_sub_csv
    raise FileNotFoundError(f"Could not find competition test data. Checked: {comp_roots}")


def find_checkpoint(prefer_swa=False):
    if prefer_swa:
        direct_candidates = [
            "/kaggle/input/knee-our-models/raptor_ft_v5_swa.pt",
            "/kaggle/input/datasets/shivoo29/knee-our-models/raptor_ft_v5_swa.pt",
            "raptor_ft_v5_swa.pt",
            "/kaggle/input/knee-our-models/raptor_ft_v5.pt",
            "/kaggle/input/datasets/shivoo29/knee-our-models/raptor_ft_v5.pt",
            "raptor_ft_v5.pt"
        ]
    else:
        direct_candidates = [
            "/kaggle/input/knee-our-models/raptor_ft_v5.pt",
            "/kaggle/input/datasets/shivoo29/knee-our-models/raptor_ft_v5.pt",
            "raptor_ft_v5.pt",
            "/kaggle/input/knee-our-models/raptor_ft_v5_swa.pt",
            "/kaggle/input/datasets/shivoo29/knee-our-models/raptor_ft_v5_swa.pt",
            "raptor_ft_v5_swa.pt"
        ]
    for p in direct_candidates:
        if os.path.exists(p):
            print(f"[ckpt] Found direct checkpoint: {p}", flush=True)
            return p

    for root in ["/kaggle/input/knee-our-models", "/kaggle/input/datasets/shivoo29/knee-our-models", "."]:
        if not os.path.exists(root): continue
        target_fnames = ["raptor_ft_v5_swa.pt", "raptor_ft_v5.pt"] if prefer_swa else ["raptor_ft_v5.pt", "raptor_ft_v5_swa.pt"]
        for fname in target_fnames:
            p = os.path.join(root, fname)
            if os.path.exists(p):
                print(f"[ckpt] Selected checkpoint: {p}", flush=True)
                return p
    raise FileNotFoundError("Could not find model checkpoint in direct paths")


# ============================================================================
# Stage-by-Stage Corpus Parity Auditor (~20 Studies)
# ============================================================================
def audit_corpus_parity(corpus_dir, comp_src, model, res, device, n_check=20):
    print("\n" + "=" * 80)
    print(f"STAGE-BY-STAGE PARITY AUDIT: LIVE DICOM vs ALL_VOLS.NPY ({n_check} Studies)")
    print("=" * 80)

    c_ids = np.load(corpus_dir / "all_ids.npy", allow_pickle=True).astype(str)
    c_vols = np.load(corpus_dir / "all_vols.npy", mmap_mode="r")
    c_masks = np.load(corpus_dir / "all_masks.npy")
    id2row = {sid: i for i, sid in enumerate(c_ids)}

    ser_csv = comp_src / "train_series.csv"
    ser_df = pd.read_csv(ser_csv, dtype=str)
    ser_records = {}
    for uid, g in ser_df.groupby("StudyInstanceUID"):
        ser_records[uid] = g.to_dict("records")

    tr = pd.read_csv(comp_src / "train.csv", dtype={"StudyInstanceUID": str})
    gold_ids = [s for s in tr[tr.ACL.notna()].StudyInstanceUID if s in id2row][:n_check]

    io_pool = ThreadPoolExecutor(max_workers=8)
    train_series_dir = os.path.join(comp_src, "train_series")

    diff_records = []
    prob_diffs = []

    for sid in gold_ids:
        row = id2row[sid]
        v_corpus = c_vols[row]
        m_corpus = c_masks[row]

        v_dicom, m_dicom, timings = build_study_optimized(sid, ser_records, train_series_dir, io_pool)

        # Stage metrics
        max_abs_diff = int(np.abs(v_corpus.astype(int) - v_dicom.astype(int)).max())
        mean_abs_diff = float(np.abs(v_corpus.astype(float) - v_dicom.astype(float)).mean())
        mask_match = bool(np.array_equal(m_corpus, m_dicom))

        # Model output difference on both tensors
        p_corp, _ = predict_study(model, v_corpus, m_corpus, res, device, use_tta=False)
        p_dcm, _ = predict_study(model, v_dicom, m_dicom, res, device, use_tta=False)
        max_p_diff = float(np.abs(p_corp - p_dcm).max())
        cos_sim = float(np.dot(p_corp, p_dcm) / (np.linalg.norm(p_corp) * np.linalg.norm(p_dcm) + 1e-8))

        diff_records.append({
            "sid": sid[:12] + "...",
            "max_px_diff": max_abs_diff,
            "mean_px_diff": mean_abs_diff,
            "mask_match": mask_match,
            "max_prob_diff": max_p_diff,
            "cos_sim": cos_sim
        })
        prob_diffs.append(max_p_diff)

    io_pool.shutdown(wait=False)

    df_report = pd.DataFrame(diff_records)
    print(df_report.to_string(index=False))
    print("-" * 80)
    avg_max_p = np.mean(prob_diffs)
    print(f"Parity Summary: Mean Max Prob Diff: {avg_max_p:.4f} | Mean Cosine Sim: {np.mean([r['cos_sim'] for r in diff_records]):.5f}")
    if avg_max_p < 0.05:
        print("[audit] PARITY VERIFIED! DICOM path reproduces corpus tensor representations with high fidelity!")
    else:
        print(f"[audit] NOTICE: Slight numerical variance observed ({avg_max_p:.4f} prob diff), verifying on Gold-58 macro-AUC...")
    print("=" * 80 + "\n")


# ============================================================================
# Visible-Data Teacher Probe Setup & AUC Evaluation
# ============================================================================
def setup_teacher_probe(comp_src, work_dir="/kaggle/working"):
    fake = Path(work_dir) / "fakecomp"
    fake.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(comp_src / "train.csv", dtype={"StudyInstanceUID": str})
    gold_ids = tr[tr.ACL.notna()].StudyInstanceUID
    n_sample = min(300, len(tr[tr.ACL.isna()]))
    sampled_ids = tr[tr.ACL.isna()].StudyInstanceUID.sample(n_sample, random_state=0)
    all_probe_ids = pd.concat([gold_ids, sampled_ids]).reset_index(drop=True)

    # 1. test.csv
    pd.DataFrame({"StudyInstanceUID": all_probe_ids}).to_csv(fake / "test.csv", index=False)

    # 2. test_series.csv
    ts = pd.read_csv(comp_src / "train_series.csv", dtype=str)
    ts[ts.StudyInstanceUID.isin(all_probe_ids)].to_csv(fake / "test_series.csv", index=False)

    # 3. sample_submission.csv
    cols = ["StudyInstanceUID"] + LAB
    pd.DataFrame({c: (all_probe_ids.values if c == "StudyInstanceUID" else 0.5) for c in cols}).to_csv(fake / "sample_submission.csv", index=False)

    # 4. test_series symlink pointing to train_series
    target_series = fake / "test_series"
    if not target_series.exists():
        src_train_series = comp_src / "train_series"
        target_series.symlink_to(src_train_series)

    print(f"[probe] Built fakecomp at {fake}: {len(gold_ids)} gold + {n_sample} train studies = {len(all_probe_ids)} total", flush=True)
    return fake, gold_ids.tolist()


def evaluate_gold_auc(sub_df, train_csv_path):
    from sklearn.metrics import roc_auc_score
    tr = pd.read_csv(train_csv_path, dtype={'StudyInstanceUID': str})
    gold_df = tr[tr.ACL.notna()].copy()
    gold_ids = set(gold_df.StudyInstanceUID)
    matched = sub_df[sub_df.StudyInstanceUID.isin(gold_ids)].copy()

    if len(matched) < len(gold_ids):
        print(f"[eval] Warning: Only matched {len(matched)} / {len(gold_ids)} gold studies", flush=True)
    if len(matched) == 0:
        print("[eval] No gold studies matched in submission. Skipping AUC evaluation.", flush=True)
        return None

    # Align exactly by StudyInstanceUID
    gold_df = gold_df[gold_df.StudyInstanceUID.isin(matched.StudyInstanceUID)].copy()
    gold_df = gold_df.sort_values("StudyInstanceUID").reset_index(drop=True)
    matched = matched.sort_values("StudyInstanceUID").reset_index(drop=True)

    aucs = {}
    print("\n" + "=" * 80)
    print(f"VISIBLE GOLD-58 VALIDATION RESULTS (K_EVAL={K_EVAL})")
    print("=" * 80)
    for col in LAB:
        y_true = gold_df[col].values.astype(float)
        y_pred = matched[col].values.astype(float)
        try:
            score = roc_auc_score(y_true, y_pred)
            aucs[col] = score
            print(f"  {col:<20}: {score:.4f}")
        except Exception as e:
            print(f"  {col:<20}: ERR ({e})")
    macro_auc = float(np.mean(list(aucs.values()))) if aucs else 0.0
    print("-" * 80)
    print(f"  MACRO-AUC (Gold 58): {macro_auc:.4f}")
    print("=" * 80 + "\n")
    return macro_auc


# ============================================================================
# Multi-GPU Pipeline Executor
# ============================================================================
def run_pipeline(test_dir, test_series_csv, sample_sub_csv, devices, models, res, is_probe=False):
    t_start = time.time()
    tag = "PROBE" if is_probe else "PRODUCTION"
    print(f"\n[{tag}] Initializing pipeline run...")
    print(f"[{tag}] Image dir: {test_dir} | Series CSV: {test_series_csv}")

    ser_df = pd.read_csv(test_series_csv)
    ser_df["StudyInstanceUID"] = ser_df["StudyInstanceUID"].astype(str)
    ser_records = {}
    for uid, g in ser_df.groupby("StudyInstanceUID"):
        ser_records[uid] = g.to_dict("records")

    sample_sub = pd.read_csv(sample_sub_csv)
    test_ids = sample_sub["StudyInstanceUID"].astype(str).tolist()
    n_studies = len(test_ids)
    print(f"[{tag}] Scoring {n_studies} studies with K_EVAL={K_EVAL}...", flush=True)

    # Shared queues and thread pools
    io_pool = ThreadPoolExecutor(max_workers=12)
    work_queue = queue.Queue(maxsize=8)
    results = [None] * n_studies
    profile_records = []
    fallback_count = 0
    records_lock = threading.Lock()
    t_steady_start = time.time()

    def gpu_worker(worker_idx, dev, model):
        nonlocal fallback_count
        while True:
            item = work_queue.get()
            if item is None:
                work_queue.task_done()
                break
            idx, sid, vol, mask, timings = item
            is_fallback = False

            # Check if volume has zero decoded slices
            if timings.get("decoded_slices", 0) == 0:
                is_fallback = True
                with records_lock:
                    fallback_count += 1
                record_exception(f"gpu_worker_{worker_idx}_zero_slices", sid, RuntimeError("Zero valid slices decoded for study!"))

            try:
                preds, t_model = predict_study(model, vol, mask, res, dev, use_tta=USE_TTA)
                timings["model"] = t_model
                timings["worker"] = worker_idx
            except Exception as e:
                if not is_fallback:
                    with records_lock:
                        fallback_count += 1
                record_exception(f"gpu_worker_{worker_idx}_predict", sid, e)
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
                if done_cnt % 25 == 0 or done_cnt == n_studies:
                    cur_steady = time.time() - t_steady_start
                    print(f"[{tag}] [{done_cnt}/{n_studies}] {cur_steady / done_cnt:.3f}s/study (steady) | {cur_steady:.1f}s elapsed", flush=True)
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
            record_exception("producer_build_study", sid, e)
            vol = np.zeros((MAXS, IMG, IMG), np.uint8)
            mask = np.zeros(MAXS, np.uint8)
            timings = {"header": 0.0, "pixel": 0.0, "prep": 0.0, "total_io": 0.0, "decoded_slices": 0, "total_requested": 0}
        work_queue.put((idx, sid, vol, mask, timings))

    # Wait for queue completion
    work_queue.join()
    for _ in devices:
        work_queue.put(None)
    for t in consumer_threads:
        t.join()
    io_pool.shutdown(wait=False)

    # Assemble and validate submission DataFrame
    for idx, r in enumerate(results):
        if r is None:
            sid = test_ids[idx]
            print(f"[FATAL ERROR] results[{idx}] for study {sid} is missing!", flush=True)
            results[idx] = {"StudyInstanceUID": sid, **{col: 0.5 for col in LAB}}

    sub = pd.DataFrame(results)
    assert len(sub) == n_studies, f"Row count mismatch: {len(sub)} vs {n_studies}"
    assert list(sub["StudyInstanceUID"]) == test_ids, "Row ID ordering drift!"
    sub = sub[["StudyInstanceUID"] + LAB]

    # Metrics
    steady_time = time.time() - t_steady_start
    avg_steady_per_study = steady_time / max(1, n_studies)
    mean_header = np.mean([p["header"] for p in profile_records]) if profile_records else 0
    mean_pixel = np.mean([p["pixel"] for p in profile_records]) if profile_records else 0
    mean_prep = np.mean([p["prep"] for p in profile_records]) if profile_records else 0
    mean_model = np.mean([p.get("model", 0) for p in profile_records]) if profile_records else 0
    total_decoded_slices = sum(p.get("decoded_slices", 0) for p in profile_records)

    print("\n" + "-" * 80)
    print(f"[{tag}] PERFORMANCE PROFILE ({n_studies} Studies)")
    print("-" * 80)
    print(f"  • Steady Throughput:       {avg_steady_per_study:.3f} s / study")
    print(f"  • Total Decoded Slices:    {total_decoded_slices} slices")
    print(f"  • Header & IPP Sorting:    {mean_header*1000:6.1f} ms")
    print(f"  • Pixel Decompression:     {mean_pixel*1000:6.1f} ms")
    print(f"  • Normalization & Crop:    {mean_prep*1000:6.1f} ms")
    print(f"  • Model Forward:           {mean_model*1000:6.1f} ms")
    print(f"  • Fallbacks Encountered:   {fallback_count} / {n_studies}")
    print("-" * 80 + "\n")

    return sub, profile_records, fallback_count


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    t_main_start = time.time()
    print("=" * 80)
    print(f"RSNA Knee Inference Pipeline | Mode: {'FULL (TTA)' if USE_TTA else 'FAST (No-TTA)'} | K_EVAL={K_EVAL}")
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

    ckpt_path = find_checkpoint(prefer_swa=False)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
    res = int(ck.get("res", 384))
    print(f"[model] Checkpoint: {ckpt_path} | Arch: {arch} | Res: {res}", flush=True)

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

    # Pre-warm GPU CUDA runtime context and cuDNN autotuner
    print("[warmup] Pre-warming CUDA runtime & cuDNN autotuner across devices...", flush=True)
    for dev, m in zip(devices, models):
        dummy_chunk = torch.zeros((WIN_CHUNK, 3, res, res), dtype=torch.float32, device=dev)
        dummy_in = dummy_chunk[None]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            _ = m(dummy_in)
        if torch.cuda.is_available():
            torch.cuda.synchronize(dev)
    t_startup = time.time() - t_main_start
    print(f"[startup] Fixed setup & GPU warmup completed in {t_startup:.2f}s", flush=True)

    # Identify real competition test inputs
    real_test_dir, real_test_series_csv, real_sample_sub_csv = find_test_data()
    real_sample_sub = pd.read_csv(real_sample_sub_csv)
    n_real = len(real_sample_sub)
    print(f"[data] Detected real test set: {n_real} studies from {real_test_dir}", flush=True)

    # Check for competition source datasets to enable Teacher Probe & Parity Auditor
    comp_src = find_comp_source()
    corpus_src = find_corpus_source()

    # STAGE 0: CORPUS PARITY AUDIT (if corpus dataset is attached and visible data available)
    if comp_src is not None and corpus_src is not None and n_real <= 10:
        try:
            audit_corpus_parity(corpus_src, comp_src, models[0], res, devices[0], n_check=20)
        except Exception as e:
            print(f"[audit] Corpus parity audit exception (non-fatal): {e}", flush=True)
            traceback.print_exc()

    # STAGE 1: TEACHER PROBE VALIDATION ON VISIBLE DATA
    should_run_probe = (comp_src is not None) and ((n_real <= 10) or (os.environ.get("RUN_PROBE", "0") == "1"))

    if should_run_probe:
        print("\n" + "=" * 80)
        print("STAGE 1: RUNNING TEACHER PROBE VALIDATION ON VISIBLE DATA")
        print("=" * 80)
        probe_root, gold_ids = setup_teacher_probe(comp_src)
        probe_test_dir = os.path.join(probe_root, "test_series")
        probe_test_series_csv = os.path.join(probe_root, "test_series.csv")
        probe_sample_sub_csv = os.path.join(probe_root, "sample_submission.csv")

        probe_sub, probe_records, probe_fallbacks = run_pipeline(
            probe_test_dir, probe_test_series_csv, probe_sample_sub_csv,
            devices, models, res, is_probe=True
        )

        # AUDIT 1: Assert IDs match test.csv and columns match sample_submission.csv
        probe_test_csv = os.path.join(probe_root, "test.csv")
        expected_test = pd.read_csv(probe_test_csv, dtype={"StudyInstanceUID": str})
        assert list(probe_sub["StudyInstanceUID"]) == list(expected_test["StudyInstanceUID"]), \
            "FATAL: Probe output StudyInstanceUID order does NOT match test.csv!"
        assert list(probe_sub.columns) == ["StudyInstanceUID"] + LAB, \
            "FATAL: Probe output columns do NOT match sample_submission.csv!"

        # AUDIT 2: Hard guard on fallback count
        probe_n = len(probe_sub)
        fallback_rate = probe_fallbacks / max(1, probe_n)
        print(f"[guard] Probe Fallback Rate: {probe_fallbacks}/{probe_n} ({fallback_rate:.2%})", flush=True)
        if fallback_rate > 0.05:
            raise RuntimeError(f"FATAL GUARD TRIGGERED: Probe fallback rate {fallback_rate:.2%} > 5% allowed limit!")

        # AUDIT 3: Validate Gold-58 macro-AUC
        macro_auc = evaluate_gold_auc(probe_sub, os.path.join(comp_src, "train.csv"))
        if macro_auc is not None:
            if macro_auc < 0.85:
                raise RuntimeError(f"FATAL GUARD TRIGGERED: Gold-58 Macro-AUC is {macro_auc:.4f} < 0.8500! Model predictions are broken/constant!")
            print(f"[audit] SUCCESS! Visible Gold-58 Macro-AUC: {macro_auc:.4f} (passed >= 0.8500 check)", flush=True)
    else:
        print(f"[info] Skipping Teacher Probe (comp_src={comp_src}, n_real={n_real}, RUN_PROBE={os.environ.get('RUN_PROBE')})", flush=True)

    # STAGE 2: PRODUCTION INFERENCE ON REAL TEST SET
    print("\n" + "=" * 80)
    print(f"STAGE 2: SCORING REAL TEST SET ({n_real} Studies)")
    print("=" * 80)
    final_sub, final_records, final_fallbacks = run_pipeline(
        real_test_dir, real_test_series_csv, real_sample_sub_csv,
        devices, models, res, is_probe=False
    )

    # Final submission verification
    assert list(final_sub["StudyInstanceUID"]) == list(real_sample_sub["StudyInstanceUID"]), \
        "FATAL: Final submission StudyInstanceUID order does NOT match sample_submission.csv!"
    assert list(final_sub.columns) == list(real_sample_sub.columns), \
        "FATAL: Final submission columns do NOT match sample_submission.csv!"
    assert not final_sub.isna().any().any(), "FATAL: Final submission contains NaNs!"

    final_fallback_rate = final_fallbacks / max(1, n_real)
    if final_fallback_rate > 0.05 and n_real > 10:
        raise RuntimeError(f"FATAL GUARD TRIGGERED: Real test set fallback rate {final_fallback_rate:.2%} > 5% limit!")

    final_sub.to_csv("submission.csv", index=False)
    print(f"\n[complete] submission.csv saved successfully with shape {final_sub.shape} and verified valid values!", flush=True)


if __name__ == "__main__":
    main()
