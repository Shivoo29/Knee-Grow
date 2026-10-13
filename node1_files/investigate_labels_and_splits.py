"""
Tasks 1–3 and 5:
1. Print the 12 label column names from train.csv and sample_submission.csv in order,
   and the mapping used in every saved .npz and in the blend matrix. Fix any mismatch.
2. Print positives per finding in the gold-58.
3. Re-run the blend evaluation with 200 random 50/50 splits.
5. What labels was CoAtNet v6 trained on?
"""

import os, json, glob, re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

KNEE = os.path.expanduser("~/projects/knee")

# ─── 1. CANONICAL LABEL ORDER FROM CSVs ───────────────────────────────────────
print("\n" + "="*70)
print("TASK 1 – LABEL COLUMN MAPPING")
print("="*70)

train_csv = pd.read_csv(os.path.join(KNEE, "ds/comp/train.csv"))
# No sample_submission.csv found in comp/, use train.csv order
NON_LABEL = {"StudyInstanceUID", "Report"}
TRAIN_CSV_LABELS = [c for c in train_csv.columns if c not in NON_LABEL]
print(f"\ntrain.csv columns ({len(train_csv.columns)}): {list(train_csv.columns)}")
print(f"train.csv LABEL COLS ({len(TRAIN_CSV_LABELS)}): {TRAIN_CSV_LABELS}")

# ─── 1b. NPZ ORDER ───────────────────────────────────────────────────────────
# The npzs store truth (58,12). We need to find what column order they use.
# They were built from train_knee.py which reads labels_avg.parquet.
labels_avg = pd.read_parquet(os.path.join(KNEE, "labels_avg.parquet"))
id_col_lap = [c for c in labels_avg.columns if "uid" in c.lower() or "study" in c.lower() or "id" in c.lower()]
LAP_LABELS = [c for c in labels_avg.columns if c not in id_col_lap]
print(f"\nlabels_avg.parquet LABEL COLS ({len(LAP_LABELS)}): {LAP_LABELS}")

# Check order match
train_set = set(TRAIN_CSV_LABELS)
lap_set   = set(LAP_LABELS)
print(f"\nIn train.csv but NOT labels_avg: {train_set - lap_set}")
print(f"In labels_avg but NOT train.csv: {lap_set - train_set}")
print(f"ORDER matches? {TRAIN_CSV_LABELS == LAP_LABELS}")

# ─── 1c. NPZ truth column order – derive from training script ─────────────────
print("\n--- Checking train_knee.py for LABEL_COLS / label_cols ---")
with open(os.path.join(KNEE, "train_knee.py")) as fh:
    src = fh.read()
# Find lines defining label column list
for i, line in enumerate(src.split("\n")):
    if re.search(r'label_col|LABEL_COL|FINDINGS|ACL', line, re.IGNORECASE):
        print(f"  L{i+1}: {line.rstrip()}")

# ─── 1d. Bi-LSTM script label order ──────────────────────────────────────────
print("\n--- Checking train_bilstm_c96.py for FINDINGS / LABEL_COLS ---")
with open(os.path.join(KNEE, "train_bilstm_c96.py")) as fh:
    bsrc = fh.read()
for i, line in enumerate(bsrc.split("\n")):
    if re.search(r'FINDING|LABEL_COL|ACL|label_col', line, re.IGNORECASE):
        print(f"  L{i+1}: {line.rstrip()}")

# ─── 2. POSITIVES PER FINDING IN GOLD-58 ─────────────────────────────────────
print("\n" + "="*70)
print("TASK 2 – POSITIVES PER FINDING IN GOLD-58")
print("="*70)

v6 = np.load(os.path.join(KNEE, "raptor_gold_v6.npz"), allow_pickle=True)
truth = v6["truth"].astype(float)   # (58, 12)
gold_ids = v6["ids"]

# Align with train.csv to verify column order
gold_train = train_csv[train_csv["StudyInstanceUID"].astype(str).isin(gold_ids.astype(str))].copy()
n_matched = len(gold_train)
print(f"\ntrain.csv gold rows matched: {n_matched}/58")

# The npz truth should correspond to LAP_LABELS (labels_avg.parquet order)
# which is what train_knee.py was trained on.
col_names_for_truth = LAP_LABELS[:truth.shape[1]]

print(f"\n{'#':>3}  {'Finding':<25} {'Positives':>10} {'Prevalence':>11} {'Is Binary?':>11} {'Min':>6} {'Max':>6}")
print("-"*74)
for j, name in enumerate(col_names_for_truth):
    col  = truth[:, j]
    npos = int((col > 0.5).sum())
    is_bin = bool(np.all((col == 0) | (col == 1)))
    print(f"{j:>3}  {name:<25} {npos:>10d} {npos/58:>10.1%} {str(is_bin):>11} {col.min():>6.3f} {col.max():>6.3f}")

# ─── 3. 200 RANDOM 50/50 SPLIT BLEND EVALUATION ──────────────────────────────
print("\n" + "="*70)
print("TASK 3 – 200 RANDOM 50/50 SPLIT BLEND EVALUATION")
print("="*70)

def rank_norm(arr):
    out = np.zeros_like(arr, dtype=float)
    for j in range(arr.shape[1]):
        out[:, j] = rankdata(arr[:, j]) / len(arr[:, j])
    return out

def macro_auc(preds, labels):
    scores = []
    for j in range(labels.shape[1]):
        if len(np.unique(labels[:, j])) > 1:
            scores.append(roc_auc_score(labels[:, j], preds[:, j]))
    return float(np.mean(scores)) if scores else 0.5

def load_rn(fname):
    d = np.load(os.path.join(KNEE, fname), allow_pickle=True)
    return rank_norm(d["pred"].astype(float))

v6_rn   = load_rn("raptor_gold_v6.npz")
v5s2_rn = load_rn("raptor_gold_v5s2.npz")
cn_rn   = load_rn("raptor_gold_cnext_base_96_swa.npz")
b96_rn  = load_rn("raptor_gold_bilstm_c96_ens.npz")

coat_ens = 0.5 * v6_rn + 0.5 * v5s2_rn

N = truth.shape[0]
NF = truth.shape[1]
NSPLITS = 200
rng = np.random.default_rng(42)

results = {
    "(a) CoAtNet dual-seed":        [],
    "(b) +ConvNeXt fixed 0.30":     [],
    "(c) +LSTM 0.10 global":        [],
    "(d) per-finding tuned (OOF)":  [],
}

W_CN   = 0.30
W_LSTM = 0.10

import time
t0 = time.time()
for split_i in range(NSPLITS):
    idx = rng.permutation(N)
    tr  = idx[:N//2]
    va  = idx[N//2:]

    y_tr  = truth[tr]
    y_val = truth[va]

    # (a) Dual CoAtNet
    results["(a) CoAtNet dual-seed"].append(macro_auc(coat_ens[va], y_val))

    # (b) +ConvNeXt fixed 0.30
    pred_b = (1 - W_CN) * coat_ens[va] + W_CN * cn_rn[va]
    results["(b) +ConvNeXt fixed 0.30"].append(macro_auc(pred_b, y_val))

    # (c) +BiLSTM fixed 0.10
    pred_c = (1 - W_CN - W_LSTM) * coat_ens[va] + W_CN * cn_rn[va] + W_LSTM * b96_rn[va]
    results["(c) +LSTM 0.10 global"].append(macro_auc(pred_c, y_val))

    # (d) Per-finding tuned on train half → evaluate on val half
    pred_d = np.zeros((len(va), NF), dtype=float)
    w1s = np.linspace(0.0, 1.0, 11)
    w2s = np.linspace(0.0, 1.0, 11)
    for j in range(NF):
        best_w = (1.0, 0.0, 0.0)
        best_auc = -1.0
        if len(np.unique(y_tr[:, j])) <= 1:
            pred_d[:, j] = coat_ens[va, j]
            continue
        for w1 in w1s:
            for w2 in w2s:
                w3 = 1.0 - w1 - w2
                if w3 < -1e-5: continue
                p = w1*coat_ens[tr,j] + w2*cn_rn[tr,j] + w3*b96_rn[tr,j]
                s = roc_auc_score(y_tr[:, j], p)
                if s > best_auc:
                    best_auc = s
                    best_w = (w1, w2, w3)
        pred_d[:, j] = best_w[0]*coat_ens[va,j] + best_w[1]*cn_rn[va,j] + best_w[2]*b96_rn[va,j]
    results["(d) per-finding tuned (OOF)"].append(macro_auc(pred_d, y_val))

elapsed = time.time() - t0
print(f"\n(200 splits completed in {elapsed:.1f}s)\n")

print(f"{'Method':<30} {'Mean':>8} {'±Std':>7} {'Median':>8} {'Min':>8} {'Max':>8}")
print("─"*75)
for name, scores in results.items():
    s = np.array(scores)
    print(f"{name:<30} {s.mean():>8.4f} {s.std():>7.4f} {np.median(s):>8.4f} {s.min():>8.4f} {s.max():>8.4f}")

print("\nGain vs (a) dual CoAtNet:")
base = np.array(results["(a) CoAtNet dual-seed"])
for name, scores in results.items():
    if name == "(a) CoAtNet dual-seed": continue
    s = np.array(scores)
    print(f"  {name}: Δ = {(s - base).mean():+.4f}")

# ─── 5. COATNET V6 LABEL PROVENANCE ──────────────────────────────────────────
print("\n" + "="*70)
print("TASK 5 – COATNET V6 LABEL PROVENANCE")
print("="*70)

# 5a. JSON metadata
with open(os.path.join(KNEE, "raptor_ft_v6.json")) as fh:
    meta = json.load(fh)
print("\nraptor_ft_v6.json:")
for k, v in meta.items():
    if isinstance(v, dict):
        print(f"  {k}:")
        for kk, vv in v.items(): print(f"    {kk}: {vv}")
    else:
        print(f"  {k}: {v}")

# 5b. labels_avg.parquet provenance
lap = pd.read_parquet(os.path.join(KNEE, "labels_avg.parquet"))
id_col = [c for c in lap.columns if "uid" in c.lower() or "study" in c.lower()][0]
label_cols = [c for c in lap.columns if c != id_col]
print(f"\nlabels_avg.parquet ({len(lap)} rows × {len(label_cols)} label cols)")
print(f"  Label columns in order: {label_cols}")
vals = lap[label_cols].values.flatten()
vals_nonan = vals[~np.isnan(vals)]
print(f"\n  Value distribution (flattened, ignoring NaN):")
print(f"    min={vals_nonan.min():.4f}  max={vals_nonan.max():.4f}")
print(f"    exactly 0.0: {(vals_nonan == 0.0).sum():,}")
print(f"    exactly 1.0: {(vals_nonan == 1.0).sum():,}")
print(f"    strictly between 0 and 1 (soft): {((vals_nonan > 0) & (vals_nonan < 1)).sum():,}")
n_nan = np.isnan(vals).sum()
n_total = len(vals)
print(f"    NaN (not-mentioned, masked): {n_nan:,} / {n_total:,} ({n_nan/n_total:.1%})")

# 5c. Source CSV check
for src_file in ["llm_labels_full.csv", "llm_labels_v2.csv",
                  "llm_labels_v4_blend.csv", "labels_llm_soft.csv"]:
    p = os.path.join(KNEE, "ds", "rsna-knee-llm-report-labels", src_file)
    if not os.path.exists(p):
        p = os.path.join(KNEE, "ds", "rsna-knee-labels", src_file)
    if os.path.exists(p):
        df = pd.read_csv(p)
        lc = [c for c in df.columns if c.lower() not in {"studyinstanceuid","report","uid"}]
        print(f"\n  {src_file}: {len(df)} rows × {len(lc)} cols")
        sample_vals = df[lc].values.flatten()
        sample_vals = sample_vals[~(sample_vals != sample_vals)]  # remove nan
        try:
            print(f"    value range: {float(np.nanmin(df[lc].values)):.3f} – {float(np.nanmax(df[lc].values)):.3f}")
        except: pass

# 5d. Masking check in train_knee.py
print("\n--- train_knee.py: NaN / loss-mask handling ---")
with open(os.path.join(KNEE, "train_knee.py")) as fh:
    lines = fh.readlines()
for i, l in enumerate(lines):
    if any(kw in l for kw in ["nan", "NaN", "isnan", "mask", "valid_mask",
                                "ignore_index", "loss_weight", "notna", "fillna"]):
        print(f"  L{i+1:4d}: {l.rstrip()}")

print("\n[ALL DONE]")
