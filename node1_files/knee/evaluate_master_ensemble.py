import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, pearsonr
import glob
import os

FINDINGS = [
    "ACL", "PCL", "MCL", "LCL",
    "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's Cyst"
]

files = {
    "CoAtNet v6 (single)": "raptor_gold_v6.npz",
    "CoAtNet v6 SWA": "raptor_gold_v6_swa.npz",
    "CoAtNet v5s2": "raptor_gold_v5s2.npz",
    "CoAtNet v5s1 SWA": "raptor_gold_v5s1_swa.npz",
    "ConvNeXt-Base 96": "raptor_gold_cnext_base_96.npz",
    "ConvNeXt-Base 96 SWA": "raptor_gold_cnext_base_96_swa.npz",
    "Bi-LSTM 44-slice": "raptor_gold_bilstm_v1_ens.npz",
    "Bi-LSTM 96-slice": "raptor_gold_bilstm_c96_ens.npz",
    "Transformer 96-slice": "raptor_gold_xfmr_c96_ens.npz",
    "EVA02-Base 96 (ep5)": "raptor_gold_eva02_base_96.npz",
}

loaded = {}
ref_ids = None
truth = None

for name, fname in files.items():
    if not os.path.exists(fname):
        continue
    try:
        d = np.load(fname, allow_pickle=True)
        pred = d['pred'].astype(np.float64)
        t = d['truth'].astype(np.float64)
        ids = d['ids']
        if ref_ids is None:
            ref_ids = ids
            truth = t
        else:
            if not np.array_equal(ids, ref_ids):
                id_map = {uid: i for i, uid in enumerate(ids)}
                reorder = [id_map[uid] for uid in ref_ids]
                pred = pred[reorder]
        loaded[name] = pred
    except Exception as e:
        print(f"Error loading {name} ({fname}): {e}")

def rank_norm(arr):
    out = np.zeros_like(arr)
    for j in range(arr.shape[1]):
        out[:, j] = rankdata(arr[:, j]) / len(arr[:, j])
    return out

def get_aucs(preds, labels):
    sc = []
    for j in range(labels.shape[1]):
        if len(np.unique(labels[:, j])) > 1:
            sc.append(roc_auc_score(labels[:, j], preds[:, j]))
        else:
            sc.append(0.5)
    return np.array(sc), np.mean(sc)

print("="*85)
print(f"{'Model Name':<30} {'Macro AUC':>10} {'Min Class':>10} {'Max Class':>10}")
print("="*85)

ranked = {}
for name, pred in loaded.items():
    scores, macro = get_aucs(pred, truth)
    ranked[name] = rank_norm(pred)
    print(f"{name:<30} {macro:10.4f} {scores.min():10.4f} {scores.max():10.4f}")
print("="*85)

# Diversity against CoAtNet v6
print("\n--- Correlation vs CoAtNet v6 ---")
v6_ref = ranked["CoAtNet v6 (single)"]
for name, r_pred in ranked.items():
    if name == "CoAtNet v6 (single)":
        continue
    corrs = [pearsonr(v6_ref[:, j], r_pred[:, j])[0] for j in range(12)]
    print(f"{name:<30} Pearson r: {np.nanmean(corrs):.4f}")

# Multi-Architecture Blends
print("\n" + "="*85)
print("MULTI-ARCHITECTURE ENSEMBLE COMBINATIONS")
print("="*85)

v6 = ranked["CoAtNet v6 (single)"]
cn = ranked["ConvNeXt-Base 96 SWA"] if "ConvNeXt-Base 96 SWA" in ranked else ranked["ConvNeXt-Base 96"]
b96 = ranked["Bi-LSTM 96-slice"]
eva = ranked.get("EVA02-Base 96 (ep5)", None)

# 1. CoAtNet dual-seed blend (v6 + v5s2)
coat_ens = 0.50 * v6 + 0.50 * ranked["CoAtNet v5s2"]
_, a_coat = get_aucs(coat_ens, truth)
print(f"1. CoAtNet Dual-Seed (v6 + v5s2):                AUC = {a_coat:.4f} (Δ {a_coat - 0.9153:+.4f})")

# 2. CoAtNet Dual-Seed + ConvNeXt-Base 96
c1 = 0.70 * coat_ens + 0.30 * cn
_, a1 = get_aucs(c1, truth)
print(f"2. CoAtNet Dual-Seed + ConvNeXt-Base 96:        AUC = {a1:.4f} (Δ {a1 - 0.9153:+.4f})")

# 3. CoAtNet Dual-Seed + ConvNeXt + Bi-LSTM 96 (Conservative rule-blend)
c2 = np.zeros_like(truth)
for j, f in enumerate(FINDINGS):
    if f in ["Lateral Meniscus"]:
        w = (0.45, 0.25, 0.30)
    elif f in ["LCL"]:
        w = (0.50, 0.35, 0.15)
    elif f in ["MCL", "PCL"]:
        w = (0.65, 0.25, 0.10)
    elif f in ["Medial OA", "PF OA"]:
        w = (0.55, 0.35, 0.10)
    elif f in ["Medial Meniscus"]:
        w = (0.65, 0.30, 0.05)
    else:
        w = (0.65, 0.35, 0.00)
    c2[:, j] = w[0] * coat_ens[:, j] + w[1] * cn[:, j] + w[2] * b96[:, j]
_, a2 = get_aucs(c2, truth)
print(f"3. CoAtNet + ConvNeXt + Bi-LSTM 96 (Rule Blend): AUC = {a2:.4f} (Δ {a2 - 0.9153:+.4f})")

# 4. If EVA-02 is ready: 4-Way Ensemble (CoAtNet + ConvNeXt + EVA-02 + Bi-LSTM 96)
if eva is not None:
    c3 = np.zeros_like(truth)
    for j, f in enumerate(FINDINGS):
        if f in ["Lateral Meniscus"]:
            w = (0.35, 0.20, 0.20, 0.25)
        elif f in ["LCL", "MCL", "PCL"]:
            w = (0.40, 0.25, 0.25, 0.10)
        elif f in ["Medial OA", "PF OA"]:
            w = (0.40, 0.25, 0.25, 0.10)
        else:
            w = (0.45, 0.25, 0.30, 0.00)
        c3[:, j] = w[0] * coat_ens[:, j] + w[1] * cn[:, j] + w[2] * eva[:, j] + w[3] * b96[:, j]
    _, a3 = get_aucs(c3, truth)
    print(f"4. 4-Way Ensemble (+ EVA-02 ep5):               AUC = {a3:.4f} (Δ {a3 - 0.9153:+.4f})")

print("="*85)
