import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

ROOT = "/home/s05814802723/projects/knee"
LAB = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

# 1. Load Gold Ground Truth
df_train = pd.read_csv(f"{ROOT}/train.csv")
gold_df = df_train.dropna(subset=LAB).copy()
y_true = gold_df[LAB].values.astype(float)
N = len(y_true)

print("=" * 85)
print(f"EVALUATING HARVESTED STUDENT FLEET ON GOLD-58 (N={N})")
print("=" * 85)

candidates = {
    "v6d24_s9":   "raptor_gold_v6d24_s9.npz",
    "v6d_s7_swa": "raptor_gold_v6d_s7_swa.npz",
    "v6_s4_swa":  "raptor_gold_v6_s4_swa.npz",
    "v6_s3_swa":  "raptor_gold_v6_s3_swa.npz",
    "v5s2_swa":   "raptor_gold_v5s2_swa.npz",
    "cnext_base": "raptor_gold_cnext_base_96_swa.npz",
    "eva02_base": "raptor_gold_eva02_base_96_swa.npz",
}

preds = {}
for name, fname in candidates.items():
    path = os.path.join(ROOT, fname)
    if not os.path.exists(path):
        print(f"Skipping {name}: {path} not found")
        continue
    data = np.load(path)
    p_key = "preds" if "preds" in data.files else ("pred" if "pred" in data.files else data.files[0])
    p = data[p_key]
    if p.shape == (N, 12):
        preds[name] = p
        aucs = [roc_auc_score(y_true[:, c], p[:, c]) for c in range(12)]
        print(f"  {name:15s} Solo Gold Macro AUC: {np.mean(aucs):.4f}")

# Rank normalization function
def to_rank(p):
    r = np.zeros_like(p)
    for c in range(12):
        r[:, c] = rankdata(p[:, c]) / float(N)
    return r

rank_preds = {name: to_rank(p) for name, p in preds.items()}

print("\n" + "=" * 85)
print("ENSEMBLE BENCHMARKS")
print("=" * 85)

# A. Sovereign 2-Student Distillation: s9 (50%) + s7 (50%)
p_duo = 0.5 * rank_preds["v6d24_s9"] + 0.5 * rank_preds["v6d_s7_swa"]
aucs_duo = [roc_auc_score(y_true[:, c], p_duo[:, c]) for c in range(12)]
print(f"1. Sovereign Distilled Duo (v6d24_s9 + v6d_s7_swa):   {np.mean(aucs_duo):.4f}")

# B. Sovereign Tri-Seed Distillation: s9 + s7 + s4 (Equal Rank)
p_tri = (rank_preds["v6d24_s9"] + rank_preds["v6d_s7_swa"] + rank_preds["v6_s4_swa"]) / 3.0
aucs_tri = [roc_auc_score(y_true[:, c], p_tri[:, c]) for c in range(12)]
print(f"2. Sovereign Tri-Seed Distillation (s9 + s7 + s4):       {np.mean(aucs_tri):.4f}")

# C. Top 4-Seed Fleet: s9 (30%) + s7 (25%) + s4 (25%) + s3 (20%)
p_quad = 0.30 * rank_preds["v6d24_s9"] + 0.25 * rank_preds["v6d_s7_swa"] + 0.25 * rank_preds["v6_s4_swa"] + 0.20 * rank_preds["v6_s3_swa"]
aucs_quad = [roc_auc_score(y_true[:, c], p_quad[:, c]) for c in range(12)]
print(f"3. Top 4-Seed Fleet (0.30 s9 + 0.25 s7 + 0.25 s4 + 0.20 s3): {np.mean(aucs_quad):.4f}")

# D. Tri-Architecture: 70% CoAtNet Tri-Seed + 15% ConvNeXt + 15% EVA-02
p_tri_arch = 0.70 * p_tri + 0.15 * rank_preds["cnext_base"] + 0.15 * rank_preds["eva02_base"]
aucs_tri_arch = [roc_auc_score(y_true[:, c], p_tri_arch[:, c]) for c in range(12)]
print(f"4. Multi-Family Fleet (70% Tri-CoAt + 15% CNext + 15% EVA-02): {np.mean(aucs_tri_arch):.4f}")

# Detailed 12-class comparison between Solo Champion (v6d24_s9) and Sovereign Tri-Seed
print("\n" + "=" * 85)
print(f"{'Finding':20s} | {'v6d24_s9 (Solo)':15s} | {'Tri-Seed Blend':15s} | {'Delta (Δ)':10s}")
print("=" * 85)
aucs_s9 = [roc_auc_score(y_true[:, c], preds["v6d24_s9"][:, c]) for c in range(12)]
for c in range(12):
    diff = aucs_tri[c] - aucs_s9[c]
    print(f"{LAB[c]:20s} | {aucs_s9[c]:15.4f} | {aucs_tri[c]:15.4f} | {diff:+10.4f}")

print("-" * 85)
print(f"{'MACRO ROC AUC':20s} | {np.mean(aucs_s9):15.4f} | {np.mean(aucs_tri):15.4f} | {np.mean(aucs_tri)-np.mean(aucs_s9):+10.4f}")
print("=" * 85)
