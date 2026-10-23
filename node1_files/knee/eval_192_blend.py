import os
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

# Ground truth
df = pd.read_csv(f"{ROOT}/train.csv")
gold = df.dropna(subset=LAB)
y_true = gold[LAB].values.astype(float)
N = len(y_true)

# Load predictions
p_192 = np.load(f"{ROOT}/pc_code192/raptor_gold_d192_s50.npz")["pred"]
p_96_s9 = np.load(f"{ROOT}/raptor_gold_v6d24_s9.npz")["pred"]
p_96_s7 = np.load(f"{ROOT}/raptor_gold_v6d_s7_swa.npz")["pred"]

def to_rank(p):
    r = np.zeros_like(p)
    for c in range(12):
        r[:, c] = rankdata(p[:, c]) / float(N)
    return r

r_192 = to_rank(p_192)
r_96_s9 = to_rank(p_96_s9)
r_96_s7 = to_rank(p_96_s7)

# Individual Macro AUCs
auc_192 = [roc_auc_score(y_true[:, c], p_192[:, c]) for c in range(12)]
auc_s9 = [roc_auc_score(y_true[:, c], p_96_s9[:, c]) for c in range(12)]
auc_s7 = [roc_auc_score(y_true[:, c], p_96_s7[:, c]) for c in range(12)]

# Blends
# 1. 192-slice (50%) + 96-slice s9 (50%)
blend_192_s9 = 0.5 * r_192 + 0.5 * r_96_s9
auc_blend_2 = [roc_auc_score(y_true[:, c], blend_192_s9[:, c]) for c in range(12)]

# 2. 192-slice (40%) + 96-slice s9 (30%) + 96-slice s7 (30%)
blend_tri = 0.40 * r_192 + 0.30 * r_96_s9 + 0.30 * r_96_s7
auc_blend_tri = [roc_auc_score(y_true[:, c], blend_tri[:, c]) for c in range(12)]

print("=" * 90)
print("12-CLASS VERIFIED PERFORMANCE BREAKDOWN & BLEND MATRIX")
print("=" * 90)
print(f"{'Finding':18s} | {'192-sl (s50)':12s} | {'96-sl (s9)':12s} | {'96-sl (s7)':12s} | {'192+96 (Dual)':13s} | {'Tri-Blend':10s}")
print("-" * 90)
for c in range(12):
    print(f"{LAB[c]:18s} | {auc_192[c]:12.4f} | {auc_s9[c]:12.4f} | {auc_s7[c]:12.4f} | {auc_blend_2[c]:13.4f} | {auc_blend_tri[c]:10.4f}")

print("=" * 90)
print(f"{'MACRO ROC AUC':18s} | {np.mean(auc_192):12.4f} | {np.mean(auc_s9):12.4f} | {np.mean(auc_s7):12.4f} | {np.mean(auc_blend_2):13.4f} | {np.mean(auc_blend_tri):10.4f}")
print("=" * 90)
