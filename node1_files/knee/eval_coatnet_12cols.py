import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

FINDINGS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture"
]

v6 = np.load('raptor_gold_v6.npz', allow_pickle=True)
v5s2 = np.load('raptor_gold_v5s2.npz', allow_pickle=True)

truth = v6['truth'].astype(float)
pred_v6 = v6['pred'].astype(float)
pred_v5s2 = v5s2['pred'].astype(float)

# Dual-seed raw average and rank-average
dual_raw = 0.5 * pred_v6 + 0.5 * pred_v5s2

def rank_norm(arr):
    out = np.zeros_like(arr, dtype=float)
    for j in range(arr.shape[1]):
        out[:, j] = rankdata(arr[:, j]) / len(arr[:, j])
    return out

dual_rank = 0.5 * rank_norm(pred_v6) + 0.5 * rank_norm(pred_v5s2)

auc_v6 = []
auc_v5s2 = []
auc_dual_raw = []
auc_dual_rank = []

for j in range(12):
    y = truth[:, j]
    auc_v6.append(roc_auc_score(y, pred_v6[:, j]))
    auc_v5s2.append(roc_auc_score(y, pred_v5s2[:, j]))
    auc_dual_raw.append(roc_auc_score(y, dual_raw[:, j]))
    auc_dual_rank.append(roc_auc_score(y, dual_rank[:, j]))

print("="*88)
print(f"{'#':<3} {'Finding':<22} {'v6':>12} {'v5s2':>12} {'Dual (Raw)':>15} {'Dual (Rank)':>15}")
print("="*88)
for j in range(12):
    print(f"{j:<3} {FINDINGS[j]:<22} {auc_v6[j]:12.4f} {auc_v5s2[j]:12.4f} {auc_dual_raw[j]:15.4f} {auc_dual_rank[j]:15.4f}")

print("-"*88)
print(f"{'':<3} {'MACRO (12 cols)':<22} {np.mean(auc_v6):12.4f} {np.mean(auc_v5s2):12.4f} {np.mean(auc_dual_raw):15.4f} {np.mean(auc_dual_rank):15.4f}")
print("="*88)
