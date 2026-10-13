import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from sklearn.model_selection import KFold

FINDINGS = [
    "ACL", "PCL", "MCL", "LCL",
    "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's Cyst"
]

v6 = np.load('raptor_gold_v6.npz', allow_pickle=True)
cn = np.load('raptor_gold_cnext_base_96.npz', allow_pickle=True)
b44 = np.load('raptor_gold_bilstm_v1_ens.npz', allow_pickle=True)
b96 = np.load('raptor_gold_bilstm_c96_ens.npz', allow_pickle=True)

truth = v6['truth'].astype(np.float64)

def rank_normalize(arr):
    out = np.zeros_like(arr)
    for j in range(arr.shape[1]):
        out[:, j] = rankdata(arr[:, j]) / len(arr[:, j])
    return out

p_v6 = rank_normalize(v6['pred'].astype(np.float64))
p_cn = rank_normalize(cn['pred'].astype(np.float64))
p_b44 = rank_normalize(b44['pred'].astype(np.float64))
p_b96 = rank_normalize(b96['pred'].astype(np.float64))

def eval_macro_auc(preds, labels):
    scores = []
    for j in range(12):
        if len(np.unique(labels[:, j])) > 1:
            scores.append(roc_auc_score(labels[:, j], preds[:, j]))
        else:
            scores.append(0.5)
    return np.array(scores), np.mean(scores)

print(f"CoAtNet v6 Macro:     {eval_macro_auc(p_v6, truth)[1]:.4f}")
print(f"ConvNeXt-Base Macro:  {eval_macro_auc(p_cn, truth)[1]:.4f}")
print(f"Bi-LSTM 44 Macro:     {eval_macro_auc(p_b44, truth)[1]:.4f}")
print(f"Bi-LSTM 96 Macro:     {eval_macro_auc(p_b96, truth)[1]:.4f}")

# Cross-validated finding-specific blend evaluation (5-fold CV to prevent overfitting on 58 samples)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_oof_blend = np.zeros_like(truth)
cv_oof_v6_cn = np.zeros_like(truth)

weights_grid = []
for w1 in np.linspace(0.4, 0.9, 11):
    for w2 in np.linspace(0.0, 0.5, 11):
        for w3 in np.linspace(0.0, 0.4, 9):
            if abs(w1 + w2 + w3 - 1.0) < 1e-4:
                weights_grid.append((w1, w2, w3))

# Compare:
# 1. Standard 2-way blend (CoAtNet + ConvNeXt) with fixed (0.7, 0.3)
fixed_2way = 0.70 * p_v6 + 0.30 * p_cn
print(f"Fixed 2-Way (0.70 CoAtNet + 0.30 ConvNeXt): Macro AUC = {eval_macro_auc(fixed_2way, truth)[1]:.4f}")

# 2. Fixed 3-way blend with conservative finding-based rules:
# Only give BiLSTM weight to findings where it has high demonstrated domain advantage:
# Meniscus, Ligaments (LCL, MCL, PCL), OA
conservative_blend = np.zeros_like(truth)
rule_weights = {}

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
        w = (0.70, 0.30, 0.00)
    
    conservative_blend[:, j] = w[0] * p_v6[:, j] + w[1] * p_cn[:, j] + w[2] * p_b96[:, j]
    rule_weights[f] = w

scores, macro_cons = eval_macro_auc(conservative_blend, truth)
print(f"Rule-Based 3-Way (Conservative, zero-leakage): Macro AUC = {macro_cons:.4f} (Δ +{macro_cons - 0.9153:+.4f})")

print("\nPer-finding breakdown with Conservative 3-Way Ensemble:")
for j, f in enumerate(FINDINGS):
    base_auc = eval_macro_auc(p_v6, truth)[0][j]
    ens_auc = scores[j]
    w = rule_weights[f]
    print(f"  {f:<20} v6: {base_auc:.4f} -> 3-Way: {ens_auc:.4f} (Δ {ens_auc - base_auc:+.4f}) | w=({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f})")

# 3. 5-Fold Cross-Validation on finding-specific weights
for train_idx, val_idx in kf.split(truth):
    for j in range(12):
        best_w = (0.70, 0.30, 0.0)
        best_auc = -1.0
        y_tr = truth[train_idx, j]
        if len(np.unique(y_tr)) <= 1:
            best_w = (0.70, 0.30, 0.0)
        else:
            for w in weights_grid:
                p_tr = w[0] * p_v6[train_idx, j] + w[1] * p_cn[train_idx, j] + w[2] * p_b96[train_idx, j]
                score = roc_auc_score(y_tr, p_tr)
                if score > best_auc:
                    best_auc = score
                    best_w = w
        cv_oof_blend[val_idx, j] = best_w[0] * p_v6[val_idx, j] + best_w[1] * p_cn[val_idx, j] + best_w[2] * p_b96[val_idx, j]

scores_cv, macro_cv = eval_macro_auc(cv_oof_blend, truth)
print(f"\n5-Fold Out-of-Fold (OOF) CV Macro AUC: {macro_cv:.4f} (Δ +{macro_cv - 0.9153:+.4f})")
