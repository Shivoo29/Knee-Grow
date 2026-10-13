import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr, pearsonr

FINDINGS = [
    "ACL", "PCL", "MCL", "LCL",
    "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's Cyst"
]

# Load predictions
models = {}
for name, fname in [
    ("CoAtNet v6", "raptor_gold_v6.npz"),
    ("ConvNeXt-Base 96", "raptor_gold_cnext_base_96.npz"),
    ("Bi-LSTM 44", "raptor_gold_bilstm_v1_ens.npz"),
    ("Bi-LSTM 96", "raptor_gold_bilstm_c96_ens.npz")
]:
    try:
        d = np.load(fname, allow_pickle=True)
        models[name] = {
            'pred': d['pred'].astype(np.float64),
            'truth': d['truth'].astype(np.float64),
            'ids': d['ids']
        }
        print(f"Loaded {name}: pred {models[name]['pred'].shape}")
    except Exception as e:
        print(f"Could not load {name} ({fname}): {e}")

# Check IDs alignment
ref_ids = models["CoAtNet v6"]['ids']
for name, data in models.items():
    if not np.array_equal(data['ids'], ref_ids):
        print(f"WARNING: {name} IDs do not match CoAtNet v6 IDs perfectly! Sorting/aligning...")
        id_to_idx = {uid: i for i, uid in enumerate(data['ids'])}
        reorder = [id_to_idx[uid] for uid in ref_ids]
        data['pred'] = data['pred'][reorder]
        data['truth'] = data['truth'][reorder]
        data['ids'] = ref_ids

truth = models["CoAtNet v6"]['truth']

def compute_aucs(pred, truth):
    aucs = []
    for j in range(truth.shape[1]):
        if len(np.unique(truth[:, j])) > 1:
            auc = roc_auc_score(truth[:, j], pred[:, j])
        else:
            auc = 0.5
        aucs.append(auc)
    return np.array(aucs), np.mean(aucs)

print("\n" + "="*70)
print(f"{'Finding':<20} " + " ".join([f"{name[:10]:>11}" for name in models.keys()]))
print("="*70)

all_aucs = {}
for name, data in models.items():
    aucs, macro = compute_aucs(data['pred'], truth)
    all_aucs[name] = (aucs, macro)

for j, f in enumerate(FINDINGS):
    row = f"{f:<20} "
    for name in models.keys():
        row += f"{all_aucs[name][0][j]:11.4f} "
    print(row)

print("-"*70)
row = f"{'MACRO ROC-AUC':<20} "
for name in models.keys():
    row += f"{all_aucs[name][1]:11.4f} "
print(row)
print("="*70)

# Diversity Analysis: Pairwise Pearson & Spearman correlation of predictions
print("\n--- Correlation Matrix against CoAtNet v6 ---")
for name, data in models.items():
    if name == "CoAtNet v6":
        continue
    p_corrs = [pearsonr(models["CoAtNet v6"]['pred'][:, j], data['pred'][:, j])[0] for j in range(12)]
    s_corrs = [spearmanr(models["CoAtNet v6"]['pred'][:, j], data['pred'][:, j])[0] for j in range(12)]
    print(f"{name:<20} Pearson: {np.nanmean(p_corrs):.4f} | Spearman: {np.nanmean(s_corrs):.4f}")

# Rank-transform helper
def rank_normalize(arr):
    out = np.zeros_like(arr)
    for j in range(arr.shape[1]):
        out[:, j] = rankdata(arr[:, j]) / len(arr[:, j])
    return out

# 1. Uniform / Scalar Blends
print("\n--- Uniform Ensembles (Rank-Averaged) ---")
v6_rank = rank_normalize(models["CoAtNet v6"]['pred'])
cn_rank = rank_normalize(models["ConvNeXt-Base 96"]['pred'])
b96_rank = rank_normalize(models["Bi-LSTM 96"]['pred'])
b44_rank = rank_normalize(models["Bi-LSTM 44"]['pred'])

# Baseline 2-way: CoAtNet + ConvNeXt
for w in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    blend = w * v6_rank + (1.0 - w) * cn_rank
    _, auc = compute_aucs(blend, truth)
    print(f"CoAtNet {w:.2f} + ConvNeXt {1-w:.2f} -> Macro AUC: {auc:.4f}")

# 3-way: CoAtNet + ConvNeXt + Bi-LSTM 96
print("\n--- 3-Way Grid Blends (CoAtNet + ConvNeXt + Bi-LSTM 96) ---")
best_scalar_auc = 0.0
best_scalar_weights = None
for w_v6 in np.linspace(0.3, 0.7, 9):
    for w_cn in np.linspace(0.3, 0.7, 9):
        for w_b in np.linspace(0.0, 0.2, 5):
            tot = w_v6 + w_cn + w_b
            if abs(tot - 1.0) < 1e-4:
                blend = w_v6 * v6_rank + w_cn * cn_rank + w_b * b96_rank
                _, auc = compute_aucs(blend, truth)
                if auc > best_scalar_auc:
                    best_scalar_auc = auc
                    best_scalar_weights = (w_v6, w_cn, w_b)

print(f"Best Scalar 3-Way: v6={best_scalar_weights[0]:.2f}, CN={best_scalar_weights[1]:.2f}, BiLSTM={best_scalar_weights[2]:.2f} -> Macro AUC: {best_scalar_auc:.4f}")

# 2. Finding-Specific Optimal Blending (Per-Class Optimization)
print("\n--- Finding-Specific Optimal Blending ---")
optimal_finding_blend = np.zeros_like(truth)
opt_weights_per_finding = []

for j in range(12):
    best_j_auc = -1.0
    best_w = None
    y_col = truth[:, j]
    p_v6 = v6_rank[:, j]
    p_cn = cn_rank[:, j]
    p_b96 = b96_rank[:, j]
    
    # Dense simplex search for finding j
    for w1 in np.linspace(0, 1, 21):
        for w2 in np.linspace(0, 1 - w1, 21):
            w3 = 1.0 - w1 - w2
            if w3 < -1e-5:
                continue
            pred_j = w1 * p_v6 + w2 * p_cn + w3 * p_b96
            score = roc_auc_score(y_col, pred_j)
            if score > best_j_auc:
                best_j_auc = score
                best_w = (w1, w2, w3)
                
    optimal_finding_blend[:, j] = best_w[0] * p_v6 + best_w[1] * p_cn + best_w[2] * p_b96
    opt_weights_per_finding.append(best_w)
    print(f"{FINDINGS[j]:<20} AUC: {best_j_auc:.4f} | Weights (v6: {best_w[0]:.2f}, CN: {best_w[1]:.2f}, LSTM96: {best_w[2]:.2f})")

_, opt_macro = compute_aucs(optimal_finding_blend, truth)
print(f"\nOPTIMAL FINDING-SPECIFIC MACRO AUC: {opt_macro:.4f} (CoAtNet v6 solo: {all_aucs['CoAtNet v6'][1]:.4f}, ConvNeXt solo: {all_aucs['ConvNeXt-Base 96'][1]:.4f})")
