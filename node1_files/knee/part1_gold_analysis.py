import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

os.environ["CUDA_VISIBLE_DEVICES"] = ""

LAB = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
       'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

ROOT = "/home/s05814802723/projects/knee"

train = pd.read_csv(f"{ROOT}/train.csv")
gold_df = train.dropna(subset=LAB)
y_true = gold_df[LAB].values.astype(float)
gold_uids = gold_df['StudyInstanceUID'].values.astype(str)
N_GOLD = len(y_true)

# Candidate gold npz files
gold_files = [
    ("v6d_s7_swa", f"{ROOT}/raptor_gold_v6d_s7_swa.npz"),
    ("v6_s4_swa",  f"{ROOT}/raptor_gold_v6_s4_swa.npz"),
    ("v6_s3_swa",  f"{ROOT}/raptor_gold_v6_s3_swa.npz"),
    ("v6_s5_swa",  f"{ROOT}/raptor_gold_v6_s5_swa.npz"),
    ("v6_swa",     f"{ROOT}/raptor_gold_v6_swa.npz"),
    ("v5s2_swa",   f"{ROOT}/raptor_gold_v5s2_swa.npz"),
    ("v5s1_swa",   f"{ROOT}/raptor_gold_v5s1_swa.npz"),
    ("v5_swa",     f"{ROOT}/raptor_gold_v5_swa.npz"),
    ("v4base_swa", f"{ROOT}/raptor_gold_v4base_swa.npz"),
    ("v6d24_s9",   f"{ROOT}/raptor_gold_v6d24_s9.npz"),
    ("xfmr_c96_ens", f"{ROOT}/raptor_gold_xfmr_c96_ens.npz")
]

models = {}
for name, path in gold_files:
    if not os.path.exists(path):
        continue
    try:
        data = np.load(path)
        keys = list(data.files)
        # determine pred key
        p_key = 'preds' if 'preds' in keys else ('pred' if 'pred' in keys else keys[0])
        p = data[p_key]
        if p.shape == (N_GOLD, 12):
            models[name] = p
        elif len(p.shape) == 1 and len(p) == N_GOLD * 12:
            models[name] = p.reshape(N_GOLD, 12)
        else:
            print(f"Skipping {name}: shape mismatch {p.shape}")
    except Exception as e:
        print(f"Error loading {name}: {e}")

model_names = list(models.keys())
M = len(model_names)
print(f"Loaded {M} models with Gold-58 predictions: {model_names}")

# 1. Individual model 12-class ROC AUC table
print("\n" + "="*100)
print(f"{'Finding':20s} | " + " | ".join(f"{m[:8]:8s}" for m in model_names) + " | Best Solo")
print("="*100)

individual_aucs = {m: [] for m in model_names}
best_solo_per_class = []

for c_idx, c_name in enumerate(LAB):
    row_str = f"{c_name:20s} | "
    best_c_auc = -1.0
    best_c_model = ""
    for m in model_names:
        auc = roc_auc_score(y_true[:, c_idx], models[m][:, c_idx])
        individual_aucs[m].append(auc)
        row_str += f"{auc:8.4f} | "
        if auc > best_c_auc:
            best_c_auc = auc
            best_c_model = m
    best_solo_per_class.append((best_c_auc, best_c_model))
    row_str += f"{best_c_auc:.4f} ({best_c_model[:8]})"
    print(row_str)

print("-" * 100)
macro_str = f"{'MACRO ROC AUC':20s} | "
for m in model_names:
    macro_str += f"{np.mean(individual_aucs[m]):8.4f} | "
oracle_upper_bound = np.mean([b[0] for b in best_solo_per_class])
macro_str += f"{oracle_upper_bound:.4f} (ORACLE UPPER)"
print(macro_str)
print("="*100)

# Convert predictions to rank percentiles and z-score logits
rank_preds = {}
zscore_preds = {}
for m in model_names:
    r_m = np.zeros_like(models[m])
    z_m = np.zeros_like(models[m])
    for c_idx in range(12):
        r_m[:, c_idx] = rankdata(models[m][:, c_idx]) / float(N_GOLD)
        # z-score of logits / probs
        p_clip = np.clip(models[m][:, c_idx], 1e-6, 1 - 1e-6)
        log_odds = np.log(p_clip / (1.0 - p_clip))
        z_m[:, c_idx] = (log_odds - log_odds.mean()) / max(log_odds.std(), 1e-6)
    rank_preds[m] = r_m
    zscore_preds[m] = z_m

# 2. Equal-weight blend across all models
# a) Raw prob mean
raw_eq = np.mean([models[m] for m in model_names], axis=0)
aucs_raw_eq = [roc_auc_score(y_true[:, c], raw_eq[:, c]) for c in range(12)]

# b) Rank percentile mean
rank_eq = np.mean([rank_preds[m] for m in model_names], axis=0)
aucs_rank_eq = [roc_auc_score(y_true[:, c], rank_eq[:, c]) for c in range(12)]

# c) Z-score mean
zscore_eq = np.mean([zscore_preds[m] for m in model_names], axis=0)
aucs_zscore_eq = [roc_auc_score(y_true[:, c], zscore_eq[:, c]) for c in range(12)]

print("\n=== BLEND COMPARISON ACROSS ALL AVAILABLE MODELS ===")
print(f"Equal-Weight Raw Prob Blend Macro AUC:   {np.mean(aucs_raw_eq):.4f}")
print(f"Equal-Weight Rank Percentile Blend Macro: {np.mean(aucs_rank_eq):.4f}")
print(f"Equal-Weight Z-Score Logit Blend Macro:  {np.mean(aucs_zscore_eq):.4f}")
print(f"Per-Class Best Model (Oracle Upper):     {oracle_upper_bound:.4f}")

# 3. Leave-One-Study-Out (LOSO) Cross-Fitted Per-Class Optimization
# We want to avoid overfitting on 58 studies, so we optimize weights out-of-fold using LOSO (58 folds of 57 train, 1 test)
# For each study i in 0..57:
#   fit weights w on y_true[-i] using Nelder-Mead on rank/prob blend
#   predict study i
print("\n=== LEAVE-ONE-STUDY-OUT (LOSO) PER-CLASS OPTIMIZATION ===")
loso_preds = np.zeros_like(y_true)

# Use top diverse models for stable weighting: v6d_s7_swa, v6_s4_swa, v6_s3_swa, v5s2_swa, v6_s5_swa, v6_swa
opt_models = ["v6d_s7_swa", "v6_s4_swa", "v6_s3_swa", "v5s2_swa", "v6_s5_swa", "v6_swa"]
opt_indices = [model_names.index(m) for m in opt_models]
K_opt = len(opt_models)

# Loss function: negative smooth AUC or cross-entropy surrogate
def neg_auc_loss(weights, X_c, y_c):
    # softmax weights
    w = np.exp(weights - np.max(weights))
    w = w / np.sum(w)
    pred = X_c @ w
    # approximate ranking loss / continuous surrogate
    # Pairwise logistic ranking loss
    pos_idx = np.where(y_c == 1)[0]
    neg_idx = np.where(y_c == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return 0.0
    diff = pred[pos_idx, None] - pred[None, neg_idx] # pos - neg should be > 0
    loss = np.mean(np.log1p(np.exp(-3.0 * diff)))
    # add Dirichlet regularization towards equal weights
    reg = 0.05 * np.sum((w - 1.0/K_opt)**2)
    return loss + reg

final_weights = {}
for c in range(12):
    X_c = np.column_stack([rank_preds[m][:, c] for m in opt_models])
    y_c = y_true[:, c]
    
    # Check if class is all 0 or 1
    if len(np.unique(y_c)) < 2:
        loso_preds[:, c] = np.mean(X_c, axis=1)
        continue
        
    for i in range(N_GOLD):
        # train set
        train_idx = [j for j in range(N_GOLD) if j != i]
        X_train, y_train = X_c[train_idx], y_c[train_idx]
        
        # init equal
        init_w = np.zeros(K_opt)
        res = minimize(neg_auc_loss, init_w, args=(X_train, y_train), method='Nelder-Mead', options={'maxiter': 100})
        w = np.exp(res.x - np.max(res.x))
        w = w / np.sum(w)
        loso_preds[i, c] = X_c[i] @ w
        
    # Fit full dataset weights for reporting
    res_full = minimize(neg_auc_loss, np.zeros(K_opt), args=(X_c, y_c), method='Nelder-Mead', options={'maxiter': 200})
    w_full = np.exp(res_full.x - np.max(res_full.x))
    w_full = w_full / np.sum(w_full)
    final_weights[LAB[c]] = {opt_models[j]: float(w_full[j]) for j in range(K_opt)}

loso_aucs = [roc_auc_score(y_true[:, c], loso_preds[:, c]) for c in range(12)]
print(f"\nLOSO Cross-Fitted Rank Blend Macro AUC: {np.mean(loso_aucs):.4f}")

# Weak classes analysis
weak_classes = ['Lateral OA', 'PF OA', 'Synovitis', 'Lateral Meniscus']
print("\n=== PERFORMANCE ON THE 4 WEAK CLASSES ===")
print(f"{'Weak Class':20s} | {'Best Solo':10s} | {'Equal Raw':10s} | {'Equal Rank':10s} | {'LOSO Opt':10s} | {'Gain vs Solo'}")
print("-" * 80)
for wc in weak_classes:
    c_idx = LAB.index(wc)
    best_s = best_solo_per_class[c_idx][0]
    eq_r = aucs_raw_eq[c_idx]
    eq_rk = aucs_rank_eq[c_idx]
    l_auc = loso_aucs[c_idx]
    gain = l_auc - best_s
    sign = "+" if gain >= 0 else ""
    print(f"{wc:20s} | {best_s:10.4f} | {eq_r:10.4f} | {eq_rk:10.4f} | {l_auc:10.4f} | {sign}{gain:.4f}")

print("\n=== OPTIMAL MODEL WEIGHTS PER CLASS (Dirichlet Regularized) ===")
for c_name, w_dict in final_weights.items():
    top_w = sorted(w_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    w_str = ", ".join(f"{k}: {v:.2f}" for k, v in top_w)
    print(f"{c_name:20s}: {w_str}")
