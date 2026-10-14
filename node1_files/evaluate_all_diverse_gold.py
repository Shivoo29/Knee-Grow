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
N_GOLD = len(y_true)

# Candidate diverse models
diverse_files = [
    # CoAtNet multi-seeds & distillation
    ("CoAtNet-v6d_s7",  f"{ROOT}/raptor_gold_v6d_s7_swa.npz"),
    ("CoAtNet-v6_s4",   f"{ROOT}/raptor_gold_v6_s4_swa.npz"),
    ("CoAtNet-v6_s3",   f"{ROOT}/raptor_gold_v6_s3_swa.npz"),
    ("CoAtNet-v6_s5",   f"{ROOT}/raptor_gold_v6_s5_swa.npz"),
    ("CoAtNet-v6_s1",   f"{ROOT}/raptor_gold_v6_swa.npz"),
    ("CoAtNet-v5s2",    f"{ROOT}/raptor_gold_v5s2_swa.npz"),
    ("CoAtNet-v6d24_s9",f"{ROOT}/raptor_gold_v6d24_s9.npz"),
    # ConvNeXt-Base
    ("ConvNeXt-Base",   f"{ROOT}/raptor_gold_cnext_base_96_swa.npz"),
    # EVA-02 Base
    ("EVA-02-Base",     f"{ROOT}/raptor_gold_eva02_base_96_swa.npz"),
    # MaxViT-B
    ("MaxViT-Base",     f"{ROOT}/raptor_gold_maxvit_b_96.npz"),
    # Compartment ROI Models
    ("ROI-CoAt-F0",     f"{ROOT}/raptor_gold_coat_roi_5fold_f0_swa.npz"),
    ("ROI-CoAt-F1",     f"{ROOT}/raptor_gold_coat_roi_5fold_f1.npz"),
    ("ROI-LatOA-F0",    f"{ROOT}/raptor_gold_coat_latoa_roi_f0.npz"),
    # Sequence Models
    ("BiLSTM-c96",      f"{ROOT}/raptor_gold_bilstm_c96_ens.npz"),
    ("Transformer-c96", f"{ROOT}/raptor_gold_xfmr_c96_ens.npz"),
]

models = {}
for name, path in diverse_files:
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue
    try:
        data = np.load(path)
        keys = list(data.files)
        p_key = 'preds' if 'preds' in keys else ('pred' if 'pred' in keys else keys[0])
        p = data[p_key]
        if p.shape == (N_GOLD, 12):
            models[name] = p
        elif len(p.shape) == 1 and len(p) == N_GOLD * 12:
            models[name] = p.reshape(N_GOLD, 12)
        else:
            print(f"Skipping {name}: shape {p.shape}")
    except Exception as e:
        print(f"Error loading {name}: {e}")

model_names = list(models.keys())
print(f"Loaded {len(model_names)} diverse architectures & models: {model_names}\n")

# Compute per-model 12-class AUCs
aucs_dict = {}
for m in model_names:
    m_aucs = [roc_auc_score(y_true[:, c], models[m][:, c]) for c in range(12)]
    aucs_dict[m] = m_aucs

df_aucs = pd.DataFrame(aucs_dict, index=LAB)
df_aucs.loc["MACRO_AUC"] = df_aucs.mean(axis=0)

# Print full table
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 150)
print(df_aucs.round(4).to_string())

# Rank normalization
rank_preds = {}
for m in model_names:
    r_m = np.zeros_like(models[m])
    for c in range(12):
        r_m[:, c] = rankdata(models[m][:, c]) / float(N_GOLD)
    rank_preds[m] = r_m

# Blends:
# 1. Pure CoAtNet Top-3
coat_top3 = ["CoAtNet-v6d_s7", "CoAtNet-v6_s4", "CoAtNet-v6_s3"]
p_coat_top3 = np.mean([rank_preds[m] for m in coat_top3], axis=0)
aucs_coat_top3 = [roc_auc_score(y_true[:, c], p_coat_top3[:, c]) for c in range(12)]

# 2. Multi-Architecture Blend: CoAtNet + ConvNeXt + EVA-02 + MaxViT + ROI-F0 + v5s2
multi_arch = ["CoAtNet-v6d_s7", "CoAtNet-v6_s4", "CoAtNet-v5s2", "ConvNeXt-Base", "EVA-02-Base", "MaxViT-Base", "ROI-CoAt-F0"]
# filter available
multi_arch = [m for m in multi_arch if m in model_names]
p_multi = np.mean([rank_preds[m] for m in multi_arch], axis=0)
aucs_multi = [roc_auc_score(y_true[:, c], p_multi[:, c]) for c in range(12)]

# 3. LOSO Cross-Fitted Per-Class Optimization across multi-architecture set
print(f"\nEvaluating Multi-Architecture Set ({len(multi_arch)} models): {multi_arch}")
K_m = len(multi_arch)
loso_multi_preds = np.zeros_like(y_true)

def neg_auc_loss(weights, X_c, y_c):
    w = np.exp(weights - np.max(weights))
    w = w / np.sum(w)
    pred = X_c @ w
    pos_idx = np.where(y_c == 1)[0]
    neg_idx = np.where(y_c == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return 0.0
    diff = pred[pos_idx, None] - pred[None, neg_idx]
    loss = np.mean(np.log1p(np.exp(-3.0 * diff)))
    reg = 0.05 * np.sum((w - 1.0/K_m)**2)
    return loss + reg

opt_weights_multi = {}
for c in range(12):
    X_c = np.column_stack([rank_preds[m][:, c] for m in multi_arch])
    y_c = y_true[:, c]
    for i in range(N_GOLD):
        train_idx = [j for j in range(N_GOLD) if j != i]
        X_train, y_train = X_c[train_idx], y_c[train_idx]
        res = minimize(neg_auc_loss, np.zeros(K_m), args=(X_train, y_train), method='Nelder-Mead', options={'maxiter': 100})
        w = np.exp(res.x - np.max(res.x))
        w = w / np.sum(w)
        loso_multi_preds[i, c] = X_c[i] @ w
    # Fit full
    res_f = minimize(neg_auc_loss, np.zeros(K_m), args=(X_c, y_c), method='Nelder-Mead', options={'maxiter': 200})
    w_f = np.exp(res_f.x - np.max(res_f.x))
    w_f = w_f / np.sum(w_f)
    opt_weights_multi[LAB[c]] = {multi_arch[j]: float(w_f[j]) for j in range(K_m)}

aucs_loso_multi = [roc_auc_score(y_true[:, c], loso_multi_preds[:, c]) for c in range(12)]

print("\n" + "="*90)
print(f"{'Finding Label':20s} | {'CoAt Top-3':11s} | {'Multi Equal':11s} | {'Multi LOSO':11s} | {'Gain vs CoAt'}")
print("="*90)
for c in range(12):
    c_name = LAB[c]
    c_top3 = aucs_coat_top3[c]
    c_mul = aucs_multi[c]
    c_loso = aucs_loso_multi[c]
    diff = c_loso - c_top3
    sign = "+" if diff >= 0 else ""
    print(f"{c_name:20s} | {c_top3:11.4f} | {c_mul:11.4f} | {c_loso:11.4f} | {sign}{diff:.4f}")

print("-" * 90)
print(f"{'MACRO ROC AUC':20s} | {np.mean(aucs_coat_top3):11.4f} | {np.mean(aucs_multi):11.4f} | {np.mean(aucs_loso_multi):11.4f} | {np.mean(aucs_loso_multi)-np.mean(aucs_coat_top3):+.4f}")
print("="*90)

print("\n=== MULTI-ARCHITECTURE WEIGHT ALLOCATION PER CLASS ===")
for c_name, w_dict in opt_weights_multi.items():
    top_w = sorted(w_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    w_str = ", ".join(f"{k}: {v:.2f}" for k, v in top_w)
    print(f"{c_name:20s}: {w_str}")
