import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

LAB = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
       "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

# Load gold truth
df = pd.read_csv("/home/s05814802723/projects/knee/train.csv")
gold_df = df.dropna(subset=LAB)
y_true = gold_df[LAB].values

models = {
    "CoAtNet_96_s9":   "/home/s05814802723/projects/knee/raptor_gold_v6d24_s9_swa.npz",
    "CoAtNet_192_s50": "/home/s05814802723/projects/knee/pc_code192/raptor_gold_d192_s50_swa.npz",
    "CNXT_T_f0_swa":   "/home/s05814802723/projects/knee/pc_code192/raptor_gold_cnxt_f0_swa.npz",
    "CNXT_T_f1_best":  "/home/s05814802723/projects/knee/pc_code192/raptor_gold_cnxt_f1.npz",
    "CNXT_S_Breakthru": "/home/s05814802723/projects/knee/pc_code192/raptor_gold_cnxt_s_breakthrough.npz"
}

preds = {}
for name, path in models.items():
    try:
        d = np.load(path)
        k = "preds" if "preds" in d.files else ("pred" if "pred" in d.files else d.files[0])
        preds[name] = d[k]
    except Exception as e:
        print(f"Failed to load {name}: {e}")

print("=" * 85)
print("INDIVIDUAL MODEL GOLD-58 MACRO AUC AND BOTTLENECK LABELS:")
print("=" * 85)
print(f"{'Model':20s} | {'Macro':6s} | {'Lat OA':6s} | {'PF OA':6s} | {'Synov':6s} | {'Lat Men':7s} | {'ACL':6s} | {'MCL':6s}")
print("-" * 85)

def eval_preds(p):
    aucs = [roc_auc_score(y_true[:, i], p[:, i]) for i in range(12)]
    return np.mean(aucs), aucs

for name, p in preds.items():
    mean_auc, aucs = eval_preds(p)
    print(f"{name:20s} | {mean_auc:.4f} | {aucs[5]:.4f} | {aucs[6]:.4f} | {aucs[8]:.4f} | {aucs[3]:.4f}  | {aucs[0]:.4f} | {aucs[1]:.4f}")

print("\n" + "=" * 85)
print("CROSS-ARCHITECTURE BLENDS (INCLUDING BREAKTHROUGH MODEL):")
print("=" * 85)

blends = {
    "CoAtNet Dual (96 + 192)":        [preds["CoAtNet_96_s9"], preds["CoAtNet_192_s50"]],
    "ConvNeXt Dual (T_f1 + S_bt)":     [preds["CNXT_T_f1_best"], preds["CNXT_S_Breakthru"]],
    "ConvNeXt Trio (T_f0 + T_f1 + S)": [preds["CNXT_T_f0_swa"], preds["CNXT_T_f1_best"], preds["CNXT_S_Breakthru"]],
    "CoAtNet-192 + CNXT-Small":       [preds["CoAtNet_192_s50"], preds["CNXT_S_Breakthru"]],
    "CoAtNet-192 + CNXT (T_f1 + S)":   [preds["CoAtNet_192_s50"], 0.5 * (preds["CNXT_T_f1_best"] + preds["CNXT_S_Breakthru"])],
    "All-5 Model Equal Blend":         [preds["CoAtNet_96_s9"], preds["CoAtNet_192_s50"], preds["CNXT_T_f0_swa"], preds["CNXT_T_f1_best"], preds["CNXT_S_Breakthru"]],
    "Weighted: CoAt(40%) + CNXT(60%)": [0.4 * preds["CoAtNet_192_s50"] + 0.3 * preds["CNXT_T_f1_best"] + 0.3 * preds["CNXT_S_Breakthru"]]
}

for b_name, b_list in blends.items():
    if len(b_list) == 1:
        p_blend = b_list[0]
    else:
        p_blend = np.mean(b_list, axis=0)
    mean_auc, aucs = eval_preds(p_blend)
    print(f"{b_name:34s} | Macro: {mean_auc:.4f} | LatOA: {aucs[5]:.4f} | PFOA: {aucs[6]:.4f} | Synov: {aucs[8]:.4f} | LatMen: {aucs[3]:.4f}")

# Full per-label breakdown for CoAtNet-192 + CNXT (T_f1 + S)
best_p = 0.5 * preds["CoAtNet_192_s50"] + 0.25 * preds["CNXT_T_f1_best"] + 0.25 * preds["CNXT_S_Breakthru"]
m_auc, all_aucs = eval_preds(best_p)
print("\n" + "=" * 85)
print(f"PER-LABEL BREAKDOWN: CoAtNet-192 (50%) + CNXT-Tiny F1 (25%) + CNXT-Small (25%) [Macro: {m_auc:.4f}]")
print("=" * 85)
for i, l in enumerate(LAB):
    print(f"  {l:18s}: {all_aucs[i]:.4f}")
