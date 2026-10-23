import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LAB = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
       "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

df = pd.read_csv("/home/s05814802723/projects/knee/train.csv").dropna(subset=LAB)
y = df[LAB].values

def load_pred(path):
    d = np.load(path)
    return d["preds"] if "preds" in d.files else d[d.files[0]]

p_coat192 = load_pred("/home/s05814802723/projects/knee/pc_code192/raptor_gold_d192_s50_swa.npz")
p_cnxt_s  = load_pred("/home/s05814802723/projects/knee/pc_code192/raptor_gold_cnxt_s_breakthrough.npz")
p_sev2_f0 = load_pred("/home/s05814802723/projects/knee/pc_code192/raptor_gold_cnxt_sev2_f0.npz")
p_sev2_f1 = load_pred("/home/s05814802723/projects/knee/pc_code192/raptor_gold_cnxt_sev2_f1.npz")

p_sev2 = 0.5 * (p_sev2_f0 + p_sev2_f1)

blends = {
    "CoAtNet-192 (Baseline)": p_coat192,
    "CNXT-Small Breakthrough": p_cnxt_s,
    "CNXT-Sev2 Dual Blend": p_sev2,
    "CoAtNet-192 + CNXT-Small": 0.5 * (p_coat192 + p_cnxt_s),
    "CoAtNet-192 + CNXT-Sev2": 0.5 * (p_coat192 + p_sev2),
    "CoAtNet-192(40%) + CNXT-S(30%) + Sev2(30%)": 0.4 * p_coat192 + 0.3 * p_cnxt_s + 0.3 * p_sev2,
}

print("=" * 80)
print(f"{'Ensemble / Model':45s} | {'Macro AUC':10s}")
print("-" * 80)
for name, p in blends.items():
    aucs = [roc_auc_score(y[:, i], p[:, i]) for i in range(12)]
    print(f"{name:45s} | {np.mean(aucs):10.4f}")

# Per label on the best blend
best_p = 0.4 * p_coat192 + 0.3 * p_cnxt_s + 0.3 * p_sev2
aucs_best = [roc_auc_score(y[:, i], best_p[:, i]) for i in range(12)]
print("\n" + "=" * 80)
print(f"Per-Label Breakdown: 3-Way Cross-Architecture Blend [Macro: {np.mean(aucs_best):.4f}]")
print("=" * 80)
for i, l in enumerate(LAB):
    print(f"  {l:18s}: {aucs_best[i]:.4f}")
