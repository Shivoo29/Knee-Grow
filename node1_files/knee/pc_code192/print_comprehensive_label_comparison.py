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

models = {
    "CoAtNet-192":      "/home/s05814802723/projects/knee/pc_code192/raptor_gold_d192_s50_swa.npz",
    "CNXT-Small-Bt":    "/home/s05814802723/projects/knee/pc_code192/raptor_gold_cnxt_s_breakthrough.npz",
    "CNXT-Tiny-v1 F1":  "/home/s05814802723/projects/knee/pc_code192/raptor_gold_cnxt_f1.npz",
    "CNXT-Sev2 F0":     "/home/s05814802723/projects/knee/pc_code192/raptor_gold_cnxt_sev2_f0.npz",
    "CNXT-Sev2 F1":     "/home/s05814802723/projects/knee/pc_code192/raptor_gold_cnxt_sev2_f1.npz",
    "CNXT-Sev2 F2":     "/home/s05814802723/projects/knee/pc_code192/raptor_gold_cnxt_sev2_f2.npz",
}

preds = {name: load_pred(path) for name, path in models.items()}
preds["Sev2 3-Fold"] = (preds["CNXT-Sev2 F0"] + preds["CNXT-Sev2 F1"] + preds["CNXT-Sev2 F2"]) / 3.0

# Multi-Architecture blend
p_multi = 0.35 * preds["CoAtNet-192"] + 0.25 * preds["CNXT-Small-Bt"] + 0.15 * preds["CNXT-Tiny-v1 F1"] + 0.25 * preds["Sev2 3-Fold"]
preds["Multi-Arch Ensemble"] = p_multi

aucs = {}
for name, p in preds.items():
    aucs[name] = [roc_auc_score(y[:, i], p[:, i]) for i in range(12)]

header = f"{'Pathology':18s} | {'CoAtNet-192':11s} | {'CNXT-Small':11s} | {'Sev2-F0':9s} | {'Sev2-F1':9s} | {'Sev2-F2':9s} | {'Sev2-3Fold':11s} | {'Multi-Ensemble':14s}"
print("=" * len(header))
print(header)
print("-" * len(header))

for i, l in enumerate(LAB):
    c192 = f"{aucs['CoAtNet-192'][i]:.4f}"
    csmall = f"{aucs['CNXT-Small-Bt'][i]:.4f}"
    sf0 = f"{aucs['CNXT-Sev2 F0'][i]:.4f}"
    sf1 = f"{aucs['CNXT-Sev2 F1'][i]:.4f}"
    sf2 = f"{aucs['CNXT-Sev2 F2'][i]:.4f}"
    s3f = f"{aucs['Sev2 3-Fold'][i]:.4f}"
    multi = f"{aucs['Multi-Arch Ensemble'][i]:.4f}"
    print(f"{l:18s} | {c192:11s} | {csmall:11s} | {sf0:9s} | {sf1:9s} | {sf2:9s} | {s3f:11s} | {multi:14s}")

print("-" * len(header))
m_c192 = f"{np.mean(aucs['CoAtNet-192']):.4f}"
m_csmall = f"{np.mean(aucs['CNXT-Small-Bt']):.4f}"
m_sf0 = f"{np.mean(aucs['CNXT-Sev2 F0']):.4f}"
m_sf1 = f"{np.mean(aucs['CNXT-Sev2 F1']):.4f}"
m_sf2 = f"{np.mean(aucs['CNXT-Sev2 F2']):.4f}"
m_s3f = f"{np.mean(aucs['Sev2 3-Fold']):.4f}"
m_multi = f"{np.mean(aucs['Multi-Arch Ensemble']):.4f}"
print(f"{'MACRO AUC':18s} | {m_c192:11s} | {m_csmall:11s} | {m_sf0:9s} | {m_sf1:9s} | {m_sf2:9s} | {m_s3f:11s} | {m_multi:14s}")
print("=" * len(header))
