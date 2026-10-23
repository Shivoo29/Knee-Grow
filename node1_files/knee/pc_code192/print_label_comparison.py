import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LAB = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
       "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

df = pd.read_csv("/home/s05814802723/projects/knee/train.csv").dropna(subset=LAB)
y = df[LAB].values

models = {
    "CoAtNet-96": "projects/knee/raptor_gold_v6d24_s9_swa.npz",
    "CoAtNet-192": "projects/knee/pc_code192/raptor_gold_d192_s50_swa.npz",
    "CNXT-Tiny F0": "projects/knee/pc_code192/raptor_gold_cnxt_f0_swa.npz",
    "CNXT-Tiny F1": "projects/knee/pc_code192/raptor_gold_cnxt_f1.npz",
    "CNXT-Small (Live Ep8)": "projects/knee/pc_code192/raptor_gold_cnxt_s_breakthrough.npz",
}

aucs = {}
for name, rel_path in models.items():
    path = f"/home/s05814802723/{rel_path}"
    d = np.load(path)
    k = "preds" if "preds" in d.files else ("pred" if "pred" in d.files else d.files[0])
    p = d[k]
    aucs[name] = [roc_auc_score(y[:, i], p[:, i]) for i in range(12)]

header = f"{'Label':18s} | {'CoAtNet-96':10s} | {'CoAtNet-192':11s} | {'CNXT-Tiny F0':12s} | {'CNXT-Tiny F1':12s} | {'CNXT-Small (Live)':18s} | {'Delta vs CoAt192':16s} | {'Delta vs TinyF1':16s}"
print(header)
print("-" * len(header))

for i, l in enumerate(LAB):
    c96 = aucs["CoAtNet-96"][i]
    c192 = aucs["CoAtNet-192"][i]
    cf0 = aucs["CNXT-Tiny F0"][i]
    cf1 = aucs["CNXT-Tiny F1"][i]
    c_live = aucs["CNXT-Small (Live Ep8)"][i]
    diff_coat = f"{c_live - c192:+.4f}"
    diff_tiny = f"{c_live - cf1:+.4f}"
    print(f"{l:18s} | {c96:10.4f} | {c192:11.4f} | {cf0:12.4f} | {cf1:12.4f} | {c_live:18.4f} | {diff_coat:16s} | {diff_tiny:16s}")

print("-" * len(header))
m_c96 = np.mean(aucs["CoAtNet-96"])
m_c192 = np.mean(aucs["CoAtNet-192"])
m_cf0 = np.mean(aucs["CNXT-Tiny F0"])
m_cf1 = np.mean(aucs["CNXT-Tiny F1"])
m_live = np.mean(aucs["CNXT-Small (Live Ep8)"])
diff_coat_m = f"{m_live - m_c192:+.4f}"
diff_tiny_m = f"{m_live - m_cf1:+.4f}"
print(f"{'MACRO AUC':18s} | {m_c96:10.4f} | {m_c192:11.4f} | {m_cf0:12.4f} | {m_cf1:12.4f} | {m_live:18.4f} | {diff_coat_m:16s} | {diff_tiny_m:16s}")
