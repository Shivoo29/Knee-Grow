#!/usr/bin/env python3
"""eda/eda_groups.py - where do our image models disagree with the labels? OOF macro/per-finding AUC by group.

Model = mean of the 4-fold OOF of ConvNeXt severity and CoAtNet oof4 (4,349 train studies). Truth = report labels
(labels_sev_v1 > 0.5), so a group with low AUC is either harder for the model or has noisier report labels.
Groups: knee side (DICOM coordinates), missing fluid-sensitive planes, scanner maker, field strength, report language.
Report text is read locally only (language guess); nothing from it is printed.
    uv run -q --with numpy --with pandas --with pyarrow --with scikit-learn --with langid python -I eda/eda_groups.py
"""
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

lab = pd.read_parquet("pcseq/labels_sev_v1.parquet").set_index("StudyInstanceUID")
F = list(lab.columns)


def oof(pat):
    zs = [np.load(sorted(glob.glob(pat.format(f=f)))[0]) for f in range(4)]
    return pd.DataFrame(np.concatenate([z["pred"] for z in zs]), index=np.concatenate([z["ids"] for z in zs]), columns=F).loc[lab.index]


rank = lambda d: d.rank(pct=True)
pred = (rank(oof("pc192/raptor_oof_cnxt_sev_f{f}_swa_fold{f}.npz")) + rank(oof("raptor_oof_oof4_f{f}_swa_fold{f}.npz"))) / 2
y = lab > 0.5

g = pd.DataFrame(index=lab.index)
side = pd.read_csv("eda/study_laterality_orientation.csv").set_index("StudyInstanceUID")["Laterality"]
g["side"] = side.reindex(lab.index).fillna("?")
ser = pd.read_csv("data/train_series.csv")
fs = ser[ser.Fluid_Sensitive == 1].groupby("StudyInstanceUID").Anatomical_Plane.agg(set)
for p in ("Axial", "Coronal", "Sagittal"):
    g[f"fs_{p[:3].lower()}"] = [("has" if p in fs.get(s, set()) else "MISSING") for s in lab.index]
site = pd.read_csv("data/site_metadata.csv").set_index("StudyInstanceUID").reindex(lab.index)
g["maker"] = site.Manufacturer.fillna("?").str.split().str[0]
g["tesla"] = site.MagneticFieldStrength.astype(str)
try:
    import langid
    rep = pd.read_csv("data/train.csv", usecols=["StudyInstanceUID", "Report"]).set_index("StudyInstanceUID").Report.reindex(lab.index)
    g["lang"] = [langid.classify(str(t))[0] for t in rep]
except ImportError:
    pass


def aucs(idx):
    out = {}
    for c in F:
        yy = y.loc[idx, c]
        if 5 <= yy.sum() <= len(yy) - 5:
            out[c] = roc_auc_score(yy, pred.loc[idx, c])
    return out


base = aucs(lab.index)
print(f"all {len(lab)} studies: macro {np.mean(list(base.values())):.4f}\n")
for col in g.columns:
    print(f"== {col}")
    for val, idx in g.groupby(col).groups.items():
        if len(idx) < 60:
            continue
        a = aucs(idx)
        common = [c for c in a if c in base]
        d = np.mean([a[c] - base[c] for c in common])
        worst = sorted(common, key=lambda c: a[c] - base[c])[:3]
        print(f"  {str(val):22s} n={len(idx):5d}  macro {np.mean(list(a.values())):.4f}  (vs all {d:+.4f})  "
              + "  ".join(f"{c} {a[c]-base[c]:+.3f}" for c in worst))
    print()
side_f = ["Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "MCL"]
print("== side-specific findings, Left vs Right")
for c in side_f:
    print(f"  {c:18s} " + "  ".join(f"{s} {aucs(g.index[g.side == s]).get(c, float('nan')):.3f}" for s in ("Left", "Right")))
