#!/usr/bin/env python3
"""eda/review_list.py - study IDs for a human look: per weak finding, the 10 most confident model-vs-label disagreements
each way (model says yes / label says no, and the reverse). IDs and scores only; open them in a local viewer.
    uv run -q --with numpy --with pandas --with pyarrow python -I eda/review_list.py
"""
import glob
import numpy as np, pandas as pd
lab = pd.read_parquet("pcseq/labels_sev_v1.parquet").set_index("StudyInstanceUID"); F = list(lab.columns)
def oof(p):
    zs = [np.load(sorted(glob.glob(p.format(f=f)))[0]) for f in range(4)]
    return pd.DataFrame(np.concatenate([z["pred"] for z in zs]), index=np.concatenate([z["ids"] for z in zs]), columns=F).loc[lab.index]
pred = (oof("pc192/raptor_oof_cnxt_sev_f{f}_swa_fold{f}.npz").rank(pct=True) + oof("raptor_oof_oof4_f{f}_swa_fold{f}.npz").rank(pct=True)) / 2
rows = []
for c in ["Lateral OA", "Lateral Meniscus", "Synovitis", "PF OA", "Effusion", "MCL"]:
    d = pred[c] - lab[c]
    for kind, ids in (("model YES, label no", d.nlargest(10).index), ("model no, label YES", d.nsmallest(10).index)):
        rows += [dict(finding=c, kind=kind, StudyInstanceUID=s, model_pct=round(pred.at[s, c], 3), label=round(lab.at[s, c], 3)) for s in ids]
pd.DataFrame(rows).to_csv("eda/review_list.csv", index=False); print(len(rows), "rows -> eda/review_list.csv")
