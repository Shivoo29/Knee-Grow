"""Series TTA pilot: does reading the unused series of the same plane + contrast, and averaging, help the fold-0 model?
Rows without an alternate keep the idea-10 "all" predictions (same model, same tool); the 510 changed rows take
node1's first-pick ("base") or alternate ("alt") predictions, so every variant shares the same unchanged rows.
    uv run -q --with numpy --with pandas --with pyarrow --with scikit-learn python -I stta/eval_stta.py   (from pcr2/)"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

lab = pd.read_parquet("../pcseq/labels_sev_v1.parquet").set_index("StudyInstanceUID")
F = list(lab.columns)
gz = np.load("raptor_gold_cnxt_r2_f0_swa.npz", allow_pickle=True)
gold = pd.DataFrame(gz["truth"], index=gz["ids"].astype(str), columns=F)
b = pd.read_parquet("stta/stta_base.parquet").set_index("StudyInstanceUID")[F]
a = pd.read_parquet("stta/stta_alt.parquet").set_index("StudyInstanceUID")[F]
variants = {"first pick": b, "alt only": a, "mean": (a + b) / 2, "max": np.maximum(a, b)}


def auc(y, p):
    return roc_auc_score(y, p) if y.nunique() > 1 else np.nan


for part, full, truth in (("OOF vs reports", "idea10/all_oof.parquet", lab > 0.5),
                          ("gold-58 vs experts", "idea10/all_gold.parquet", gold > 0.5)):
    P = pd.read_parquet(full).set_index("StudyInstanceUID")[F]
    ch = P.index.intersection(b.index)
    print(f"\n== {part}: {len(P)} studies, {len(ch)} with an alternate | node1 vs PC first-pick max |diff| "
          f"{np.abs(P.loc[ch].values - b.loc[ch].values).max():.4f}")
    print(f"{'variant':11s} macro  changed-only | " + " ".join(f"{c[:7]:>7s}" for c in F))
    for name, v in variants.items():
        Q = P.copy()
        Q.loc[ch] = v.loc[ch].values
        y = truth.loc[Q.index]
        per = [auc(y[c], Q[c]) for c in F]
        sub = np.nanmean([auc(y.loc[ch, c], Q.loc[ch, c]) for c in F])
        print(f"{name:11s} {np.nanmean(per):.4f}  {sub:.4f}       | " + " ".join(f"{x:7.3f}" for x in per))
