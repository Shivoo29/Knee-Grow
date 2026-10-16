#!/usr/bin/env python3
"""Decorrelation check for a candidate blend member (numbers only: no ids, no text).

Per finding: AUC of A, AUC of B, AUC of the rank blend (1-w)*A + w*B, within-class Spearman of A vs B
(positives and negatives separately), and on positive-negative pairs: fix = share of the pairs A orders
wrongly that B orders rightly, brk = share of the pairs A orders rightly that B orders wrongly.
A useful new member has rho well below ~0.85 and fix clearly above brk.

    .venv/bin/python -I tools/decorr.py --a 'raptor_oof_oof4_f*_swa_fold*.npz' \
        --b 'pc192/raptor_oof_cnxt_f*_swa_fold*.npz' --truth labels_hybrid_v1.parquet
    --truth gold:ds/comp/train.csv   -> the 58 expert-labelled studies (Report column never read)
"""
import argparse
import glob

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
       'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def _table(path):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(
        path, usecols=["StudyInstanceUID"] + LAB, dtype={"StudyInstanceUID": str})
    return df.set_index(df["StudyInstanceUID"].astype(str))[LAB]


def load_preds(spec):
    files = sorted(glob.glob(spec))
    if not files:
        raise SystemExit(f"no files match {spec}")
    if files[0].endswith(".npz"):
        parts = []
        for f in files:
            z = np.load(f, allow_pickle=True)
            parts.append(pd.DataFrame(z["pred"], columns=LAB, index=z["ids"].astype(str)))
        df = pd.concat(parts)
    else:
        df = _table(files[0])
    if df.index.duplicated().any():
        raise SystemExit(f"{spec}: duplicate study ids")
    return df


def load_truth(spec):
    if spec.startswith("gold:"):
        return _table(spec[5:]).dropna()          # gold rows are the only fully labelled train rows
    return _table(spec)


def _pairs(y, a, b):
    d_a = a[y == 1][:, None] - a[y == 0][None, :]
    d_b = b[y == 1][:, None] - b[y == 0][None, :]
    wrong, right = d_a < 0, d_a > 0
    fix = float((d_b[wrong] > 0).mean()) if wrong.any() else 0.0
    brk = float((d_b[right] < 0).mean()) if right.any() else 0.0
    return fix, brk


def compare(A, B, T, w=0.3):
    ids = A.index.intersection(B.index).intersection(T.index)
    if len(ids) == 0:
        raise SystemExit("no common study ids between --a, --b and --truth")
    rows = []
    for c in LAB:
        y = (T.loc[ids, c].values >= 0.5).astype(int)
        if y.min() == y.max():
            continue
        a = rankdata(A.loc[ids, c].values) / len(ids)
        b = rankdata(B.loc[ids, c].values) / len(ids)
        fix, brk = _pairs(y, a, b)
        rows.append(dict(finding=c, n_pos=int(y.sum()),
                         auc_a=roc_auc_score(y, a), auc_b=roc_auc_score(y, b),
                         auc_blend=roc_auc_score(y, (1 - w) * a + w * b),
                         rho_pos=spearmanr(a[y == 1], b[y == 1])[0],
                         rho_neg=spearmanr(a[y == 0], b[y == 0])[0], fix=fix, brk=brk))
    return pd.DataFrame(rows).set_index("finding"), len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--w", type=float, default=0.3)
    a = ap.parse_args()
    r, n = compare(load_preds(a.a), load_preds(a.b), load_truth(a.truth), a.w)
    pd.set_option("display.width", 160)
    print(f"common studies: {n} | blend weight on B: {a.w}")
    print(r.round(4).to_string())
    print("MACRO", r.drop(columns="n_pos").mean().round(4).to_dict())


if __name__ == "__main__":
    main()
