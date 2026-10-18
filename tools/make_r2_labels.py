#!/usr/bin/env python3
"""tools/make_r2_labels.py - round-2 soft labels: report labels mixed with our image models' out-of-fold predictions.

Self-training on competition data only (no external data): each family's 4-fold OOF probabilities are Platt-calibrated
per finding to the report labels, the families are averaged, and the result is mixed with the report labels:
    r2 = alpha * report + (1 - alpha) * image_oof
The same calibration applied to each family's gold-58 predictions (held out from every fold) scores the mix on gold
for a grid of alphas, so the label gain is measured before anything trains on it.

    uv run -q --with numpy --with pandas --with pyarrow --with scikit-learn python -I tools/make_r2_labels.py \
        --family cnxt_sev='pc192/raptor_{kind}_cnxt_sev_f{f}_swa*.npz' --family coat='raptor_{kind}_oof4_f{f}_swa*.npz' \
        --alpha 0.5 --out labels_r2.parquet
"""
import argparse
import glob

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="pcseq/labels_sev_v1.parquet")
ap.add_argument("--gold_report", default="labels/labels_clef_sev.csv", help="report labels for the 58 gold studies")
ap.add_argument("--family", action="append", required=True, help="name=pattern with {kind} (oof|gold) and {f} (fold)")
ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--out", required=True)
a = ap.parse_args()

base = pd.read_parquet(a.base).set_index("StudyInstanceUID")
F = list(base.columns)
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def one(pattern):
    hits = sorted(glob.glob(pattern))
    assert len(hits) == 1, (pattern, hits)
    return np.load(hits[0])


oof_cal, gold_cal, gold_truth, gold_ids = [], [], None, None
for spec in a.family:
    name, pat = spec.split("=", 1)
    zs = [one(pat.format(kind="oof", f=f) .replace("_swa*", "_swa_fold%d*" % f)) for f in range(4)]
    oof = pd.DataFrame(np.concatenate([z["pred"] for z in zs]), index=np.concatenate([z["ids"] for z in zs]), columns=F)
    oof = oof.loc[base.index]  # every labelled study has exactly one OOF prediction
    gz = [np.load(p) for f in range(4) for p in sorted(glob.glob(pat.format(kind="gold", f=f)))[:1]]  # a fold may lack gold preds
    assert gz, name + ": no gold predictions"
    gold = np.mean([z["pred"] for z in gz], axis=0)
    gold_truth, gold_ids = gz[0]["truth"], gz[0]["ids"]
    co, cg = np.zeros(oof.shape), np.zeros(gold.shape)
    for j, c in enumerate(F):  # Platt per finding against the soft report label (weights = soft targets)
        x, y = logit(oof[c].values)[:, None], base[c].values
        m = LogisticRegression(C=1e6).fit(np.r_[x, x], np.r_[np.ones(len(y)), np.zeros(len(y))], sample_weight=np.r_[y, 1 - y])
        co[:, j] = m.predict_proba(x)[:, 1]
        cg[:, j] = m.predict_proba(logit(gold[:, j])[:, None])[:, 1]
    oof_cal.append(co)
    gold_cal.append(cg)
    print(f"{name}: OOF macro AUC vs report labels "
          f"{np.mean([roc_auc_score(base[c] > 0.5, oof[c]) for c in F]):.4f} | gold {np.mean([roc_auc_score(gold_truth[:, j], gold[:, j]) for j in range(12)]):.4f}")

img = np.mean(oof_cal, axis=0)
img_gold = np.mean(gold_cal, axis=0)
rep_gold = pd.read_csv(a.gold_report).set_index("StudyInstanceUID").loc[gold_ids, F].values
gauc = lambda p: np.mean([roc_auc_score(gold_truth[:, j], p[:, j]) for j in range(12)])
print("gold-58 macro AUC of the label mix (report weight alpha):")
for al in (1.0, 0.7, 0.5, 0.3, 0.0):
    print(f"  alpha {al:.1f}: {gauc(al * rep_gold + (1 - al) * img_gold):.4f}")
per = [(c, roc_auc_score(gold_truth[:, j], rep_gold[:, j]), roc_auc_score(gold_truth[:, j], (a.alpha * rep_gold + (1 - a.alpha) * img_gold)[:, j]))
       for j, c in enumerate(F)]
print("per finding, report -> mix at alpha %.1f:" % a.alpha, ", ".join(f"{c} {r:.3f}->{m:.3f}" for c, r, m in per))

r2 = a.alpha * base.values + (1 - a.alpha) * img
assert np.isfinite(r2).all() and r2.min() >= 0 and r2.max() <= 1 and r2.shape == base.shape
pd.DataFrame(r2.astype(np.float32), index=base.index, columns=F).reset_index().to_parquet(a.out, index=False)
print("wrote", a.out, r2.shape, "mean", np.round(r2.mean(0), 3).tolist())
