#!/usr/bin/env python3
"""Score multi-model ensembles on Gold-58 using saved raptor_gold_*.npz files."""
import sys, glob, os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
       "Medial OA", "Lateral OA", "PF OA", "Effusion",
       "Synovitis", "Baker's", "Contusion", "Fracture"]

def load_npz(path):
    d = np.load(path)
    pred_key = "pred" if "pred" in d else "preds"
    truth_key = "truth" if "truth" in d else "targets"
    return d[pred_key], d[truth_key]

def calc_auc(preds, targets):
    aucs = []
    for j in range(12):
        y_t = targets[:, j]
        y_p = preds[:, j]
        if len(np.unique(y_t)) > 1:
            aucs.append(roc_auc_score(y_t, y_p))
        else:
            aucs.append(np.nan)
    return float(np.nanmean(aucs)), aucs

def main():
    files = sys.argv[1:]
    if not files:
        files = glob.glob("raptor_gold_*swa*.npz")
    print(f"Loading {len(files)} models:")
    preds_list = []
    targets = None
    names = []
    for f in sorted(files):
        if not os.path.exists(f):
            print(f"  Missing: {f}")
            continue
        p, t = load_npz(f)
        score, _ = calc_auc(p, t)
        print(f"  • {os.path.basename(f)}: Gold Macro-AUC = {score:.4f}")
        preds_list.append(p)
        names.append(os.path.basename(f))
        if targets is None:
            targets = t

    if len(preds_list) < 2:
        return

    # Prob blend
    prob_blend = np.mean(preds_list, axis=0)
    prob_score, p_aucs = calc_auc(prob_blend, targets)
    print(f"\n[EQUAL PROB BLEND] ({len(preds_list)} models): {prob_score:.4f}")

    # Rank blend
    rank_preds = []
    for p in preds_list:
        rp = np.zeros_like(p)
        for j in range(12):
            rp[:, j] = rankdata(p[:, j]) / len(p)
        rank_preds.append(rp)
    rank_blend = np.mean(rank_preds, axis=0)
    rank_score, r_aucs = calc_auc(rank_blend, targets)
    print(f"[EQUAL RANK BLEND] ({len(preds_list)} models): {rank_score:.4f}")

    print("\nPer-Class Breakdown (Rank Blend):")
    for name, auc in zip(LAB, r_aucs):
        print(f"  {name:18s}: {auc:.4f}")

if __name__ == "__main__":
    main()
