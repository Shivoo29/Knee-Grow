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

