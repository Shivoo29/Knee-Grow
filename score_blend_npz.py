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
