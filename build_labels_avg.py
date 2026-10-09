#!/usr/bin/env python3
"""Evaluate blends on gold-58 and generate labels_avg.parquet for training."""
import os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
       "Medial OA", "Lateral OA", "PF OA", "Effusion",
       "Synovitis", "Baker's", "Contusion", "Fracture"]

tr = pd.read_csv("data/train.csv")
tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
gold_df = tr[tr[LAB].notna().all(axis=1)].copy().set_index("StudyInstanceUID")
gold_ids = set(gold_df.index)
all_train_ids = [u for u in tr["StudyInstanceUID"].unique() if u not in gold_ids]
assert len(all_train_ids) == 4349, f"Expected 4349 non-gold studies, got {len(all_train_ids)}"

def load_clean(path):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    id_col = [c for c in df.columns if "StudyInstanceUID" in c or c.lower() == "id"][0]
    df = df.copy()
    df["StudyInstanceUID"] = df[id_col].astype(str)
    col_map = {c: c.replace("_", " ").strip() for c in df.columns if c.replace("_", " ").strip() in LAB}
    df = df.rename(columns=col_map).set_index("StudyInstanceUID")
    return df[LAB]

# Load key tables
v4 = load_clean("data/labels/llm_labels_v4_blend.csv")
v2 = load_clean("data/labels/llm_labels_v2.csv")
full = load_clean("data/labels/llm_labels_full.csv")
dread = load_clean("data/labels/labels_llm_soft.parquet")
nartaa = load_clean("data/labels/labels_v1_blend.parquet")
jev_hybrid = load_clean("data/labels/HYBRID_labels_scores.csv")
jev_final2 = load_clean("data/labels/FINAL2_labels_scores.csv")

# 1. Scorable average on gold-58 (tables with gold rows: v4, v2, full, jev_hybrid, jev_final2)
scorable_tables = [v4, v2, full, jev_hybrid, jev_final2]
scorable_names = ["v4_blend", "v2", "full", "jev_hybrid", "jev_final2"]

# Check individual and blended AUCs on gold-58
overlap_gold = [u for u in gold_df.index if all(u in t.index for t in scorable_tables)]
y_true = gold_df.loc[overlap_gold, LAB].values.astype(float)

def get_macro_auc(preds_df):
    y_pred = preds_df.loc[overlap_gold, LAB].values.astype(float)
    aucs = {}
    for j, col in enumerate(LAB):
        val_true = y_true[:, j]
        val_pred = y_pred[:, j]
        valid = ~np.isnan(val_pred) & ~np.isnan(val_true)
        if len(np.unique(val_true[valid])) > 1:
            aucs[col] = float(roc_auc_score(val_true[valid], val_pred[valid]))
        else:
            aucs[col] = float("nan")
    return float(np.nanmean(list(aucs.values()))), aucs

