#!/usr/bin/env python3
"""Evaluate public label tables against gold-58 and construct averaged labels."""
import os, glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
       "Medial OA", "Lateral OA", "PF OA", "Effusion",
       "Synovitis", "Baker's", "Contusion", "Fracture"]

tr = pd.read_csv("data/train.csv")
tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
gold_df = tr[tr[LAB].notna().all(axis=1)].copy().set_index("StudyInstanceUID")
gold_ids = set(gold_df.index)
print(f"[gold] loaded {len(gold_df)} gold-standard studies")

def evaluate_table(name, df):
    id_col = [c for c in df.columns if "StudyInstanceUID" in c or c.lower() == "id"][0]
    df = df.copy()
    df["StudyInstanceUID"] = df[id_col].astype(str)
    
    col_map = {}
    for c in df.columns:
        clean = c.replace("_", " ").strip()
        for target in LAB:
            if clean.lower() == target.lower():
                col_map[c] = target
    df = df.rename(columns=col_map)
    df = df.set_index("StudyInstanceUID")
    
    missing_cols = [c for c in LAB if c not in df.columns]
    if missing_cols:
        return {"name": name, "status": f"missing columns: {missing_cols}", "n_rows": len(df), "df": None}
        
    overlap_gold = [u for u in gold_df.index if u in df.index]
    n_gold = len(overlap_gold)
    
    if n_gold == 0:
        return {"name": name, "status": "unscorable (0 gold rows)", "n_rows": len(df), "n_gold": 0, "df": df[LAB]}
        
    y_true = gold_df.loc[overlap_gold, LAB].values.astype(float)
    y_pred = df.loc[overlap_gold, LAB].values.astype(float)
    
    # Exact match check
    exact_matches = np.all(np.isclose(y_pred, y_true, atol=1e-4), axis=1)
    exact_rate = float(np.mean(exact_matches))
    
    aucs = {}
    for j, col in enumerate(LAB):
        val_true = y_true[:, j]
        val_pred = y_pred[:, j]
        valid = ~np.isnan(val_pred) & ~np.isnan(val_true)
        if len(np.unique(val_true[valid])) > 1:
            aucs[col] = float(roc_auc_score(val_true[valid], val_pred[valid]))
        else:
            aucs[col] = float("nan")
    macro_auc = float(np.nanmean(list(aucs.values())))
    
    return {
        "name": name,
        "status": "scorable",
        "n_rows": len(df),
        "n_gold": n_gold,
        "exact_rate": exact_rate,
        "macro_auc": macro_auc,
        "aucs": aucs,
        "df": df[LAB]
    }

candidate_files = [
    # stevenleehans
    ("stevenleehans_v4_blend", "data/labels/llm_labels_v4_blend.csv"),
    ("stevenleehans_v2", "data/labels/llm_labels_v2.csv"),
    ("stevenleehans_full", "data/labels/llm_labels_full.csv"),
    # dreaddevelopment
    ("dread_llm_soft", "data/labels/labels_llm_soft.parquet"),
    # nartaa
    ("nartaa_v1_blend", "data/labels/labels_v1_blend.parquet"),
    ("nartaa_v0raw", "data/labels/labels_v0raw.parquet"),
    ("nartaa_v1_gemlow", "data/labels/labels_v1_gemlow.parquet"),
    ("nartaa_v1_perlabel", "data/labels/labels_v1_perlabel.parquet"),
    ("nartaa_train_labels_v0", "data/labels/train_labels_v0.parquet"),
    ("nartaa_train_labels_v1", "data/labels/train_labels_v1.parquet"),
    # riadmohamed42 (JEV)
    ("jev_final2_scores", "data/labels/FINAL2_labels_scores.csv"),
    ("jev_hybrid_scores", "data/labels/HYBRID_labels_scores.csv"),
]

results = []
tables = {}

for name, path in candidate_files:
