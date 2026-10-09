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

auc_v4, aucs_v4 = get_macro_auc(v4)
print(f"Baseline stevenleehans v4_blend AUC: {auc_v4:.4f}")

# Blend 1: simple mean of all 5 scorable tables
blend_all_scorable = sum([t.loc[overlap_gold, LAB] for t in scorable_tables]) / len(scorable_tables)
auc_blend_all, aucs_blend_all = get_macro_auc(blend_all_scorable)
print(f"Blend of 5 scorable tables AUC:     {auc_blend_all:.4f}")

# Blend 2: v4_blend + jev_hybrid (the two strongest)
blend_top2 = (v4.loc[overlap_gold, LAB] + jev_hybrid.loc[overlap_gold, LAB]) / 2.0
auc_top2, aucs_top2 = get_macro_auc(blend_top2)
print(f"Blend of Top-2 (v4 + jev_hybrid):   {auc_top2:.4f}")

# 2. Build consensus training labels for all 4,349 studies
# Sources: stevenleehans (v4_blend, v2, full), dread_llm_soft, nartaa_v1_blend, jev_hybrid, jev_final2
train_tables = {
    "v4_blend": v4,
    "v2": v2,
    "dread_soft": dread,
    "nartaa_blend": nartaa,
    "jev_hybrid": jev_hybrid,
}

print(f"\nBuilding labels_avg across {len(train_tables)} diverse source tables:")
for name, t in train_tables.items():
    print(f"  - {name}: {sum(u in t.index for u in all_train_ids)}/4349 present")

# Compute mean per finding across all sources for each study
stacked = []
for name, t in train_tables.items():
    aligned = t.reindex(all_train_ids)
    stacked.append(aligned.values)

stacked = np.stack(stacked, axis=0) # shape (N_tables, 4349, 12)
avg_values = np.nanmean(stacked, axis=0) # mean across tables
# Clip to valid range [0, 1]
avg_values = np.clip(avg_values, 0.0, 1.0)

labels_avg = pd.DataFrame(avg_values, index=all_train_ids, columns=LAB)
labels_avg.index.name = "StudyInstanceUID"
labels_avg = labels_avg.reset_index()

# Verify shape and nulls
assert len(labels_avg) == 4349
assert labels_avg[LAB].isna().sum().sum() == 0
assert not any(u in gold_ids for u in labels_avg["StudyInstanceUID"])

labels_avg.to_parquet("labels_avg.parquet", index=False)
print(f"\n[output] Saved labels_avg.parquet: {labels_avg.shape} (4,349 rows x 13 cols)")

# Also write comparison summary
print("\n=== Per-Finding AUC Comparison vs Gold-58 ===")
print(f"{'Finding':18s} | {'v4_blend':8s} | {'jev_hybrid':10s} | {'5-table avg':11s}")
print("-" * 55)
for col in LAB:
    print(f"{col:18s} | {aucs_v4[col]:8.4f} | {aucs_top2[col]:10.4f} | {aucs_blend_all[col]:11.4f}")
print(f"{'Macro AUC':18s} | {auc_v4:8.4f} | {auc_top2:10.4f} | {auc_blend_all:11.4f}")
