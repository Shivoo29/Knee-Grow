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
    if not os.path.exists(path):
        continue
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    res = evaluate_table(name, df)
    results.append(res)
    if res.get("df") is not None:
        tables[name] = res["df"]

print("\n=== Evaluation against Gold-58 ===")
print(f"{'Table':26s} | {'Rows':5s} | {'Gold':4s} | {'Exact':5s} | {'Macro AUC':9s}")
print("-" * 62)
for r in results:
    if r["status"] == "scorable":
        print(f"{r['name']:26s} | {r['n_rows']:5d} | {r['n_gold']:4d} | {r['exact_rate']:5.3f} | {r['macro_auc']:9.4f}")
    else:
        print(f"{r['name']:26s} | {r['n_rows']:5d} | {r.get('n_gold', 0):4d} | {'-':5s} | {r['status']}")

# Print detailed per-finding AUCs for scorable tables
print("\n=== Per-Finding AUCs (Scorable Tables) ===")
header = f"{'Finding':18s}" + "".join([f" | {r['name'][:10]:10s}" for r in results if r['status'] == 'scorable'])
print(header)
print("-" * len(header))
for col in LAB:
    row_str = f"{col:18s}"
    for r in results:
        if r["status"] == "scorable":
            v = r["aucs"].get(col, float("nan"))
            row_str += f" | {v:10.4f}"
    print(row_str)

# Build averaged labels across independent high-quality tables
# Candidate high quality sources:
# 1. stevenleehans_v4_blend (best single, 0.893)
# 2. stevenleehans_v2 (0.887)
# 3. dread_llm_soft (independent prompt from Dread)
# 4. nartaa_v1_blend (independent blend from nartaa)
# 5. jev_final2_scores (independent JEV verification)

# Let's check overlap of study IDs across all 4,349 non-gold training studies
all_study_ids = [u for u in tr["StudyInstanceUID"].unique() if u not in gold_ids]
print(f"\nNon-gold training studies target: {len(all_study_ids)} (expected 4349)")

# Check coverage for each table
print("\nCoverage of 4,349 non-gold studies:")
for name, tbl in tables.items():
    cov = sum(u in tbl.index for u in all_study_ids)
    print(f"  {name:26s}: {cov}/{len(all_study_ids)} ({cov/len(all_study_ids)*100:.1f}%)")
