#!/usr/bin/env python3
"""
create_2fold_split.py

Creates a balanced 2-fold cross-validation split across all 4,407 RSNA Knee studies:
- Fold 0: 2,175 studies (for training Fold 0 models / validation on Fold 0)
- Fold 1: 2,174 studies (for training Fold 1 models / validation on Fold 1)
- Fold -1: 58 Gold studies (reserved strictly as independent evaluation holdout)

Ensures label distribution parity between Fold 0 and Fold 1 across all 12 abnormalities.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def main():
    train_path = "data/train.csv"
    labels_avg_path = "labels_avg.parquet"
    out_json_path = "data/labels/folds_2fold_v1.json"

    print(f"Loading {train_path}...")
    train_df = pd.read_csv(train_path)
    
    # 58 Gold studies have non-null ground truth labels in train.csv
    gold_mask = train_df["ACL"].notna()
    gold_uids = set(train_df.loc[gold_mask, "StudyInstanceUID"].tolist())
    print(f"Found {len(gold_uids)} Gold studies (assigned to fold -1).")

    print(f"Loading {labels_avg_path}...")
    labels_df = pd.read_parquet(labels_avg_path)
    
    # Filter out gold studies if any exist in labels_df
    non_gold_df = labels_df[~labels_df["StudyInstanceUID"].isin(gold_uids)].copy().reset_index(drop=True)
    print(f"Found {len(non_gold_df)} non-gold studies for 2-fold splitting.")

    target_cols = [c for c in non_gold_df.columns if c != "StudyInstanceUID"]

    # Stratified multi-label approximation: rank by sum of probabilities + primary findings
    sort_key = non_gold_df[target_cols].sum(axis=1) + non_gold_df["ACL"] * 2.0 + non_gold_df["Lateral OA"] * 2.0
    sorted_indices = np.argsort(sort_key.values)

    # Alternate assignment (snake split) for near-perfect distribution matching
    fold_assignments = np.zeros(len(non_gold_df), dtype=int)
    for rank, idx in enumerate(sorted_indices):
        fold_assignments[idx] = 0 if (rank % 2 == 0) else 1

    non_gold_df["fold"] = fold_assignments

    print("\n--- Split Verification ---")
    print(f"Fold 0 count: {(non_gold_df['fold'] == 0).sum()}")
    print(f"Fold 1 count: {(non_gold_df['fold'] == 1).sum()}")

    print("\nMean probability comparison across 12 findings:")
    summary = []
    for col in target_cols:
        m0 = non_gold_df.loc[non_gold_df["fold"] == 0, col].mean()
        m1 = non_gold_df.loc[non_gold_df["fold"] == 1, col].mean()
        diff = abs(m0 - m1)
        summary.append({"Finding": col, "Fold 0": round(m0, 4), "Fold 1": round(m1, 4), "Diff": round(diff, 5)})
    print(pd.DataFrame(summary).to_string(index=False))

    # Construct final mapping
    folds_dict = {}
    for uid in gold_uids:
        folds_dict[uid] = -1
    for _, row in non_gold_df.iterrows():
        folds_dict[row["StudyInstanceUID"]] = int(row["fold"])

    assert len(folds_dict) == len(train_df), f"Expected {len(train_df)} studies, got {len(folds_dict)}"

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump({"folds": folds_dict}, f, indent=2)
    print(f"\nSaved split to {out_json_path}")

    # Also save a copy at project root for convenient training access
    with open("folds_2fold_v1.json", "w") as f:
        json.dump({"folds": folds_dict}, f, indent=2)
    print("Saved copy to folds_2fold_v1.json")

if __name__ == "__main__":
    main()
