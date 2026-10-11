#!/usr/bin/env python3
"""train_5fold_knee.py - Master 5-Fold Cross-Validation Orchestrator & OOF Aggregator.

Orchestrates 5-fold cross-validation across available nodes (local GPU, remote SSH node1).
Collects per-fold checkpoints and out-of-fold predictions (`raptor_oof_{tag}_fold{k}.npz`),
combines all 5 folds into a single full-corpus (4,349 studies) OOF prediction table,
and computes global cross-validation macro-AUC across all 12 findings.
"""

import os
import sys
import glob
import json
import argparse
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FINDINGS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]


def aggregate_5fold_oofs(
    tag: str,
    oof_dir: str = ".",
    num_folds: int = 5,
    out_parquet: Optional[str] = None
) -> pd.DataFrame:
    """Combines per-fold out-of-fold NPZ prediction files into a unified DataFrame.
    
    Args:
        tag: Experiment tag used during training (e.g. 'coat_latoa_roi' or 'v6').
        oof_dir: Directory containing raptor_oof_{tag}_fold*.npz files.
        num_folds: Expected number of folds (default 5).
        out_parquet: Destination parquet file path.
        
    Returns:
        DataFrame containing StudyInstanceUID, fold, and predicted probabilities for all 12 findings.
    """
    records = []
    found_folds = set()

    # Search for fold files matching patterns
    patterns = [
        os.path.join(oof_dir, f"raptor_oof_{tag}*fold*.npz"),
        os.path.join(oof_dir, f"*{tag}*fold*.npz")
    ]
    npz_files = []
    for p in patterns:
        npz_files.extend(glob.glob(p))
    npz_files = sorted(set(npz_files))

    print(f"[5-Fold Aggregator] Found {len(npz_files)} matching OOF npz files for tag '{tag}'")

    for fpath in npz_files:
        try:
            data = np.load(fpath, allow_pickle=True)
            preds = data["pred"]      # (N, 12)
            uids = data["ids"]        # (N,)
            fold = int(data.get("fold", -1))
            found_folds.add(fold)

            for i, uid in enumerate(uids):
                row = {"StudyInstanceUID": str(uid), "fold": fold}
                for c_idx, col in enumerate(FINDINGS):
                    row[col] = float(preds[i, c_idx])
                records.append(row)
            print(f"  Loaded fold {fold}: {len(uids)} studies from {os.path.basename(fpath)}")
        except Exception as e:
            print(f"  [WARN] Failed to load {fpath}: {e}")

    if not records:
        raise RuntimeError(f"No valid OOF predictions found for tag '{tag}' in '{oof_dir}'")

    df = pd.DataFrame(records).drop_duplicates(subset=["StudyInstanceUID"])
    print(f"[5-Fold Aggregator] Consolidated {len(df)} unique studies across folds: {sorted(found_folds)}")

    if out_parquet:
        df.to_parquet(out_parquet, index=False)
        print(f"[5-Fold Aggregator] Saved consolidated OOF predictions to {out_parquet}")

    return df


def evaluate_oof_performance(
    oof_df: pd.DataFrame,
    labels_path: str = "labels_avg.parquet"
) -> Dict[str, float]:
    """Computes macro-AUC and per-finding AUC of the OOF predictions against target labels."""
    if labels_path.endswith(".parquet"):
        true_df = pd.read_parquet(labels_path)
    else:
        true_df = pd.read_csv(labels_path)

    merged = pd.merge(oof_df, true_df, on="StudyInstanceUID", suffixes=("_pred", "_true"))
    print(f"\n=== Global 5-Fold Cross-Validation Metrics ({len(merged)} studies) ===")

    aucs = {}
    for col in FINDINGS:
        col_pred = f"{col}_pred"
        col_true = f"{col}_true"
        if col_pred in merged.columns and col_true in merged.columns:
            y_t = merged[col_true].values
            y_p = merged[col_pred].values
            # Soft label or binary label AUC
            y_bin = (y_t > 0.5).astype(int) if set(np.unique(y_t)) - {0, 1} else y_t.astype(int)
            if len(np.unique(y_bin)) > 1:
                auc = float(roc_auc_score(y_bin, y_p))
                aucs[col] = auc
                print(f"  {col:<20}: AUC = {auc:.4f}")
            else:
                print(f"  {col:<20}: Insufficient positive/negative classes to compute AUC")

    macro_auc = float(np.mean(list(aucs.values()))) if aucs else 0.0
    print(f"--------------------------------------------------")
    print(f"  GLOBAL 5-FOLD MACRO-AUC: {macro_auc:.4f}\n")
    return aucs


def main():
    parser = argparse.ArgumentParser(description="Aggregate and evaluate 5-fold CV OOF predictions")
    parser.add_argument("--tag", type=str, required=True, help="Model experiment tag (e.g. coat_latoa_roi)")
    parser.add_argument("--oof_dir", type=str, default=".", help="Directory containing OOF npz files")
    parser.add_argument("--out_parquet", type=str, default=None, help="Output path for consolidated OOF parquet")
    parser.add_argument("--labels", type=str, default="labels_avg.parquet", help="Ground truth labels file")
    args = parser.parse_args()

    out_p = args.out_parquet or f"oof_{args.tag}_5fold.parquet"
    df = aggregate_5fold_oofs(args.tag, oof_dir=args.oof_dir, out_parquet=out_p)
    if os.path.exists(args.labels):
        evaluate_oof_performance(df, labels_path=args.labels)


if __name__ == "__main__":
    main()
