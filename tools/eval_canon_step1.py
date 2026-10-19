#!/usr/bin/env python3
"""Evaluate Step 1 of Knee-Side Canonicalization:
Compares plain vs mirrored inference on Fold 0 held-out studies.

Variants evaluated:
  1. Left knees plain (unmodified)
  2. Left knees mirrored (anatomical mirror)
  3. Right knees plain (unmodified)
  4. Right knees mirrored (control)
  5. Flip TTA: (plain + mirrored) / 2

Metrics reported:
  - Per-finding AUC vs labels_sev_v1.parquet (> 0.5)
  - Macro AUC over 5 side-specific findings: [Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, MCL]
  - Macro AUC over other 7 findings: [ACL, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture]
  - Overall 12-finding Macro AUC
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LAB = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"
]
SIDE_SPECIFIC = ["Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "MCL"]
OTHER_7 = [c for c in LAB if c not in SIDE_SPECIFIC]


def compute_aucs(truth_df, pred_df, study_ids):
    sub_truth = truth_df.loc[study_ids]
    sub_pred = pred_df.loc[study_ids]
    
    aucs = {}
    for col in LAB:
        y_true = (sub_truth[col].values > 0.5).astype(int)
        y_score = sub_pred[col].values.astype(float)
        if len(np.unique(y_true)) > 1:
            aucs[col] = float(roc_auc_score(y_true, y_score))
        else:
            aucs[col] = float("nan")
    
    side_macro = float(np.nanmean([aucs[c] for c in SIDE_SPECIFIC]))
    other_macro = float(np.nanmean([aucs[c] for c in OTHER_7]))
    all_macro = float(np.nanmean([aucs[c] for c in LAB]))
    
    return aucs, side_macro, other_macro, all_macro


def format_table(results):
    cols = ["Variant", "N", "Side-Specific (5)", "Other (7)", "Macro (12)"]
    rows = []
    for r in results:
        rows.append([
            r["name"],
            str(r["n"]),
            f"{r['side']:.4f}",
            f"{r['other']:.4f}",
            f"{r['macro']:.4f}"
        ])
    
    widths = [max(len(row[i]) for row in [cols] + rows) for i in range(len(cols))]
    hdr = " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    sep = "-+-".join("-" * widths[i] for i in range(len(cols)))
    lines = [hdr, sep]
    for row in rows:
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(cols))))
    return "\n".join(lines)


def format_per_finding_table(results):
    cols = ["Finding", "Category"] + [r["name"] for r in results]
    rows = []
    for col in LAB:
        cat = "Side" if col in SIDE_SPECIFIC else "Other"
        row = [col, cat]
        for r in results:
            val = r["aucs"].get(col, float("nan"))
            row.append(f"{val:.4f}")
        rows.append(row)
    
    widths = [max(len(row[i]) for row in [cols] + rows) for i in range(len(cols))]
    hdr = " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    sep = "-+-".join("-" * widths[i] for i in range(len(cols)))
    lines = [hdr, sep]
    for row in rows:
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(cols))))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", required=True, help="OOF plain predictions parquet")
    ap.add_argument("--mirrored", required=True, help="OOF mirrored predictions parquet")
    ap.add_argument("--labels", required=True, help="labels parquet (e.g. labels_sev_v1.parquet)")
    ap.add_argument("--laterality", required=True, help="study_laterality_orientation.csv")
    args = ap.parse_args()

    plain_df = pd.read_parquet(args.plain).set_index("StudyInstanceUID")
    mirr_df = pd.read_parquet(args.mirrored).set_index("StudyInstanceUID")
    labels_df = pd.read_parquet(args.labels).set_index("StudyInstanceUID")
    lat_df = pd.read_csv(args.laterality).set_index("StudyInstanceUID")

    common_ids = [u for u in plain_df.index if u in mirr_df.index and u in labels_df.index and u in lat_df.index]
    print(f"Total evaluated studies: {len(common_ids)}")

    left_ids = [u for u in common_ids if str(lat_df.loc[u, "Laterality"]).strip().lower() == "left"]
    right_ids = [u for u in common_ids if str(lat_df.loc[u, "Laterality"]).strip().lower() == "right"]
    print(f"  Left knees:  {len(left_ids)}")
    print(f"  Right knees: {len(right_ids)}")

    # Flip TTA dataframe: average of plain and mirrored
    tta_df = (plain_df.loc[common_ids, LAB] + mirr_df.loc[common_ids, LAB]) / 2.0

    configs = [
        ("Left Plain", plain_df, left_ids),
        ("Left Mirrored", mirr_df, left_ids),
        ("Right Plain", plain_df, right_ids),
        ("Right Mirrored (Control)", mirr_df, right_ids),
        ("Left Flip-TTA", tta_df, left_ids),
        ("Right Flip-TTA", tta_df, right_ids),
        ("All Plain", plain_df, common_ids),
        ("All Flip-TTA", tta_df, common_ids),
    ]

    results = []
    for name, p_df, uids in configs:
        aucs, side_m, other_m, all_m = compute_aucs(labels_df, p_df, uids)
        results.append({
            "name": name,
            "n": len(uids),
            "aucs": aucs,
            "side": side_m,
            "other": other_m,
            "macro": all_m
        })

    print("\n" + "=" * 65)
    print("SUMMARY RESULTS (MACRO AUC)")
    print("=" * 65)
    print(format_table(results))

    print("\n" + "=" * 65)
    print("HYPOTHESIS CHECK & SUCCESS GATE")
    print("=" * 65)
    left_plain_side = results[0]["side"]
    left_mirr_side = results[1]["side"]
    delta_left_side = left_mirr_side - left_plain_side

    right_plain_side = results[2]["side"]
    right_mirr_side = results[3]["side"]
    delta_right_side = right_mirr_side - right_plain_side

    print(f"Left Knees Side-Specific (5 findings):")
    print(f"  Plain:    {left_plain_side:.4f}")
    print(f"  Mirrored: {left_mirr_side:.4f}")
    print(f"  Gap (Mirrored - Plain): {delta_left_side:+.4f} (Threshold: >= +0.0050)")

    print(f"\nRight Knees Side-Specific (5 findings, Control):")
    print(f"  Plain:    {right_plain_side:.4f}")
    print(f"  Mirrored: {right_mirr_side:.4f}")
    print(f"  Gap (Mirrored - Plain): {delta_right_side:+.4f} (Expected: <= 0)")

    gate_passed = (delta_left_side >= 0.005) and (delta_right_side <= 0.002)
    print(f"\nSUCCESS GATE STATUS: {'>>> PASSED <<<' if gate_passed else '>>> DID NOT MEET GATE <<<'}")

    all_plain_macro = results[6]["macro"]
    all_tta_macro = results[7]["macro"]
    print(f"\nFlip-TTA Impact on All Fold-0 Knees:")
    print(f"  Plain Macro:    {all_plain_macro:.4f}")
    print(f"  Flip-TTA Macro: {all_tta_macro:.4f}")
    print(f"  Gain:           {all_tta_macro - all_plain_macro:+.4f}")

    print("\n" + "=" * 65)
    print("PER-FINDING AUC BREAKDOWN")
    print("=" * 65)
    print(format_per_finding_table(results))
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
