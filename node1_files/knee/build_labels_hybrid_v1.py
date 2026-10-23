#!/usr/bin/env python3
"""build_labels_hybrid_v1.py - Construct the clean hybrid label table for RSNA Knee 2026.

Teammate-Verified Rules (Updated 17-Oct-2026):
1. Base Table: Start from labels_avg.parquet (captures consensus across 31 public tables, 0.810 on Lateral OA).
2. ONLY MCL is spliced from LLM:
   - MCL in labels_avg suffers from 14.7% severity over-call (crashed MCL to 0.872).
   - LLM / Distilled MCL is clean at 4.9% prevalence (reaches 0.954 - 0.964 MCL).
   - Lateral OA is NOT spliced: LLM scored 0.774 vs 0.810 for labels_avg, so Lateral OA is kept from labels_avg!
3. Synovitis Effusion-Conditional Down-weighting:
   - Reports rarely address synovitis (83.7% not mentioned).
   - When effusion is positive (Effusion >= 0.5) but report is silent/negative on synovitis, downweight loss weight (W = 0.35).
   - This prevents false-negative reports from pulling down confident image predictions.
"""
import os
import sys
import numpy as np
import pandas as pd

KNEE = "/home/s05814802723/projects/knee"
LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA',
    'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def main():
    print("=" * 80)
    print("RE-BUILDING LABELS_HYBRID_V1.PARQUET & LABEL_WEIGHTS_HYBRID_V1.PARQUET (CORRECTED)")
    print("=" * 80)

    p_avg = os.path.join(KNEE, "labels_avg.parquet")
    p_llm = os.path.join(KNEE, "ds", "rsna-knee-labels", "labels_llm_soft.parquet")
    p_train = os.path.join(KNEE, "train.csv")

    df_avg = pd.read_parquet(p_avg).set_index("StudyInstanceUID")
    df_llm = pd.read_parquet(p_llm).set_index("StudyInstanceUID")
    df_train = pd.read_csv(p_train).set_index("StudyInstanceUID")

    print(f"Loaded labels_avg: {len(df_avg)} rows")
    print(f"Loaded labels_llm_soft: {len(df_llm)} rows")
    print(f"Loaded train.csv: {len(df_train)} rows")

    # Start with labels_avg
    df_hybrid = df_avg.copy()

    # 1. Splicing ONLY MCL from clean LLM
    print("\n[1] Splicing clean MCL from labels_llm_soft (and keeping Lateral OA from labels_avg)...")
    old_mcl_prev = (df_hybrid['MCL'] >= 0.5).mean() * 100
    df_hybrid['MCL'] = df_llm['MCL'].clip(0.05, 0.95)
    new_mcl_prev = (df_hybrid['MCL'] >= 0.5).mean() * 100
    print(f"  MCL >= 0.5 prevalence adjusted: {old_mcl_prev:.1f}% -> {new_mcl_prev:.1f}%")
    print(f"  Lateral OA: Retained from labels_avg (current prevalence: {(df_hybrid['Lateral OA']>=0.5).mean()*100:.1f}%, benchmark: 0.810)")

    # 2. Synovitis Effusion-Conditional Downweighting
    print("\n[2] Constructing Synovitis Effusion-Conditional Loss Weight Matrix...")
    df_weights = pd.DataFrame(1.0, index=df_hybrid.index, columns=LABELS)

    if "Report" in df_train.columns:
        reports = df_train["Report"].fillna("").astype(str).str.lower()
        syn_mentioned = reports.str.contains(r"synovi|pannus|synovial", regex=True)
        
        # Condition: Report does not mention synovitis AND effusion is elevated (>= 0.35)
        # In these cases, a negative synovitis target is highly suspect (false negative).
        downweighted = 0
        for uid in df_weights.index:
            if uid in reports.index:
                is_syn_silent = not syn_mentioned.loc[uid]
                eff_prob = df_hybrid.loc[uid, 'Effusion']
                if is_syn_silent and eff_prob >= 0.35:
                    df_weights.loc[uid, 'Synovitis'] = 0.35
                    downweighted += 1
        print(f"  Downweighted {downweighted} / {len(df_weights)} ({downweighted/len(df_weights)*100:.1f}%) silent synovitis cases with effusion to W=0.35")

    # 3. Save corrected files
    out_labels = os.path.join(KNEE, "labels_hybrid_v1.parquet")
    out_weights = os.path.join(KNEE, "label_weights_hybrid_v1.parquet")

    df_hybrid.reset_index().to_parquet(out_labels)
    df_weights.reset_index().to_parquet(out_weights)

    print(f"\n[3] Successfully wrote corrected files:")
    print(f"  {out_labels} ({os.path.getsize(out_labels)/1024:.1f} KB)")
    print(f"  {out_weights} ({os.path.getsize(out_weights)/1024:.1f} KB)")

    print("\n" + "=" * 80)
    print("FINAL PREVALENCE AUDIT (>= 0.5 POSITIVE %)")
    print("=" * 80)
    for col in LABELS:
        p_avg = (df_avg[col] >= 0.5).mean() * 100
        p_hyb = (df_hybrid[col] >= 0.5).mean() * 100
        print(f"  {col:20s} | labels_avg: {p_avg:5.1f}% | labels_hybrid_v1: {p_hyb:5.1f}%")

if __name__ == "__main__":
    main()
