import numpy as np
import pandas as pd

LAB = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
       "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

# 1. Gold-58 expert ground truth from train.csv
train_df = pd.read_csv("/home/s05814802723/projects/knee/train.csv")
gold_mask = train_df[LAB].notna().all(axis=1)
gold_df = train_df[gold_mask].copy()

print("=" * 80)
print(f"1. EXPERT GOLD GROUND TRUTH (n={len(gold_df)} studies)")
print("=" * 80)
print(f"{'Finding / Pathology':20s} | {'Positive (1)':14s} | {'Negative (0)':14s} | {'Positive %':12s}")
print("-" * 80)
for l in LAB:
    pos = int((gold_df[l] == 1.0).sum())
    neg = int((gold_df[l] == 0.0).sum())
    pct = (pos / len(gold_df)) * 100.0
    print(f"{l:20s} | {pos:14d} | {neg:14d} | {pct:11.1f}%")

# 2. Full 4,407 Cohort from result_labels.csv
df_res = pd.read_csv("/home/s05814802723/projects/knee/result_labels.csv")
print("\n" + "=" * 80)
print(f"2. FULL COHORT: result_labels.csv (n={len(df_res)} studies)")
print("   Thresholded at probability >= 0.5 for binary presence:")
print("=" * 80)
print(f"{'Finding / Pathology':20s} | {'Prob >= 0.5':14s} | {'Prob < 0.5':14s} | {'Pos % (>=0.5)':14s} | {'Mean Prob':10s}")
print("-" * 80)
for l in LAB:
    pos = int((df_res[l] >= 0.5).sum())
    neg = int((df_res[l] < 0.5).sum())
    pct = (pos / len(df_res)) * 100.0
    mean_p = df_res[l].mean()
    print(f"{l:20s} | {pos:14d} | {neg:14d} | {pct:13.1f}% | {mean_p:9.3f}")

# 3. Full 4,349 Cohort from labels_avg.parquet (the old noisy set)
df_avg = pd.read_parquet("/home/s05814802723/projects/knee/labels_avg.parquet")
print("\n" + "=" * 80)
print(f"3. OLD TRAINING LABELS: labels_avg.parquet (n={len(df_avg)} studies)")
print("=" * 80)
print(f"{'Finding / Pathology':20s} | {'Prob >= 0.5':14s} | {'Prob < 0.5':14s} | {'Pos % (>=0.5)':14s} | {'Mean Prob':10s}")
print("-" * 80)
for l in LAB:
    pos = int((df_avg[l] >= 0.5).sum())
    neg = int((df_avg[l] < 0.5).sum())
    pct = (pos / len(df_avg)) * 100.0
    mean_p = df_avg[l].mean()
    print(f"{l:20s} | {pos:14d} | {neg:14d} | {pct:13.1f}% | {mean_p:9.3f}")
