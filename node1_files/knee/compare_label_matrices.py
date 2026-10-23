import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

LAB = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
       "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

# 1. Load files
df_res = pd.read_csv("/home/s05814802723/projects/knee/result_labels.csv")
df_avg = pd.read_parquet("/home/s05814802723/projects/knee/labels_avg.parquet")
df_hyb = pd.read_parquet("/home/s05814802723/projects/knee/labels_hybrid_v1.parquet")
train_df = pd.read_csv("/home/s05814802723/projects/knee/train.csv")

uid_col = df_res.columns[0]
train_uid_col = train_df.columns[0]

# Gold studies
gold_mask = train_df[LAB].notna().all(axis=1)
gold_uids = set(train_df.loc[gold_mask, train_uid_col].values)
train_uids = [u for u in df_res[uid_col].values if u not in gold_uids]

print(f"Total rows in result_labels: {len(df_res)}")
print(f"Gold studies: {len(gold_uids)}, Training non-gold studies: {len(train_uids)}")

# Align training studies
df_res_train = df_res.set_index(uid_col).loc[train_uids]
df_avg_train = df_avg.loc[train_uids] if train_uids[0] in df_avg.index else df_avg
df_hyb_train = df_hyb.loc[train_uids] if train_uids[0] in df_hyb.index else df_hyb

print("=" * 90)
print("COMPARISON: result_labels.csv vs labels_avg.parquet (on 4,349 Training Studies)")
print("=" * 90)
print(f"{'Finding':18s} | {'Mean (result)':14s} | {'Mean (labels_avg)':18s} | {'Pearson r':10s} | {'Spearman r':10s}")
print("-" * 90)

for l in LAB:
    r_vals = df_res_train[l].values
    a_vals = df_avg_train[l].values
    pr, _ = pearsonr(r_vals, a_vals)
    sr, _ = spearmanr(r_vals, a_vals)
    print(f"{l:18s} | {r_vals.mean():14.4f} | {a_vals.mean():18.4f} | {pr:10.4f} | {sr:10.4f}")

# Gold-58 analysis
gold_df = train_df.loc[gold_mask].set_index(train_uid_col)
df_res_gold = df_res.set_index(uid_col).loc[gold_df.index]

print("\n" + "=" * 90)
print("GOLD-58 DETAILED EVALUATION OF result_labels.csv")
print("=" * 90)
print(f"{'Finding':18s} | {'Gold Truth Mean':16s} | {'result_labels Mean':18s} | {'Gold AUC':10s}")
print("-" * 90)

aucs = []
for l in LAB:
    y_true = gold_df[l].values
    y_pred = df_res_gold[l].values
    auc = roc_auc_score(y_true, y_pred)
    aucs.append(auc)
    print(f"{l:18s} | {y_true.mean():16.4f} | {y_pred.mean():18.4f} | {auc:10.4f}")

print("-" * 90)
print(f"{'MACRO AUC':18s} | {'':16s} | {'':18s} | {np.mean(aucs):10.4f}")
