import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LAB = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
       "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

# 1. Load result_labels.csv
df_res = pd.read_csv("/home/s05814802723/projects/knee/result_labels.csv")
print("=" * 80)
print("1. RESULT_LABELS.CSV INTEGRITY & SHAPE CHECK")
print("=" * 80)
print(f"Shape: {df_res.shape}")
print(f"Columns: {list(df_res.columns)}")
print(f"Null count: {df_res.isna().sum().sum()}")
uid_col = df_res.columns[0]
print(f"UID column: '{uid_col}', unique: {df_res[uid_col].nunique()}")

# 2. Compare with train.csv
train_df = pd.read_csv("/home/s05814802723/projects/knee/train.csv")
print(f"Matches train.csv length: {len(df_res) == len(train_df)}")
order_match = (df_res[uid_col].values == train_df[train_df.columns[0]].values).all()
print(f"Matches train.csv UID order exactly: {order_match}")

# 3. Mean probabilities across all 4,407 rows
print("\n" + "=" * 80)
print("2. MEAN PROBABILITIES ACROSS ALL 4,407 ROWS")
print("=" * 80)
for col in LAB:
    if col in df_res.columns:
        print(f"  {col:18s}: {df_res[col].mean():.4f} (min: {df_res[col].min():.3f}, max: {df_res[col].max():.3f})")

# 4. Gold-58 Evaluation
# Load expert ground truth from train.csv (studies where all 12 labels are non-null)
gold_mask = train_df[LAB].notna().all(axis=1)
gold_train = train_df[gold_mask].copy()
gold_uids = gold_train[train_df.columns[0]].values
y_gold = gold_train[LAB].values

print("\n" + "=" * 80)
print(f"3. GOLD-58 EVALUATION (n={len(gold_train)} studies)")
print("   Checking how well result_labels.csv matches expert ground truth:")
print("=" * 80)

# Merge or filter df_res for gold_uids
df_res_gold = df_res[df_res[uid_col].isin(gold_uids)].copy()
# Sort in same order as gold_train
df_res_gold = df_res_gold.set_index(uid_col).reindex(gold_uids)
p_res = df_res_gold[LAB].values

aucs_res = []
for i, l in enumerate(LAB):
    try:
        a = roc_auc_score(y_gold[:, i], p_res[:, i])
        aucs_res.append(a)
    except Exception as e:
        aucs_res.append(0.5)

# Also compare against labels_avg and labels_hybrid_v1 on Gold-58 if available
try:
    df_avg = pd.read_parquet("/home/s05814802723/projects/knee/labels_avg.parquet")
    df_avg_gold = df_avg.loc[gold_uids] if gold_uids[0] in df_avg.index else df_avg[df_avg.index.isin(gold_uids)].reindex(gold_uids)
    p_avg = df_avg_gold[LAB].values
    aucs_avg = [roc_auc_score(y_gold[:, i], p_avg[:, i]) for i in range(12)]
except Exception as e:
    aucs_avg = None

try:
    df_hyb = pd.read_parquet("/home/s05814802723/projects/knee/labels_hybrid_v1.parquet")
    df_hyb_gold = df_hyb.loc[gold_uids] if gold_uids[0] in df_hyb.index else df_hyb[df_hyb.index.isin(gold_uids)].reindex(gold_uids)
    p_hyb = df_hyb_gold[LAB].values
    aucs_hyb = [roc_auc_score(y_gold[:, i], p_hyb[:, i]) for i in range(12)]
except Exception as e:
    aucs_hyb = None

print(f"{'Pathology':18s} | {'result_labels':13s} | {'labels_avg':10s} | {'labels_hybrid':13s}")
print("-" * 65)
for i, l in enumerate(LAB):
    s_res = f"{aucs_res[i]:.4f}"
    s_avg = f"{aucs_avg[i]:.4f}" if aucs_avg else "N/A"
    s_hyb = f"{aucs_hyb[i]:.4f}" if aucs_hyb else "N/A"
    print(f"{l:18s} | {s_res:13s} | {s_avg:10s} | {s_hyb:13s}")

print("-" * 65)
m_res = np.mean(aucs_res)
m_avg = f"{np.mean(aucs_avg):.4f}" if aucs_avg else "N/A"
m_hyb = f"{np.mean(aucs_hyb):.4f}" if aucs_hyb else "N/A"
print(f"{'MACRO AUC':18s} | {m_res:13.4f} | {m_avg:10s} | {m_hyb:13s}")
