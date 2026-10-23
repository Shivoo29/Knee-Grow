import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA',
    'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture'
]
WEAK = ['Lateral OA', 'PF OA', 'Synovitis', 'Lateral Meniscus']

print("=" * 80)
print("1. GOLD-58 BENCHMARK GROUND TRUTH STATS")
print("=" * 80)
train_csv = '/home/s05814802723/projects/knee/train.csv'
df_train = pd.read_csv(train_csv)
df_gold = df_train.dropna(subset=LABELS).copy()
print(f"Total Gold Studies: {len(df_gold)}")

for col in LABELS:
    pos = int(df_gold[col].sum())
    prev = pos / len(df_gold) * 100
    flag = " <*** WEAK TARGET" if col in WEAK else ""
    print(f"  {col:20s}: {pos:2d} / 58 ({prev:5.1f}%){flag}")

print("\n" + "=" * 80)
print("2. LABEL TABLES AUDIT: PREVALENCE & MEANS ON ~4,349 STUDIES")
print("=" * 80)
files = {
    'labels_avg': '/home/s05814802723/projects/knee/labels_avg.parquet',
    'labels_distil_v1': '/home/s05814802723/projects/knee/labels_distil_v1.parquet',
    'labels_v4': '/home/s05814802723/projects/knee/labels_v4.parquet',
    'labels_llm_soft': '/home/s05814802723/projects/knee/ds/rsna-knee-labels/labels_llm_soft.parquet'
}

dataframes = {}
for name, p in files.items():
    if os.path.exists(p):
        df = pd.read_parquet(p)
        dataframes[name] = df
        print(f"\n--- {name} (N={len(df)}) ---")
        for col in WEAK:
            if col in df.columns:
                vals = df[col].dropna().values.astype(float)
                mean_v = np.mean(vals)
                prev_50 = np.mean(vals >= 0.5) * 100
                gold_prev = df_gold[col].mean() * 100
                print(f"  {col:20s} | Mean: {mean_v:.3f} | >=0.5 Prev: {prev_50:5.1f}% | Gold Prev: {gold_prev:5.1f}%")

print("\n" + "=" * 80)
print("3. PEARSON CORRELATION & DISAGREEMENT: LABELS_AVG vs LABELS_DISTIL_V1")
print("=" * 80)
if 'labels_avg' in dataframes and 'labels_distil_v1' in dataframes:
    d_avg = dataframes['labels_avg']
    d_dst = dataframes['labels_distil_v1']
    if 'study_id' in d_avg.columns: d_avg = d_avg.set_index('study_id')
    if 'study_id' in d_dst.columns: d_dst = d_dst.set_index('study_id')
    common = list(set(d_avg.index).intersection(set(d_dst.index)))
    print(f"Common Studies: {len(common)}")
    for col in LABELS:
        if col in d_avg.columns and col in d_dst.columns:
            s_avg = d_avg.loc[common, col].values.astype(float)
            s_dst = d_dst.loc[common, col].values.astype(float)
            r = np.corrcoef(s_avg, s_dst)[0, 1]
            mae = np.mean(np.abs(s_avg - s_dst))
            bin_agree = np.mean((s_avg >= 0.5) == (s_dst >= 0.5)) * 100
            flag = " <*** WEAK TARGET" if col in WEAK else ""
            print(f"  {col:20s} | Pearson r: {r:5.3f} | MAE: {mae:5.3f} | Bin Agree: {bin_agree:5.1f}%{flag}")

print("\n" + "=" * 80)
print("4. GOLD-58 PREDICTIONS AUDIT: D192_S50 vs V6D24_S9")
print("=" * 80)
gold_npz = {
    'd192_s50_swa': '/home/s05814802723/projects/knee/pc_code192/raptor_gold_d192_s50_swa.npz',
    'v6d24_s9_swa': '/home/s05814802723/projects/knee/pc_code/raptor_gold_v6d24_s9_swa.npz',
    'v6d24_s9_ep':  '/home/s05814802723/projects/knee/raptor_gold_v6d24_s9.npz'
}
for name, p in gold_npz.items():
    if os.path.exists(p):
        data = np.load(p)
        preds = data['probs'] if 'probs' in data else (data['preds'] if 'preds' in data else data[list(data.keys())[0]])
        targets = df_gold[LABELS].values.astype(float)
        print(f"\n--- Model: {name} (Shape: {preds.shape}) ---")
        for idx, col in enumerate(LABELS):
            auc = roc_auc_score(targets[:, idx], preds[:, idx])
            flag = " <*** WEAK TARGET" if col in WEAK else ""
            # Count false positives and false negatives at threshold 0.5
            fp = np.sum((preds[:, idx] >= 0.5) & (targets[:, idx] == 0))
            fn = np.sum((preds[:, idx] < 0.5) & (targets[:, idx] == 1))
            print(f"  {col:20s}: AUC {auc:6.4f} | FP: {fp:2d} | FN: {fn:2d}{flag}")
