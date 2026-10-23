import os
import pandas as pd
import numpy as np

LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA',
    'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture'
]
TARGET_COLS = ['Lateral OA', 'PF OA', 'Synovitis', 'Lateral Meniscus']

train_csv_path = '/home/s05814802723/projects/knee/train.csv'
if not os.path.exists(train_csv_path):
    train_csv_path = '/home/s05814802723/projects/knee/ds/rsna-knee-dataset/train.csv'

print('=' * 80)
print('1. GOLD-58 BENCHMARK AUDIT (N=58)')
print('=' * 80)
if os.path.exists(train_csv_path):
    df_train = pd.read_csv(train_csv_path)
    df_gold = df_train.dropna(subset=LABELS)
    print(f'Total Gold studies: {len(df_gold)}')
    for col in LABELS:
        pos = int(df_gold[col].sum())
        prev = pos / len(df_gold) * 100
        marker = ' [TARGET WEAK CLASS]' if col in TARGET_COLS else ''
        print(f'{col:20s}: {pos:2d} / {len(df_gold)} ({prev:5.1f}%){marker}')

print('\n' + '=' * 80)
print('2. LABEL TABLE COMPARISON ACROSS LARGE COHORT (~4349 studies)')
print('=' * 80)
tables = {
    'labels_avg': '/home/s05814802723/projects/knee/labels_avg.parquet',
    'labels_distil_v1': '/home/s05814802723/projects/knee/labels_distil_v1.parquet',
    'labels_v4': '/home/s05814802723/projects/knee/labels_v4.parquet',
    'labels_llm_soft': '/home/s05814802723/projects/knee/ds/rsna-knee-labels/labels_llm_soft.parquet'
}

summary = []
loaded_dfs = {}
for name, p in tables.items():
    if os.path.exists(p):
        try:
            df = pd.read_parquet(p)
            loaded_dfs[name] = df
            row = {'Table': name, 'N': len(df)}
            for col in LABELS:
                if col in df.columns:
                    s = df[col].dropna()
                    mean_val = s.mean()
                    row[col] = f'{mean_val:.3f}'
            summary.append(row)
        except Exception as e:
            print(f'Error reading {name}: {e}')

df_summary = pd.DataFrame(summary)
print(df_summary[['Table', 'N'] + TARGET_COLS].to_string(index=False))

print('\nFull label table summary:')
print(df_summary.to_string(index=False))

print('\n' + '=' * 80)
print('3. DISAGREEMENT AND CORRELATION BETWEEN LABELS_AVG AND LABELS_DISTIL_V1')
print('=' * 80)
if 'labels_avg' in loaded_dfs and 'labels_distil_v1' in loaded_dfs:
    d1 = loaded_dfs['labels_avg']
    d2 = loaded_dfs['labels_distil_v1']
    if 'study_id' in d1.columns:
        d1 = d1.set_index('study_id')
    if 'study_id' in d2.columns:
        d2 = d2.set_index('study_id')
    common = list(set(d1.index).intersection(set(d2.index)))
    print(f'Common studies: {len(common)}')
    for col in LABELS:
        if col in d1.columns and col in d2.columns:
            s1 = d1.loc[common, col].astype(float)
            s2 = d2.loc[common, col].astype(float)
            corr = np.corrcoef(s1, s2)[0, 1]
            mae = np.mean(np.abs(s1 - s2))
            bin_agree = np.mean((s1 >= 0.5) == (s2 >= 0.5)) * 100
            marker = ' [TARGET WEAK CLASS]' if col in TARGET_COLS else ''
            print(f'{col:20s} | Pearson r: {corr:5.3f} | MAE: {mae:5.3f} | Bin Agree: {bin_agree:5.1f}%{marker}')

print('\n' + '=' * 80)
print('4. GOLD-58 PREDICTIONS AUDIT ON D192_S50 AND V6D24_S9')
print('=' * 80)
# Check if raptor_gold_d192_s50.npz and raptor_gold_v6d24_s9.npz exist
pred_files = {
    'd192_s50': '/home/s05814802723/projects/knee/pc_code192/raptor_gold_d192_s50_swa.npz',
    'v6d24_s9': '/home/s05814802723/projects/knee/pc_code/raptor_gold_v6d24_s9_swa.npz'
}
for tag, pf in pred_files.items():
    if not os.path.exists(pf):
        # try alternate locations
        alt = f'/home/s05814802723/projects/knee/{os.path.basename(pf)}'
        if os.path.exists(alt):
            pf = alt
    if os.path.exists(pf):
        try:
            data = np.load(pf)
            print(f'\n{tag} from {pf}:')
            print('Keys:', list(data.keys()))
            if 'probs' in data and 'targets' in data:
                probs = data['probs']
                targets = data['targets']
                from sklearn.metrics import roc_auc_score
                for idx, col in enumerate(LABELS):
                    y_true = targets[:, idx]
                    y_pred = probs[:, idx]
                    if len(np.unique(y_true)) > 1:
                        auc = roc_auc_score(y_true, y_pred)
                        # False positive rate at 0.5 or worst false positives
                        fp = np.sum((y_pred >= 0.5) & (y_true == 0))
                        fn = np.sum((y_pred < 0.5) & (y_true == 1))
                        marker = ' [TARGET WEAK CLASS]' if col in TARGET_COLS else ''
                        print(f'  {col:20s}: AUC {auc:6.4f} | FP (>=0.5): {fp:2d} | FN (<0.5): {fn:2d}{marker}')
        except Exception as e:
            print(f'Error reading {pf}: {e}')
