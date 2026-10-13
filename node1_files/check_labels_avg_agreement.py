import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

gold = pd.read_csv('/home/s05814802723/projects/knee/ds/rsna-knee-hpo-assets/gold_58.csv')
labels_avg = pd.read_parquet('/home/s05814802723/projects/knee/labels_avg.parquet')
findings = [c for c in gold.columns if c not in ['StudyInstanceUID', 'fold']]

mg = gold.merge(labels_avg, on='StudyInstanceUID', suffixes=('_gold', '_avg'))
print('Finding-by-finding AUC of labels_avg against Gold-58:')
aucs = []
for f in findings:
    y_true = mg[f + '_gold'].values
    y_pred = mg[f + '_avg'].values
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred)
        aucs.append(auc)
        print(f'  {f:18s}: {auc:.4f} (pos={int(y_true.sum())})')
print(f'Macro AUC: {np.mean(aucs):.4f}')
