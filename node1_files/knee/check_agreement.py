import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

gold = pd.read_csv('/home/s05814802723/projects/knee/ds/gold_58.csv')
labels_avg = pd.read_parquet('/home/s05814802723/projects/knee/labels_avg.parquet')
llm_full = pd.read_csv('/home/s05814802723/projects/knee/ds/rsna-knee-llm-report-labels/llm_labels_full.csv')

findings = [c for c in gold.columns if c not in ['StudyInstanceUID', 'fold']]

mg_avg = gold.merge(labels_avg, on='StudyInstanceUID', suffixes=('_gold', '_avg'))
mg_llm = gold.merge(llm_full, on='StudyInstanceUID', suffixes=('_gold', '_llm'))

print(f'Matching gold in labels_avg: {len(mg_avg)}/58')
print(f'Matching gold in llm_full: {len(mg_llm)}/58')

print('\nFinding-by-finding AUC of LLM labels against Gold-58:')
for f in findings:
    y_true = mg_llm[f + '_gold'].values
    y_pred = mg_llm[f + '_llm'].values
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred)
        print(f'  {f:18s}: {auc:.4f} (pos={int(y_true.sum())})')
    else:
        print(f'  {f:18s}: N/A')

