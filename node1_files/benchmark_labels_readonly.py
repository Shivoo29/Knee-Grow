import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

LAB = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
       'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

train = pd.read_csv('/home/s05814802723/projects/knee/train.csv')
gold = train.dropna(subset=LAB)
print(f"Loaded Gold ground truth: {len(gold)} studies.")

files = [
    '/home/s05814802723/projects/knee/labels_distil_v1.parquet',
    '/home/s05814802723/projects/knee/labels_avg.parquet',
    '/home/s05814802723/projects/knee/labels_v4.parquet',
    '/home/s05814802723/projects/knee/ds/rsna-knee-labels/labels_llm_soft.parquet',
]

for p in files:
    try:
        df = pd.read_parquet(p)
        mg = gold.merge(df, on='StudyInstanceUID', suffixes=('_gold', '_pred'))
        print(f"\n--- {p.split('/')[-1]} (Matches {len(mg)} Gold Studies) ---")
        aucs = {}
        for c in LAB:
            pred_c = c + '_pred' if c + '_pred' in mg.columns else c
            y_t = mg[c + '_gold'].values.astype(float)
            y_p = mg[pred_c].values.astype(float)
            if len(np.unique(y_t)) > 1:
                aucs[c] = roc_auc_score(y_t, y_p)
            else:
                aucs[c] = float('nan')
        for c, a in aucs.items():
            print(f"  {c:18s}: {a:.4f}")
        macro = np.nanmean(list(aucs.values()))
        print(f"  ==> MACRO ROC AUC: {macro:.4f}")
    except Exception as e:
        print(f"Error on {p}: {e}")
