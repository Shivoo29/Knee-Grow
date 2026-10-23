import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LAB = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
       'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

gold_files = {
    'v6d_s7': 'raptor_gold_v6d_s7_swa.npz',
    'v6_s4':  'raptor_gold_v6_s4_swa.npz',
    'v6_s3':  'raptor_gold_v6_s3_swa.npz',
    'v6_s5':  'raptor_gold_v6_s5_swa.npz',
    'v6_s1':  'raptor_gold_v6_swa.npz',
    'v5s2':   'raptor_gold_v5s2_swa.npz'
}

preds = {}
y_true = None

for name, fname in gold_files.items():
    try:
        data = np.load(f'/home/s05814802723/projects/knee/{fname}')
        # print keys
        keys = list(data.files)
        p_key = 'preds' if 'preds' in keys else ('pred' if 'pred' in keys else keys[0])
        preds[name] = data[p_key]
        if y_true is None:
            if 'targets' in keys:
                y_true = data['targets']
            elif 'y' in keys:
                y_true = data['y']
            elif 'y_true' in keys:
                y_true = data['y_true']
    except Exception as e:
        print(f'Error loading {fname}: {e}')

if y_true is None:
    # load from train.csv
    df = pd.read_csv('/home/s05814802723/projects/knee/train.csv')
    gold_df = df.dropna(subset=LAB)
    y_true = gold_df[LAB].values

print(f'Gold evaluation on {len(y_true)} studies across 12 labels:')
for name, p in preds.items():
    aucs = []
    for i in range(12):
        try:
            aucs.append(roc_auc_score(y_true[:, i], p[:, i]))
        except Exception:
            pass
    print(f'  {name:10s} Solo Gold Macro AUC: {np.mean(aucs):.4f}')

# Equal blend
p_all = np.mean(list(preds.values()), axis=0)
aucs_all = [roc_auc_score(y_true[:, i], p_all[:, i]) for i in range(12)]
print(f'\n==> ALL 6-MODEL EQUAL BLEND Macro AUC: {np.mean(aucs_all):.4f}')

# Top 3 (v6d_s7, v6_s4, v6_s3)
top3 = [preds['v6d_s7'], preds['v6_s4'], preds['v6_s3']]
p_top3 = np.mean(top3, axis=0)
aucs_top3 = [roc_auc_score(y_true[:, i], p_top3[:, i]) for i in range(12)]
print(f'==> TOP 3 SEED BLEND (v6d_s7 + v6_s4 + v6_s3) Macro AUC: {np.mean(aucs_top3):.4f}')
