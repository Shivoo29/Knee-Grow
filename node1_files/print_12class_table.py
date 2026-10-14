import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LAB = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
       'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

df = pd.read_csv('/home/s05814802723/projects/knee/train.csv')
gold_df = df.dropna(subset=LAB)
y_true = gold_df[LAB].values

gold_files = {
    'v6d_s7': 'raptor_gold_v6d_s7_swa.npz',
    'v6_s4':  'raptor_gold_v6_s4_swa.npz',
    'v6_s3':  'raptor_gold_v6_s3_swa.npz',
}

preds = {}
for name, fname in gold_files.items():
    data = np.load(f'/home/s05814802723/projects/knee/{fname}')
    p_key = 'preds' if 'preds' in data.files else data.files[0]
    preds[name] = data[p_key]

top3_ens = np.mean([preds['v6d_s7'], preds['v6_s4'], preds['v6_s3']], axis=0)

print(f"{'Finding Label':20s} | {'v6d_s7':8s} | {'v6_s4':8s} | {'v6_s3':8s} | {'Top-3 Blend':11s}")
print("-" * 65)
for i, name in enumerate(LAB):
    auc_s7 = roc_auc_score(y_true[:, i], preds['v6d_s7'][:, i])
    auc_s4 = roc_auc_score(y_true[:, i], preds['v6_s4'][:, i])
    auc_s3 = roc_auc_score(y_true[:, i], preds['v6_s3'][:, i])
    auc_ens = roc_auc_score(y_true[:, i], top3_ens[:, i])
    print(f"{name:20s} | {auc_s7:8.4f} | {auc_s4:8.4f} | {auc_s3:8.4f} | {auc_ens:11.4f}")

print("-" * 65)
m_s7 = np.mean([roc_auc_score(y_true[:, i], preds['v6d_s7'][:, i]) for i in range(12)])
m_s4 = np.mean([roc_auc_score(y_true[:, i], preds['v6_s4'][:, i]) for i in range(12)])
m_s3 = np.mean([roc_auc_score(y_true[:, i], preds['v6_s3'][:, i]) for i in range(12)])
m_ens = np.mean([roc_auc_score(y_true[:, i], top3_ens[:, i]) for i in range(12)])
print(f"{'MACRO ROC AUC':20s} | {m_s7:8.4f} | {m_s4:8.4f} | {m_s3:8.4f} | {m_ens:11.4f}")
