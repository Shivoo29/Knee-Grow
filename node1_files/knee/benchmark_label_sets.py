import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

gold = pd.read_csv('/home/s05814802723/projects/knee/ds/rsna-knee-hpo-assets/gold_58.csv')
findings = [c for c in gold.columns if c not in ['StudyInstanceUID', 'fold']]

cand_files = [
    '/home/s05814802723/projects/knee/labels_avg.parquet',
    '/home/s05814802723/projects/knee/labels_v4.parquet',
    '/home/s05814802723/projects/knee/ds/rsna-knee-llm-report-labels/llm_labels_full.csv',
    '/home/s05814802723/projects/knee/ds/rsna-knee-llm-report-labels/llm_labels_v2.csv',
    '/home/s05814802723/projects/knee/ds/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv',
    '/home/s05814802723/projects/knee/ds/rsna-knee-labels/labels_llm_soft.csv',
    '/home/s05814802723/projects/knee/ds/rsna-knee-hpo-assets/labels_v1_blend.parquet',
    '/home/s05814802723/projects/knee/ds/rsna-knee-hpo-assets/labels_v1_perlabel.parquet',
]

print("=== BENCHMARKING LABEL SETS AGAINST GOLD-58 ===")
for path in cand_files:
    if not os.path.exists(path):
        continue
    try:
        df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    except Exception as e:
        continue
    
    if 'StudyInstanceUID' not in df.columns:
        continue
    mg = gold.merge(df, on='StudyInstanceUID', suffixes=('_gold', '_pred'))
    if len(mg) < 50:
        continue
    aucs = []
    lat_oa = None
    pf_oa = None
    syn = None
    for f in findings:
        pred_col = f + '_pred' if f + '_pred' in mg.columns else f
        if pred_col not in mg.columns:
            continue
        y_true = mg[f + '_gold'].values
        y_pred = mg[pred_col].values
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_pred)
            aucs.append(auc)
            if f == 'Lateral OA': lat_oa = auc
            elif f == 'PF OA': pf_oa = auc
            elif f == 'Synovitis': syn = auc
    if len(aucs) == 12:
        m = np.mean(aucs)
        lat_s = f"{lat_oa:.4f}" if lat_oa is not None else "N/A"
        pf_s = f"{pf_oa:.4f}" if pf_oa is not None else "N/A"
        syn_s = f"{syn:.4f}" if syn is not None else "N/A"
        print(f"{os.path.basename(path):32s} | Macro: {m:.4f} | LatOA: {lat_s} | PFOA: {pf_s} | Syn: {syn_s} | N={len(df)}")
