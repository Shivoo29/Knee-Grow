import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

os.environ["CUDA_VISIBLE_DEVICES"] = ""

LAB = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
       'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

ROOT = "/home/s05814802723/projects/knee"

# 1. List files in rsna-knee-labels and related dirs
print("=== FILES IN ds/rsna-knee-labels/ ===")
p_labels = f"{ROOT}/ds/rsna-knee-labels"
if os.path.exists(p_labels):
    for f in os.listdir(p_labels):
        full = os.path.join(p_labels, f)
        print(f"  {f:30s} ({os.path.getsize(full)/1024:.1f} KB)")

print("\n=== FILES IN ds/rsna-knee-llm-report-labels/ ===")
p_llm = f"{ROOT}/ds/rsna-knee-llm-report-labels"
if os.path.exists(p_llm):
    for f in os.listdir(p_llm):
        full = os.path.join(p_llm, f)
        print(f"  {f:30s} ({os.path.getsize(full)/1024:.1f} KB)")

# Load train and Gold-58
train = pd.read_csv(f"{ROOT}/train.csv")
gold = train.dropna(subset=LAB)
gold_uids = set(gold['StudyInstanceUID'].astype(str))
print(f"\nGold 58 studies loaded: {len(gold_uids)}")

# 2. Check all label files across repository for Gold-58 overlap and compute ROC AUC if found
cand_files = glob.glob(f"{ROOT}/**/*.parquet", recursive=True) + glob.glob(f"{ROOT}/**/*.csv", recursive=True)
cand_files = [f for f in cand_files if "label" in f.lower() or "llm" in f.lower()]

print("\n=== CHECKING LABEL FILES FOR GOLD-58 OVERLAP & ROC AUC ===")
for p in sorted(set(cand_files)):
    rel = os.path.relpath(p, ROOT)
    if "venv" in rel or ".git" in rel:
        continue
    try:
        df = pd.read_parquet(p) if p.endswith('.parquet') else pd.read_csv(p)
    except Exception:
        continue
    
    uid_col = None
    for c in ['StudyInstanceUID', 'study_id', 'UID', 'id']:
        if c in df.columns:
            uid_col = c
            break
    if uid_col is None:
        continue
        
    df_uids = set(df[uid_col].astype(str))
    overlap = gold_uids.intersection(df_uids)
    print(f"File: {rel:45s} | Rows: {len(df):5d} | Overlap with Gold-58: {len(overlap):2d}")
    
    if len(overlap) >= 30: # If it has substantial gold overlap, test ROC AUC!
        mg = gold.merge(df, left_on='StudyInstanceUID', right_on=uid_col, suffixes=('_gold', '_pred'))
        aucs = {}
        for c in LAB:
            # find gold column
            gold_c = c + '_gold' if c + '_gold' in mg.columns else (c if c in gold.columns else None)
            if gold_c not in mg.columns:
                continue
            # find pred column
            pred_col = None
            for dc in mg.columns:
                if dc == gold_c:
                    continue
                if dc.lower().replace(" ", "").replace("_", "") == c.lower().replace(" ", "").replace("_", ""):
                    pred_col = dc
                    break
                if dc == c + '_pred':
                    pred_col = dc
                    break
            if pred_col is not None and pred_col in mg.columns:
                y_t = mg[gold_c].values.astype(float)
                y_p = mg[pred_col].values.astype(float)
                if len(np.unique(y_t)) > 1:
                    aucs[c] = roc_auc_score(y_t, y_p)
        if aucs:
            print("   ---> Measured ROC AUC against Gold-58:")
            for c in LAB:
                if c in aucs:
                    print(f"        {c:20s}: {aucs[c]:.4f}")
            print(f"        {'MACRO ROC AUC':20s}: {np.mean(list(aucs.values())):.4f}")

# 3. Print mean, std, min, max, and >0.5 prevalence for labels_distil_v1.parquet vs train.csv
print("\n" + "="*85)
print("=== LABELS_DISTIL_V1.PARQUET (4,349 STUDIES) VS TRAIN.CSV (58 GOLD STUDIES) ===")
print("="*85)
p_distil = f"{ROOT}/labels_distil_v1.parquet"
df_distil = pd.read_parquet(p_distil)

print(f"{'Finding':20s} | {'Distil Mean':11s} {'Std':6s} {'Min':6s} {'Max':6s} {'>0.5 Prev':9s} | {'Gold Mean':10s} {'>0.5 Prev':9s}")
print("-" * 85)
for c in LAB:
    # Distil stats
    v_dist = df_distil[c].values.astype(float)
    d_mean = np.mean(v_dist)
    d_std  = np.std(v_dist)
    d_min  = np.min(v_dist)
    d_max  = np.max(v_dist)
    d_prev = np.mean(v_dist > 0.5)
    
    # Gold stats
    v_gold = gold[c].values.astype(float)
    g_mean = np.mean(v_gold)
    g_prev = np.mean(v_gold > 0.5)
    
    print(f"{c:20s} | {d_mean:11.4f} {d_std:6.3f} {d_min:6.3f} {d_max:6.3f} {d_prev:8.1%} | {g_mean:10.4f} {g_prev:8.1%}")
