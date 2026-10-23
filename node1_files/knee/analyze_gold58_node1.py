import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr

ROOT = "/home/s05814802723/projects/knee"
LAB = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

print("=" * 90)
print("DEEP EXHAUSTIVE DEDUCTION & ANALYSIS OF THE RSNA KNEE 'GOLD-58' BENCHMARK")
print("=" * 90)

df = pd.read_csv(f"{ROOT}/train.csv")
gold = df.dropna(subset=LAB).copy()
print(f"Total training cohort studies: {len(df)}")
print(f"Studies with complete ground truth across all 12 abnormalities (Gold-58): {len(gold)}")

# 1. Label prevalence and exact counts
print("\n" + "=" * 90)
print("1. EXACT LABEL PREVALENCE & CLASS BALANCE (N=58)")
print("=" * 90)
stats = []
for c in LAB:
    pos = int((gold[c] == 1.0).sum())
    neg = int((gold[c] == 0.0).sum())
    prev = pos / len(gold)
    stats.append({
        "Finding": c,
        "Positives": pos,
        "Negatives": neg,
        "Prevalence (%)": round(prev * 100, 2),
        "Minority Class Size": min(pos, neg)
    })

df_stats = pd.DataFrame(stats)
print(df_stats.to_string(index=False))

# 2. Number of positive findings per patient (co-morbidity / multi-label density)
gold["num_positives"] = gold[LAB].sum(axis=1)
print("\n" + "=" * 90)
print("2. MULTI-LABEL CO-MORBIDITY: NUMBER OF POSITIVE FINDINGS PER STUDY")
print("=" * 90)
pos_counts = gold["num_positives"].value_counts().sort_index()
for k, v in pos_counts.items():
    print(f"  {int(k):2d} abnormalities: {v:2d} studies ({v/len(gold)*100:5.1f}%)")
print(f"\nMean abnormalities per knee: {gold['num_positives'].mean():.2f} (std={gold['num_positives'].std():.2f}, min={int(gold['num_positives'].min())}, max={int(gold['num_positives'].max())})")

# 3. Label correlation matrix
print("\n" + "=" * 90)
print("3. PEARSON CORRELATION MATRIX BETWEEN 12 FINDINGS IN GOLD-58")
print("=" * 90)
corr = gold[LAB].corr()
print(corr.round(2).to_string())

# 4. Check DICOM & Series metadata for Gold-58
series_df = pd.read_csv(f"{ROOT}/ds/comp/train_series.csv")
gold_series = series_df[series_df["StudyInstanceUID"].isin(gold["StudyInstanceUID"])]
print("\n" + "=" * 90)
print("4. DICOM & SCANNER METADATA FOR GOLD-58")
print("=" * 90)
print(f"Total series for Gold-58 studies: {len(gold_series)} (average {len(gold_series)/58:.2f} series/study)")
print("\nAnatomical Planes in Gold-58 series:")
print(gold_series["Anatomical_Plane"].value_counts().to_string())
print("\nFluid Sensitivity in Gold-58 series:")
print(gold_series["Fluid_Sensitive"].value_counts().to_string())

# Check site / folds distribution for Gold-58 if available
folds_path = f"{ROOT}/ds/folds_site.json"
if os.path.exists(folds_path):
    with open(folds_path) as f:
        site_folds = json.load(f)
    print("\n" + "=" * 90)
    print("5. FOLD DISTRIBUTION OF GOLD-58 IN SITE-GROUPED 5-FOLD CV")
    print("=" * 90)
    fold_dist = {}
    for fold_idx, (tr_ids, val_ids) in enumerate(site_folds if isinstance(site_folds, list) else site_folds.items()):
        val_set = set(val_ids)
        g_in_val = len(set(gold["StudyInstanceUID"]).intersection(val_set))
        fold_dist[f"Fold {fold_idx}"] = g_in_val
    print("Gold-58 studies per validation fold:", fold_dist)

# 6. Evaluate all available saved models on Gold-58
print("\n" + "=" * 90)
print("6. PERFORMANCE COMPARISON OF ALL TRAINED MODELS ON GOLD-58")
print("=" * 90)
y_true = gold[LAB].values.astype(float)
npz_files = [f for f in os.listdir(ROOT) if f.startswith("raptor_gold_") and f.endswith(".npz")]

results = {}
for nf in sorted(npz_files):
    tag = nf.replace("raptor_gold_", "").replace(".npz", "")
    try:
        data = np.load(os.path.join(ROOT, nf), allow_pickle=True)
        keys = list(data.files)
        p_key = "preds" if "preds" in keys else ("pred" if "pred" in keys else keys[0])
        p = data[p_key]
        if p.shape == (58, 12):
            aucs = [roc_auc_score(y_true[:, c], p[:, c]) for c in range(12)]
            results[tag] = {
                "Macro_AUC": np.mean(aucs),
                "aucs": aucs,
                "preds": p
            }
    except Exception as e:
        pass

df_res = pd.DataFrame({
    tag: [round(results[tag]["Macro_AUC"], 4)] + [round(results[tag]["aucs"][c], 4) for c in range(12)]
    for tag in results
}, index=["MACRO_AUC"] + LAB)

pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 160)
print(df_res.to_string())

# 7. Correlation among top models on Gold-58
print("\n" + "=" * 90)
print("7. PREDICTION CORRELATION (INTER-MODEL DIVERSITY) ON GOLD-58")
print("=" * 90)
top_models = sorted(results.keys(), key=lambda k: results[k]["Macro_AUC"], reverse=True)[:6]
print("Top 6 models by Gold Macro AUC:", top_models)

matrix = np.zeros((len(top_models), len(top_models)))
for i, m1 in enumerate(top_models):
    for j, m2 in enumerate(top_models):
        corr_list = [pearsonr(results[m1]["preds"][:, c], results[m2]["preds"][:, c])[0] for c in range(12)]
        matrix[i, j] = np.mean(corr_list)

df_corr = pd.DataFrame(matrix, index=top_models, columns=top_models)
print(df_corr.round(4).to_string())

print("\n" + "=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
