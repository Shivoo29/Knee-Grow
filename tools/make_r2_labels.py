#!/usr/bin/env python3
"""tools/make_r2_labels.py - round-2 soft labels: report labels mixed with our image models' out-of-fold predictions.

Self-training on competition data only (no external data): each family's 4-fold OOF probabilities are Platt-calibrated
per finding to the report labels, the families are averaged, and the result is mixed with the report labels:
    r2 = alpha * report + (1 - alpha) * image_oof
The same calibration applied to each family's gold-58 predictions (held out from every fold) scores the mix on gold
for a grid of alphas, so the label gain is measured before anything trains on it.

    uv run -q --with numpy --with pandas --with pyarrow --with scikit-learn python -I tools/make_r2_labels.py \
        --family cnxt_sev='pc192/raptor_{kind}_cnxt_sev_f{f}_swa*.npz' --family coat='raptor_{kind}_oof4_f{f}_swa*.npz' \
        --alpha 0.5 --out labels_r2.parquet
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def get_study_languages(train_csv_path, lang_cache="data/study_languages.csv"):
    if lang_cache and os.path.exists(lang_cache):
        return pd.read_csv(lang_cache).set_index("StudyInstanceUID")["Language"]
    import langid
    rep_df = pd.read_csv(train_csv_path, usecols=["StudyInstanceUID", "Report"]).set_index("StudyInstanceUID")
    # Rule 4.b: classify locally, never print report text
    raw_langs = pd.Series([langid.classify(str(r))[0] for r in rep_df["Report"]], index=rep_df.index)
    counts = raw_langs.value_counts()
    mapped = raw_langs.map(lambda l: l if counts[l] >= 60 else "other")
    if lang_cache:
        os.makedirs(os.path.dirname(lang_cache) or ".", exist_ok=True)
        pd.DataFrame({"StudyInstanceUID": mapped.index, "Language": mapped.values}).to_csv(lang_cache, index=False)
    return mapped


ap = argparse.ArgumentParser()
ap.add_argument("--base", default="pcseq/labels_sev_v1.parquet")
ap.add_argument("--gold_report", default="labels/labels_clef_sev.csv", help="report labels for the 58 gold studies")
ap.add_argument("--family", action="append", required=True, help="name=pattern with {kind} (oof|gold) and {f} (fold)")
ap.add_argument("--alpha", type=float, default=0.4)
ap.add_argument("--lang_alpha", action="store_true", help="enable language-aware alpha: alpha_cell = clip(alpha + k*d, 0.15, alpha)")
ap.add_argument("--lang_k", type=float, default=None, help="specific k factor; if None, evaluates k in (2, 4, 6) and picks best gold macro AUC")
ap.add_argument("--train_csv", default="data/train.csv", help="path to train.csv with StudyInstanceUID and Report")
ap.add_argument("--lang_cache", default="data/study_languages.csv", help="path to cached study languages")
ap.add_argument("--out", required=True)
a = ap.parse_args()

base = pd.read_parquet(a.base).set_index("StudyInstanceUID")
F = list(base.columns)
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def one(pattern):
    hits = sorted(glob.glob(pattern))
    assert len(hits) == 1, (pattern, hits)
    return np.load(hits[0])


oof_cal, gold_cal, gold_truth, gold_ids = [], [], None, None
for spec in a.family:
    name, pat = spec.split("=", 1)
    zs = [one(pat.format(kind="oof", f=f) .replace("_swa*", "_swa_fold%d*" % f)) for f in range(4)]
    oof = pd.DataFrame(np.concatenate([z["pred"] for z in zs]), index=np.concatenate([z["ids"] for z in zs]), columns=F)
    oof = oof.loc[base.index]  # every labelled study has exactly one OOF prediction
    gz = [np.load(p) for f in range(4) for p in sorted(glob.glob(pat.format(kind="gold", f=f)))[:1]]  # a fold may lack gold preds
    assert gz, name + ": no gold predictions"
    gold = np.mean([z["pred"] for z in gz], axis=0)
    gold_truth, gold_ids = gz[0]["truth"], gz[0]["ids"]
    co, cg = np.zeros(oof.shape), np.zeros(gold.shape)
    for j, c in enumerate(F):  # Platt per finding against the soft report label (weights = soft targets)
        x, y = logit(oof[c].values)[:, None], base[c].values
        m = LogisticRegression(C=1e6).fit(np.r_[x, x], np.r_[np.ones(len(y)), np.zeros(len(y))], sample_weight=np.r_[y, 1 - y])
        co[:, j] = m.predict_proba(x)[:, 1]
        cg[:, j] = m.predict_proba(logit(gold[:, j])[:, None])[:, 1]
    oof_cal.append(co)
    gold_cal.append(cg)
    print(f"{name}: OOF macro AUC vs report labels "
          f"{np.mean([roc_auc_score(base[c] > 0.5, oof[c]) for c in F]):.4f} | gold {np.mean([roc_auc_score(gold_truth[:, j], gold[:, j]) for j in range(12)]):.4f}")

img = np.mean(oof_cal, axis=0)
img_gold = np.mean(gold_cal, axis=0)
rep_gold = pd.read_csv(a.gold_report).set_index("StudyInstanceUID").loc[gold_ids, F].values
gauc = lambda p: np.mean([roc_auc_score(gold_truth[:, j], p[:, j]) for j in range(12)])
print("gold-58 macro AUC of the label mix (report weight alpha):")
for al in (1.0, 0.7, 0.5, 0.4, 0.3, 0.0):
    print(f"  alpha {al:.1f}: {gauc(al * rep_gold + (1 - al) * img_gold):.4f}")
per = [(c, roc_auc_score(gold_truth[:, j], rep_gold[:, j]), roc_auc_score(gold_truth[:, j], (a.alpha * rep_gold + (1 - a.alpha) * img_gold)[:, j]))
       for j, c in enumerate(F)]
print("per finding, report -> mix at alpha %.1f:" % a.alpha, ", ".join(f"{c} {r:.3f}->{m:.3f}" for c, r, m in per))

if not a.lang_alpha:
    r2 = a.alpha * base.values + (1 - a.alpha) * img
else:
    study_lang = get_study_languages(a.train_csv, a.lang_cache)
    base_lang = [study_lang.get(u, "other") for u in base.index]
    gold_lang = [study_lang.get(u, "other") for u in gold_ids]
    unique_langs = sorted(list(set(base_lang)))

    base_aucs = {c: roc_auc_score(base[c].values > 0.5, img[:, j]) for j, c in enumerate(F)}
    plain_gold_macro = gauc(a.alpha * rep_gold + (1 - a.alpha) * img_gold)
    plain_per_gold = [roc_auc_score(gold_truth[:, j], (a.alpha * rep_gold + (1 - a.alpha) * img_gold)[:, j]) for j in range(12)]

    k_list = [a.lang_k] if a.lang_k is not None else [2.0, 4.0, 6.0]
    best_k = None
    best_gold = -1.0
    best_alpha_tab = None

    print("\n" + "=" * 65)
    print("LANGUAGE-AWARE ALPHA GRID EVALUATION ON GOLD-58")
    print(f"Baseline plain alpha={a.alpha:.1f} gold macro-AUC: {plain_gold_macro:.4f}")
    print("=" * 65)

    for k in k_list:
        alpha_tab = {}
        num_changed = 0
        total_cells = len(unique_langs) * len(F)
        for l in unique_langs:
            mask_l = np.array([bl == l for bl in base_lang])
            row_alphas = []
            for j, c in enumerate(F):
                y_cell = base[c].values[mask_l] > 0.5
                pred_cell = img[mask_l, j]
                if len(np.unique(y_cell)) > 1:
                    cell_auc = roc_auc_score(y_cell, pred_cell)
                    d = cell_auc - base_aucs[c]
                else:
                    d = 0.0
                al = float(np.clip(a.alpha + k * d, 0.15, a.alpha))
                if al < a.alpha:
                    num_changed += 1
                row_alphas.append(al)
            alpha_tab[l] = row_alphas

        alpha_gold_mat = np.array([alpha_tab.get(l, [a.alpha] * len(F)) for l in gold_lang])
        mix_gold = alpha_gold_mat * rep_gold + (1.0 - alpha_gold_mat) * img_gold
        gold_k_macro = gauc(mix_gold)
        per_k = [roc_auc_score(gold_truth[:, j], mix_gold[:, j]) for j in range(12)]

        print(f"\n--- k = {k:.1f} (changed {num_changed}/{total_cells} cells) ---")
        print(f"Gold macro-AUC: {gold_k_macro:.4f} (diff vs plain: {gold_k_macro - plain_gold_macro:+.4f})")
        print("Per-finding: " + " ".join(f"{c[:5]}:{v:.3f}({v-pv:+.3f})" for c, v, pv in zip(F, per_k, plain_per_gold)))

        if gold_k_macro > best_gold:
            best_gold = gold_k_macro
            best_k = k
            best_alpha_tab = alpha_tab

    print("\n" + "=" * 65)
    print(f"SELECTED BEST k = {best_k:.1f} (Gold-58 macro: {best_gold:.4f})")
    print("=" * 65)
    print("\n--- Alpha Table (numbers and language codes only) ---")
    alpha_df = pd.DataFrame(
        [[round(v, 3) for v in best_alpha_tab[l]] for l in unique_langs],
        index=unique_langs, columns=F
    )
    print(alpha_df.to_string())
    print("=" * 65 + "\n")

    alpha_base_mat = np.array([best_alpha_tab.get(l, [a.alpha] * len(F)) for l in base_lang])
    r2 = alpha_base_mat * base.values + (1.0 - alpha_base_mat) * img

assert np.isfinite(r2).all() and r2.min() >= 0 and r2.max() <= 1 and r2.shape == base.shape
pd.DataFrame(r2.astype(np.float32), index=base.index, columns=F).reset_index().to_parquet(a.out, index=False)
print("wrote", a.out, r2.shape, "mean", np.round(r2.mean(0), 3).tolist())
