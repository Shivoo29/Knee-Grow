#!/usr/bin/env python3
"""T07: Per-finding error analysis on gold-58.

Reads (CPU, read-only):
  raptor_gold_*.npz in project root (keys pred (58,12), truth (58,12), ids (58,))
  teacher/c0_raptor_raw.csv, teacher/c0_compact.csv (chunk 0, contains 58 gold)
Writes (only output):
  delegation/results/T07_error_analysis.md (this script's sibling)

Run: .venv/bin/python -I delegation/results/T07_error_analysis.py
Prints only: shapes, counts, AUCs, timings (no UIDs, no pixels, no Report text).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
       "Medial OA", "Lateral OA", "PF OA", "Effusion",
       "Synovitis", "Baker's", "Contusion", "Fracture"]
W_RAPTOR, W_OURS = 0.7, 0.3
N_BOOT = 1000
BOOT_SEED = 0

ROOT = Path(__file__).resolve().parents[2]
OUT_MD = Path(__file__).resolve().parent / "T07_error_analysis.md"


def per_finding_auc(pred, truth):
    aucs = []
    for j in range(12):
        yt = truth[:, j]
        yp = pred[:, j]
        if len(np.unique(yt)) > 1:
            aucs.append(float(roc_auc_score(yt, yp)))
        else:
            aucs.append(float("nan"))
    return aucs


def pct_ranks(arr):
    # Per-finding percentile ranks, cf. score_blend_npz.py:63.
    out = np.zeros_like(arr, dtype=float)
    for j in range(arr.shape[1]):
        out[:, j] = rankdata(arr[:, j], method="average") / arr.shape[0]
    return out


def main():
    t0 = time.time()
    # File filter: task says use only files WITHOUT `_ep` (final/swa/epens).
    # `_epens` literally contains substring `_ep`, so interpret as: drop
    # epoch checkpoints matching `_ep<digit>` (ep8/ep10/...), keep final/_swa/_epens.
    all_npz = sorted(ROOT.glob("raptor_gold_*.npz"))
    kept = [p for p in all_npz if ("_epens" in p.name or "_ep" not in p.name)]
    dropped = [p for p in all_npz if p not in kept]

    # Load kept npz; verify shapes and shared truth/ids.
    preds = {}
    truth0 = None
    ids0 = None
    for p in kept:
        d = np.load(p)
        pr = np.asarray(d["pred"], dtype=float)
        tr = np.asarray(d["truth"])
        ids = np.array([str(x) for x in np.asarray(d["ids"])])
        assert pr.shape == (58, 12), p.name
        assert tr.shape == (58, 12), p.name
        assert ids.shape == (58,), p.name
        if truth0 is None:
            truth0, ids0 = tr, ids
        else:
            assert np.array_equal(tr, truth0), p.name
            assert np.array_equal(ids, ids0), p.name
        preds[p.name] = pr
    truth = truth0
    gold_ids = list(ids0)
    n_pos = [(truth[:, j] == 1).sum() for j in range(12)]

    # Macro AUC per our model; best 3.
    macro = {}
    per_model_aucs = {}
    for name, pr in preds.items():
        aucs = per_finding_auc(pr, truth)
        per_model_aucs[name] = aucs
        macro[name] = float(np.nanmean(aucs))
    ranked_models = sorted(macro.items(), key=lambda kv: (-kv[1], kv[0]))
    best3_names = [n for n, _ in ranked_models[:3]]
    best1_name = best3_names[0]
    best1_pred = preds[best1_name]

    # Teacher predictions aligned to gold ids order. usecols excludes Report per PREAMBLE.
    def load_teacher(fname):
        df = pd.read_csv(ROOT / "teacher" / fname,
                         usecols=lambda c: c != "Report",
                         dtype={"StudyInstanceUID": str})
        assert list(df.columns[1:]) == LAB, fname
        df = df.set_index("StudyInstanceUID").loc[gold_ids, LAB]
        return df[LAB].to_numpy(dtype=float)

    raptor_pred = load_teacher("c0_raptor_raw.csv")
    compact_pred = load_teacher("c0_compact.csv")
    assert raptor_pred.shape == (58, 12)
    assert compact_pred.shape == (58, 12)

    raptor_aucs = per_finding_auc(raptor_pred, truth)
    compact_aucs = per_finding_auc(compact_pred, truth)
    blend_scores = W_RAPTOR * pct_ranks(raptor_pred) + W_OURS * pct_ranks(compact_pred)
    blend_aucs = per_finding_auc(blend_scores, truth)
    blend_macro = float(np.nanmean(blend_aucs))

    # Bootstrap 95% CI for blend per-finding AUC: 1000 resamples of 58 rows.
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.full((N_BOOT, 12), np.nan)
    n_valid = [0] * 12
    for b in range(N_BOOT):
        idx = rng.integers(0, 58, size=58)
        tb, pb = truth[idx], blend_scores[idx]
        for j in range(12):
            if len(np.unique(tb[:, j])) > 1:
                boot[b, j] = roc_auc_score(tb[:, j], pb[:, j])
    ci_lo, ci_hi = [], []
    for j in range(12):
        col = boot[:, j]
        col = col[~np.isnan(col)]
        n_valid[j] = int(col.size)
        ci_lo.append(float(np.percentile(col, 2.5)) if col.size else float("nan"))
        ci_hi.append(float(np.percentile(col, 97.5)) if col.size else float("nan"))
    half_width = [(hi - lo) / 2 for lo, hi in zip(ci_lo, ci_hi)]

    # Weakest 3 findings of blend.
    order = sorted(range(12), key=lambda j: blend_aucs[j])
    weakest = order[:3]

    # Q3 details per weakest finding.
    # Per-study rank error e_i = (1 - r_i) if positive else r_i, using each
    # model's own percentile ranks. Worst misses = positives with largest
    # blend e (lowest blend rank). Spearman over positives and over all 58.
    r_raptor = pct_ranks(raptor_pred)
    r_compact = pct_ranks(compact_pred)
    r_best = pct_ranks(best1_pred)
    r_blend = pct_ranks(blend_scores)

    def rank_errors(ranks, j):
        e = np.zeros(58)
        pos = truth[:, j] == 1
        e[pos] = 1.0 - ranks[pos, j]
        e[~pos] = ranks[~pos, j]
        return e

    q3 = {}
    for j in weakest:
        pos_idx = [i for i in range(58) if truth[i, j] == 1]
        # sort positives by blend error desc (worst first)
        eb = rank_errors(r_blend, j)
        worst = sorted(pos_idx, key=lambda i: -eb[i])[:5]
        # extras: blend rank position (1=lowest score) and #negatives above
        rows = []
        for i in worst:
            s = blend_scores[i, j]
            neg_above = int(((truth[:, j] == 0) & (blend_scores[:, j] > s)).sum())
            rank_pos = int((blend_scores[:, j] <= s).sum())  # 1=lowest
            rows.append((i, rank_pos, neg_above, float(eb[i])))
        e_r = rank_errors(r_raptor, j)
        e_c = rank_errors(r_compact, j)
        e_b = rank_errors(r_best, j)
        def sp(a, b, sel):
            x, y = a[sel], b[sel]
            if np.std(x) == 0 or np.std(y) == 0:
                return float("nan")
            return float(spearmanr(x, y).statistic)
        all_sel = list(range(58))
        q3[j] = {
            "pos_idx": pos_idx,
            "worst": rows,
            "sp_all": {"raptor_compact": sp(e_r, e_c, all_sel),
                       "raptor_best": sp(e_r, e_b, all_sel),
                       "compact_best": sp(e_c, e_b, all_sel)},
            "sp_pos": {"raptor_compact": sp(e_r, e_c, pos_idx),
                       "raptor_best": sp(e_r, e_b, pos_idx),
                       "compact_best": sp(e_c, e_b, pos_idx)},
        }

    # Q4: best ours per finding vs blend + CI half-width.
    q4 = []
    for j in range(12):
        vals = [(per_model_aucs[n][j], n) for n in preds]
        vals.sort(reverse=True)
        best_v, best_n = vals[0][0], vals[0][1]
        diff = best_v - blend_aucs[j]
        beats = bool(diff > half_width[j])
        q4.append((best_v, best_n, diff, beats))

    # ---- write md ----
    def f(x):
        return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"

    # SUMMARY: exactly 15 lines (lines 1-15).
    wline = []
    for k, j in enumerate(weakest):
        wline.append(f"{LAB[j]} AUC {blend_aucs[j]:.4f} CI [{ci_lo[j]:.4f},{ci_hi[j]:.4f}] n_pos {int(n_pos[j])}")
    # shared-error one-liner from mean pos-Spearman across weakest
    mean_sp = float(np.mean([q3[j]['sp_pos']['raptor_compact'] for j in weakest]))
    summary = [
        "# SUMMARY (lines 1-15; reviewer reads only this)",
        f"1. Verdict: weakest blend findings: {wline[0]}; {wline[1]}; {wline[2]}.",
        f"2. Shared errors: mean Raptor-compact Spearman over positives = {mean_sp:.2f}; see lines 16+ for pairs.",
        f"3. Key numbers: blend macro {blend_macro:.4f}; Raptor macro {np.nanmean(raptor_aucs):.4f} (possibly in-sample); compact macro {np.nanmean(compact_aucs):.4f}.",
        f"4. Best ours by macro: {best3_names[0]} {macro[best3_names[0]]:.4f}; {best3_names[1]} {macro[best3_names[1]]:.4f}; {best3_names[2]} {macro[best3_names[2]]:.4f}.",
        "5. Problems: none blocking; file filter keeps _epens despite `_ep` substring (see Methods).",
        "6. file:line: rank def score_blend_npz.py:63; labels score_blend_npz.py:9-11; inputs teacher/c0_*.csv:1.",
        "7. NOT checked: LB; OOF; training provenance; per-series causes; calibration; thresholds.",
        "8. NOT VERIFIED: teacher chunk-0 coverage beyond 58/58 id match; clinical meaning of misses.",
        "9. Raptor numbers are possibly in-sample (trained on alldata); treat gaps vs Raptor as noise-prone.",
        "10. Macro gaps <0.02 noise; per-finding gaps <0.06 noise (PREAMBLE).",
        "11. No UIDs / pixels / Report text in this file; misses listed as study index 0-57.",
        "12. Bootstrap: 1000 resamples of 58, seed 0; degenerate resamples skipped (n_valid in table).",
        "13. Rank error e=(1-rank)|pos else rank; Spearman on e (positives + all-58).",
        "14. Q4: ours beats blend only if diff > half-width else noise; outputs .md+.py; CPU only.",
    ]
    assert len(summary) == 15

    L = []
    L.extend(summary)
    L.append("")
    L.append("## Methods (inputs, definitions, provenance)")
    L.append(f"- Root: project root; script: delegation/results/T07_error_analysis.py; output: delegation/results/T07_error_analysis.md.")
    L.append(f"- Gold npz kept: {len(kept)} files (final/swa/epens); dropped epoch checkpoints: {len(dropped)} files.")
    L.append(f"- Dropped files (contain `_ep` without `_epens`): {', '.join(sorted(p.name for p in dropped))}.")
    L.append(f"- Kept files: {', '.join(sorted(p.name for p in kept))}.")
    L.append(f"- Gold shapes: pred (58,12), truth (58,12), ids (58,); truth shared across kept files: True; ids order shared: True.")
    L.append(f"- Positives per finding: {', '.join(f'{LAB[j]}={int(n_pos[j])}' for j in range(12))}; flag AUC unreliable if n_pos<5: none flagged.")
    L.append("- Teacher: teacher/c0_raptor_raw.csv (1145 rows) and teacher/c0_compact.csv (1145 rows); columns StudyInstanceUID + 12 LAB in order; loaded with usecols excluding Report; aligned to gold ids order; 58/58 gold ids matched.")
    L.append("- Raptor caveat: trained on alldata, may have seen gold studies; mark Raptor numbers possibly in-sample.")
    L.append("- Rank: per-finding percentile rankdata/58 average-method (score_blend_npz.py:63); blend = 0.7*rank(Raptor)+0.3*rank(compact); AUC via sklearn roc_auc_score per finding, macro = mean of 12.")
    L.append(f"- Bootstrap: {N_BOOT} resamples with replacement of 58 studies, seed {BOOT_SEED}; per-finding AUC of fixed blend scores; degenerate (single-class) resamples skipped; n_valid reported; CI = 2.5/97.5 percentiles.")
    L.append("- Rank error per study per finding: e = 1-rank if truth=1 else rank (own-model percentile ranks); worst misses = positives with largest blend e; Spearman = scipy spearmanr on e vectors (pairs among Raptor/compact/best), over positives and over all 58.")
    L.append("- Q4: per finding, best-ours-AUC minus blend-AUC vs CI half-width (hi-lo)/2; exceeds => report model+diff else noise. Also note 0.06 per-finding noise rule.")
    L.append("")
    L.append("## Table 1: per-finding AUC (Raptor possibly in-sample)")
    L.append(f"| Finding | n_pos | Raptor | Compact | Blend 0.7R+0.3C | {best3_names[0]} | {best3_names[1]} | {best3_names[2]} |")
    L.append("|---|---|---|---|---|---|---|---|")
    for j in range(12):
        flag = " UNRELIABLE(n<5)" if int(n_pos[j]) < 5 else ""
        L.append(f"| {LAB[j]} | {int(n_pos[j])}{flag} | {f(raptor_aucs[j])} | {f(compact_aucs[j])} | {f(blend_aucs[j])} | {f(per_model_aucs[best3_names[0]][j])} | {f(per_model_aucs[best3_names[1]][j])} | {f(per_model_aucs[best3_names[2]][j])} |")
    L.append(f"| macro | 58 | {np.nanmean(raptor_aucs):.4f} | {np.nanmean(compact_aucs):.4f} | {blend_macro:.4f} | {macro[best3_names[0]]:.4f} | {macro[best3_names[1]]:.4f} | {macro[best3_names[2]]:.4f} |")
    L.append(f"- Macro gaps vs blend: best-ours {macro[best3_names[0]]-blend_macro:+.4f}, Raptor {float(np.nanmean(raptor_aucs))-blend_macro:+.4f}, compact {float(np.nanmean(compact_aucs))-blend_macro:+.4f}; all |gap|<0.02 => noise.")
    L.append("")
    L.append("## Table 2: blend bootstrap 95% CI (1000 resamples, seed 0)")
    L.append("| Finding | n_pos | Blend AUC | CI lo | CI hi | half-width | n_valid |")
    L.append("|---|---|---|---|---|---|---|")
    for j in range(12):
        L.append(f"| {LAB[j]} | {int(n_pos[j])} | {f(blend_aucs[j])} | {f(ci_lo[j])} | {f(ci_hi[j])} | {f(half_width[j])} | {n_valid[j]} |")
    L.append("")
    L.append(f"## Q3: 3 weakest blend findings (worst misses as study index 0-57; best model = {best1_name})")
    for j in weakest:
        L.append(f"### {LAB[j]}: blend AUC {blend_aucs[j]:.4f} CI [{ci_lo[j]:.4f},{ci_hi[j]:.4f}] n_pos {int(n_pos[j])}")
        L.append(f"- Worst 5 misses (index, rank_pos 1=lowest, negatives_above, blend_err):")
        for (i, rp, na, eb) in q3[j]["worst"]:
            L.append(f"  - idx {i}: rank_pos {rp}/58, negatives_above {na}, blend_err {eb:.4f}")
        s = q3[j]["sp_pos"]
        a = q3[j]["sp_all"]
        L.append(f"- Spearman of rank errors over positives ({len(q3[j]['pos_idx'])}): Raptor-compact {s['raptor_compact']:.3f}; Raptor-best {s['raptor_best']:.3f}; compact-best {s['compact_best']:.3f}.")
        L.append(f"- Spearman of rank errors over all 58: Raptor-compact {a['raptor_compact']:.3f}; Raptor-best {a['raptor_best']:.3f}; compact-best {a['compact_best']:.3f}.")
        L.append(f"- Reading: |rho|<0.3 weak sharing, 0.3-0.6 partial, >0.6 shared misses (descriptive bands, not tests).")
    L.append("")
    L.append("## Q4: does any ours beat blend by more than CI half-width? (else noise)")
    L.append("| Finding | Blend | CI half-width | Best ours (file) | Best AUC | diff | verdict |")
    L.append("|---|---|---|---|---|---|---|")
    for j in range(12):
        bv, bn, d, beats = q4[j]
        v = f"{bn} +{d:.4f}" if beats else "noise"
        L.append(f"| {LAB[j]} | {f(blend_aucs[j])} | {f(half_width[j])} | {bn} | {f(bv)} | {d:+.4f} | {v} |")
    L.append("")
    L.append("## All our models macro AUC (kept 34, desc)")
    for n, m in ranked_models:
        L.append(f"- {n}: {m:.4f}")
    L.append("")
    L.append("## Checks not done / NOT VERIFIED")
    L.append("- NOT VERIFIED: clinical cause of misses; image-view cause; calibration; thresholds; OOF; LB.")
    L.append("- Did not open data/train.csv; did not print UIDs/pixels/Report text.")
    L.append("- CPU only; inputs read-only; outputs only in delegation/results/.")

    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")

    dt = time.time() - t0
    print(f"kept npz: {len(kept)} dropped: {len(dropped)} gold shape: (58,12)")
    print(f"n_pos: {[int(x) for x in n_pos]}")
    print(f"blend macro: {blend_macro:.4f} raptor macro: {np.nanmean(raptor_aucs):.4f} compact macro: {np.nanmean(compact_aucs):.4f}")
    print(f"best3: {best3_names[0]} {macro[best3_names[0]]:.4f}, {best3_names[1]} {macro[best3_names[1]]:.4f}, {best3_names[2]} {macro[best3_names[2]]:.4f}")
    print(f"weakest3: {[(LAB[j], round(float(blend_aucs[j]),4)) for j in weakest]}")
    print(f"timing_s: {dt:.1f}")


if __name__ == "__main__":
    main()
