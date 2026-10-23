#!/usr/bin/env python3
"""CPU-only Phase 0 audit for per-label teacher routing.

Each --teacher input must be an NPZ with aligned study-level OOF predictions,
truth, and IDs (default keys: pred, truth, ids). The predictions must come
from models that did not train on those studies. Site IDs are supplied via CSV.

Example:
  python routing_check.py \
    --teacher s0=pred_s0_oof.npz --teacher s3=pred_s3_oof.npz \
    --sites site_metadata.csv --out routing_phase0.md

Requires numpy, pandas, scipy, and scikit-learn. This script does not load
models, access a GPU, or alter any input files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score


DEFAULT_LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def parse_teacher(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("teacher must be NAME=PATH")
    name, path = spec.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("teacher must be NAME=PATH")
    return name, Path(path)


def load_npz(path: Path, pred_key: str, truth_key: str, id_key: str):
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in (pred_key, truth_key, id_key) if k not in z.files]
        if missing:
            raise ValueError(f"{path}: missing NPZ keys {missing}; found {z.files}")
        pred = np.asarray(z[pred_key], dtype=np.float64)
        truth = np.asarray(z[truth_key], dtype=np.float64)
        ids = np.asarray(z[id_key]).astype(str)
    if pred.ndim != 2 or truth.shape != pred.shape or ids.shape != (pred.shape[0],):
        raise ValueError(f"{path}: expected pred/truth (N,L), ids (N,); got "
                         f"{pred.shape}, {truth.shape}, {ids.shape}")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"{path}: duplicate study IDs")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError(f"{path}: predictions/truth contain NaN or infinity")
    if not np.isin(truth, [0.0, 1.0]).all():
        raise ValueError(f"{path}: truth must be binary 0/1")
    if np.any((pred < 0) | (pred > 1)):
        raise ValueError(f"{path}: pred values must be probabilities in [0,1]")
    return pred, truth, ids


def auc_vector(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    out = np.full(y.shape[1], np.nan, dtype=float)
    for j in range(y.shape[1]):
        if np.unique(y[:, j]).size == 2:
            out[j] = roc_auc_score(y[:, j], p[:, j])
    return out


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    a = auc_vector(y, p)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def rank_mean(pred: np.ndarray) -> np.ndarray:
    """Per-label mean percentile rank, robust to teacher calibration scales."""
    n, k, labels = pred.shape
    out = np.empty((n, labels), dtype=float)
    for j in range(labels):
        ranks = np.column_stack([
            rankdata(pred[:, t, j], method="average") / (n + 1.0)
            for t in range(k)
        ])
        out[:, j] = ranks.mean(axis=1)
    return out


def centered_teacher_auc(aucs: np.ndarray) -> np.ndarray:
    """Remove each teacher's overall AUC level before measuring label effects."""
    return aucs - np.nanmean(aucs, axis=1, keepdims=True)


def grouped_bootstrap_noise(y, pred, groups, observed_auc, n_boot, rng):
    uniq = np.unique(groups)
    errors = []
    # Sample site clusters with replacement; duplicate clusters duplicate rows.
    rows_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([rows_by_group[g] for g in sampled])
        boot = np.vstack([auc_vector(y[rows], pred[rows, t])
                          for t in range(pred.shape[1])])
        errors.append(centered_teacher_auc(boot - observed_auc))
    err = np.asarray(errors)
    return np.nanvar(err, axis=0, ddof=1)


def split_half_analysis(y, pred, groups, names, n_splits, rng):
    uniq = np.unique(groups)
    if len(uniq) < 4:
        raise ValueError(f"Need >=4 site groups for split-half checks; found {len(uniq)}")
    k, labels = pred.shape[1], pred.shape[2]
    diffs_route_best, diffs_route_random, diffs_route_ens = [], [], []
    wins_a, wins_b = [], []
    test_auc_records = []
    bootstrap_split = None
    for _ in range(n_splits):
        perm = rng.permutation(uniq)
        ga, gb = set(perm[:len(perm)//2]), set(perm[len(perm)//2:])
        if bootstrap_split is None:
            bootstrap_split = (tuple(ga), tuple(gb))
        ia = np.flatnonzero(np.isin(groups, list(ga)))
        ib = np.flatnonzero(np.isin(groups, list(gb)))
        if len(ia) == 0 or len(ib) == 0:
            continue
        aa = np.vstack([auc_vector(y[ia], pred[ia, t]) for t in range(k)])
        ab = np.vstack([auc_vector(y[ib], pred[ib, t]) for t in range(k)])
        wa = np.nanargmax(aa, axis=0)
        wb = np.nanargmax(ab, axis=0)
        wins_a.extend(wa.tolist())
        wins_b.extend(wb.tolist())
        # Select on A, score on B, and reverse. Route picks one teacher per label.
        for sel, ev, eval_rows in ((aa, ab, ib), (ab, aa, ia)):
            winner = np.nanargmax(sel, axis=0)
            routed = np.column_stack([pred[eval_rows, winner[j], j]
                                      for j in range(labels)])
            best_idx = int(np.nanargmax(np.nanmean(sel, axis=1)))
            best = pred[eval_rows, best_idx, :]
            random_idx = rng.integers(0, k, size=labels)
            random_routed = np.column_stack([pred[eval_rows, random_idx[j], j]
                                             for j in range(labels)])
            ens = rank_mean(pred[eval_rows])
            ar = macro_auc(y[eval_rows], routed)
            abest = macro_auc(y[eval_rows], best)
            arand = macro_auc(y[eval_rows], random_routed)
            aens = macro_auc(y[eval_rows], ens)
            diffs_route_best.append(ar - abest)
            diffs_route_random.append(ar - arand)
            diffs_route_ens.append(ar - aens)
            test_auc_records.append((ar, abest, arand, aens))
    if not diffs_route_best:
        raise ValueError("No valid site-grouped half splits")
    stability = float(np.mean(np.asarray(wins_a) == np.asarray(wins_b)))
    records = np.asarray(test_auc_records)
    # These intervals describe variation over repeated split directions; they are
    # not a substitute for a held-out multi-seed/site hierarchical bootstrap.
    summary = {}
    for name, values in (("route_minus_best", diffs_route_best),
                         ("route_minus_random", diffs_route_random),
                         ("route_minus_ensemble", diffs_route_ens)):
        summary[name] = {
            "mean": float(np.mean(values)),
            "split_2.5pct": float(np.quantile(values, .025)),
            "split_97.5pct": float(np.quantile(values, .975)),
        }
    summary["mean_macro_auc"] = {
        "routed": float(records[:, 0].mean()),
        "best_single_selected_on_other_half": float(records[:, 1].mean()),
        "random_routing": float(records[:, 2].mean()),
        "uniform_rank_ensemble": float(records[:, 3].mean()),
    }
    summary["winner_stability"] = stability
    summary["chance_stability"] = 1.0 / k
    # Hierarchical site bootstrap for a fixed site split: resample site clusters
    # independently on selection/evaluation sides, reselect routing winners, and
    # score on resampled held-out clusters. The two directions are paired.
    ga, gb = bootstrap_split
    by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    boot_delta = []
    for _ in range(500):
        direction_deltas = []
        for sel_groups, eval_groups in ((ga, gb), (gb, ga)):
            sg = rng.choice(sel_groups, size=len(sel_groups), replace=True)
            eg = rng.choice(eval_groups, size=len(eval_groups), replace=True)
            si = np.concatenate([by_group[g] for g in sg])
            ei = np.concatenate([by_group[g] for g in eg])
            sel_auc = np.vstack([auc_vector(y[si], pred[si, t]) for t in range(k)])
            if not np.isfinite(sel_auc).any(axis=0).all():
                continue
            winners = np.nanargmax(sel_auc, axis=0)
            chosen = int(np.nanargmax(np.nanmean(sel_auc, axis=1)))
            routed = np.column_stack([pred[ei, winners[j], j] for j in range(labels)])
            best = pred[ei, chosen, :]
            routed_score, best_score = macro_auc(y[ei], routed), macro_auc(y[ei], best)
            if np.isfinite(routed_score) and np.isfinite(best_score):
                direction_deltas.append(routed_score-best_score)
        if direction_deltas:
            boot_delta.append(float(np.mean(direction_deltas)))
    if boot_delta:
        summary["route_minus_best_group_bootstrap_ci"] = [
            float(np.quantile(boot_delta, .025)), float(np.quantile(boot_delta, .975))]
        summary["group_bootstrap_replicates"] = len(boot_delta)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", action="append", type=parse_teacher, required=True,
                    help="teacher name and NPZ path, e.g. s0=pred_s0_oof.npz; repeat")
    ap.add_argument("--sites", required=True, type=Path, help="CSV containing study ID and site ID")
    ap.add_argument("--id-col", default="StudyInstanceUID")
    ap.add_argument("--site-col", default="site_id")
    ap.add_argument("--pred-key", default="pred")
    ap.add_argument("--truth-key", default="truth")
    ap.add_argument("--npz-id-key", default="ids")
    ap.add_argument("--labels", default=None,
                    help="comma-separated label names; default RSNA 12-label order")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--out", type=Path, default=Path("routing_phase0.md"))
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    if len(args.teacher) < 2:
        ap.error("provide at least two teachers")
    names = [x[0] for x in args.teacher]
    if len(set(names)) != len(names):
        ap.error("teacher names must be unique")
    if args.bootstrap < 20 or args.splits < 20:
        ap.error("use at least 20 bootstrap replicates and 20 split repetitions")

    loaded = [load_npz(path, args.pred_key, args.truth_key, args.npz_id_key)
              for _, path in args.teacher]
    ref_pred, truth, ids = loaded[0]
    preds = [ref_pred]
    for (name, path), (p, y, this_ids) in zip(args.teacher[1:], loaded[1:]):
        if p.shape != ref_pred.shape:
            raise ValueError(f"{name}: prediction shape {p.shape} differs from {ref_pred.shape}")
        if not np.array_equal(this_ids, ids):
            raise ValueError(f"{name}: IDs/order differ from first teacher; align OOF files first")
        if not np.array_equal(y, truth):
            raise ValueError(f"{name}: truth differs from first teacher")
        preds.append(p)
    pred = np.stack(preds, axis=1)  # N x teachers x labels
    labels = args.labels.split(",") if args.labels else DEFAULT_LABELS[:pred.shape[2]]
    if len(labels) != pred.shape[2]:
        raise ValueError(f"got {pred.shape[2]} labels but {len(labels)} names")

    sites_df = pd.read_csv(args.sites, dtype={args.id_col: str})
    if args.id_col not in sites_df or args.site_col not in sites_df:
        raise ValueError(f"site CSV needs columns {args.id_col!r} and {args.site_col!r}")
    sites_df = sites_df[[args.id_col, args.site_col]].drop_duplicates(args.id_col)
    site_map = dict(zip(sites_df[args.id_col].astype(str), sites_df[args.site_col].astype(str)))
    missing = [sid for sid in ids if sid not in site_map or site_map[sid] in ("", "nan", "None")]
    if missing:
        raise ValueError(f"missing site IDs for {len(missing)} prediction studies; example IDs: {missing[:3]}")
    groups = np.asarray([site_map[sid] for sid in ids])
    if np.unique(groups).size < 4:
        raise ValueError(f"need at least 4 site groups; found {np.unique(groups).size}")

    per_label_pos = truth.sum(axis=0).astype(int)
    per_label_neg = (truth.shape[0] - per_label_pos).astype(int)
    bad = [labels[j] for j in range(len(labels)) if per_label_pos[j] == 0 or per_label_neg[j] == 0]
    if bad:
        raise ValueError(f"labels without both classes in full input: {bad}")

    rng = np.random.default_rng(args.seed)
    point = np.vstack([auc_vector(truth, pred[:, t]) for t in range(pred.shape[1])])
    centered = centered_teacher_auc(point)
    obs_var = np.nanvar(centered, axis=0, ddof=1)
    noise_var = grouped_bootstrap_noise(truth, pred, groups, point,
                                        args.bootstrap, rng)
    shrink = np.divide(np.maximum(obs_var-noise_var, 0), obs_var,
                       out=np.zeros_like(obs_var), where=obs_var > 0)

    pred_corr_by_label = []
    for j in range(pred.shape[2]):
        ranks = np.column_stack([
            rankdata(pred[:, t, j], method="average")
            for t in range(pred.shape[1])
        ])
        corr = np.corrcoef(ranks, rowvar=False)
        pred_corr_by_label.append(corr)
    offdiag = np.triu_indices(len(names), 1)
    mean_pred_corr = [float(np.nanmean(c[offdiag])) if len(names) > 1 else float("nan")
                      for c in pred_corr_by_label]
    split = split_half_analysis(truth, pred, groups, names, args.splits, rng)

    rows = []
    for j, label in enumerate(labels):
        winner = int(np.nanargmax(point[:, j]))
        rows.append({
            "label": label,
            "positive": int(per_label_pos[j]),
            "negative": int(per_label_neg[j]),
            "best_teacher_full_oof": names[winner],
            "best_auc_descriptive": float(point[winner, j]),
            "observed_centered_teacher_variance": float(obs_var[j]),
            "group_bootstrap_noise_variance": float(noise_var[j]),
            "shrink_factor": float(shrink[j]),
            "mean_teacher_prediction_spearman": mean_pred_corr[j],
        })

    results = {
        "n_studies": int(len(ids)), "n_sites": int(np.unique(groups).size),
        "teachers": names, "labels": labels, "split_half": split,
        "per_label": rows,
        "teacher_macro_auc": {names[i]: float(np.nanmean(point[i])) for i in range(len(names))},
        "teacher_label_auc": {names[i]: point[i].tolist() for i in range(len(names))},
        "warning": "Full-OOF best-teacher values are descriptive and selection-biased. "
                   "Split-half intervals describe split variability, not an independent "
                   "multi-seed/site hierarchical confidence interval.",
    }

    lines = ["# Phase 0: Per-label Teacher Routing Check", "",
             f"- Studies: {len(ids)}; site clusters: {np.unique(groups).size}",
             f"- Teachers: {', '.join(names)}", "- Compute: CPU only", "",
             "## Per-label metrics", "",
             "| Label | Pos | Neg | Best teacher (descriptive) | Best AUC | Centered observed var | Group bootstrap noise var | Shrink | Mean teacher prediction corr |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['label']} | {r['positive']} | {r['negative']} | {r['best_teacher_full_oof']} | "
                     f"{r['best_auc_descriptive']:.4f} | {r['observed_centered_teacher_variance']:.6g} | "
                     f"{r['group_bootstrap_noise_variance']:.6g} | {r['shrink_factor']:.3f} | "
                     f"{r['mean_teacher_prediction_spearman']:.3f} |")
    lines += ["", "## Site-grouped split-half routing", "",
              f"Split-half winner stability: **{split['winner_stability']:.3f}** "
              f"(chance = {split['chance_stability']:.3f}).", "",
              "Mean macro-AUC on opposite halves:", "",
              "| Method | Mean macro-AUC |", "|---|---:|"]
    for key, title in (("routed", "Per-label routing"),
                       ("best_single_selected_on_other_half", "Best single selected on selection half"),
                       ("random_routing", "Random per-label routing"),
                       ("uniform_rank_ensemble", "Uniform rank ensemble")):
        lines.append(f"| {title} | {split['mean_macro_auc'][key]:.4f} |")
    lines += ["", "Paired differences over repeated split directions (2.5–97.5% split quantiles; descriptive, not inferential CIs):", "",
              "| Difference | Mean | 2.5% | 97.5% |", "|---|---:|---:|---:|"]
    for key, title in (("route_minus_best", "Routing − best single"),
                       ("route_minus_random", "Routing − random routing"),
                       ("route_minus_ensemble", "Routing − ensemble")):
        d = split[key]
        lines.append(f"| {title} | {d['mean']:+.4f} | {d['split_2.5pct']:+.4f} | {d['split_97.5pct']:+.4f} |")
    if "route_minus_best_group_bootstrap_ci" in split:
        lo, hi = split["route_minus_best_group_bootstrap_ci"]
        lines += ["", f"Fixed-split hierarchical site bootstrap for routing − best single: "
                  f"**[{lo:+.4f}, {hi:+.4f}]** "
                  f"({split['group_bootstrap_replicates']} valid replicates; winners refit in each replicate)."]
    lines += ["", "## Interpretation guardrails", "",
              "The full-OOF per-label winners are selection-biased. Use split-half results for the routing signal. "
              "The displayed split quantiles are not a hierarchical confidence interval over student seeds and sites; "
              "this Phase 0 script does not estimate student-seed variance. Do not claim Phase 1 evidence from this report.", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    json_path = args.json_out or args.out.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} and {json_path}")
    print(f"N={len(ids)} studies, {np.unique(groups).size} sites, {len(names)} teachers")
    print(f"routing-best={split['route_minus_best']['mean']:+.4f}; "
          f"stability={split['winner_stability']:.3f} vs chance {1/len(names):.3f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
