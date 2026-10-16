#!/usr/bin/env python3
"""Hybrid training labels: a base table with chosen findings taken from another table (prints counts only).

Why MCL: CoAtNet-384 gold-58 MCL AUC is 0.872 trained on labels_avg vs 0.964 on labels_llm_soft (0.954 distilled);
labels_avg marks 14.7 % MCL-positive vs 4.9 % (host counts only acute high-grade tears). Weights of overridden
findings are reset to 1: the 31-table ambiguity mask describes the replaced column, not the new one.

    .venv/bin/python -I tools/build_hybrid_labels.py --base labels_avg.parquet \
        --override data/labels/labels_llm_soft.parquet --findings MCL \
        --weights label_weights_consensus_v1.parquet --out_weights label_weights_hybrid_v1.parquet \
        --out labels_hybrid_v1.parquet
"""
import argparse

import pandas as pd


def build(base, override, findings, weights=None):
    b = base.set_index("StudyInstanceUID")
    o = override.set_index("StudyInstanceUID")
    missing = b.index.difference(o.index)
    if len(missing):
        raise SystemExit(f"override table lacks {len(missing)} base studies")
    for c in findings:
        b[c] = o.loc[b.index, c].astype("float32").values
    w = None
    if weights is not None:
        w = weights.set_index("StudyInstanceUID").copy()
        for c in findings:
            w["w_" + c] = 1.0
        w = w.reset_index()
    return b.reset_index(), w


def soften_synovitis(labels, weights, w_neg=0.3):
    """Reports rarely address synovitis (forum: 83.7 % 'not addressed'), so a synovitis negative next to a
    positive effusion is often a silent positive. Down-weight those negatives to w_neg (never raise a weight)."""
    lab = labels.set_index("StudyInstanceUID")
    w = weights.set_index("StudyInstanceUID").copy()
    silent = (lab.loc[w.index, "Synovitis"] < 0.5) & (lab.loc[w.index, "Effusion"] >= 0.5)
    w.loc[silent, "w_Synovitis"] = w.loc[silent, "w_Synovitis"].clip(upper=w_neg)
    return w.reset_index(), int(silent.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--override", required=True)
    ap.add_argument("--findings", required=True, help="comma list, e.g. MCL")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out_weights", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--syn_neg_w", type=float, default=None, help="down-weight synovitis negatives with positive effusion to this weight")
    a = ap.parse_args()
    fs = [f for f in a.findings.split(",") if f]
    base = pd.read_parquet(a.base)
    out, w = build(base, pd.read_parquet(a.override), fs, pd.read_parquet(a.weights) if a.weights else None)
    if a.syn_neg_w is not None and w is not None:
        w, n = soften_synovitis(out, w, a.syn_neg_w)
        print(f"synovitis: {n} negatives with positive effusion down-weighted to {a.syn_neg_w}")
    out.to_parquet(a.out, index=False)
    for c in fs:
        print(f"{c}: positive share {float((base[c] >= 0.5).mean()):.3f} -> {float((out[c] >= 0.5).mean()):.3f}")
    if w is not None:
        assert a.out_weights, "--out_weights is required with --weights"
        w.to_parquet(a.out_weights, index=False)
        print(f"weights: zero share {float((w.filter(like='w_').values == 0).mean()):.3f} (overridden findings at 1)")


if __name__ == "__main__":
    main()
