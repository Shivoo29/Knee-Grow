"""
judge_image.py - LOCKED judge for Loop 2 (imaging).

Metric: the competition's macro ROC-AUC across the 12 findings.
Findings with a single class in the evaluation slice are skipped and
reported, matching standard practice for small sets.

Gold-58 protocol from L2 onward: model SELECTION runs on silver
validation only; gold_check() is the rationed reality check and logs
every consultation to gold_uses.log. During L1 (skeleton phase) the
gold set is the sanctioned playground and no logging applies.
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
            "Medial OA", "Lateral OA", "PF OA", "Effusion",
            "Synovitis", "Baker's", "Contusion", "Fracture"]


def load_gold_labels(train_csv: str = "train.csv") -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    gold = df.dropna(subset=FINDINGS).copy()
    for f in FINDINGS:
        gold[f] = gold[f].astype(int)
    return gold[["StudyInstanceUID"] + FINDINGS].reset_index(drop=True)


def macro_auc(y_df: pd.DataFrame, p_df: pd.DataFrame) -> dict:
    """Both frames: StudyInstanceUID + 12 finding columns. p = probabilities."""
    m = y_df.merge(p_df, on="StudyInstanceUID", suffixes=("", "_p"))
    per, valid = {}, []
    for f in FINDINGS:
        y = m[f].to_numpy()
        p = m[f + "_p"].to_numpy(dtype=float)
        if len(np.unique(y)) < 2:
            per[f] = None
            continue
        a = float(roc_auc_score(y, p))
        per[f] = round(a, 4)
        valid.append(a)
    return {"n": int(len(m)),
            "n_findings_scored": len(valid),
            "macro_auc": round(float(np.mean(valid)), 4) if valid else 0.0,
            "per_finding": per}


def gold_check(p_df: pd.DataFrame, reason: str,
               train_csv: str = "train.csv") -> dict:
    """Rationed from L2 onward. Every call is logged with a reason."""
    with open("gold_uses.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{reason}\n")
    print(f"GOLD CONSULTED: {reason} (logged to gold_uses.log)")
    return macro_auc(load_gold_labels(train_csv), p_df)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = pd.DataFrame({"StudyInstanceUID": [str(i) for i in range(40)]})
    for f in FINDINGS:
        y[f] = rng.integers(0, 2, 40)
    perfect = y.copy()
    perfect[FINDINGS] = perfect[FINDINGS].astype(float)
    s1 = macro_auc(y, perfect)
    noise = y.copy()
    for f in FINDINGS:
        noise[f] = rng.random(40)
    s2 = macro_auc(y, noise)
    print("perfect:", s1["macro_auc"], "| random:", s2["macro_auc"])
    assert s1["macro_auc"] == 1.0 and 0.2 < s2["macro_auc"] < 0.8
    print("self-test passed")
