#!/usr/bin/env python3
"""tools/build_label_weights.py - per-(study, finding) weights for noisy-label down-weighting.

    teacher = 0.7 * Raptor + 0.3 * compact     (same recipe as build_targets.py)
    weight  = 0.0 where |teacher - labels_avg| > thr, else 1.0

    .venv/bin/python tools/build_label_weights.py --thr 0.7 --out label_weights_v1.parquet

Reads teacher/c{0,1,2,3}_raptor_raw.csv + teacher/c{0..3}_compact.csv (dedupe on
StudyInstanceUID, same as build_targets.py) and labels_avg.parquet. Gold
studies are not in labels_avg, so they never get a weight row. Writes columns
StudyInstanceUID, w_<finding> x12. Prints per-finding fraction set to 0
(counts only). CSVs are loaded with usecols excluding Report.
"""
import argparse
import os

import numpy as np
import pandas as pd

LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
       'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def _load_kind(teacher_dir, kind):
    parts = [pd.read_csv(os.path.join(teacher_dir, f'c{k}_{kind}.csv'),
                         usecols=lambda c: c != 'Report',
                         dtype={'StudyInstanceUID': str}) for k in range(4)]
    return pd.concat(parts).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')[LAB]


def build_label_weights(teacher_dir, labels_path, thr, out_path):
    """Returns the weights DataFrame (StudyInstanceUID + w_<finding> cols)."""
    labels = pd.read_parquet(labels_path).set_index('StudyInstanceUID')[LAB]
    teacher = 0.7 * _load_kind(teacher_dir, 'raptor_raw') + 0.3 * _load_kind(teacher_dir, 'compact')
    missing = labels.index.difference(teacher.index)
    assert len(missing) == 0, f'{len(missing)} label studies have no teacher prediction'
    teacher = teacher.loc[labels.index]
    assert teacher.notna().all().all() and ((teacher >= 0) & (teacher <= 1)).all().all()
    w = pd.DataFrame({f'w_{c}': (np.abs(teacher[c].values - labels[c].values) <= thr).astype(np.float32)
                      for c in LAB}, index=labels.index)
    w.reset_index().to_parquet(out_path, index=False)
    return w


def main():
    ap = argparse.ArgumentParser(description="Build per-(study,finding) label weights")
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--out", default="label_weights_v1.parquet")
    ap.add_argument("--teacher_dir", default="teacher")
    ap.add_argument("--labels", default="labels_avg.parquet")
    a = ap.parse_args()
    w = build_label_weights(a.teacher_dir, a.labels, a.thr, a.out)
    n = len(w)
    parts = []
    for c in LAB:
        col = w["w_" + c]
        parts.append(f"{c[:5]} {float((col == 0).mean()):.3f}({int((col == 0).sum())})")
    print(f"rows {n} thr {a.thr} -> {os.path.basename(a.out)}", " ".join(parts), flush=True)


if __name__ == "__main__":
    main()
