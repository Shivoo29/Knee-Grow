#!/usr/bin/env python3
"""Task 9: Clef report grades -> soft labels (prints counts only, never text or ids).

label[finding] = sum over report grades of P(grade) * V[finding][grade], where V is the chance that the expert
image readers call the finding positive given that grade. V is fixed from the host rubric (discussion 733343:
borderline graded negative; OA needs >50 % loss over ~1 cm; effusion and Baker's moderate or large; MCL acute
high-grade only) and forum evidence (reports under-call effusion and fracture). It is never fitted on gold-58,
so gold-58 stays an honest check of severity labels against presence-only labels:
    .venv/bin/python -I tools/severity_labels.py --grades grades_clef.parquet \
        --out labels_clef_sev.parquet --any_out labels_clef_any.parquet
    .venv/bin/python -I tools/decorr.py --a labels_clef_any.parquet --b labels_clef_sev.parquet \
        --truth gold:ds/comp/train.csv
Findings that pass are spliced into the training labels with tools/build_hybrid_labels.py.
"""
import argparse

import pandas as pd

V = {
    'ACL': {'not_mentioned': 0.05, 'intact': 0.02, 'sprain_or_low_grade': 0.15, 'possible_tear': 0.4,
            'high_grade_partial': 0.85, 'complete_tear': 0.95, 'graft_intact': 0.1, 'graft_torn': 0.85},
    'MCL': {'not_mentioned': 0.02, 'intact': 0.02, 'low_grade_or_chronic': 0.1, 'moderate_partial': 0.4,
            'acute_high_grade_partial': 0.85, 'acute_complete_tear': 0.95, 'possible_tear': 0.3},
    'Medial Meniscus': {'not_mentioned': 0.05, 'normal': 0.03, 'degeneration_only': 0.1, 'possible_tear': 0.45,
                        'definite_tear': 0.92, 'morphologic_abnormality': 0.8, 'post_surgical': 0.5},
    'Lateral Meniscus': {'not_mentioned': 0.05, 'normal': 0.03, 'degeneration_only': 0.1, 'possible_tear': 0.45,
                         'definite_tear': 0.92, 'morphologic_abnormality': 0.8, 'post_surgical': 0.5},
    'Medial OA': {'not_mentioned': 0.05, 'normal': 0.02, 'low_grade': 0.12, 'high_grade_small': 0.35,
                  'high_grade_large': 0.9, 'unspecified': 0.4},
    'Lateral OA': {'not_mentioned': 0.05, 'normal': 0.02, 'low_grade': 0.12, 'high_grade_small': 0.35,
                   'high_grade_large': 0.9, 'unspecified': 0.4},
    'PF OA': {'not_mentioned': 0.05, 'normal': 0.02, 'low_grade': 0.12, 'high_grade_small': 0.35,
              'high_grade_large': 0.9, 'unspecified': 0.4},
    'Effusion': {'not_mentioned': 0.1, 'none_or_physiologic': 0.05, 'trace_or_small': 0.3, 'moderate': 0.85,
                 'large': 0.95, 'unspecified': 0.55},
    'Synovitis': {'not_mentioned': 0.15, 'none': 0.05, 'possible_or_mild': 0.45, 'definite': 0.9},
    "Baker's": {'not_mentioned': 0.05, 'none': 0.02, 'small': 0.3, 'moderate_or_large': 0.9, 'ruptured': 0.8,
                'unspecified': 0.55},
    'Contusion': {'not_mentioned': 0.1, 'none': 0.05, 'traumatic_contusion': 0.92, 'degenerative_edema': 0.5,
                  'other_edema': 0.6, 'edema_with_fracture': 0.5},
    'Fracture': {'not_mentioned': 0.08, 'none': 0.04, 'acute_fracture': 0.92, 'insufficiency_or_stress': 0.7,
                 'old_or_healed': 0.25, 'possible': 0.5},
}
NORMAL = {'not_mentioned', 'intact', 'normal', 'none', 'none_or_physiologic'}
PRESENCE = {f: {o: float(o not in NORMAL) for o in opts} for f, opts in V.items()}


def to_labels(grades, table):
    """Expected value of the table over each finding's grade probabilities; refuses option-set mismatches."""
    out = pd.DataFrame({"StudyInstanceUID": grades["StudyInstanceUID"].astype(str).values})
    for f, vals in table.items():
        cols = [c for c in grades.columns if c.startswith(f + "|")]
        opts = [c.split("|", 1)[1] for c in cols]
        if sorted(opts) != sorted(vals):
            raise SystemExit(f"{f}: grade options {sorted(opts)} != table options {sorted(vals)}")
        out[f] = (grades[cols].values * [vals[o] for o in opts]).sum(1).astype("float32")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grades", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--any_out", default=None, help="also write presence-only labels (P(finding mentioned as abnormal))")
    a = ap.parse_args()
    g = pd.read_parquet(a.grades)
    sev = to_labels(g, V)
    sev.to_parquet(a.out, index=False)
    anyp = to_labels(g, PRESENCE)
    if a.any_out:
        anyp.to_parquet(a.any_out, index=False)
    print(f"{len(sev)} studies | finding: presence share >= 0.5 -> severity share >= 0.5 (mean severity label)")
    for f in V:
        print(f"  {f:17s} {float((anyp[f] >= 0.5).mean()):.3f} -> {float((sev[f] >= 0.5).mean()):.3f} "
              f"({float(sev[f].mean()):.3f})")


if __name__ == "__main__":
    main()
