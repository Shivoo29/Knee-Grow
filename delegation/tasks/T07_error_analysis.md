# T07: Per-finding error analysis on gold-58 (which findings lose the most AUC?)

Read `delegation/PREAMBLE.md` first and obey it. CPU only, read-only except your outputs. Time-box: 2 hours.

## Why
Macro AUC = mean of 12 per-finding AUCs. We want to build specialist models only for the 2-3 weakest findings.
First we need to know which ones are weak, and whether all models fail on the same studies.

## Inputs (local)
- `raptor_gold_*.npz` in the project root: our models' gold-58 predictions. Keys: `pred` (58,12), `truth` (58,12),
  `ids` (58,). Columns in this order: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA,
  Effusion, Synovitis, Baker's, Contusion, Fracture. Use only files WITHOUT `_ep` in the name (final/swa/epens).
- `teacher/c0_raptor_raw.csv` (public Raptor CoAtNet) and `teacher/c0_compact.csv` (public Heliosli compact
  ConvNeXt): predictions on chunk 0, which contains all 58 gold studies (StudyInstanceUID + 12 columns).
  CAVEAT: Raptor was trained on "alldata" and may have seen the gold studies; mark its numbers "possibly in-sample".
- Gold truth: take it from any npz `truth` + `ids` (do NOT open data/train.csv; it has the Report column).

## Do
Write `delegation/results/T07_error_analysis.py` (run with `.venv/bin/python -I`), producing
`delegation/results/T07_error_analysis.md` with:
1. Table: rows = findings, columns = AUC of Raptor, compact, 0.7*rank(Raptor)+0.3*rank(compact), and our best 3
   models by macro AUC (name them by file). Also positives count per finding (AUC with < 5 positives is unreliable;
   flag it).
2. Bootstrap 95% CI (1000 resamples of the 58 studies) for each finding's AUC of the blend.
3. The 3 weakest findings of the blend, and for each: which studies are the worst misses (by rank of a positive
   among negatives, list only the study index 0-57, not the UID), and whether Raptor, compact and our best model
   miss the SAME studies (Spearman of their per-study rank errors).
4. For each finding: does any of our models beat the blend by more than the CI half-width? (Say "noise" otherwise.)

First 15 lines of the .md = SUMMARY: weakest 3 findings with AUC and CI, and one line on whether models share errors.
No LB predictions. Print no UIDs, no pixel data, no report text.
