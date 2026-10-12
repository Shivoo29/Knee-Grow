# Task for zcode (GLM-5.3): series-count robustness test of our inference preprocessing

## HARD RULES (competition data security — violating these can disqualify us)
1. NEVER open, print, cat, head, or read the `Report` column of any CSV (data/train.csv has it). If you load train.csv, immediately drop `Report` (`usecols=` without it).
2. NEVER print pixel arrays or save images. Print only shapes, dtypes, counts, min/max/mean numbers.
3. Do not upload anything, do not use the internet with project data.
4. Do not modify existing files. Write only new files under `tests/`.
5. Do not touch the remote machine `node1` and do not kill any process.

## Context
Project root: `~/Documents/still_learning/rsna-knee-abnormality-detection` (python: `.venv/bin/python`).
Kaggle competition: knee MRI, each study = several DICOM series (Sagittal/Coronal/Axial, various sequences).
Our inference turns a study into a fixed 96-slot volume: slots Sag 26+22, Cor 18+12, Ax 18 (see `rsna_preproc.py`,
`build_corpus96_stream.py`, and the inference code used in `kaggle_submission_top10_champion/` and `sovereign_fast_dual_fleet.py`).
Series metadata: `data/train_series.csv`, `data/test_series.csv` (plane, sequence flags per series).

Train studies have 3 to 14 series (mean 5.5; 675 have 4, 434 have 8+). The hidden test will have the same variety.
A past bug: a wrong folder made every input all-zero and the notebook silently scored 0.500. We must make sure odd studies
do not silently produce empty/garbage slots.

## What to do
1. Read the slot-selection logic (how series are chosen per slot) in `rsna_preproc.py` and the inference script. Write a
   short explanation (≤15 lines) in `tests/SERIES_SELECTION.md`: which series wins a slot when several qualify; what happens
   when a plane or the fluid-sensitive series is missing.
2. Using ONLY `data/train_series.csv` metadata (no DICOM needed), write `tests/test_series_selection.py` that runs the
   selection function on every train study's series list and reports counts:
   - studies where any slot group gets no series (empty slot group), per group
   - studies where selection is non-deterministic (run twice with shuffled series order; compare chosen SeriesInstanceUIDs)
   - studies with 3 series and with ≥10 series: what was chosen (print UIDs truncated to last 6 chars, never paths with reports)
   If the selection function needs pixel data, refactor nothing: instead mock the DICOM header fields it reads from the CSV.
3. Run it, put the summary numbers in `tests/RESULTS.md`, and list any bug you found with file:line and a proposed fix
   (do not apply the fix).

Deliverable: `tests/SERIES_SELECTION.md`, `tests/test_series_selection.py`, `tests/RESULTS.md`. Keep the code short and plain.
