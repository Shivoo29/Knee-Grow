# T04: Left/right canonicalisation of knees (RSNA 2024 "split LR" idea)

Read `delegation/PREAMBLE.md` first and obey it. Exceptions for THIS task are listed under "Where you work".
Time-box: 4 hours. Then write the results file with what you have.

## Goal
Make every study look like the same knee to the model, so that the medial compartment is always on the same side
(sagittal: same end of the slice stack; coronal/axial: same side of the image). Findings that depend on this:
Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA. Today nothing does this: `train_knee.py:277` says
"NO flips (laterality is signal)", and slices are only sorted by `dot(ImagePositionPatient, r x c)` in
`rsna_preproc.py:56-62`, whose direction depends on each series' orientation.

Background (verified by the 3rd place RSNA 2024 lumbar writeup, discussion 539453): "For the right-side data, we
reversed the order of sagittal slices and horizontally flipped the axial images."
Note: `~/projects/knee/anatomical_mirror.py` on another machine claims "+0.003 to +0.004 LB" for mirror TTA. That is
NOT VERIFIED; our own test of mirror TTA showed no gain. Do not reuse that claim. This task is about training-time
canonicalisation, not TTA.

## Facts measured on 300 random training series headers (Oct 14)
- `ImageLaterality`/`Laterality` present: 145 of 300 (L 70, R 75), empty in 155.
- `SeriesDescription` contains LT/RT/LEFT/RIGHT tokens in 39 of 300.
- A study has several series; coverage per study is higher than per series (NOT measured).
- Corpus layout (`rsna_preproc.py:20-29`): 96 slices = slots [Sagittal fluid 26, Sagittal non-fluid 22, Coronal fluid 18,
  Coronal non-fluid 12, Axial 18]. `ds/corpus96/slots.csv` maps (StudyInstanceUID, slot_idx) to SeriesInstanceUID,
  `n_picked`, `is_padded`. `all_vols.npy` is (N, 96, H, W) uint8, `all_masks.npy` (N, 96) marks real vs padded slices,
  `all_ids.npy` gives the row order. Do NOT load `all_vols.npy` fully (62 GB); you do not need it.

## Design (implement exactly this; ask in the results file if something blocks it)
Two independent levels, both per (study, slot):

1. Orientation canonicalisation (always possible, needs only IOP/IPP):
   - Patient coordinates are LPS: +x = patient left.
   - Sagittal slot: compute the x of the first and last picked slice (`ImagePositionPatient[0]`). Flag
     `rev = 1` if x decreases along the stored order, so after reversing, x always increases.
   - Coronal/axial slot: row direction `r = IOP[:3]`. Flag `hflip = 1` if `r[0] < 0`, so after flipping, image
     columns always go towards +x.
2. Side canonicalisation (only when the side is known):
   - Side from, in order: `ImageLaterality`, `Laterality`, SeriesDescription tokens; majority vote over the study's
     series; `U` if unknown or conflicting.
   - For a LEFT knee, invert both flags of every slot (medial lies towards -x for a left knee, +x for a right knee),
     so every known knee ends up looking like a right knee. `U` studies keep level 1 only.

Flags are applied at LOAD time (no corpus rebuild):
- Sagittal `rev`: reverse only the real slices of that slot (use the mask; keep padded slices at the end).
- Coronal/axial `hflip`: `np.flip(slice, axis=-1)` for every slice of that slot.

## Deliverables (all inside the worktree, see below)
1. `laterality.py` with pure functions, no I/O inside them:
   - `side_from_headers(list_of_dicts) -> 'L'|'R'|'U'` (dict keys: ImageLaterality, Laterality, SeriesDescription)
   - `slot_flags(plane: str, iop: list[float], first_ipp: list[float], last_ipp: list[float], side: str) -> int`
     (returns 1 to reverse/flip, else 0)
   - `apply_flags(vol: np.ndarray (96,H,W), mask: np.ndarray (96,), flags: np.ndarray (5,)) -> np.ndarray`
2. `build_laterality_flags.py`: reads DICOM headers (`stop_before_pixels=True`) for the series listed in
   `ds/corpus96/slots.csv`, from a zip path given by `--zip` (the 265 GB competition zip; it is on node1, you do NOT
   run this script, Claude runs it there). Writes `laterality_flags.csv`: StudyInstanceUID, side, side_source,
   f0..f4. Must stream: one `dcmread` per needed file, never extract the zip. Print only counts.
3. Hook in `train_knee.py` (in the worktree copy): new flag `--lat_flags PATH`; when given, the dataset applies
   `apply_flags` to `vol` right after `vol = self.vols[row]` (`train_knee.py:250`). Without the flag, behaviour is
   byte-identical to today.
4. Hook in `kaggle_base954_fleet/fleet_v6.py` (worktree copy): at inference the same flags must be computed from the
   TEST DICOM headers that the script already reads, then applied before windows are built. Enabled only when the
   checkpoint has `ck.get("lat_flags") == True` (and `train_knee.py` must save `"lat_flags": bool(a.lat_flags)` in
   its three checkpoint dicts next to `"res": a.res`). `fleet_v6.py` must stay one self-contained file, so copy the
   three functions into it and add a test that asserts the copies are identical to `laterality.py` (compare source
   text via `inspect.getsource`).
5. `tests/test_laterality.py`, synthetic data only, at least:
   - sagittal stack with x decreasing gets reversed; x increasing does not; LEFT inverts it.
   - coronal with `r[0] < 0` gets flipped; LEFT inverts it; `U` keeps level 1 only.
   - padded slices stay at the end after a reverse (mask with 20 real + 6 padded in slot 0).
   - `apply_flags` with all-zero flags returns an identical array.
   - side vote: {L, L, R} -> `U` (any conflict -> `U`); {L, L} -> `L`; no tags -> `U`.
   - fleet_v6 copies identical to laterality.py.
   pytest is NOT installed (do not install it). Write plain `def test_*` functions plus a `__main__` block that runs them all, and run `.venv/bin/python tests/test_laterality.py`. Paste the exact output into results.

## Where you work (exceptions to PREAMBLE)
- Create a git worktree: `git worktree add ../knee-lat -b laterality` from the project root, and edit ONLY inside
  `../knee-lat`. You may edit `train_knee.py` and `kaggle_base954_fleet/fleet_v6.py` THERE (not in the main folder).
- Do not commit; the owner commits. Do not run training. Do not touch `ds/` in the main folder except reading
  `ds/corpus96/slots.csv` (symlink it or pass its path).
- Results file: `delegation/results/T04_laterality.md` in the MAIN folder. First 15 lines = SUMMARY per PREAMBLE.

## What Claude does after you (not your job)
Runs `build_laterality_flags.py` on node1, checks side coverage, trains one CoAtNet version 6 distilled student with
`--lat_flags` (seed 12) and compares gold-58 per finding against CoAtNet version 6 seed 8 (0.9296).
