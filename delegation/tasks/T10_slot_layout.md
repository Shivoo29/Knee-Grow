# T10: Configurable slot layout (96 -> 192 slices) in training and Kaggle inference

Read `delegation/PREAMBLE.md` first and obey it. Exceptions under "Where you work". Time-box: 5 hours.

## Why
Our corpus keeps 96 slices per study in 5 fixed slots [Sagittal fluid 26, Sagittal non-fluid 22, Coronal fluid 18,
Coronal non-fluid 12, Axial 18]. Series have a median of 30 slices, so we throw away ~35% of slices (148 -> 96 per
study) and our "3 adjacent slices" windows are really 1.5-2 slices apart. A new corpus with layout
[40, 40, 36, 36, 40] = 192 slices (keeps ~90% of slices; built by `build_corpus96_stream.py --slot_sizes`) is being
built on node1 at `~/projects/knee/ds/corpus192/`. The code below must handle any 5-slot layout.

## Where the 96 layout is hard-coded (grep to confirm)
- `train_knee.py`: `SLOT_BOUNDS = [(0, 26), (26, 48), (48, 66), (66, 78), (78, 96)]`, `slot_of`, `pick_centers(max_d=96)`,
  `mirror_volume` (uses SLOT_BOUNDS), `--k_eval` default usage (94 = 96 - 2), `self._D` from the volume (already dynamic).
- `kaggle_base954_fleet/fleet_v6.py`: its own `SLOTS` / `MAXS = 96` / `SAGITTAL_SLOTS` / `K_EVAL = 94` / `SLOT_BOUNDS`
  copy, the per-slot slice picking that builds the test volume from DICOMs, `pick_centers(..., max_d=MAXS)`.
- `rsna_preproc.py` `SLOTS` (only if fleet_v6 imports nothing from it; check).

## Build
1. `train_knee.py`: new flag `--slot_sizes` (default "26,22,18,12,18"). Derive `SLOT_BOUNDS` from it at startup (one
   function `bounds_from_sizes(sizes)`), use it in `slot_of`, `mirror_volume`, `pick_centers` (max_d = sum(sizes)).
   Assert at startup that the corpus volume depth == sum(sizes). Save `"slot_sizes": a.slot_sizes` in every checkpoint
   dict and JSON next to `"slots"`. `--k_eval` stays a flag; document that 190 = 192 - 2 is the full-coverage value.
2. `fleet_v6.py`: read `ck.get("slot_sizes", "26,22,18,12,18")`; build the test volume with those slot sizes using
   exactly the same picking rule as `build_corpus96_stream.py` (`linspace` over the same SPAN, duplicates when the
   series is shorter); `MAXS` and `K_EVAL = MAXS - 2` follow from it. Old checkpoints (no key) must behave exactly as
   today. The fleet copy of `bounds_from_sizes` / `slot_of` / `pick_centers` must be source-identical to train_knee's.
3. Keep everything else unchanged (ROI, seq_head, clean_windows, slot_emb, norm).

## Tests (`tests/test_slot_layout.py`, plain `def test_*` + `__main__` runner, no pytest)
- default sizes: `bounds_from_sizes` == the old SLOT_BOUNDS; `slot_of` identical to the old function for 0..95.
- 192 layout: bounds [(0,40),(40,80),(80,116),(116,152),(152,192)]; `slot_of(39)==0`, `slot_of(40)==1`, `slot_of(191)==4`.
- `mirror_volume` on a synthetic (192,8,8) volume with the 192 layout: sagittal real slices reversed per slot,
  coronal/axial flipped; mirror twice == identity.
- fleet test-volume builder: for a synthetic series of 30 slices and slot size 40, picked indices == the indices
  `build_corpus96_stream.py` would pick (import or copy its picking function and compare).
- old checkpoint path (no slot_sizes key) in fleet: window builder output identical to the current main-branch code
  (use `git show HEAD:kaggle_base954_fleet/fleet_v6.py` as the reference, like tests/test_roi_inference.py does).
- train/fleet copies identical.
CPU only. Do NOT run training or GPU jobs (training runs on this PC and on node1).

## Where you work
`git worktree add ../knee-slots -b slots` from the project root; edit only there. Do not commit.
Results: `delegation/results/T10_slot_layout.md` in the MAIN folder; first 15 lines = SUMMARY.
