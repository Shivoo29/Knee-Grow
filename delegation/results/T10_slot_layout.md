# SUMMARY (lines 1-15; reviewer reads only this)
1. Verdict: configurable 5-slot layout built in worktree ../knee-slots (branch slots, no commit); 6/6 CPU tests pass; no training/GPU job run.
2. Changed: train_knee.py +36 (bounds_from_sizes, dynamic max_d, --slot_sizes, depth assert, ckpt keys, k_eval doc); fleet_v6.py +53/-19 (same 3 funcs, picking helper, span 0.02/0.98, ckpt-driven rebind); new tests/test_slot_layout.py.
3. Key numbers: default bounds byte-equal old; 192 bounds [(0,40),(40,80),(80,116),(116,152),(152,192)]; picks n=30/k=40 lo=0 hi=28 == corpus; old-path windows+TTA bitwise 0.00e+00.
4. Problems: none blocking; ONE deliberate behavior change: fleet slice span 0.06/0.94 -> corpus 0.02/0.98 (see lines 16+).
5. file:line (worktree): train bounds train_knee.py:187-231 + startup 682-692 + keys 6 sites; fleet helper 248-255, copies ~370-412, rebind 1153-1166.
6. NOT checked: real 192-slice training; Kaggle end-to-end on corpus192; LB impact of the span change.
7. NOT VERIFIED: corpus192 contents/depth/order on node1; 190-window runtime vs 9h limit.
8. Main folder untouched (git status: no modifications); main .venv used read-only; jobs untouched (no ssh/pkill, CPU only).
9. Old checkpoints (no slot_sizes key) re-derive byte-identical layout consts (96/94/bounds/sizes/map); window builder output bitwise equal to HEAD.
10. rsna_preproc.py SLOTS untouched (nothing imports it; verified); its 0.02/0.98 span matches the new fleet rule.
11. --k_eval stays a flag; 190 = 192 - 2 documented in help; depth assert fails loudly on corpus/layout mismatch.
12. Copies identical: bounds_from_sizes/slot_of/pick_centers source-equal train vs fleet (3451ch-class precedent kept).
13. No UIDs/pixels/Report text in tests or this file; shapes/counts/scalars only; no CSVs opened.
14. Time-box: ~1.5h of 5h used.

## Span decision (deliberate behavior change, read before merging)
- Before: fleet picked test slices over span [0.06, 0.94] (`git show HEAD:...fleet_v6.py`, old lines ~302-305) while the corpus builder and the SSOT `rsna_preproc.py:31-33,135-143` use [0.02, 0.98]. Train/inference disagreed on peripheral slices.
- Task text requires "exactly the same picking rule as build_corpus96_stream.py (linspace over the same SPAN...)" and its test pins fleet picks == corpus picks (n=30/k=40 -> lo=0 hi=28, verified in test 4). Satisfying both forces the fleet span to 0.02/0.98.
- Consequence: old checkpoints pick slightly wider peripheral slices at inference than before (volume-level change; window builder given the same volume is bitwise identical, test 5). LB impact NOT VERIFIED and cannot be checked here.
- `rsna_preproc.py` left untouched: no file imports it (fleet_v6 and train_knee carry their own copies), so its SLOTS literal is dormant.

## Build details (worktree ../knee-slots)
- `train_knee.py:187-231`: `bounds_from_sizes(sizes)`; `SLOT_BOUNDS` derived; `slot_of` generalized; `pick_centers(..., max_d=None)` defaults to layout depth; `mirror_volume` uses the global (docstring de-hard-coded from 96).
- `train_knee.py:546-548,682-692`: `--slot_sizes` (default "26,22,18,12,18"); startup parses, asserts 5 positive ints, rebinds global `SLOT_BOUNDS`, asserts `corpus depth == sum(sizes)` via `_open_corpus()` (mmap, cheap), prints layout line. `--k_eval` help documents 94/190 full-coverage values.
- Checkpoint dicts (6 sites): `"slot_sizes": a.slot_sizes` next to `"slots"` in all 4 .pt dicts + 2 JSON sidecars.
- `fleet_v6.py:~370-412`: identical copies of the 3 functions (test-extracted equality). `_pick_slot_slice_indices(n, k)` (`:248-255`) used by `build_study_optimized`; `K_EVAL` comment updated.
- `fleet_v6.py:1153-1166` (in `main`): layout from `ck.get("slot_sizes", "26,22,18,12,18")` with assert; rebinds `SLOTS/MAXS/K_EVAL/SLOT_SIZES/SLOT_BOUNDS/_b_slots/_s_map` (TTA map included). No-key checkpoints reproduce 96/94/bounds exactly. `SAGITTAL_SLOTS=(0,1)` and 5-wide `slot_emb` unchanged (slot indices, not sizes).
- Untouched by design: ROI/seq_head/clean_windows/slot_emb/norm paths; `self._D`-dynamic train code needed no changes.

## Tests (`tests/test_slot_layout.py`, main .venv `python -I`, CPU, 0.1s)
- Exact output: `PASS default_bounds ... / PASS layout_192 ... slot_of(39)==0, slot_of(40)==1, slot_of(191)==4 / PASS mirror_192 ... identity ok / PASS picking_matches_corpus picks=40 lo=0 hi=28 source_block ok / PASS old_ckpt_path_unchanged consts ok windows+tta bitwise equal / PASS copies_identical 3 funcs identical / done 6/6 pass`. No command failed.
- HEAD reference imported from `git show HEAD:...` to a temp module (mirrors test_roi_inference.py); slot_of compared 0..95 plus edges -1/200; centers compared clean on/off + slots filter.
