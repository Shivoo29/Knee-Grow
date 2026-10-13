# T05: Data recipe for the next CoAtNet run (clean windows, mirror + contrast augmentation, slot embedding)

Read `delegation/PREAMBLE.md` first and obey it. Exceptions for THIS task are under "Where you work".
Time-box: 5 hours. Then write the results file with what you have.

## Verified problems (Claude checked these in code on Oct 14)
1. `train_knee.py:217-229` `_centers`: the comment says "window centers must have both neighbors valid", but the code
   only checks the first/last valid index. Windows therefore mix two slots (e.g. last Sagittal-fluid slice + first
   Sagittal-non-fluid slice in one RGB window) and include black padded slices. `fleet_v6.py` `predict_study`
   (around line 336-345) has the same logic, so training and inference match, but both are wrong.
2. Slot layout (`rsna_preproc.py:20-29`): 96 slices = [Sagittal fluid 26, Sagittal non-fluid 22, Coronal fluid 18,
   Coronal non-fluid 12, Axial 18]. Slot 3 (Coronal non-fluid) is empty in 25.5% of studies (`ds/corpus96/slots.csv`).
   `all_masks.npy` (N,96) marks real slices.
3. Augmentation is only a +-10% brightness gain (`train_knee.py:277-279`). The comment "NO flips (laterality is
   signal)" is wrong: labels are medial/lateral, not left/right; a mirrored right knee is a valid left knee with the
   same labels.
4. The attention head gets no information about which slot/plane a window comes from (`train_knee.py:323-362`).
5. `fleet_v6.py` writes 0.5 for a study whose prediction fails and only counts it (around line 640-660); no limit.

## What to build (all behind new flags; with no new flag, behaviour must be byte-identical to today)
`train_knee.py`:
- `--clean_windows`: a window center c is allowed only if c-1, c, c+1 are all real (mask > 0) AND in the same slot.
  Put the slot boundaries in one function `slot_of(index) -> int` derived from the sizes [26,22,18,12,18].
- `--mirror_p FLOAT` (default 0): training only, per STUDY (not per window), with probability p mirror the whole
  study before windows are cut: reverse the order of the REAL slices inside each Sagittal slot (slots 0,1; padded
  slices stay where they are) and `np.flip(axis=-1)` every slice of Coronal/Axial slots (2,3,4).
- `--contrast_aug`: training only, per study, random gamma in [0.8, 1.25] applied to the [0,1] windows, then the
  existing brightness gain. Clamp to [0,1].
- `--slot_emb`: `nn.Embedding(5, F_dim)` (init zeros) added to each window feature before the head's LayerNorm,
  using the slot of the window's CENTER slice. The dataset must return the slot ids with the windows; keep the
  forward signature backward compatible (`forward(x, slots=None)`; `None` = no embedding).
- Save `clean_windows`, `slot_emb` (bools) in all checkpoint dicts, next to `"res": a.res, "norm": a.norm`.

`kaggle_base954_fleet/fleet_v6.py`:
- Read `ck.get("clean_windows", False)` and `ck.get("slot_emb", False)`; when true, pick centers the same way and pass
  slot ids. Old checkpoints (no keys) must run exactly as today. `fleet_v6.py` must stay one self-contained file:
  copy `slot_of` and the center selection into it, and add a test that the copies are identical (compare
  `inspect.getsource` text, or the extracted function source).
- Fallback guard: if more than 3% of studies fall back to 0.5, raise `RuntimeError` at the end (before writing the
  CSV), printing the count.

`tests/test_data_recipe.py` (pytest is NOT installed; plain `def test_*` + a `__main__` runner; run with
`.venv/bin/python tests/test_data_recipe.py`, paste exact output):
- clean centers never straddle a slot boundary and never touch a masked slice (synthetic mask with slot 3 empty).
- without `--clean_windows` the old center list is unchanged (compare against a copy of today's function).
- mirror: sagittal real slices reversed, padded slices untouched, coronal/axial flipped; mirror twice = identity.
- slot ids returned match `slot_of(center)`.
- `forward(x)` and `forward(x, slots=None)` give identical outputs; with `--slot_emb` and zero-init embedding the
  output equals the no-embedding output at init.
- train and fleet copies of `slot_of`/center selection are identical.
- fallback guard raises at 4% and does not raise at 2%.
Also run `./run_pc_dinov2_s11.sh`? NO. Do not run any training or GPU job (a training job is running on this PC).

## Where you work (exceptions to PREAMBLE)
- `git worktree add ../knee-recipe -b recipe` from the project root; edit ONLY inside `../knee-recipe`
  (you may edit `train_knee.py` and `kaggle_base954_fleet/fleet_v6.py` there). Another agent works in `../knee-lat`
  on laterality (T04); do not touch it.
- Do not commit. Results file: `delegation/results/T05_data_recipe.md` in the MAIN folder; first 15 lines = SUMMARY.
- Print no pixel values; synthetic arrays only in tests.

## After you (Claude)
Reviews the diff, runs a 64-study speed check on node1 with `--win_chunk 0` vs 14, then launches one CoAtNet
version 6 run with `--clean_windows --mirror_p 0.5 --contrast_aug --slot_emb` after seed 9 finishes.
