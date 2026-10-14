# T08: Run ROI-crop specialist checkpoints in Kaggle inference (fleet_v6.py)

Read `delegation/PREAMBLE.md` first and obey it. Exceptions under "Where you work". Time-box: 4 hours.

## Context
We are training "specialist" CoAtNets on the PC (`run_pc_spec_oa.sh`): only coronal windows (`--slots 2,3`) and
compartment ROI crops (`--roi --roi_mode compartment`). Training crops come from `roi_boxes.npz` (per-slice tissue
box `sbox` (N,96,4) and centroid `cen` (N,96,2)), built by `build_roi_boxes.py` from the corpus. Inference
(`kaggle_base954_fleet/fleet_v6.py`) has NO ROI support today, so these checkpoints cannot run on Kaggle yet.
Already supported in fleet_v6: `slots` (ck["slots"], e.g. "2,3"), `clean_windows`, `slot_emb`, `norm`.

## Build
In `fleet_v6.py` (one self-contained file, no new imports beyond numpy/torch/cv2 already used):
1. Copy the per-slice box/centroid computation from `build_roi_boxes.py` (the function used inside
   `_process_chunk`) and `_roi_square_box` + `_roi_compartments` from `train_knee.py` (grep for them), unchanged.
2. When the checkpoint says ROI (`ck.get("roi")` true; also read `roi_mode`, `roi_pad` (default 0.06),
   `roi_overlap` (default 0.12)), `predict_study` must build windows exactly like `train_knee.py`
   `StudyWindows.__getitem__` does in eval mode: centers from `pick_centers(..., slots=...)`, box from the CENTER
   slice, compartment mode = 2 crops (left, right) per center, each resized to `res`; and pass slot ids if
   `slot_emb`. Number of windows = 2 x centers; cap total windows at 2 x K_EVAL.
3. Boxes must be computed at inference from the same uint8 volume the model sees (the 96-slice volume built from
   test DICOMs), with the same function as training.
4. Checkpoints without `roi` keys must behave exactly as today.
5. In `train_knee.py`: add `"roi": a.roi, "roi_mode": a.roi_mode, "roi_pad": a.roi_pad, "roi_overlap": a.roi_overlap`
   next to `"slots": a.slots` in every checkpoint dict (do not change anything else).
6. Script `tools/add_roi_keys.py CKPT --roi_mode compartment`: loads a .pt, adds those 4 keys (+ keeps all others),
   saves to `CKPT` with suffix `_roi.pt`. (The run in progress was started before step 5.)

## Tests (`tests/test_roi_inference.py`, plain functions + `__main__` runner, no pytest)
- Box function copy in fleet_v6 is source-identical to `build_roi_boxes.py`'s; `_roi_square_box` and
  `_roi_compartments` identical to train_knee's (compare extracted source text).
- On 3 real studies from `ds/corpus96/all_vols.npy` (mmap, read 3 rows only; print only shapes/ints): boxes computed
  by the fleet copy equal `roi_boxes.npz` rows for those studies (match by `all_ids.npy` order and the npz's ids
  key if present).
- Window tensor for one study from fleet `predict_study`'s window builder equals `StudyWindows(train=False)`'s
  output for the same study, same k (max abs diff < 1e-6). Build `StudyWindows` with roi_sbox/roi_cen from the npz.
- Old checkpoint path (no roi keys): window builder output unchanged vs the current main-branch code.
Do NOT run any GPU job; a training job is running on this PC. CPU only.

## Where you work
`git worktree add ../knee-roi -b roi` from the project root; edit only there. Do not commit.
Results: `delegation/results/T08_roi_inference.md` in the MAIN folder; first 15 lines = SUMMARY.
