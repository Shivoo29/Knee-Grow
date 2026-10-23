# T0Y — Axial fluid-sensitive corpus variant: preflight stopped

## Step 0 — Preconditions

### Disk and running jobs

- `df -h ~/projects/knee`: 642 GB available; the 120 GB requirement passes.
- `tmux ls`: `kneegrow` and the read-only `t0x_monitor` session only. No `train_knee`, distillation, v6 seed, or corpus-build process was found in the process check.
- The source zip and `ds/corpus96/slots.csv` exist. `train_series.csv` is at `ds/comp/train_series.csv` and also inside the raw zip.

### Training input discovery

`train_knee.py` is at `~/projects/knee/train_knee.py`, so `HERE` is `~/projects/knee`.

- `_open_corpus()` first checks `HERE/all_vols.npy`. If it exists, it loads `HERE/all_vols.npy` and `HERE/all_masks.npy` directly and does not consult `RAPTOR_INPUT`.
- Only if `HERE/all_vols.npy` is absent does it call `_find()` under `RAPTOR_INPUT`. The single-corpus path uses `all_vols.npy` + `all_masks.npy`; the two-part fallback expects `all_vols.npy`, `extra_vols.npy`, `all_masks.npy`, and `extra_masks.npy`.
- `_open_ids()` likewise first checks `HERE/all_ids.npy`; only if absent does it search `RAPTOR_INPUT` for `all_ids.npy` (or `all_ids.npy` + `extra_ids.npy`).
- `_find()` walks the `RAPTOR_INPUT` tree by basename. `comp/train.csv` is not used by `_open_corpus()` or `_open_ids()`; the distilled labels are supplied separately with `--labels labels_distil_v1.parquet`.

The project-root files currently exist and point to the old corpus:

- `all_vols.npy` → `ds/corpus96/all_vols.npy`
- `all_masks.npy` → `ds/corpus96/all_masks.npy`
- `all_ids.npy` → `ds/corpus96/all_ids.npy`

**Stop condition:** with the existing `train_knee.py` location and these root symlinks, setting only `RAPTOR_INPUT=$HOME/projects/knee/ds_axfs` would still load the old corpus. Making the new corpus work would require changing the loader, changing/removing existing root links, or running an unchanged copy of `train_knee.py` from a different directory. Those are outside the Step 0 condition that it work with only a different `RAPTOR_INPUT`, and altering the existing links/code is prohibited. Per the instructions, I stopped here.

### Slot and picker references found

The repository search excluded virtual environments, caches, and `.git`; it found these files with slot lists, picker code, or direct uses of `SLOTS`/the picker:

- `build_corpus96_stream.py` — corpus `SLOTS` and `_pick_series_for_slot` definitions.
- `rsna_preproc.py` — `SLOTS` and `pick_series_for_slot` definitions.
- `pc_code/rsna_preproc.py` — inference preprocessing slot/picker implementation.
- `pc_code/kaggle_base954_fleet/fleet_v6.py` — fleet inference slot/picker implementation.
- `submission_kernel.py` — slot/picker implementation.
- `kaggle_downloads/rsna-knee-student-fleet-96-slice/rsna-knee-student-fleet-96-slice.py`.
- `kaggle_downloads/rsna-knee-v6-fast-96-slice-swa/rsna-knee-v6-fast-96-slice-swa.py`.
- `kaggle_downloads/rsna-knee-v5-blend-2-seeds/rsna-knee-v5-blend-2-seeds.py`.
- `kaggle_downloads/rsna-knee-v5-blend-3-seeds/rsna-knee-v5-blend-3-seeds.py`.
- `kaggle_downloads/rsna-knee-5-family-grand-ensemble/run_5family_ensemble.py` and its notebook; also `kaggle_downloads/5_family/rsna-knee-5-family-grand-ensemble.ipynb`.
- `kaggle_downloads/rsna-knee-blend-plus-d4/run_raptor_blend.py` and its notebook.
- `kaggle_downloads/rsna-knee-clean-rank-blend/run_raptor_v6.py` and its notebook.
- `kaggle_downloads/rsna-knee-grand-ensemble-v1/run_raptor_blend.py` and its notebook.
- `kaggle_downloads/rsna-knee-grand-ensemble-v2/run_raptor_blend.py` and its notebook.
- `eval_gold58_raw_dicom.py`, `eval_res_tta.py`, `test_fleet_runner.py`, `verify_inference_preproc.py`, and the teacher-probe notebooks also reference the slot/picker interface.

## Steps not run

Steps 1–4 were not run. No affected-study CSV, `ds_axfs/`, copied or patched corpus, versioned preprocessing/builder/inference scripts, or training launcher was created. No dataset/corpus file, existing code, GPU, or training process was changed.

## Final status

**NOT READY — stopped at Step 0 because `train_knee.py` resolves the existing project-root corpus symlinks before `RAPTOR_INPUT`; the requested new-corpus-only environment override cannot select `ds_axfs`.**


## T0Y continuation update — stopped at Step 3 verification failure (2026-10-15)

### Step 0
- Disk precheck passed: 642 GB free before the copy; post-copy check showed 584 GB free.
- The original project-root all_vols/all_masks/all_ids links take precedence over RAPTOR_INPUT. To preserve the old paths and select ds_axfs, an unchanged, byte-identical copy of train_knee.py was placed under train_axfs/ (no old corpus links there). The new launcher was not created because a later required verification failed.
- Active GPU training jobs were visible during the work: oof4_f2 and recipe_s22. They were not touched. All audit/rebuild work ran at nice 19, CPU only, at most 16 workers.
- Slot and picker source locations were inventoried in the Step 0 section above.

### Step 1
- 4,407 studies; 450 affected (10.21%); 9 gold studies affected.
- train_series.csv: 24,371 rows, zero missing Fluid_Sensitive flags. test_series.csv: 15 rows across 3 dummy studies, zero missing flags.
- Applying the new picker changes only those 450 studies. The requested affected list is delegation/results/T0Y_affected_studies.csv with columns StudyUID, old_series, new_series, is_gold.
- Missing-value policy in new code copies: invalid/missing flag => None; attempt explicit 0/1 preference only when valid, otherwise fallback to first eligible candidate.

### Step 2
- Physical copy ds_axfs/corpus96 created; source ds/corpus96 was not written.
- Copy verification passed: SHA-256 matched for all_masks.npy and all_ids.npy; 200 randomly sampled all_vols.npy studies compared byte-for-byte.
- All 450 affected studies were fully rebuilt from the zip. Determinism passed for slots 0–3 (slices 0–77 and masks 0–77) in all 450 studies.
- Only slot 4 (slices 78–95 and mask positions 78–95) was patched in ds_axfs. Post-check passed for 20 random affected studies (selected axial series has Fluid_Sensitive=1 and slot is populated) and 20 random unaffected studies (all 96 image slices and masks remain byte-identical).
- ds_axfs footprint: 59 GB.
- New copies created: build_corpus96_stream_axfs.py, rsna_preproc_axfs.py, kaggle_students/fleet_v6_axfs.py, and byte-identical train_axfs/train_knee.py.

### Step 3 — FAILED; stop rule applied
- Train-side vs ds_axfs slot parity was not completed. The first call to rsna_preproc_axfs.pick_series_for_slot failed with NameError: name 'pd' is not defined in its new _fluid_flag helper.
- Per T0Y, stopped immediately after this verification failure. No code fix, inference parity rerun, raw DICOM pixel parity run, or dummy test picker run was performed.
- Therefore pixel parity max/mean differences are NOT AVAILABLE; required max-diff <= 1 criterion remains unverified.
- The 3 dummy test studies remain unverified by the new picker.

### Step 4
- Not reached because Step 3 failed. No run_distil_axfs.sh was created and no training command was launched.
- Existing active training commands show that clean-window and slot-embedding options are in current use, but T05 compatibility for this new corpus has not been verified here.

Runtime: copy plus rebuild/determinism and post-check took approximately 3 minutes after the zip TOC was indexed; failed Step 3 picker check occurred before pixel parity. Header failures: none were encountered during these rebuilds; raw inference pixel decode failure count is not available because that stage was not run.

FINAL STATUS: NOT READY — AXFS corpus construction and determinism passed, but required inference picker parity failed at rsna_preproc_axfs.py due to missing pandas import; stop rule followed.



## T0Y completion update (2026-10-15)

The earlier Step 3 stop was resolved after user authorization. The preprocessing helper was corrected to detect non-finite/missing flags with NumPy, avoiding an undeclared pandas dependency. The audit then resumed and all remaining required checks passed.

### Step 0 — loader and unchanged references
- The training copy is now byte-identical to pc_code/train_knee.py (SHA-256 659374863764396129998ca403e1305e1ba6a115735fb6e438f84767efed16e9), which contains the T05 clean-window and slot-embedding options.
- Its unchanged _find/_open_corpus/_open_ids code expects paired all/extra arrays when it searches RAPTOR_INPUT. Added zero-row extra_vols.npy, extra_masks.npy, and extra_ids.npy under ds_axfs/corpus96. A CPU-only loader check confirmed RAPTOR_INPUT=ds_axfs resolves exactly 4,407 x 96 volumes, masks, and IDs. No source loader code or old symlinks were changed.
- The complete slot/picker source inventory remains in the Step 0 section above.

### Step 1 — affected studies
- 4,407 total studies; 450 affected (10.21%); 9 gold studies affected.
- train_series.csv: 24,371 rows, 0 missing Fluid_Sensitive flags. test_series.csv: 15 rows, 3 studies, 0 missing flags.
- Applying the new slot-4 rule changes exactly those 450 studies.
- Affected list: delegation/results/T0Y_affected_studies.csv (StudyUID, old_series, new_series, is_gold).

### Step 2 — copy and patch
- Physical ds_axfs copy: approximately 59 GB. 584 GB free remained after copying.
- Copy verification passed: all_masks.npy SHA-256 9efda498c68c1d6ea26afb9463a468170905d31de3abb640cff44e071431a93b; all_ids.npy SHA-256 0eefe4a70605e35adec7d9e53685a841c534d7b9c0c51894c6a2c47756bf6c60. 200 sampled all_vols.npy studies were byte-identical before patching.
- Full-study rebuild of all 450 affected studies completed without a study failure. Slots 0–3 and masks 0–77 were byte-identical for all 450.
- Patched only slot 4 slices/masks 78–95 in ds_axfs. 20/20 random affected studies use an axial fluid-sensitive series and contain valid pixels; 20/20 random unaffected studies remain byte-identical across all 96 slices and masks.
- The original ds/corpus96 and its root links remain unchanged.

### Step 3 — parity
- rsna_preproc_axfs picker: 22,035/22,035 slot series match (100%).
- kaggle_students/fleet_v6_axfs.py picker: 22,035/22,035 slot series match (100%).
- Both pickers ran on all 3 dummy test studies (15 test-series rows); no missing fluid flags.
- Raw-DICOM end-to-end preprocessing on 10 affected + 10 unaffected studies; repository tolerance max absolute difference <= 1:
  - Slot 0: max 1; mean 0.00029386
  - Slot 1: max 1; mean 0.00010646
  - Slot 2: max 1; mean 0.00015884
  - Slot 3: max 1; mean 0.00008446
  - Slot 4: max 1; mean 0.00088652
- Failures: header decode 0; pixel decode 0; studies with a selected slot but no series files 0. All parity checks pass.

### Step 4 — prepared, not launched
- Launcher: run_distil_axfs.sh. Shell syntax check passed. The training loader copy is in train_axfs/train_knee.py.
- Exact command:
  export RAPTOR_INPUT=$HOME/projects/knee/ds_axfs RAPTOR_PRELOAD=1
  python train_axfs/train_knee.py --ckpt timm --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --bs 2 --accum 8 --k 12 --k_eval 94 --epochs 16 --patience 5 --win_chunk 14 --clean_windows --slot_emb --labels labels_distil_v1.parquet --seed 11 --tag v6d_axfs_s11
- This combines T05 clean-window and slot-embedding options in the same run. Launcher applies nice -n 19. It was not executed; no training was started.
- New files: build_corpus96_stream_axfs.py, rsna_preproc_axfs.py, kaggle_students/fleet_v6_axfs.py, train_axfs/train_knee.py, run_distil_axfs.sh, and ds_axfs/slots_axfs.csv.
- Runtime: full copy approximately 2 minutes; affected rebuild/determinism approximately 3 minutes after ZIP indexing; raw inference parity 0.71 minutes; loader compatibility check under 1 second. Rebuild study failures 0/450; raw parity header/pixel failures 0/20 studies.

FINAL STATUS: READY — AXFS corpus copy/patch, unchanged-loader input discovery, both picker parities, dummy test picker, and raw-DICOM pixel parity all passed. Training launcher is prepared and intentionally not launched.

