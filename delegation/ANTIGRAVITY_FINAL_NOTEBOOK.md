# Task: build "Final 2" Kaggle submission notebook (RSNA Knee Abnormality Detection)

Deadline: Oct 22 2026 23:59 UTC (Oct 23 05:29 IST). Kaggle queue is 4-6 h, so the notebook must be ready by Wed Oct 21.
Repo: /home/codemaster29/Documents/still_learning/rsna-knee-abnormality-detection. Read chat.md sections 72-79 first
(newest facts), then memory notes. Do NOT submit; the user submits. Do NOT add Co-Authored-By lines to commits.

## Goal
Final 1 already exists = our best public recipe, 0.958 ("blendmine" fork, see chat.md: -w1 R6 dataset, bundle
fe055406fad6a12f, FLIP_TTA True, per-finding W_RAPTOR_C). Do not change it.
Build Final 2 = that same notebook + our own ConvNeXt group swapped to the new all-data models, rank-blended at
weight W (prepare two versions: W 0.2 and W 0.1; earlier W 0.3 scored 0.956, W 0.2 scored 0.957).

## What goes into our own-model group
1. All-data ConvNeXt seeds, round-2 labels: seeds 310, 311, 312, 313, 314, 315 (gold-58: 0.9312, 0.9346, 0.9307 SWA,
   0.9257, 0.9309, 315 = idea-9 PU weights, check its json). 310/311/314 are on Kaggle; 312 is on the PC
   (pcr2/raptor_ft_cnxt_r2_all_s312*.pt), 313 on node1 (~/projects/knee/pc_code192/), 315 should be at Kaggle dataset
   knee-cnxt-pu-all-s315. Verify each exists; upload missing ones as private datasets (shivoo29/...).
   Use the SWA checkpoint where one exists. If runtime does not fit, drop 313 first (weakest).
2. Axial slot prefers fat-suppressed series in the TEST corpus builder: when choosing the series for the axial slot,
   prefer Fluid_Sensitive==1 (fallback: any axial). Measured on 97 fold-0 studies: PF OA 0.878->0.893, Lateral OA
   0.838->0.852, Effusion 0.854->0.868, no losses. On test there is no series csv with flags: derive fluid-sensitive from
   DICOM headers the same way the notebook already classifies sagittal/coronal FS (check how it fills the FS slots).
3. Lateral OA scanner adjustment (only our model's Lateral OA column, before rank blending):
   node1 ~/projects/knee/cpu_ideas/latoa_scanner_prior.json (copy it locally). key = Manufacturer + "|" +
   ManufacturerModelName from the DICOM header; new_logit = coef_logit * logit(p) + coef_prior * prior[key]
   (unknown key -> prior 0 or the json's default; read scanner_prior.py on node1 to match exactly). Gain: OOF 0.841->0.851.
4. Keep flip TTA as the base notebook does for its own groups; for our group, follow whatever the existing own-model
   group does (chat.md: no-TTA test was 0.9284 vs 0.9289, TTA costs ~50% time).

## Hard constraints
- 9 h Kaggle limit, 2x T4: keep the existing KILL_AT_H 8.4 guard; budget ~1.0 h per 4-ckpt ConvNeXt group without TTA.
  Run on the 3 visible test studies, extrapolate s/study to ~1,300 studies, report the projection.
- fp16 (not bf16) on T4. Decode DICOMs once.
- No fastMRI- or OAI-derived models in Final 2 (memory rule). Do not touch report text.
- Never print or log Kaggle keys/tokens; never put them in cells or files.
- Do not delete anything on node1/PC without asking the user.

## Deliverables
- Two notebook versions (W 0.2, W 0.1) pushed as private Kaggle kernels, commit run COMPLETE on the visible test,
  submission.csv shape and no NaNs checked, runtime projection < 8 h.
- A short chat.md section 80: datasets used (names + versions), checkpoints, W, runtime projection, what was dropped.
- Tell the user which version to submit first (W 0.2) and the final-pair plan: Final 1 = existing 0.958 recipe,
  Final 2 = whichever of W 0.2 / W 0.1 scores higher.
