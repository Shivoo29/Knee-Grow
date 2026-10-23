# Phase 0 Teacher-Routing Audit Report

**Date / Timestamp:** 2026-10-16T12:03:01.767650  
**Target Server Alias:** `knee` (`node1.intelunnati`)  
**Audit Status:** STOPPED / INCONCLUSIVE (Rule 5 Triggered)

---

## 1. Environment & Script Preflight
- **SSH Alias:** `knee` (configured in `~/.ssh/config` pointing to `node1.intelunnati`).
- **Script Located:** `/home/s05814802723/projects/knee/routing_check.py`
- **Script Input Requirements:**
  - `--teacher NAME=PATH`: Study-level OOF prediction NPZ files with keys `pred`, `truth`, `ids`.
  - Predictions must be out-of-fold probabilities on `[0, 1]`.
  - Truth must be binary `{0, 1}`.
  - `--sites PATH`: CSV containing study IDs and site cluster IDs.
  - Requires >= 4 unique site groups for site-grouped split-half routing and hierarchical bootstrap.

---

## 2. Inventory of Candidate Teachers
From the repository metadata (`raptor_ft_*.json`, `STATUS.md`, `sk_update_10oct.md`, and `pc_ckpts/`), seven candidate teacher models were identified:

1. **`coat224_v1`**: CoAtNet-1 @ 224px (44 slices, seed 0). Trained on `labels_avg.parquet` on all $N=4,349$ training studies. Evaluated solely on Gold-58 ($N=58$).
2. **`r50_v1`**: ResNet-50 @ 384px (44 slices, seed 0). Trained on `labels_avg.parquet` on all $N=4,349$ training studies. Evaluated solely on Gold-58 ($N=58$).
3. **`convnext_s1`**: ConvNeXt-Small @ 384px (44 slices, seed 1). Trained on `labels_avg.parquet` on all $N=4,349$ training studies. Evaluated solely on Gold-58 ($N=58$).
4. **`convnext_s2`**: ConvNeXt-Small @ 384px (44 slices, seed 2). Trained on `labels_avg.parquet` on all $N=4,349$ training studies. Evaluated solely on Gold-58 ($N=58$).
5. **`effnetv2_s1`**: EfficientNetV2-S @ 384px (44 slices, seed 1). Trained on `labels_avg.parquet` on all $N=4,349$ training studies. Evaluated solely on Gold-58 ($N=58$).
6. **`dinov2_s11`**: DINOv2-Small @ 384px (44 slices, seed 11). Trained on `labels_avg.parquet` on all $N=4,349$ training studies. Evaluated solely on Gold-58 ($N=58$).
7. **`v6_s4`** (or `v5` / `v6` fleet): CoAtNet-2 @ 384px (96 slices, seed 4). Trained on `labels_avg.parquet` on all $N=4,349$ training studies. Evaluated solely on Gold-58 ($N=58$).

---

## 3. OOF and Prediction Scale Verification
- **Study ID & Scale Alignment:** The 58 gold studies across all `raptor_gold_*.npz` files share identical study IDs, 12 RSNA labels, and probabilities in [0, 1].
- **Out-Of-Fold Check:**
  - **None** of the candidate teachers possess complete 5-fold or site-grouped out-of-fold (OOF) predictions across the study cohort.
  - All candidate teachers were trained on the entire $N=4,349$ training set without K-fold partitioning.
  - Training set predictions for these teachers are in-sample (memorized) and violate Rule 4: *"Check that the predictions are genuinely out-of-fold for each study. Do not use in-sample predictions or predictions from a teacher trained on its evaluation fold."*
  - The only OOF NPZ files found in the project are incomplete single-fold fragments:
    - `raptor_oof_coat_roi_5fold_f0_fold0.npz` (Fold 0 only, $N=873$)
    - `raptor_oof_test_smoke_compartment_fold0.npz` (Fold 0 only, $N=873$)
    - `pc_code/raptor_oof_oof4_f2_fold2.npz` (Fold 2 only, $N=1087$)
    - `pc_code/raptor_oof_oof4_f3_fold3.npz` (Fold 3 only, $N=1087$)

---

## 4. Audit Rule 5 Enforcement & Decision
> **Audit Rule 5:**
> *"If the available score matrix is only from the 58 gold studies, or if site-grouped OOF predictions/group IDs are unavailable, stop and report that Phase 0 cannot support a routing decision. Do not substitute the gold-58 table."*

### Decision: **INCONCLUSIVE / HALT**
1. **Rule 5 Triggered:** The only prediction matrix available across multiple teachers is strictly limited to the 58 gold studies (`raptor_gold_*.npz`).
2. **Gold-58 Substitution Prohibited:** As established in the codebase documentation (`docs/status.md`, `sk_update_10oct.md`), gold-58 sample size ($N=58$) has a high standard error per finding (+/- 0.02 to +/- 0.07 macro-AUC). Substituting gold-58 would produce selection-biased, unstable winners that do not generalize to held-out sites.
3. **Recommendation:**
   - Drop per-label teacher routing.
   - Use uniform ensemble-mean targets (rank or probability mean) across models for distillation.
   - Do not launch Phase 1 or student training under a per-label routed scheme until genuine site-grouped 5-fold OOF predictions are generated.
