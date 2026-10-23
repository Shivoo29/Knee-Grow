# Mirror Evidence Report

**Date:** October 19, 2026  
**Environment:** Unnati Cluster Node 1 (`172.27.92.5`), `/home/s05814802723/projects/knee`  
**Mode:** READ-ONLY Forensic Audit (No GPU, no training, no inference, tmux untouched, existing files unmodified).

---

## ITEM 1: SEARCH FOR EVALUATIONS COMPARING MIRROR TTA VS WITHOUT TTA
**Status:** **NOT CHECKED**

* **Investigation Scope:**
  Searched all log files (`*.log`), text files (`*.txt`, `*.out`), receipts (`*.json`), and markdown files (`*.md`) across the entire repository for evaluations comparing a model with mirror TTA (`anatomical_mirror.py`, `eval_res_tta.py`) vs without TTA, or forward-pass vs mirror-pass predictions.
* **Findings:**
  1. **`eval_res_tta.py` Attempted but Crashed:**
     - Script `/home/s05814802723/projects/knee/eval_res_tta.py` was constructed to evaluate 96-slice models on Gold-58 with and without mirror TTA across resolutions 384, 352, and 320.
     - However, the script failed on line 35 upon execution, as documented in `/home/s05814802723/projects/knee/eval_res_tta.log`:
       ```python
       RuntimeError: The expanded size of the tensor (484) must match the existing size (576) at non-singleton dimension 3. Target sizes: [94, 8, 484, 484]. Tensor sizes: [1, 8, 576, 576]
       ```
       Due to this relative position bias dimension mismatch in CoAtNet at non-384 resolutions, the script terminated at epoch 0 and never output or saved any comparative AUC metrics.
  2. **`anatomical_mirror.py` Docstring Claim:**
     - The docstring of `/home/s05814802723/projects/knee/anatomical_mirror.py:13-14` states:
       `"Empirically proven to yield a consistent +0.003 to +0.004 macro-AUC boost on the public LB."`
     - However, there are zero saved result files, logs, or tables containing the corresponding offline Gold-58 or OOF per-class breakdown for forward-only vs mirror-only or with-TTA vs without-TTA.
* **Reason for NOT CHECKED:** No completed offline evaluation comparing mirror TTA vs without TTA was ever logged or saved to disk.

---

## ITEM 2: AUDIT OF SAVED .NPZ PREDICTION FILES FOR FORWARD VS MIRROR ARRAYS
**Status:** **NOT CHECKED**

* **Investigation Scope:**
  Audited all 221 saved `.npz` prediction and evaluation files across `/home/s05814802723/projects/knee/`, `/home/s05814802723/projects/knee/pc_code/`, and `/home/s05814802723/projects/knee/pc_code192/` to inspect their internal dictionary keys.
* **Findings:**
  - Zero `.npz` files contain separate forward-only and mirror-only prediction arrays (e.g., keys such as `fwd`, `mirror`, `forward`, `tta`, or `plain`).
  - All saved prediction files contain only a single 12-column output prediction array under the standard keys `'pred'` or `'preds'`.
* **Reason for NOT CHECKED:** Separate forward-pass and mirror-pass prediction arrays were not saved in any checkpoint `.npz` file.

---

## ITEM 3: RUN SCRIPTS, MIRROR_P VALUES, AND IDENTICAL ARCHITECTURE/FOLD COMPARISON
**Status:** **VERIFIED**

### 3.1 Catalog of Run Scripts and Configured `mirror_p`

| Script Path | Model Family / Architecture | Configured `mirror_p` | Target Tag |
| :--- | :--- | :---: | :--- |
| `pc_code192/run_cnxt_sev2_folds.sh` | ConvNeXt-Tiny (`convnext_tiny.in12k_ft_in1k`, 192 slices, 2-scale zoom) | **0.5** | `cnxt_sev2_f$f` (Folds 0, 1, 2, 3) |
| `pc_code192/run_breakthrough_model.sh` | ConvNeXt-Small (`convnext_small.fb_in22k_ft_in1k`, 192 slices, 2-scale zoom) | **0.5** | `cnxt_s_breakthrough` |
| `pc_code192/run_cnxt_folds.sh` | ConvNeXt-Tiny (`convnext_tiny.in12k_ft_in1k`, 192 slices, 2-scale zoom) | **0.5** | `cnxt_f$f` (Folds 0, 1, 2, 3) |
| `pc_code192/run_cnxt_r2.sh` | ConvNeXt-Tiny (`convnext_tiny.in12k_ft_in1k`, 192 slices, 2-scale zoom) | **0.5** | `cnxt_r2_f$f` |
| `pc_code192/run_cnxt_r2s2.sh` | ConvNeXt-Tiny (`convnext_tiny.in12k_ft_in1k`, 192 slices, 2-scale zoom) | **0.5** | `cnxt_r2s2_f$f` |
| `pc_code192/run_cnxt_r2all.sh` | ConvNeXt-Tiny (`convnext_tiny.in12k_ft_in1k`, 192 slices, 2-scale zoom) | **0.5** | `cnxt_r2_all_s$s$` |
| `pc_code192/run_node1_192k24.sh` | CoAtNet-384 (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, 192 slices) | **0.5** | `d192k24_s$S` (stopped ep 0) |
| `pc_code/run_node1_oof.sh` | CoAtNet-384 (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, 96 slices) | **0.5** | `oof4_f$f` (Folds 2, 3) |
| `pc_code/run_node1_final.sh` | CoAtNet-384 (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, 96 slices) | **0.5** | `final_s$S` |
| `pc_code/run_node1_recipe.sh` | CoAtNet-384 (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, 96 slices) | **0.5** | `recipe_s22` |
| `pc_code192/run_node1_192.sh` | CoAtNet-384 (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, 192 slices) | **0.0 (default)** | `d192_s$S` (Seed 50 completed) |
| `run_distil24.sh` | CoAtNet-384 (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, 96 slices) | **0.0 (default)** | `v6d24_s9` |
| `run_distil.sh` | CoAtNet-384 (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, 96 slices) | **0.0 (default)** | `v6d_s7` |
| `run_v6_seeds.sh` | CoAtNet-384 (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, 96 slices) | **0.0 (default)** | `v6_s3`, `v6_s4`, `v6_s5` |
| `run_5fold_fast_champion.sh` | CoAtNet-384 (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, 96 slices) | **0.0 (default)** | `coat_roi_5fold_f$f` |
| `run_maxvit_c96.sh` | MaxViT-Base (`maxvit_base_tf_384.in1k`, 96 slices) | **0.0 (default)** | `maxvit_b_96` |
| `train_specialist_f0.sh` | CoAtNet-384 (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, 96 slices) | **0.0 (default)** | `coat_latoa_roi_f0` |

### 3.2 Is There a Run with `mirror_p=0` on the Same Architecture and Fold as a `mirror_p=0.5` Run?
* **Answer:** **NO.**
* **Details:**
  1. **ConvNeXt-Tiny (192 slices, two-scale zoom):** Every completed fold run (`cnxt_f0`, `cnxt_f1`, `cnxt_sev2_f0`, `cnxt_sev2_f1`, `cnxt_sev2_f2`, `cnxt_r2_f2`, `cnxt_r2s2_f0`, `cnxt_r2s2_f1`) was launched via scripts containing `--mirror_p 0.5`. There was no run of this architecture and fold configuration with `mirror_p=0.0`. The only ConvNeXt run with `mirror_p=0.0` was `convnext_s1`, which was a completely different configuration (96 slices global 384x384, single-scale, no fold CV, full 4,349 training).
  2. **CoAtNet-384 (192 slices):** `raptor_ft_d192_s50` was run with `mirror_p=0.0` (default). Its paired script `run_node1_192k24.sh` (`--mirror_p 0.5`, seed 51) was started but terminated at epoch 0 (`train_coat_d192k24_s51.log`, 353 bytes), leaving zero completed epochs or predictions.
  3. **CoAtNet-384 (96 slices):** The runs with `mirror_p=0.5` (`oof4_f2`, `oof4_f3`, `final_s41`) were trained under the `pc_code/` recipe (distilled soft targets, 4-fold site split), whereas the `mirror_p=0.0` runs (`v6d_s7`, `v6d24_s9`, `v6_s3-s5`) were trained under the earlier full-corpus student distillation pipeline without fold CV.
* **Conclusion:** No identical architecture and fold pair exists with one trained at `mirror_p=0` and the other at `mirror_p=0.5`.

---

## ITEM 4: AUDIT OF `train_specialist_pfoa.py:59`
**Status:** **VERIFIED**

In `/home/s05814802723/projects/knee/train_specialist_pfoa.py`:
```python
43: def __getitem__(self, idx):
44:     uid = self.ids[idx]
45:     vol = self.vols[idx].astype(np.float32) / 255.0  # (18, 224, 224)
46:     K, H, W = vol.shape
...
57:     if self.train:
58:         # Horizontal flip augmentation
59:         if np.random.rand() > 0.5:
60:             x = np.flip(x, axis=-1).copy()
```

### 4.1 Which Array Axis is Flipped?
* **Array Axis:** **`axis=-1`** (the trailing dimension of the `(18, 3, 224, 224)` tensor `x`, corresponding to the image width / column dimension).

### 4.2 What That Axis Represents Anatomically:
* **For Axial Input:**
  - `AxialDataset` loads `axial_vols.npy` (18-slice transverse acquisition covering the patellofemoral joint).
  - From DICOM `ImageOrientationPatient` for axial series (`pd_tse_tra_d`), row vector direction cosines are `[0.988, 0.145, 0.053]`.
  - The image horizontal width axis (`axis=-1`) is aligned with the **Patient Left-Right axis** ($+X$).
  - **Anatomical effect:** Flipping `axis=-1` on Axial slices inverts the **Medial Patellar Facet $\leftrightarrow$ Lateral Patellar Facet** and reflects the lateral trochlear ridge onto the medial side.
* **For Sagittal Input:**
  - From DICOM `ImageOrientationPatient` for sagittal series (`pd_tse_sag_d`), row vector direction cosines are `[0.158, 0.985, 0.066]`.
  - The image horizontal width axis (`axis=-1`) is aligned with the **Patient Anterior-Posterior axis** ($+Y$).
  - **Anatomical effect:** Flipping `axis=-1` on Sagittal slices inverts **Anterior $\leftrightarrow$ Posterior** (reflecting the anterior patella to the posterior popliteal fossa, creating a biologically impossible joint).

### 4.3 Do That Specialist's Predictions Feed Any Submission or Ensemble?
* **Answer:** **NO.**
* **Evidence:**
  - Grepping the entire repository confirms `/home/s05814802723/projects/knee/train_specialist_pfoa.py` is referenced **nowhere else**.
  - Its target checkpoint `/home/s05814802723/projects/knee/specialist_pfoa/specialist_pfoa_best.pt` does not exist on disk.
  - It does not participate in `submission_kernel.py`, `eval_cross_architecture_blend.py`, `print_comprehensive_label_comparison.py`, or any saved ensemble `.npz` file.

---
