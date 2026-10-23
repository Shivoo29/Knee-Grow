---
license: mit
library_name: kneeseg
tags:
- medical
- medical-segmentation
- mri
- segmentation
- knee
- ski10
- random-forest
metrics:
- dice
---

# Knee Bone & Cartilage Segmentation Models (SKI10 Subchondral Surface Refiner & Articular Atlas)

This repository contains trained models for 4-structure 3D MRI Knee Bone and Cartilage Segmentation (`Femur`, `Femoral Cartilage`, `Tibia`, `Tibial Cartilage`) on the [SKI10 dataset](https://ski10.grand-challenge.org/), using the `kneeseg` Python package (`>= 0.2.4`).

- **GitHub**: [https://github.com/wq2012/kneeseg](https://github.com/wq2012/kneeseg)
- **PyPI**: [https://pypi.org/project/kneeseg/](https://pypi.org/project/kneeseg/)
- **Hugging Face Collection**: [Knee MRI Segmentation Collection](https://huggingface.co/collections/wq2012/knee-mri-segmentation)
- **Sibling Models**:
  - [wq2012/knee_3d_mri_segmentation_SKI10_subchondral_atlas](https://huggingface.co/wq2012/knee_3d_mri_segmentation_SKI10_subchondral_atlas) (**This repository — Recommended SKI10 4-structure model**)
  - [wq2012/knee_3d_mri_segmentation_OAI_downsampled_subchondral_atlas](https://huggingface.co/wq2012/knee_3d_mri_segmentation_OAI_downsampled_subchondral_atlas) (Recommended OAI 6-structure model)
  - [wq2012/knee_3d_mri_segmentation_SKI10_OAI_subchondral_atlas](https://huggingface.co/wq2012/knee_3d_mri_segmentation_SKI10_OAI_subchondral_atlas) (Unified Cross-Domain SKI10 + OAI model)
  - [wq2012/knee_3d_mri_segmentation_SKI10](https://huggingface.co/wq2012/knee_3d_mri_segmentation_SKI10) (Baseline v1 SKI10 model)
  - [wq2012/knee_3d_mri_segmentation_OAI_downsampled](https://huggingface.co/wq2012/knee_3d_mri_segmentation_OAI_downsampled) (Baseline v1 OAI downsampled model)

## Model Details

- **Architecture**:
  1. **Bone Pass 1 (`bone_rf_p1.joblib`, `stage='v2_p1'`, `10.07 MB`)**: Joint-Anchored Physical-Space Random Forest (`62` features: percentile-normalized MRI intensity, multi-scale physical Gaussian smoothed intensities and gradient magnitudes, joint-anchored 3D coordinate frames, 3D probabilistic bone atlases, and physical-millimeter Random Shift Intensity Differences) cascaded with a 1D Cortical Ray Enclosure Global Corrector.
  2. **Bone Pass 2 + Subchondral Surface Refiner (`bone_rf_p2.joblib`, `stage='v2_p2'`, `292.81 MB`)**: Full-resolution Boundary-Shell Auto-Context Forest (`77` features) cascaded with a binary (`classes_ = [0, 1]`, where `0 = Non-Bone` and `1 = Bone`, in contrast to the 4-class multi-bone baseline formulation in `BoneClassifier.train()`) Subchondral Articular Surface Refiner (`RefBone V1`: `108` features; `RefBone V2`: `156` surface-normal ray intensity/gradient features + dual whole-shell `hgbt_all`/`et_all` and articular subchondral `hgbt_sub`/`et_sub` HistGradientBoosting / ExtraTrees ensembles, blended `0.35 * V1 + 0.65 * V2` for femur and `0.50 * V1 + 0.50 * V2` for tibia) that resolves partial-volume blending along the femoral condyles and tibial plateau.
  3. **Cartilage Pass 1 (`cartilage_rf_p1.joblib`, `stage='v6_p1'`, `42.05 MB`)**: 4-Class Compartment-Specialist Semantic Context Forest + HistGradientBoosting / ExtraTrees ensemble (`183` features; `0=Background`, `1=Subchondral Bone/Meniscus`, `2=Target Cartilage`, `3=Opposing Cartilage`) using exact physical-space Signed Distance Transforms (`d_F, d_T`), surface-normal ray intensity profiles (`s in [-2.5, +4.0] mm`) capturing the dark cortical bone plate and bright cartilage layer, 4D articular coordinate probabilistic atlases (`36 x 36 x 28 x 24` medial/lateral compartment frames), 18 anatomical surface landmarks per bone, structure tensor eigenvalues, and physical-space RSID.
  4. **Cartilage Pass 2 + Out-Of-Fold Subchondral Calibrator (`cartilage_rf_p2.joblib`, `stage='v6_p2'`, `1,212.65 MB`)**: Dual-Bone-Geometry Auto-Context Ensemble (`m_v5` with `481` features on `RefBone V1` geometry, `m_v6` with `481` features on `RefBone V2` geometry, and `m_v7` tibial specialist with `429`/`597` features) + 4-fold Out-Of-Fold multi-family Stage-3 boundary classifier (`127`/`137` features in `s3_clean`, `151`/`167` features in `f3_models`, and `103` features in `pure_models`) + `StandardScaler + Ridge` calibrator (`calib_model`, fit strictly on the 80 training scans' Out-Of-Fold predictions with base threshold `th_base = 0.508`, selecting `10` of `45` scan-level statistics with `Ridge(alpha=20.0)` for `th` and `9` of `17` statistics with `Ridge(alpha=10.0)` for `bp`, plus a Pass-2 probability-weighted bonus `- 0.12 * p_t2` for tibial cartilage; femoral cartilage uses fixed `w = 0.25, bp = 0.14, th = 0.40`).
- **Labels**:
  - `0`: Background
  - `1`: Femur
  - `2`: Femoral Cartilage
  - `3`: Tibia
  - `4`: Tibial Cartilage
- **Dataset Split (`ski10_full_split.json`)**:
  - **Training**: Strictly trained on **80 cases** from `SKI10/TrainingData` using 4-fold Out-Of-Fold stacking and Leave-One-Out atlas construction.
  - **Evaluation**: Evaluated on the **20 held-out cases** from `SKI10/TrainingData` with **zero train-eval overlap** and zero disk-cache lookups. Because ground-truth labels for the 50 official `SKI10/TestData` scans are not publicly released, all quantitative SKI10 ablation and reporting are conducted on this 80/20 split of the 100 labeled `TrainingData` scans (all tree weights, LOO 4D atlases, and OOF Ridge calibrator coefficients are fit strictly on the 80 training scans, while ensemble family architecture and global blend weights were selected on the 80/20 split).
- **Inference Compute Requirements**:
  - **Hardware**: Pure CPU (`numpy`, `scipy`, `scikit-learn`); no GPU required.
  - **Runtime & Peak RAM**: `~45–90 seconds` per volume on an 8-core CPU (`n_jobs=-1`), `~4.0–5.5 GB` peak RAM (`~1.56 GB` total compressed checkpoint footprint).

## Performance Comparison (Strict 80/20 Non-Overlapping SKI10 Split)

Evaluated across the 20 held-out evaluation volumes (`ski10_full_split.json`, saved in `evaluation_report.json` and `EvalData_labels/`). Both population standard deviation (`ddof=0`, matching `np.std` in `evaluation_report.json`) and sample standard deviation (`ddof=1`) are reported alongside 10,000-resample percentile bootstrap 95% confidence intervals (`seed=42`):

| Structure | Baseline v1 DSC (80/20 Split) | Subchondral Atlas DSC (`ddof=0` / `ddof=1`) | 95% Bootstrap CI | Absolute Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Femur (`1`)** | 0.9046 ± 0.0361 | **0.9689 ± 0.0161** (`0.0165` `ddof=1`) | `[0.9613, 0.9751]` | **+0.0643 (+6.43%)** |
| **Tibia (`3`)** | 0.9292 ± 0.0260 | **0.9742 ± 0.0080** (`0.0082` `ddof=1`) | `[0.9705, 0.9774]` | **+0.0450 (+4.50%)** |
| **Femoral Cartilage (`2`)** | 0.6767 ± 0.0481 | **0.8026 ± 0.0504** (`0.0517` `ddof=1`) | `[0.7793, 0.8225]` | **+0.1259 (+12.59%)** |
| **Tibial Cartilage (`4`)** | 0.6411 ± 0.0540 | **0.8001 ± 0.0404** (`0.0414` `ddof=1`) | `[0.7815, 0.8165]` | **+0.1590 (+15.90%)** |
| **2-Bone Mean** | 0.9169 | **0.9715** | — | **+0.0546 (+5.46%)** |
| **2-Cartilage Mean** | 0.6589 | **0.8013** | — | **+0.1424 (+14.24%)** |
| **Overall 4-Structure Mean** | 0.7879 | **0.8864** | — | **+0.0985 (+9.85%)** |

> **Note on v1 (`wq2012/knee_3d_mri_segmentation_SKI10`)**: The baseline v1 model on the strict 80/20 split achieved `0.9046 / 0.9292 / 0.6767 / 0.6411` DSC (and `0.9155 / 0.9383 / 0.7095 / 0.6799` DSC when evaluated with train-eval overlap on 100 training scans). This Subchondral Atlas model achieves **`0.9689 / 0.9742 / 0.8026 / 0.8001` DSC** on the strict 80/20 held-out split with zero train-eval overlap.

### Per-Case Held-Out Evaluation Results (20 Test Volumes)

| Volume | Femur DSC | Tibia DSC | Femoral Cartilage DSC | Tibial Cartilage DSC |
| :--- | :--- | :--- | :--- | :--- |
| `image-004.mhd` | 0.9577 | 0.9749 | 0.8064 | 0.8111 |
| `image-005.mhd` | 0.9241 | 0.9787 | 0.8406 | 0.8083 |
| `image-012.mhd` | 0.9679 | 0.9737 | 0.8247 | 0.8087 |
| `image-014.mhd` | 0.9854 | 0.9830 | 0.8560 | 0.8507 |
| `image-015.mhd` | 0.9814 | 0.9753 | 0.8423 | 0.7918 |
| `image-018.mhd` | 0.9681 | 0.9774 | 0.7108 | 0.8099 |
| `image-028.mhd` | 0.9694 | 0.9736 | 0.8181 | 0.7477 |
| `image-029.mhd` | 0.9824 | 0.9790 | 0.8340 | 0.7965 |
| `image-030.mhd` | 0.9660 | 0.9834 | 0.7313 | 0.7727 |
| `image-032.mhd` | 0.9755 | 0.9625 | 0.7190 | 0.7198 |
| `image-036.mhd` | 0.9689 | 0.9806 | 0.8013 | 0.8529 |
| `image-055.mhd` | 0.9665 | 0.9752 | 0.8000 | 0.8352 |
| `image-065.mhd` | 0.9806 | 0.9549 | 0.8617 | 0.8360 |
| `image-070.mhd` | 0.9821 | 0.9819 | 0.8459 | 0.8239 |
| `image-076.mhd` | 0.9827 | 0.9710 | 0.8192 | 0.8390 |
| `image-082.mhd` | 0.9678 | 0.9851 | 0.8108 | 0.7929 |
| `image-087.mhd` | 0.9790 | 0.9784 | 0.8254 | 0.8213 |
| `image-089.mhd` | 0.9275 | 0.9584 | 0.6774 | 0.6985 |
| `image-095.mhd` | 0.9684 | 0.9680 | 0.7954 | 0.7659 |
| `image-098.mhd` | 0.9762 | 0.9686 | 0.8315 | 0.8181 |
| **Mean ± Std (`ddof=0` / `ddof=1`)** | **0.9689 ± 0.0161 (`0.0165`)** | **0.9742 ± 0.0080 (`0.0082`)** | **0.8026 ± 0.0504 (`0.0517`)** | **0.8001 ± 0.0404 (`0.0414`)** |

## Usage

### 1. Install Library
```bash
pip install --upgrade kneeseg
```

### 2. Load Models and Run End-to-End Inference
```python
import numpy as np
from huggingface_hub import hf_hub_download
from kneeseg.io import load_volume, save_volume
from kneeseg.bone_rf import BoneClassifier
from kneeseg.rf_seg import CartilageClassifier

REPO_ID = "wq2012/knee_3d_mri_segmentation_SKI10_subchondral_atlas"

# 1. Download and load the 4 cascade checkpoints
bone_p1 = BoneClassifier()
bone_p2 = BoneClassifier()
cart_p1 = CartilageClassifier()
cart_p2 = CartilageClassifier()

bone_p1.load(hf_hub_download(repo_id=REPO_ID, filename="bone_rf_p1.joblib"))
bone_p2.load(hf_hub_download(repo_id=REPO_ID, filename="bone_rf_p2.joblib"))
cart_p1.load(hf_hub_download(repo_id=REPO_ID, filename="cartilage_rf_p1.joblib"))
cart_p2.load(hf_hub_download(repo_id=REPO_ID, filename="cartilage_rf_p2.joblib"))

# 2. Load a 3D knee MRI volume (.mhd/.raw)
image, spacing = load_volume("path/to/image-004.mhd", return_spacing=True)

# 3. Run Bone Pass 1 -> Bone Pass 2 (with Subchondral Surface Refiner)
#    Note: bone_p2.predict returns (labels_3d, continuous_prob_map_4d)
_, prob_b1 = bone_p1.predict(image, spacing=spacing)
pred_b2, prob_b2 = bone_p2.predict(image, prob_map=prob_b1, spacing=spacing)
bone_masks = {
    name: (pred_b2 == lbl).astype(np.uint8)
    for name, lbl in bone_p2.bone_Map.items()
}

# 4. Run Cartilage Pass 1 -> Cartilage Pass 2 (with Stage 3 & Subchondral Calibrator)
prob_c1 = cart_p1.predict_proba_map(image, bone_masks, spacing=spacing)
cart_pred, _ = cart_p2.predict(image, bone_masks, prob_map=prob_c1, spacing=spacing)

# 5. Merge into final multi-class label volume (1=Femur, 2=FemCart, 3=Tibia, 4=TibCart, 5=Patella, 6=PatCart)
final_seg = np.zeros_like(pred_b2, dtype=np.uint8)
final_seg[pred_b2 == 1] = 1
final_seg[pred_b2 == 2] = 3
final_seg[pred_b2 == 3] = 5
mask_cart = cart_pred > 0
final_seg[mask_cart] = cart_pred[mask_cart]
save_volume(final_seg, "prediction-004.mhd", metadata={"spacing": spacing})
```

## Files & Checkpoint Footprint (`1.56 GB` Total)

| File | Size | Description |
| :--- | :--- | :--- |
| `bone_rf_p1.joblib` | `10.07 MB` (`9.61 MiB`) | Bone Segmentation Pass 1 (`stage='v2_p1'`, Joint-Anchored Physical-Space Forest + 1D Cortical Ray Corrector) |
| `bone_rf_p2.joblib` | `292.81 MB` (`279.25 MiB`) | Bone Segmentation Pass 2 (`stage='v2_p2'`, Boundary-Shell Auto-Context Forest + Binary Subchondral Surface Refiner `RefBone V1 + V2`) |
| `cartilage_rf_p1.joblib` | `42.05 MB` (`40.10 MiB`) | Cartilage Segmentation Pass 1 (`stage='v6_p1'`, 4-Class Compartment-Specialist Semantic Context Forest + 4D Articular Atlas) |
| `cartilage_rf_p2.joblib` | `1,212.65 MB` (`1,156.47 MiB`) | Cartilage Segmentation Pass 2 (`stage='v6_p2'`, Auto-Context Ensemble + Stage-3 Boundary Classifier + Ridge Calibrator) |
| `ski10_full_split.json` | `2.14 KB` | Exact 80-train / 20-eval split definition |
| `evaluation_report.json` | `4.90 KB` | Verified per-structure and per-case DSC evaluation metrics across the 20 held-out SKI10 test cases |
| `EvalData_labels/` | `20 cases` | Predicted `.mhd/.raw` label volumes for the 20 held-out SKI10 evaluation cases |

## Citation

**Plain Text:**

> Quan Wang, Dijia Wu, Le Lu, Meizhu Liu, Kim L. Boyer, and Shaohua Kevin Zhou.
> "Semantic Context Forests for Learning-Based Knee Cartilage Segmentation in 3D MR Images."
> MICCAI 2013: Workshop on Medical Computer Vision.

> Quan Wang.
> Exploiting Geometric and Spatial Constraints for Vision and Lighting Applications.
> Ph.D. dissertation, Rensselaer Polytechnic Institute, 2014.

**BibTeX:**

```bibtex
@inproceedings{wang2013semantic,
  title={Semantic context forests for learning-based knee cartilage segmentation in 3D MR images},
  author={Wang, Quan and Wu, Dijia and Lu, Le and Liu, Meizhu and Boyer, Kim L and Zhou, Shaohua Kevin},
  booktitle={International MICCAI Workshop on Medical Computer Vision},
  pages={105--115},
  year={2013},
  organization={Springer}
}

@phdthesis{wang2014exploiting,
  title={Exploiting Geometric and Spatial Constraints for Vision and Lighting Applications},
  author={Quan Wang},
  year={2014},
  school={Rensselaer Polytechnic Institute}
}
```
