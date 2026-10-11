# RSNA Knee Abnormality Detection 2026

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.11+](https://img.shields.io/badge/PyTorch-2.11+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

High-performance, top-tier solution pipeline for the **RSNA Knee Abnormality Detection Challenge** (Kaggle 2026). The system predicts 12 multi-label pathological abnormalities across 3D multi-planar MRI scans (Sagittal, Coronal, and Axial) using high-resolution hybrid vision backbones, dedicated anatomical specialists, compartment region-of-interest (ROI) pooling, and strict FP32 percentile-rank ensembling.

---

## Table of Contents
1. [Competition Objective & Findings](#competition-objective--findings)
2. [Architectural Highlights](#architectural-highlights)
3. [End-to-End Pipeline Overview](#end-to-end-pipeline-overview)
4. [3D Volume Preprocessing Engine](#3d-volume-preprocessing-engine)
5. [Model Architecture & Sovereign Fleet](#model-architecture--sovereign-fleet)
6. [Targeted Anatomical Specialists](#targeted-anatomical-specialists)
7. [Validation Strategy & 5-Fold Cross Validation](#validation-strategy--5-fold-cross-validation)
8. [Inference, TTA & Rank Blending](#inference-tta--rank-blending)
9. [Project Directory Layout](#project-directory-layout)
10. [Quickstart & CLI Reproduction](#quickstart--cli-reproduction)
11. [Benchmark Leaderboard & Ablations](#benchmark-leaderboard--ablations)
12. [Forensic Engineering & Safeguards](#forensic-engineering--safeguards)
13. [Hardware & Infrastructure](#hardware--infrastructure)
14. [Author & License](#author--license)

---

## Competition Objective & Findings

The goal is to predict the presence or absence of **12 pathological conditions** across 4,407 knee MRI examinations:

| Index | Finding | Clinical Category | Primary Anatomical View |
|:---:|:---|:---|:---|
| 0 | **ACL Tear** | Ligamentous Injury | Sagittal / Coronal |
| 1 | **MCL Injury** | Ligamentous Injury | Coronal |
| 2 | **Medial Meniscal Tear** | Meniscal Tear | Sagittal / Coronal |
| 3 | **Lateral Meniscal Tear** | Meniscal Tear | Sagittal / Coronal |
| 4 | **Medial Compartment OA** | Degenerative Joint Disease | Coronal |
| 5 | **Lateral Compartment OA** | Degenerative Joint Disease | Coronal |
| 6 | **Patellofemoral OA (PF OA)** | Degenerative Joint Disease | Axial (Patellofemoral joint) |
| 7 | **Joint Effusion** | Fluid Collection | Sagittal / Axial (Fluid sensitive) |
| 8 | **Synovitis** | Inflammatory Change | Axial / Sagittal |
| 9 | **Baker's Cyst** | Fluid Collection | Axial / Sagittal |
| 10 | **Bone Contusion** | Osseous Trauma | Sagittal / Coronal (Fat-suppressed) |
| 11 | **Fracture** | Osseous Trauma | Multi-planar |

* **Evaluation Metric**: Unweighted Macro ROC-AUC across all 12 abnormalities.
* **Label Challenge**: Only **58 studies** contain ground-truth human expert gold annotations; the remaining **4,349 studies** contain free-text unstructured radiology reports only.

---

## Architectural Highlights

- **ImagePositionPatient (IPP) 3D Normal Sorting**: Sorts DICOM slices along the true anatomical slice normal vector, preventing slice inversions caused by irregular scan acquisition.
- **96-Slice Multi-Planar Container**: Assembles 5 canonical anatomical slots into a unified volumetric representation:
  - Slot 0: Sagittal Fat-Suppressed (26 slices)
  - Slot 1: Sagittal Standard (22 slices)
  - Slot 2: Coronal Fat-Suppressed (18 slices)
  - Slot 3: Coronal Standard (12 slices)
  - Slot 4: Axial (18 slices)
- **Targeted Anatomical Specialists**:
  - **Pure Axial PF OA Specialist**: 18-slice dedicated network reaching **0.7825 PF OA AUC** and **0.7944 Gold Macro-AUC** (+0.0309 AUC lift on patellofemoral osteoarthritis).
  - **Compartment-ROI Lateral Specialist**: Foreground medial vs. lateral joint splitting via `roi_boxes.npz`, breaking the 0.810 Lateral OA ceiling.
- **Anatomical-Mirror Test-Time Augmentation (TTA)**: Slot-aware 3D reflection (Sagittal slice-depth reversal + Coronal/Axial width flip) without altering diagnostic orientation.
- **Strict FP32 Numerical Stability**: Completely eliminates Tesla T4 hardware FP16 LayerNorm/Softmax NaNs in multi-head attention.
- **Percentile Rank Calibration**: Blends heterogeneous models in normalized rank space $[0, 1]$, preventing raw probability logit mismatch and model dilution.

---

## End-to-End Pipeline Overview

```
                      [ Raw Multi-Planar DICOM Series ]
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
      [ IPP Slice Normal Sort ]              [ Metadata Extraction ]
                 │                                       │
      [ Per-Slot 2%/98% Normalization ]                  │
                 │                                       │
      [ 140mm Physical Knee Crop ]                       │
                 │                                       │
                 ▼                                       ▼
       [ 96-Slice Multi-Planar Container ]   [ 18-Slice Axial Cache ]
                 │                                       │
      ┌──────────┴──────────┐                            │
      ▼                     ▼                            ▼
[ Sovereign CoAtNet-384 ] [ ConvNeXt-Base-384 ]   [ Axial PF OA Specialist ]
(Compartment-ROI Mode)   (Full Volume 96)         (Pure 18-Slice ResNet/Conv)
      │                     │                            │
      └──────────┬──────────┘                            │
                 ▼                                       │
       [ 5-Fold OOF Predictions ]                        │
                 │                                       │
                 ▼                                       ▼
     [ Anatomical Mirror TTA ]              [ Specialist Prediction ]
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
                    [ Percentile Rank Transform ]
                                     │
                    [ Weighted Consensus Blend ]
                      (75% Consensus + 25% Ours)
                                     │
                                     ▼
                           [ submission.csv ]
```

---

## 3D Volume Preprocessing Engine

1. **Physical Aspect Ratio & Center Crop**: Crops each DICOM frame to a calibrated physical field-of-view ($140\text{mm} \times 140\text{mm}$) using `PixelSpacing`, centered on the knee joint center-of-mass.
2. **Per-Slot Dynamic Range Clipping**: Percentiles (2% and 98%) are calculated independently for each anatomical slot rather than globally, preserving local contrast differences between fluid-sensitive T2/PDFS and T1 anatomical sequences.
3. **Isotropic Resampling**: Slices are resampled to $384 \times 384$ using bicubic interpolation.
4. **Volume Layout**: Shape $(96, 384, 384)$, uint8 memory footprint with accompanying slice validity mask $(96,)$.

---

## Model Architecture & Sovereign Fleet

The primary architecture utilizes **hybrid attention-convolution backbones** configured for volumetric slice sequence processing:

* **CoAtNet-384 (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`)**:
  - ImageNet-12k pretrained, fine-tuned at $384 \times 384$.
  - Depthwise convolutions in early stages capture fine trabecular and cartilage morphology; self-attention in late stages captures long-range joint relationships.
  - Multi-window temporal evaluation ($K=42$ to $94$ windows) with top-3 SWA trajectory averaging.
* **ConvNeXt-Base (`convnext_base.fb_in22k_ft_in1k_384`)**:
  - ImageNet-22k pretrained, 87.6M parameters.
  - High capacity modernized CNN backbone providing architectural diversity to pair with CoAtNet.
* **EfficientNetV2-S (`tf_efficientnetv2_s.in21k_ft_in1k`)**:
  - Fast auxiliary representation for low-compute baseline evaluation.

---

## Targeted Anatomical Specialists

### 1. Pure Axial PF OA Specialist (`train_specialist_pfoa.py`)
Standard 96-slice models allocate only 18 slices to the axial plane, resulting in under-representation of the patellofemoral compartment. We engineered a dedicated specialist network:
- **Input**: Dedicated 18-slice pure axial cache (`ds/axial_pfoa/`).
- **Performance**:
  - Epoch 9: **0.7825 PF OA AUC** (Record high).
  - Epoch 12: **0.7944 Gold Macro-AUC**.
  - Verified lift on Gold-58: **+0.0309 AUC** on Patellofemoral Osteoarthritis.

### 2. Compartment-ROI Lateral Specialist (`roi_localize.py`)
Lateral compartment findings (Lateral OA, Lateral Meniscus) suffer from Left vs. Right knee orientation ambiguity:
- Automatically detects knee joint center-of-mass and computes bounding boxes separating Medial and Lateral compartments with calibrated overlap (`0.12`).
- Integrated into `train_5fold_knee.py` via `--roi --roi_mode compartment --roi_boxes roi_boxes.npz`.
- Achieves **0.8124 Lateral OA AUC** and **0.7876 Lateral Meniscus AUC**.

---

## Validation Strategy & 5-Fold Cross Validation

- **Site & Patient Group Stratification**:
  - Grouped by `PatientID` / `StudyInstanceUID` to ensure zero patient leakage across folds.
  - Stratified across clinical imaging sites (extracted from DICOM header attributes) to prevent hospital distribution shift.
- **OOF Aggregation (`purify_labels_oof.py`)**:
  - Aggregates out-of-fold probability predictions across all 5 folds.
  - Distills soft pseudo-targets to purify noise in free-text LLM report extractions.

---

## Inference, TTA & Rank Blending

### Anatomical-Mirror Test-Time Augmentation (`anatomical_mirror.py`)
Standard horizontal flipping alters knee anatomy (turning a Medial Meniscus into Lateral). Our slot-aware anatomical mirror:
- Reverses depth order for Sagittal slices (simulating mirror trajectory along the joint axis).
- Performs horizontal flipping for Coronal and Axial slices with simultaneous label swapping between Medial and Lateral compartments.

### Percentile Rank Blending (`sovereign_fast_runner.py`)
Raw predicted probabilities from different model architectures occupy disparate calibration landscapes. Arithmetic averaging causes probability compression and dilution. We compute:

$$R_m = \frac{\text{rank}(P_m) - 1}{N - 1}$$

$$P_{\text{final}} = w_{\text{consensus}} \cdot R_{\text{consensus}} + \sum_{m} w_m \cdot R_m$$

---

## Project Directory Layout

```
.
├── README.md                          # Complete project documentation
├── chat.md                            # Comprehensive experiment journal & handoff log
├── experiment_ledger.csv              # Machine-readable experiment & submission ledger
├── requirements.txt                   # Reproducible Python dependencies
├── pyproject.toml                     # Build and packaging configuration
├── Makefile                           # Automated development & execution workflows
│
├── train_5fold_knee.py                # High-throughput 5-fold cross validation trainer
├── run_5fold_fast_champion.sh         # Node-1 automated 5-fold execution script
├── train_specialist_pfoa.py           # Pure axial PF OA specialist training pipeline
├── extract_axial_pfoa.py              # 18-slice axial cache generation script
├── run_local_pfoa_specialist.sh       # Local GPU specialist execution script
│
├── roi_localize.py                    # Knee compartment ROI detection & localization
├── build_roi_boxes.py                 # Full dataset ROI bounding box precomputation
├── purify_labels_oof.py               # OOF teacher-to-student distillation purifier
├── anatomical_mirror.py               # Slot-aware 3D reflection TTA engine
│
├── kaggle_submission_champion.py      # Self-contained champion FP32 inference kernel
├── sovereign_fast_runner.py           # Fast multi-model sovereign fleet runner (<20 min)
├── status.sh                          # One-command multi-machine telemetry monitor
│
├── data/                              # Competition tabular metadata and label matrices
│   ├── train.csv                      # Official competition training index
│   ├── test.csv                       # Competition test index
│   ├── sample_submission.csv          # Submission schema template
│   └── labels/                        # Distilled pseudo-labels & gold evaluation sets
│
└── tests/                             # Automated unit testing suite
    ├── test_anatomical_mirror.py      # Tests for 3D reflection invariance
    ├── test_purify_labels.py          # Tests for soft target purification
    └── test_5fold_pipeline.py         # Tests for OOF aggregation and fold integrity
```

---

## Quickstart & CLI Reproduction

### 1. Environment Installation
```bash
git clone https://github.com/Shivoo29/Knee-Grow.git
cd Knee-Grow
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. High-Throughput 5-Fold Training (Node-1 L4 GPU)
```bash
# High-throughput CoAtNet-384 with compartment ROI and RAM preloading
bash run_5fold_fast_champion.sh
```

### 3. Dedicated Axial PF OA Specialist Training (Local RTX 4070S)
```bash
# 1. Precompute 18-slice axial cache
python3 extract_axial_pfoa.py --out_dir ds/axial_pfoa

# 2. Train specialist network
bash run_local_pfoa_specialist.sh
```

### 4. Running Automated Tests
```bash
pytest tests/ -v
```

---

## Benchmark Leaderboard & Ablations

| Model / Architecture | Input / Resolution | Gold-58 Macro-AUC | Key Finding Strengths |
|:---|:---:|:---:|:---|
| **ResNet-50 Baseline** | 44 Slices @ 336px | `0.8445` | Fast convergence, low VRAM |
| **CoAtNet-224 (`rmlp_1`)** | 44 Slices @ 224px | `0.8823` | High efficiency |
| **CoAtNet-384 `v4base`** | 44 Slices @ 384px | `0.8925` | ACL (0.933), MedMen (0.975), Baker's (0.995) |
| **CoAtNet-384 `v5` SWA** | 44 Slices @ 384px | `0.9152` | MedOA (0.970), Effusion (0.959) |
| **CoAtNet-384 `v6` SWA** | 96 Slices @ 384px | `0.9252` | Full volume coverage |
| **Lateral Compartment ROI `f0`** | 96 Slices + ROI | `0.7625` (Ep 2) | **Lateral OA (0.8104)**, **Lat Meniscus (0.7876)** |
| **Pure Axial PF OA Specialist** | 18 Slices (Axial) | **`0.7944`** | **PF OA (0.7825)**, Sub-0.50 Loss |
| **Sovereign + Specialist Blend** | 96 Slices + Specialists | **`0.8584` (PF OA)** | **+0.0309 AUC Lift on Patellofemoral OA** |

---

## Forensic Engineering & Safeguards

Throughout development on Kaggle's hidden evaluation environment, several critical production safeguards were established:

1. **Strict FP32 Execution**: Kaggle Tesla T4 GPUs exhibit numerical overflow in multi-head attention softmax and LayerNorm under FP16 autocast. All inference code runs in strict `torch.float32`.
2. **Directory Resolution Without Recursive Globbing**: Avoided `glob.glob(..., recursive=True)` over 250,000 files; implemented $O(1)$ direct folder checks taking $< 0.0001\text{s}$ per series.
3. **Sub-6-Hour Submission Time Budget**: Kaggle enforces a hard 9-hour execution limit. Heavy multi-model stacks risk timeouts on ~1,300 hidden test studies. Our inference pipeline is optimized to finish in $\approx 5.1$ hours, guaranteeing a 3.9-hour safety margin.
4. **Zero Silent Fallback**: Ensured all assertion guards and error handlers fail loudly rather than silently copying default public baselines.

---

## Hardware & Infrastructure

* **Primary Cluster Node (Node-1)**:
  - GPU: NVIDIA L4 (24 GB GDDR6)
  - CPU: Intel Xeon Gold 6530 (64 vCPUs)
  - Memory: 251 GB RAM
  - Environment: PyTorch 2.11 / CUDA 12.8 / Conda `knee`
* **Local Development Station (PC)**:
  - GPU: NVIDIA GeForce RTX 4070 SUPER (12 GB GDDR6X)
  - CPU: AMD Ryzen 5 7600
  - Memory: 32 GB DDR5 RAM
  - Environment: PyTorch 2.11 / CUDA 12.8 / `.venv`

---

## Author & License

* **Author**: Shivoo29 ([GitHub: @Shivoo29](https://github.com/Shivoo29))
* **Contact**: `2004skj@gmail.com`
* **License**: Released under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) License.
