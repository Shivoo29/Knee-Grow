# Knee-Grow
## Overview

Solution pipeline for the RSNA Knee Abnormality Detection challenge, identifying 12 pathological findings from 3D MRI volumes.

## Problem Formulation

Multi-label classification across sagittal, coronal, and axial series detecting tears, ligament injuries, and degenerative changes.

## Dataset Structure

```
data/
├── train.csv
├── train_series.csv
├── test.csv
└── labels/
```

## Preprocessing

- DICOM HU windowing and intensity normalization
- Resampling to isotropic resolution
- 96-slice volume interpolation

## Label Engineering

- LLM report distillation with Qwen
- Soft pseudo-labels calibration
- Outlier and hedging phrase penalties

## Model Architectures

1. ResNet-50 (2.5D feature extractor)
2. CoAtNet-224 (Hybrid attention)
3. ConvNeXt-Small (Modernized ConvNet)
4. EfficientNetV2

## Loss Functions & Optimization

- Class-balanced Binary Cross-Entropy
- AdamW with weight decay 1e-2
- Cosine Annealing with linear warmup

## Stochastic Weight Averaging (SWA)

SWA applied from epoch 8 onwards to enhance generalization across clinical scanners.

## Validation Strategy

- Stratified 5-Fold Group Cross Validation
- Mean Multi-Label ROC-AUC evaluation

## Out-Of-Fold Gold Matrices

Predictions saved as NPZ arrays for Nelder-Mead ensemble weight optimization.

## Submission Pipeline

Fast CPU/GPU inference kernels with test-time augmentation (TTA) and ONNX/PyTorch runtime.

## Training Commands

```bash
bash run_resnet.sh
bash run_coat224.sh
bash run_convnext_pc.sh
```

