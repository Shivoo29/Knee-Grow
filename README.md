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

