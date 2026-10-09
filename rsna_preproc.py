#!/usr/bin/env python3
"""rsna_preproc.py - Canonical Unified Preprocessing Engine for RSNA Knee 2026.

Single Source of Truth (SSOT) shared identically between:
1. Offline Corpus Builder (`build_corpus96_stream.py`) -> training tensors
2. Online Kaggle Submission Kernel -> test inference tensors

Guarantees 100% mathematical parity (zero pixel drift, identical slot ordering,
identical spatial sorting, identical dynamic range clipping).
"""
import io
import numpy as np
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

# Canonical 96-slice contract (Nartaa / Dread 0.949 frontier recipe)
IMG = 384
CROP_MM = 140.0
SLOTS = [
    ("Sagittal", 1, 26),   # Sagittal Fluid-Sensitive (Fat-Suppressed)
    ("Sagittal", 0, 22),   # Sagittal Non-Fluid-Sensitive (Proton Density / T1 / T2)
    ("Coronal",  1, 18),   # Coronal Fluid-Sensitive
    ("Coronal",  0, 12),   # Coronal Non-Fluid-Sensitive
    ("Axial",   -1, 18),   # Axial (any fluid-sensitivity preference)
]
SLOT_SIZES = [s[2] for s in SLOTS]
MAXS = sum(SLOT_SIZES)  # 96 slices total
SAGITTAL_SLOTS = (0, 1)

# Sampling window span within each series (0.02 to 0.98 to capture collateral ligaments & meniscal roots)
SPAN_LO = 0.02
