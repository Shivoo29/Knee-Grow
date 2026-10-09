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
SPAN_HI = 0.98

# Percentile contrast windowing
NORM_LO_PCT = 2.0
NORM_HI_PCT = 98.0


def parse_dicom_header(source):
    """Parses spatial position and pixel spacing from a DICOM file or bytes buffer.
    
    Args:
        source: file path (str/Path) or bytes / io.BytesIO.
    Returns:
        (pos, ps, is_valid):
            pos: float projection of ImagePositionPatient onto slice normal (r x c)
            ps: float pixel spacing in mm
            is_valid: bool indicating successful read
    """
    try:
        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(source)
        h = pydicom.dcmread(source, stop_before_pixels=True)
        iop = getattr(h, 'ImageOrientationPatient', None)
        ipp = getattr(h, 'ImagePositionPatient', None)
        if iop is not None and ipp is not None and len(iop) == 6:
            r = np.array(iop[:3], dtype=np.float64)
            c = np.array(iop[3:], dtype=np.float64)
            n = np.cross(r, c)
            pos = float(np.dot(np.array(ipp, dtype=np.float64), n))
        else:
            pos = float(getattr(h, 'InstanceNumber', 0) or 0)
        ps_val = getattr(h, 'PixelSpacing', None)
        ps = float(ps_val[0]) if ps_val is not None else 0.5
        return pos, ps, True
    except Exception:
        return 0.0, 0.5, False


def decode_dicom_pixel(source):
    """Reads pixel data with modality LUT and Monochrome inversion.
    
    Args:
        source: file path (str/Path) or bytes / io.BytesIO.
    Returns:
        2D float32 numpy array or None if decoding fails.
    """
    try:
        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(source)
        d = pydicom.dcmread(source, force=True)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a
    except Exception:
        return None


def mm_crop_and_resize(a, ps, crop_mm=CROP_MM, target_res=IMG):
    """Extracts a centered physical FOV box and resizes via INTER_AREA.
    
    Args:
        a: 2D float32 image array
        ps: pixel spacing in mm
        crop_mm: physical box width/height in mm (default: 140.0)
