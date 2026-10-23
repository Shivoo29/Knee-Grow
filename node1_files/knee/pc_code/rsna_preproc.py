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
        target_res: output square dimension (default: 384)
    Returns:
        2D float32 image array of shape (target_res, target_res)
    """
    h, w = a.shape
    cpx = int(round(crop_mm / max(ps, 1e-3)))
    cpx = max(1, min(cpx, min(h, w)))
    y0 = (h - cpx) // 2
    x0 = (w - cpx) // 2
    cropped = a[y0:y0 + cpx, x0:x0 + cpx]
    return cv2.resize(cropped, (target_res, target_res), interpolation=cv2.INTER_AREA)


def _fluid_flag(v):
    """0/1 Fluid_Sensitive flag; None for missing/NaN. (`x or -1` turned a real 0 into -1.)"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else int(f)


def pick_series_for_slot(series_rows, plane, fluid, used_series):
    """Selects the best available series matching plane and fluid-sensitivity criteria."""
    plane_norm = str(plane).strip().capitalize()
    cands = [
        r for r in series_rows
        if str(r.get('Anatomical_Plane', '')).strip().capitalize() == plane_norm
        and str(r.get('SeriesInstanceUID', '')) not in used_series
    ]
    if fluid in (0, 1):
        pref = [r for r in cands if _fluid_flag(r.get('Fluid_Sensitive')) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def pick_slice_indices(num_slices, k):
    """Computes evenly spaced slice picks within [SPAN_LO, SPAN_HI]."""
    if num_slices <= 0:
        return []
    if num_slices == 1:
        return [0] * k
    lo = int(num_slices * SPAN_LO)
    hi = max(lo, int(num_slices * SPAN_HI) - 1)
    return np.linspace(lo, hi, k).round().astype(int).tolist()


def normalize_and_pack_slot(arrs, pss, target_res=IMG):
    """Applies per-slot 2%/98% contrast percentile normalization and mm-crop.
    
    Args:
        arrs: list of 2D float32 arrays (or None for failed decodes)
        pss: list of float pixel spacings in mm
        target_res: output resolution (default: 384)
    Returns:
        3D uint8 array of shape (len(arrs), target_res, target_res)
    """
    k = len(arrs)
    out = np.zeros((k, target_res, target_res), dtype=np.uint8)
    valid_arrs = [a for a in arrs if a is not None]
    if valid_arrs:
        allpx = np.concatenate([a.ravel() for a in valid_arrs])
        loq, hiq = np.percentile(allpx, [NORM_LO_PCT, NORM_HI_PCT])
    else:
        loq, hiq = 0.0, 1.0

    denom = max(hiq - loq, 1e-6)
    for i, (a, ps) in enumerate(zip(arrs, pss)):
        if a is None:
            continue
        aw = np.clip((a - loq) / denom, 0.0, 1.0)
        aw = mm_crop_and_resize(aw, ps, target_res=target_res)
        out[i] = (aw * 255.0).astype(np.uint8)
    return out
