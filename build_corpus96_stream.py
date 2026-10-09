#!/usr/bin/env python3
"""build_corpus96_stream.py - Stream 96-slice native 384px training corpus directly from zip.

Key Design Decisions:
1. Zero Disk Unzip: Reads DICOM bytes directly from `rsna-knee-abnormality-detection.zip`
   in-memory using Python's `zipfile.ZipFile`, avoiding the need for 247 GB unzipped storage.
2. 96-Slice Slot Contract (Nartaa / Dread 0.949 recipe):
   SLOTS = [
       ("Sagittal", 1, 26),  # Sagittal Fluid-Sensitive (Fat-Sat)
       ("Sagittal", 0, 22),  # Sagittal Non-Fluid-Sensitive
       ("Coronal",  1, 18),  # Coronal Fluid-Sensitive
       ("Coronal",  0, 12),  # Coronal Non-Fluid-Sensitive
       ("Axial",   -1, 18),  # Axial (any fluid preference)
   ]  Total = 96 slices per study @ native 384px.
3. IPP Plane Normal Projection:
   Calculates slice normal n = r x c from ImageOrientationPatient, then projects
   ImagePositionPatient onto n: pos = dot(ipp, n). Slices sorted by true 3D spatial position.
4. Physical Crop:
   140.0 mm centered field-of-view, resized to (384, 384) via INTER_AREA.
5. Slot Audit Logging (slots.csv):
   Records every study x slot filling decision: which series was chosen, how many slices
   were available, whether it was padded or empty. Crucial for verifying PF OA / Lat OA coverage.
6. Multi-threaded / Multiprocess Worker Pool:
   Each worker process opens an independent ZipFile handle to enable lock-free parallel extraction.
"""

import os
import io
import sys
import time
import zipfile
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import cv2

# Configuration matching the 0.949 frontier recipe
IMG = 384
CROP_MM = 140.0
SLOTS = [
    ("Sagittal", 1, 26),
    ("Sagittal", 0, 22),
    ("Coronal", 1, 18),
    ("Coronal", 0, 12),
    ("Axial", -1, 18)
]
MAXS = sum(s[2] for s in SLOTS)  # 96 slices

# Global worker state for multiprocessing
_ZIP_HANDLE = None
_ZIP_PATH = None
_FILE_MAP = None
_SERIES_MAP = None


def _init_worker(zip_path, file_map, series_map):
    """Initialize worker process with open ZipFile handle."""
    global _ZIP_HANDLE, _ZIP_PATH, _FILE_MAP, _SERIES_MAP
    _ZIP_PATH = zip_path
    _ZIP_HANDLE = zipfile.ZipFile(zip_path, mode='r')
    _FILE_MAP = file_map
    _SERIES_MAP = series_map


def _read_dcm_header_from_zip(z, filename):
    """Read DICOM header without decompressing pixel data."""
    try:
        data = z.read(filename)
        h = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True)
        iop = getattr(h, 'ImageOrientationPatient', None)
        ipp = getattr(h, 'ImagePositionPatient', None)
        if iop is not None and ipp is not None and len(iop) == 6:
            r = np.array(iop[:3], dtype=np.float64)
            c = np.array(iop[3:], dtype=np.float64)
            n = np.cross(r, c)
            pos = float(np.dot(np.array(ipp, dtype=np.float64), n))
        else:
            pos = float(getattr(h, 'InstanceNumber', 0) or 0)
        ps = getattr(h, 'PixelSpacing', None)
        ps = float(ps[0]) if ps is not None else 0.5
        return (pos, filename, ps, True)
    except Exception:
        return (0.0, filename, 0.5, False)


def _read_dcm_pixel_from_zip(z, filename):
    """Read and decode DICOM pixel array with modality LUT."""
    data = z.read(filename)
    d = pydicom.dcmread(io.BytesIO(data))
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
        a = a.max() - a
    return a


def _crop_and_resize(a, ps):
    """140.0 mm centered physical crop and resize to IMG x IMG."""
    h, w = a.shape
    cpx = int(round(CROP_MM / max(ps, 1e-3)))
    cpx = min(cpx, min(h, w))
    y0 = (h - cpx) // 2
    x0 = (w - cpx) // 2
    cropped = a[y0:y0 + cpx, x0:x0 + cpx]
    return cv2.resize(cropped, (IMG, IMG), interpolation=cv2.INTER_AREA)


def _pick_series_for_slot(series_rows, plane, fluid, used_series):
    """Select best series for a given plane & fluid-sensitivity requirement."""
    cands = [r for r in series_rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used_series]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get('Fluid_Sensitive', 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def _process_single_study(study_id):
    """Process all slots for one StudyInstanceUID. Runs in worker process."""
    global _ZIP_HANDLE, _FILE_MAP, _SERIES_MAP
    z = _ZIP_HANDLE
    series_rows = _SERIES_MAP.get(study_id, [])

    vol = np.zeros((MAXS, IMG, IMG), dtype=np.uint8)
    slot_records = []
    used_series = set()
    idx = 0

    for slot_idx, (plane, fluid, k) in enumerate(SLOTS):
        r = _pick_series_for_slot(series_rows, plane, fluid, used_series)
        record = {
            "StudyInstanceUID": study_id,
            "slot_idx": slot_idx,
            "slot_plane": plane,
            "slot_fluid": fluid,
            "slot_k": k,
            "filled": False,
            "SeriesInstanceUID": "",
            "series_plane": "",
            "series_fluid": -1,
            "n_available": 0,
            "n_picked": 0,
            "is_padded": True,
        }

        if r is None:
            # Slot cannot be filled (no matching series)
            idx += k
            slot_records.append(record)
            continue

        ser_id = str(r['SeriesInstanceUID'])
        used_series.add(ser_id)
        record["SeriesInstanceUID"] = ser_id
        record["series_plane"] = r.get("Anatomical_Plane", "")
        record["series_fluid"] = int(r.get("Fluid_Sensitive", -1) or -1)

        # Get list of DICOM files inside the zip for this series
        files = _FILE_MAP.get(ser_id, [])
        record["n_available"] = len(files)

        if not files:
            idx += k
            slot_records.append(record)
            continue

        # 1. Read headers and sort by plane-normal IPP projection
        hdrs = [_read_dcm_header_from_zip(z, f) for f in files]
        hdrs.sort(key=lambda x: x[0])  # sort by pos
        sorted_files = [(f, ps) for _, f, ps, ok in hdrs if ok]
        if not sorted_files:
            sorted_files = [(f, 0.5) for _, f, _, _ in hdrs]

        n = len(sorted_files)
        med_ps = float(np.median([ps for _, ps in sorted_files])) if sorted_files else 0.5

        # 2. Wide span sampling [0.02, 0.98] to retain peripheral slices (menisci & collaterals)
        lo, hi = int(n * 0.02), int(n * 0.98) - 1
        hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k

        # 3. Read picked slice pixels
        arrs = []
        pss = []
        for p in picks:
            fp, ps = sorted_files[min(p, n - 1)]
            try:
                arrs.append(_read_dcm_pixel_from_zip(z, fp))
                pss.append(ps if ps > 0 else med_ps)
            except Exception:
                arrs.append(None)
                pss.append(med_ps)

        valid_arrs = [a for a in arrs if a is not None]
        if valid_arrs:
            allpx = np.concatenate([a.ravel() for a in valid_arrs])
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0

        # 4. Normalize, crop, resize and place into volume
        n_placed = 0
        for a, ps in zip(arrs, pss):
            if idx >= MAXS:
                break
            if a is None:
                idx += 1
                continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0.0, 1.0)
            aw = _crop_and_resize(aw, ps)
            vol[idx] = (aw * 255.0).astype(np.uint8)
            idx += 1
            n_placed += 1

        record["filled"] = (n_placed > 0)
        record["n_picked"] = n_placed
        record["is_padded"] = (n < k or n_placed < k)
        slot_records.append(record)

