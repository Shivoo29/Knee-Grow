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
