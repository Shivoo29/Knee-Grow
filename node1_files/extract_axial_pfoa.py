#!/usr/bin/env python3
"""extract_axial_pfoa.py - Extract dedicated 18-slice Axial Cache for Patellofemoral OA (PF OA).

Streams directly from raw competition ZIP archive without unzipping to disk.
Focuses 100% of volume capacity on the Axial plane covering the entire
patellofemoral joint from superior to inferior patellar pole.

Produces a compact ~3.9 GB dataset:
  - axial_vols.npy: (N, 18, 224, 224) uint8
  - axial_masks.npy: (N, 18) uint8
  - axial_ids.npy: (N,) string array
"""

import os
import io
import sys
import time
import zipfile
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import cv2

# Global worker state for lock-free parallel extraction
_ZIP_HANDLE = None
_ZIP_PATH = None
_FILE_MAP = None
_SERIES_MAP = None
_RES = 224
_CROP_MM = 140.0
_K = 18


def _init_worker(zip_path, file_map, series_map, res, crop_mm, k):
    """Initialize worker process with independent ZipFile handle."""
    global _ZIP_HANDLE, _ZIP_PATH, _FILE_MAP, _SERIES_MAP, _RES, _CROP_MM, _K
    _ZIP_PATH = zip_path
    _ZIP_HANDLE = zipfile.ZipFile(zip_path, mode='r')
    _FILE_MAP = file_map
    _SERIES_MAP = series_map
    _RES = res
    _CROP_MM = crop_mm
    _K = k


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


def _crop_and_resize(a, ps, crop_mm, res):
    """Center-crop crop_mm FOV and resize to res x res."""
    H, W = a.shape
    crop_px = int(round(crop_mm / ps))
    crop_px = max(16, min(crop_px, min(H, W)))

    cy, cx = H // 2, W // 2
    y0 = max(0, cy - crop_px // 2)
    x0 = max(0, cx - crop_px // 2)
    y1 = min(H, y0 + crop_px)
    x1 = min(W, x0 + crop_px)

    cropped = a[y0:y1, x0:x1]
    if cropped.shape[0] != res or cropped.shape[1] != res:
        cropped = cv2.resize(cropped, (res, res), interpolation=cv2.INTER_AREA)
    return cropped


def _process_single_study(study_id):
    """Extract 18 pure-axial slices for one StudyInstanceUID."""
    global _ZIP_HANDLE, _FILE_MAP, _SERIES_MAP, _RES, _CROP_MM, _K
    z = _ZIP_HANDLE
    series_rows = _SERIES_MAP.get(study_id, [])

    vol = np.zeros((_K, _RES, _RES), dtype=np.uint8)
    mask = np.zeros(_K, dtype=np.uint8)

    # Filter to Axial series only
    axial_series = [r for r in series_rows if str(r.get("Anatomical_Plane", "")).lower() == "axial"]
    if not axial_series:
        return study_id, vol, mask

    # Prefer Fluid_Sensitive == 1
    fs_axial = [r for r in axial_series if int(r.get("Fluid_Sensitive", -1) or -1) == 1]
    best_series = fs_axial[0] if fs_axial else axial_series[0]

    ser_id = str(best_series["SeriesInstanceUID"])
    files = _FILE_MAP.get(ser_id, [])
    if not files:
        return study_id, vol, mask

    # 1. Read headers and sort by normal projection
    hdrs = [_read_dcm_header_from_zip(z, f) for f in files]
    hdrs.sort(key=lambda x: x[0])
    sorted_files = [(f, ps) for _, f, ps, ok in hdrs if ok]
    if not sorted_files:
        sorted_files = [(f, 0.5) for _, f, _, _ in hdrs]

    n = len(sorted_files)
    med_ps = float(np.median([ps for _, ps in sorted_files])) if sorted_files else 0.5

    # 2. Sample _K evenly spaced slices across the entire volume
    lo, hi = 0, n - 1
    picks = np.linspace(lo, hi, _K).round().astype(int) if n > 1 else [0] * _K

    # 3. Read picked slices
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

    # 4. Normalize and crop
    for idx, (a, ps) in enumerate(zip(arrs, pss)):
        if a is None:
            continue
        aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0.0, 1.0)
        aw = _crop_and_resize(aw, ps, _CROP_MM, _RES)
        vol[idx] = (aw * 255.0).astype(np.uint8)
        mask[idx] = 1

    return study_id, vol, mask


def build_file_map_from_zip(zip_path):
    """Scan ZIP directory structure once in RAM to map SeriesInstanceUID -> file paths."""
    print(f"[zip] opening archive {zip_path} for TOC indexation...", flush=True)
    t0 = time.time()
    file_map = {}
    with zipfile.ZipFile(zip_path, 'r') as z:
        names = z.namelist()
        print(f"[zip] indexed {len(names)} entries in {time.time()-t0:.1f}s", flush=True)
        for name in names:
            if not name.endswith(".dcm"):
                continue
            parts = name.split("/")
            if len(parts) >= 3:
                series_id = parts[-2]
                file_map.setdefault(series_id, []).append(name)
    print(f"[zip] found {len(file_map)} unique DICOM series in {time.time()-t0:.1f}s", flush=True)
    return file_map


def main():
    parser = argparse.ArgumentParser(description="Extract dedicated 18-slice Axial cache for PF OA")
    parser.add_argument("--zip", type=str, default="/home/s05814802723/projects/knee/raw/rsna-knee-abnormality-detection.zip")
    parser.add_argument("--train_csv", type=str, default="/home/s05814802723/projects/knee/train.csv")
    parser.add_argument("--series_csv", type=str, default="/home/s05814802723/projects/knee/ds/comp/train_series.csv")
    parser.add_argument("--out_dir", type=str, default="/home/s05814802723/projects/knee/ds/axial_pfoa")
    parser.add_argument("--res", type=int, default=224, help="Output slice resolution (default 224px)")
    parser.add_argument("--k", type=int, default=18, help="Slices per study (default 18)")
    parser.add_argument("--workers", type=int, default=16, help="Worker processes")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[axial_extract] loading metadata...", flush=True)
    train_df = pd.read_csv(args.train_csv)
    study_ids = train_df["StudyInstanceUID"].astype(str).tolist()
    if args.limit > 0:
        study_ids = study_ids[:args.limit]
    n_studies = len(study_ids)

    series_df = pd.read_csv(args.series_csv)
    series_df["StudyInstanceUID"] = series_df["StudyInstanceUID"].astype(str)
    series_df["SeriesInstanceUID"] = series_df["SeriesInstanceUID"].astype(str)

    series_map = {}
    for sid, group in series_df.groupby("StudyInstanceUID"):
        series_map[sid] = group.to_dict("records")

    file_map = build_file_map_from_zip(args.zip)

    vols_path = os.path.join(args.out_dir, "axial_vols.npy")
    masks_path = os.path.join(args.out_dir, "axial_masks.npy")
    ids_path = os.path.join(args.out_dir, "axial_ids.npy")

    expected_gb = (n_studies * args.k * args.res * args.res) / (1024**3)
    print(f"[axial_extract] Target array: {n_studies} studies x {args.k} slices x {args.res}x{args.res} ({expected_gb:.2f} GB)")

    vols_mmap = np.lib.format.open_memmap(vols_path, mode='w+', dtype=np.uint8, shape=(n_studies, args.k, args.res, args.res))
    masks_arr = np.zeros((n_studies, args.k), dtype=np.uint8)
    saved_ids = []

    t0 = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(args.zip, file_map, series_map, args.res, 140.0, args.k)) as executor:
        for sid, vol, mask in executor.map(_process_single_study, study_ids):
            vols_mmap[completed] = vol
            masks_arr[completed] = mask
            saved_ids.append(sid)
            completed += 1

            if completed % 256 == 0 or completed == n_studies:
                vols_mmap.flush()
                elapsed = time.time() - t0
                rate = completed / elapsed
                print(f"[axial_extract] Processed {completed}/{n_studies} ({completed/n_studies*100:.1f}%) "
                      f"in {elapsed:.1f}s ({rate:.1f} studies/s)", flush=True)

    vols_mmap.flush()
    np.save(masks_path, masks_arr)
    np.save(ids_path, np.array(saved_ids, dtype=object))

    elapsed = time.time() - t0
    print(f"[axial_extract] [SUCCESS] Extracted {completed} studies in {elapsed/60:.1f} min! Saved to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
