#!/usr/bin/env python3
"""extract_orientation_laterality.py - Parallel DICOM orientation and laterality scanner.

Computes 3D physical coordinate laterality and slice sorting parameters:
  - Slice Normal: n = r x c
  - Physical slice projection: k = IPP . n
  - Physical Center in LPS coordinates:
      P_center = IPP + (Rows / 2 * ps_r) * r + (Cols / 2 * ps_c) * c
  - Laterality Determination:
      X_center > 0 => Left knee
      X_center < 0 => Right knee

Runs entirely on CPU in the background without affecting GPU training.
"""
import os
import sys
import io
import time
import zipfile
import csv
import numpy as np

KNEE = "/home/s05814802723/projects/knee"
ZIP_PATH = os.path.join(KNEE, "raw", "rsna-knee-abnormality-detection.zip")
OUT_CSV = os.path.join(KNEE, "ds", "study_laterality_orientation.csv")

def main():
    try:
        import pydicom
    except ImportError:
        print("ERROR: pydicom not found.", flush=True)
        sys.exit(1)

    t0 = time.time()
    print("=" * 80, flush=True)
    print("EXTRACTING DICOM ORIENTATION & PHYSICAL LATERALITY (CPU SCAN)", flush=True)
    print(f"Zip: {ZIP_PATH}", flush=True)
    print(f"Output: {OUT_CSV}", flush=True)
    print("=" * 80, flush=True)

    if not os.path.exists(ZIP_PATH):
        print(f"ERROR: {ZIP_PATH} not found.", flush=True)
        sys.exit(1)

    # 1. Index one coronal/axial/sagittal DICOM per study
    print("[1] Indexing first valid DICOM for each study...", flush=True)
    study_to_dcm = {}
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        for name in zf.namelist():
            if not name.endswith('.dcm'):
                continue
            parts = name.split('/')
            study_uid = None
            for i, p in enumerate(parts):
                if p == 'train_series' and i + 1 < len(parts):
                    study_uid = parts[i + 1]
                    break
            if study_uid and study_uid not in study_to_dcm:
                study_to_dcm[study_uid] = name

    n_studies = len(study_to_dcm)
    print(f"[2] Discovered {n_studies} studies. Parsing headers in parallel chunks...", flush=True)

    rows = []
    left_count = 0
    right_count = 0

    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        for i, (study_uid, dcm_path) in enumerate(study_to_dcm.items()):
            try:
                b = zf.read(dcm_path)
                ds = pydicom.dcmread(io.BytesIO(b), stop_before_pixels=True)

                iop = getattr(ds, 'ImageOrientationPatient', None)
                ipp = getattr(ds, 'ImagePositionPatient', None)
                ps = getattr(ds, 'PixelSpacing', None)
                rows_px = getattr(ds, 'Rows', 512)
                cols_px = getattr(ds, 'Columns', 512)

                if iop is not None and ipp is not None and len(iop) == 6 and len(ipp) == 3:
                    r = np.array(iop[:3], dtype=np.float64)
                    c = np.array(iop[3:], dtype=np.float64)
                    n = np.cross(r, c)
                    ipp_arr = np.array(ipp, dtype=np.float64)
                    
                    ps_r = float(ps[0]) if ps is not None else 0.5
                    ps_c = float(ps[1]) if ps is not None else 0.5

                    # 3D Center in LPS coordinates
                    center = ipp_arr + (rows_px / 2.0 * ps_r) * r + (cols_px / 2.0 * ps_c) * c
                    x_center = float(center[0])
                    y_center = float(center[1])
                    z_center = float(center[2])

                    # X_center > 0 => Left knee, X_center < 0 => Right knee
                    lat = "Left" if x_center > 0 else "Right"
                    if lat == "Left": left_count += 1
                    else: right_count += 1

                    k_pos = float(np.dot(ipp_arr, n))

                    rows.append({
                        "StudyInstanceUID": study_uid,
                        "Laterality": lat,
                        "X_center": round(x_center, 2),
                        "Y_center": round(y_center, 2),
                        "Z_center": round(z_center, 2),
                        "Normal_X": round(float(n[0]), 4),
                        "Normal_Y": round(float(n[1]), 4),
                        "Normal_Z": round(float(n[2]), 4),
                        "Slice_K": round(k_pos, 2),
                        "PixelSpacing_R": round(ps_r, 4),
                        "PixelSpacing_C": round(ps_c, 4),
                        "Rows": int(rows_px),
                        "Cols": int(cols_px),
                    })
                else:
                    rows.append({
                        "StudyInstanceUID": study_uid,
                        "Laterality": "Unknown",
                        "X_center": 0.0, "Y_center": 0.0, "Z_center": 0.0,
                        "Normal_X": 0.0, "Normal_Y": 0.0, "Normal_Z": 0.0,
                        "Slice_K": 0.0, "PixelSpacing_R": 0.5, "PixelSpacing_C": 0.5,
                        "Rows": 512, "Cols": 512
                    })
            except Exception as e:
                rows.append({
                    "StudyInstanceUID": study_uid,
                    "Laterality": "Error",
                    "X_center": 0.0, "Y_center": 0.0, "Z_center": 0.0,
                    "Normal_X": 0.0, "Normal_Y": 0.0, "Normal_Z": 0.0,
                    "Slice_K": 0.0, "PixelSpacing_R": 0.5, "PixelSpacing_C": 0.5,
                    "Rows": 512, "Cols": 512
                })

            if (i + 1) % 500 == 0 or i == n_studies - 1:
                el = time.time() - t0
                eta = el / (i + 1) * (n_studies - i - 1)
                print(f"  [{i+1}/{n_studies}] {el:.0f}s elapsed (ETA {eta:.0f}s) | Left: {left_count}, Right: {right_count}", flush=True)

    # Write output CSV
    fieldnames = [
        "StudyInstanceUID", "Laterality", "X_center", "Y_center", "Z_center",
        "Normal_X", "Normal_Y", "Normal_Z", "Slice_K", "PixelSpacing_R", "PixelSpacing_C",
        "Rows", "Cols"
    ]
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 80, flush=True)
    print(f"COMPLETED SCAN IN {time.time()-t0:.1f}s", flush=True)
    print(f"Total Studies Processed: {len(rows)}", flush=True)
    print(f"Discovered: {left_count} Left Knees ({left_count/len(rows)*100:.1f}%), {right_count} Right Knees ({right_count/len(rows)*100:.1f}%)", flush=True)
    print(f"Saved: {OUT_CSV} ({os.path.getsize(OUT_CSV)/1024:.1f} KB)", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
