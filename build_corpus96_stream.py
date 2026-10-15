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

    mask = (vol.reshape(MAXS, -1).sum(axis=1) > 0).astype(np.uint8)
    return study_id, vol, mask, slot_records


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
            # expected format: train/<study_id>/<series_id>/<sop>.dcm
            # or <study_id>/<series_id>/<sop>.dcm
            if len(parts) >= 3:
                series_id = parts[-2]
                file_map.setdefault(series_id, []).append(name)
    print(f"[zip] found {len(file_map)} unique DICOM series in {time.time()-t0:.1f}s", flush=True)
    return file_map


def main():
    parser = argparse.ArgumentParser(description="Build 96-slice 384px corpus by streaming from ZIP.")
    parser.add_argument("--zip", type=str, required=True, help="Path to raw competition ZIP archive")
    parser.add_argument("--train_csv", type=str, default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--series_csv", type=str, default="data/train_series.csv", help="Path to train_series.csv")
    parser.add_argument("--out_dir", type=str, default="ds/corpus96", help="Output directory for .npy corpus")
    parser.add_argument("--workers", type=int, default=16, help="Number of worker processes")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on study count for testing")
    parser.add_argument("--slot_sizes", type=str, default="", help="comma list overriding the 5 slot sizes, e.g. 40,40,36,36,40 (keeps more native slices)")
    args = parser.parse_args()

    if not os.path.exists(args.zip):
        raise FileNotFoundError(f"ZIP archive not found at: {args.zip}")
    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"train.csv not found at: {args.train_csv}")
    if not os.path.exists(args.series_csv):
        raise FileNotFoundError(f"train_series.csv not found at: {args.series_csv}")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.slot_sizes:   # workers are forked after this, so they inherit the new layout
        global SLOTS, MAXS
        sizes = [int(v) for v in args.slot_sizes.split(",")]
        assert len(sizes) == len(SLOTS), sizes
        SLOTS = [(pl, fl, k) for (pl, fl, _), k in zip(SLOTS, sizes)]
        MAXS = sum(sizes)
        print(f"[slots] layout {sizes} -> {MAXS} slices per study", flush=True)

    # Load train and series metadata
    print("[data] loading metadata...", flush=True)
    train_df = pd.read_csv(args.train_csv)
    study_ids = train_df["StudyInstanceUID"].astype(str).tolist()
    if args.limit > 0:
        study_ids = study_ids[:args.limit]
    n_studies = len(study_ids)

    series_df = pd.read_csv(args.series_csv)
    series_df["StudyInstanceUID"] = series_df["StudyInstanceUID"].astype(str)
    series_df["SeriesInstanceUID"] = series_df["SeriesInstanceUID"].astype(str)

    # Group series by StudyInstanceUID into a dictionary of dict-records
    series_map = {}
    for sid, group in series_df.groupby("StudyInstanceUID"):
        series_map[sid] = group.to_dict("records")

    # Index ZIP contents in RAM
    file_map = build_file_map_from_zip(args.zip)

    # Allocate memory-mapped output arrays
    vols_path = os.path.join(args.out_dir, "all_vols.npy")
    masks_path = os.path.join(args.out_dir, "all_masks.npy")
    ids_path = os.path.join(args.out_dir, "all_ids.npy")
    slots_csv_path = os.path.join(args.out_dir, "slots.csv")

    print(f"[corpus] allocating memmap storage for {n_studies} studies x {MAXS} slices x {IMG}x{IMG}...", flush=True)
    total_gb = (n_studies * MAXS * IMG * IMG) / (1024**3)
    print(f"[corpus] expected array size: {total_gb:.2f} GB at {vols_path}", flush=True)

    # Open memmapped array for zero-RAM streaming write
    vols_mmap = np.lib.format.open_memmap(vols_path, mode='w+', dtype=np.uint8, shape=(n_studies, MAXS, IMG, IMG))
    masks_arr = np.zeros((n_studies, MAXS), dtype=np.uint8)
    all_slots_records = []
    saved_ids = []

    print(f"[stream] launching {args.workers} workers...", flush=True)
    t0 = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(args.zip, file_map, series_map)) as executor:
        futures = {executor.submit(_process_single_study, sid): (idx, sid) for idx, sid in enumerate(study_ids)}

        for future in as_completed(futures):
            orig_idx, sid = futures[future]
            try:
                out_sid, vol, mask, slot_recs = future.result()
                vols_mmap[orig_idx] = vol
                masks_arr[orig_idx] = mask
                all_slots_records.extend(slot_recs)
                saved_ids.append((orig_idx, out_sid))
                completed += 1

                if completed % 50 == 0 or completed == n_studies:
                    elapsed = time.time() - t0
                    speed = completed / max(elapsed, 1e-3)
                    eta_s = (n_studies - completed) / max(speed, 1e-3)
                    print(f"[{completed}/{n_studies}] {completed*100/n_studies:.1f}% | "
                          f"{speed:.2f} studies/s | ETA: {eta_s/60:.1f}m", flush=True)
            except Exception as e:
                print(f"[ERROR] study {sid} failed: {e}", file=sys.stderr, flush=True)

    # Flush memmapped vols to disk
    vols_mmap.flush()
    del vols_mmap

    # Sort IDs and masks to ensure perfect index ordering
    saved_ids.sort(key=lambda x: x[0])
    ordered_ids = np.array([sid for _, sid in saved_ids], dtype=object)

    np.save(masks_path, masks_arr)
    np.save(ids_path, ordered_ids)

    # Save slots audit CSV
    slots_df = pd.DataFrame(all_slots_records)
    slots_df.to_csv(slots_csv_path, index=False)
    print(f"[audit] saved slot coverage record to {slots_csv_path} ({len(slots_df)} rows)", flush=True)

    # Print validation & summary statistics
    total_time = time.time() - t0
    print("\n" + "=" * 60)
    print(f"CORPUS 96-SLICE BUILD COMPLETE in {total_time/60:.1f}m ({total_time/n_studies:.2f} s/study)")
    print("=" * 60)
    print(f"Output files:")
    print(f"  Volumes:  {vols_path} ({os.path.getsize(vols_path) / (1024**3):.2f} GB)")
    print(f"  Masks:    {masks_path} ({os.path.getsize(masks_path) / (1024**2):.2f} MB)")
    print(f"  IDs:      {ids_path} ({len(ordered_ids)} entries)")
    print(f"  Audit:    {slots_csv_path}")

    # Slot fill rate summary
    fill_summary = slots_df.groupby("slot_plane")["filled"].agg(
        total_slots="count",
        filled_slots="sum",
        fill_rate=lambda x: f"{x.mean()*100:.1f}%"
    )
    print("\nSlot Fill Rates by Anatomical Plane:")
    print(fill_summary)
    print("=" * 60)


if __name__ == "__main__":
    main()
