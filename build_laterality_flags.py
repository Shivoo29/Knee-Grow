#!/usr/bin/env python3
"""build_laterality_flags.py - Stream DICOM headers from competition ZIP to generate laterality flags.

RSNA 2026 Knee Abnormality Detection.
Deliverable 2 for T04: Left/right canonicalisation.
Reads DICOM headers with stop_before_pixels=True.
Never extracts the zip. Writes laterality_flags.csv.
"""

import os
import io
import sys
import zipfile
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import pydicom

from laterality import side_from_headers, slot_flags

# Global worker state
_ZIP_HANDLE = None
_FILE_MAP = None


def _init_worker(zip_path, file_map):
    global _ZIP_HANDLE, _FILE_MAP
    _ZIP_HANDLE = zipfile.ZipFile(zip_path, mode="r")
    _FILE_MAP = file_map


def _read_dcm_hdr(z, filename):
    """Read essential spatial and laterality tags from DICOM file in zip."""
    try:
        data = z.read(filename)
        h = pydicom.dcmread(
            io.BytesIO(data),
            stop_before_pixels=True,
            specific_tags=[
                "ImageOrientationPatient",
                "ImagePositionPatient",
                "InstanceNumber",
                "ImageLaterality",
                "Laterality",
                "SeriesDescription",
            ],
            force=True,
        )
        iop = getattr(h, "ImageOrientationPatient", None)
        ipp = getattr(h, "ImagePositionPatient", None)
        if iop is not None and ipp is not None and len(iop) == 6:
            r = np.array(iop[:3], dtype=np.float64)
            c = np.array(iop[3:], dtype=np.float64)
            n = np.cross(r, c)
            pos = float(np.dot(np.array(ipp, dtype=np.float64), n))
            iop_list = [float(v) for v in iop]
            ipp_list = [float(v) for v in ipp]
        else:
            pos = float(getattr(h, "InstanceNumber", 0) or 0)
            iop_list = None
            ipp_list = None

        il = getattr(h, "ImageLaterality", None)
        lat = getattr(h, "Laterality", None)
        sd = getattr(h, "SeriesDescription", None)

        return {
            "filename": filename,
            "pos": pos,
            "iop": iop_list,
            "ipp": ipp_list,
            "ImageLaterality": str(il) if il is not None else None,
            "Laterality": str(lat) if lat is not None else None,
            "SeriesDescription": str(sd) if sd is not None else None,
            "ok": True,
        }
    except Exception:
        return {"filename": filename, "pos": 0.0, "iop": None, "ipp": None, "ok": False}


def _determine_side_and_source(series_headers: list[dict]) -> tuple[str, str]:
    """Determine side and report which metadata attribute provided it."""
    side = side_from_headers(series_headers)
    if side == "U":
        # Check if conflict or none
        any_l = False
        any_r = False
        for d in series_headers:
            s = side_from_headers([d])
            if s == "L":
                any_l = True
            elif s == "R":
                any_r = True
        return ("U", "conflict" if (any_l and any_r) else "none")

    # Trace which source produced the agreed side
    for source in ("ImageLaterality", "Laterality", "SeriesDescription"):
        sample_dicts = [{source: d.get(source)} for d in series_headers]
        if side_from_headers(sample_dicts) == side:
            return (side, source)

    return (side, "unknown")


def _process_study(study_slots: list[dict]):
    """Process all slots of a single study."""
    global _ZIP_HANDLE, _FILE_MAP
    z = _ZIP_HANDLE
    sid = study_slots[0]["StudyInstanceUID"]

    # Slot records map slot_idx -> slot info
    slots_by_idx = {int(s["slot_idx"]): s for s in study_slots}

    series_hdrs = {}
    picked_slice_info = {}

    for s_idx in range(5):
        s_info = slots_by_idx.get(s_idx)
        if not s_info or not s_info.get("filled", False):
            continue
        ser_uid = str(s_info.get("SeriesInstanceUID") or "")
        if not ser_uid or ser_uid not in _FILE_MAP:
            continue

        files = _FILE_MAP[ser_uid]
        if not files:
            continue

        hdrs = [_read_dcm_hdr(z, f) for f in files]
        hdrs.sort(key=lambda x: x["pos"])
        valid_hdrs = [h for h in hdrs if h.get("ok")]
        if not valid_hdrs:
            valid_hdrs = hdrs

        # Store series metadata for laterality vote
        first_valid = valid_hdrs[0]
        series_hdrs[ser_uid] = {
            "ImageLaterality": first_valid.get("ImageLaterality"),
            "Laterality": first_valid.get("Laterality"),
            "SeriesDescription": first_valid.get("SeriesDescription"),
        }

        # Calculate picks matching corpus builder wide-span [0.02, 0.98]
        n = len(valid_hdrs)
        k = int(s_info.get("slot_k", 20))
        lo, hi = int(n * 0.02), int(n * 0.98) - 1
        hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k

        first_pick = valid_hdrs[min(picks[0], n - 1)]
        last_pick = valid_hdrs[min(picks[-1], n - 1)]

        picked_slice_info[s_idx] = {
            "plane": str(s_info.get("slot_plane") or ""),
            "iop": first_valid.get("iop"),
            "first_ipp": first_pick.get("ipp"),
            "last_ipp": last_pick.get("ipp"),
        }

    # Determine study side
    side, side_source = _determine_side_and_source(list(series_hdrs.values()))

    # Compute slot flags f0..f4
    flags = [0, 0, 0, 0, 0]
    for s_idx in range(5):
        info = picked_slice_info.get(s_idx)
        if info is not None:
            flags[s_idx] = slot_flags(
                plane=info["plane"],
                iop=info["iop"],
                first_ipp=info["first_ipp"],
                last_ipp=info["last_ipp"],
                side=side,
            )

    return {
        "StudyInstanceUID": sid,
        "side": side,
        "side_source": side_source,
        "f0": flags[0],
        "f1": flags[1],
        "f2": flags[2],
        "f3": flags[3],
        "f4": flags[4],
    }


def main():
    parser = argparse.ArgumentParser(description="Build laterality_flags.csv by streaming headers from ZIP.")
    parser.add_argument("--zip", type=str, required=True, help="Path to raw competition ZIP archive")
    parser.add_argument("--slots_csv", type=str, default="ds/corpus96/slots.csv", help="Path to slots.csv")
    parser.add_argument("--out", type=str, default="laterality_flags.csv", help="Output CSV path")
    parser.add_argument("--workers", type=int, default=16, help="Worker processes")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on study count for testing")
    args = parser.parse_args()

    if not os.path.exists(args.zip):
        raise FileNotFoundError(f"ZIP archive not found: {args.zip}")
    if not os.path.exists(args.slots_csv):
        raise FileNotFoundError(f"slots.csv not found: {args.slots_csv}")

    # Index ZIP filenames by series ID
    file_map = {}
    with zipfile.ZipFile(args.zip, "r") as z:
        for name in z.namelist():
            if not name.endswith(".dcm"):
                continue
            parts = name.split("/")
            if len(parts) >= 2:
                series_id = parts[-2]
                file_map.setdefault(series_id, []).append(name)

    # Load slots.csv
    slots_df = pd.read_csv(args.slots_csv)
    grouped_studies = [group.to_dict("records") for _, group in slots_df.groupby("StudyInstanceUID")]
    if args.limit > 0:
        grouped_studies = grouped_studies[: args.limit]

    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.zip, file_map),
    ) as executor:
        futures = {executor.submit(_process_study, study): study[0]["StudyInstanceUID"] for study in grouped_studies}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)

    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values("StudyInstanceUID").reset_index(drop=True)
    out_df.to_csv(args.out, index=False)

    # Print only counts
    total = len(out_df)
    counts_side = out_df["side"].value_counts().to_dict()
    counts_src = out_df["side_source"].value_counts().to_dict()
    f0_sum = int(out_df["f0"].sum())
    f1_sum = int(out_df["f1"].sum())
    f2_sum = int(out_df["f2"].sum())
    f3_sum = int(out_df["f3"].sum())
    f4_sum = int(out_df["f4"].sum())

    print(f"Total studies: {total}")
    print(f"Side counts: {counts_side}")
    print(f"Source counts: {counts_src}")
    print(f"Flag sums: f0={f0_sum}, f1={f1_sum}, f2={f2_sum}, f3={f3_sum}, f4={f4_sum}")


if __name__ == "__main__":
    main()
