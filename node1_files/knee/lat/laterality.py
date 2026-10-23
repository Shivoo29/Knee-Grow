"""laterality.py - Pure functions for knee laterality and orientation canonicalisation.

RSNA 2026 Knee Abnormality Detection.
Deliverable 1 for T04: Left/right canonicalisation.
No I/O inside these functions.
"""

import re
import numpy as np

# Canonical 96-slice slot layout (Nartaa / Dread 96-slice contract)
# Slots: [Sagittal FS (26), Sagittal non-FS (22), Coronal FS (18), Coronal non-FS (12), Axial (18)]
SLOT_BOUNDS = [(0, 26), (26, 48), (48, 66), (66, 78), (78, 96)]
SAGITTAL_SLOTS = (0, 1)
CORONAL_AXIAL_SLOTS = (2, 3, 4)

_LEFT_RE = re.compile(r"\b(LT|LEFT)\b", re.IGNORECASE)
_RIGHT_RE = re.compile(r"\b(RT|RIGHT)\b", re.IGNORECASE)


def side_from_headers(list_of_dicts: list[dict]) -> str:
    """Determine knee side ('L', 'R', or 'U') from DICOM series headers.

    Priority order per series dict:
      1. ImageLaterality ('L'/'LEFT' -> 'L', 'R'/'RIGHT' -> 'R')
      2. Laterality ('L'/'LEFT' -> 'L', 'R'/'RIGHT' -> 'R')
      3. SeriesDescription tokens (LT/LEFT -> 'L', RT/RIGHT -> 'R')

    Voting rules across series:
      - All agreeing 'L' -> 'L'
      - All agreeing 'R' -> 'R'
      - Any conflict (e.g. both 'L' and 'R' present) -> 'U'
      - No tags found -> 'U'
    """
    if not list_of_dicts:
        return "U"

    votes = []
    for d in list_of_dicts:
        if not isinstance(d, dict):
            continue
        series_side = None

        # 1. ImageLaterality
        il = str(d.get("ImageLaterality") or "").strip().upper()
        if il in ("L", "LEFT"):
            series_side = "L"
        elif il in ("R", "RIGHT"):
            series_side = "R"

        # 2. Laterality
        if series_side is None:
            lat = str(d.get("Laterality") or "").strip().upper()
            if lat in ("L", "LEFT"):
                series_side = "L"
            elif lat in ("R", "RIGHT"):
                series_side = "R"

        # 3. SeriesDescription tokens
        if series_side is None:
            sd = str(d.get("SeriesDescription") or "").strip()
            if sd:
                has_l = bool(_LEFT_RE.search(sd))
                has_r = bool(_RIGHT_RE.search(sd))
                if has_l and not has_r:
                    series_side = "L"
                elif has_r and not has_l:
                    series_side = "R"
                elif has_l and has_r:
                    series_side = "U"

        if series_side in ("L", "R"):
            votes.append(series_side)
        elif series_side == "U":
            votes.append("U")

    if not votes:
        return "U"
    if "U" in votes or ("L" in votes and "R" in votes):
        return "U"
    if all(v == "L" for v in votes):
        return "L"
    if all(v == "R" for v in votes):
        return "R"
    return "U"


def slot_flags(plane: str, iop: list[float], first_ipp: list[float], last_ipp: list[float], side: str) -> int:
    """Compute canonicalisation flag (0 or 1) for a slot.

    Level 1: Orientation canonicalisation:
      - Sagittal: rev = 1 if x decreases along stored order (last_x < first_x), else 0.
      - Coronal/Axial: hflip = 1 if r[0] < 0 (IOP[0] < 0), else 0.
    Level 2: Side canonicalisation (if side is known):
      - If side == 'L': invert flag (1 - flag), so every knee matches a right knee.
      - If side == 'R' or 'U': keep flag as is.

    Returns:
      1 if reverse/flip should be applied, 0 otherwise.
    """
    p = str(plane).strip().lower()
    if p.startswith("sag"):
        x0 = float(first_ipp[0]) if first_ipp is not None and len(first_ipp) > 0 else 0.0
        x1 = float(last_ipp[0]) if last_ipp is not None and len(last_ipp) > 0 else 0.0
        flag = 1 if x1 < x0 else 0
    else:
        r0 = float(iop[0]) if iop is not None and len(iop) > 0 else 0.0
        flag = 1 if r0 < 0.0 else 0

    if str(side).strip().upper() == "L":
        flag = 1 - flag

    return int(flag)


def apply_flags(vol: np.ndarray, mask: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """Apply canonicalisation flags to a (96, H, W) volume.

    Args:
      vol: numpy array of shape (96, H, W)
      mask: numpy array of shape (96,) marking real (>0) vs padded (==0) slices
      flags: array-like of shape (5,) with flags for the 5 slots:
             [Sag FS, Sag non-FS, Cor FS, Cor non-FS, Axial]
             - For sagittal slots (0, 1): flag=1 reverses real slices; padded remain at end.
             - For coronal/axial slots (2, 3, 4): flag=1 horizontally flips (np.flip axis=-1).

    Returns:
      Canonicalised volume of shape (96, H, W).
    """
    flags_arr = np.asarray(flags)
    if not np.any(flags_arr):
        return vol.copy()

    out = vol.copy()
    for slot_idx, (start, end) in enumerate(SLOT_BOUNDS):
        if slot_idx >= len(flags_arr) or not flags_arr[slot_idx]:
            continue
        if slot_idx in SAGITTAL_SLOTS:
            slot_mask = mask[start:end]
            n_real = int(np.sum(slot_mask > 0))
            if n_real > 1:
                out[start : start + n_real] = out[start : start + n_real][::-1]
        else:
            out[start:end] = np.flip(out[start:end], axis=-1)
    return out
