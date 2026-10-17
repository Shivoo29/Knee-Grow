#!/usr/bin/env python3
"""test_laterality.py - Synthetic unit tests for knee laterality canonicalisation.

RSNA 2026 Knee Abnormality Detection.
Deliverable 5 for T04: Left/right canonicalisation.
Runs with plain Python standard library (no pytest).
"""

import sys
import os
import inspect
from unittest.mock import MagicMock
import numpy as np

# Mock missing DICOM / CV2 packages in local venv so fleet_v6 can be imported
for mod in ["pydicom", "pydicom.pixel_data_handlers", "pydicom.pixel_data_handlers.util", "cv2"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Ensure worktree root and fleet directory are importable
HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.dirname(HERE)
sys.path.insert(0, WORKTREE)
sys.path.insert(0, os.path.join(WORKTREE, "kaggle_base954_fleet"))

import laterality
import fleet_v6


def test_sagittal_orientation():
    """Sagittal stack with x decreasing gets reversed; x increasing does not; LEFT inverts it."""
    # Sagittal with x decreasing along slice order
    first_ipp = [50.0, 0.0, 0.0]
    last_ipp = [-50.0, 0.0, 0.0]

    # Right or Unknown: x decreasing -> flag = 1 (reverse to make x increase)
    assert laterality.slot_flags("Sagittal", iop=None, first_ipp=first_ipp, last_ipp=last_ipp, side="R") == 1
    assert laterality.slot_flags("Sagittal", iop=None, first_ipp=first_ipp, last_ipp=last_ipp, side="U") == 1

    # Left: inverts the decision -> 1 - 1 = 0
    assert laterality.slot_flags("Sagittal", iop=None, first_ipp=first_ipp, last_ipp=last_ipp, side="L") == 0

    # Sagittal with x increasing along slice order
    first_ipp_inc = [-50.0, 0.0, 0.0]
    last_ipp_inc = [50.0, 0.0, 0.0]

    # Right or Unknown: x increasing -> flag = 0 (already increasing)
    assert laterality.slot_flags("Sagittal", iop=None, first_ipp=first_ipp_inc, last_ipp=last_ipp_inc, side="R") == 0
    assert laterality.slot_flags("Sagittal", iop=None, first_ipp=first_ipp_inc, last_ipp=last_ipp_inc, side="U") == 0

    # Left: inverts the decision -> 1 - 0 = 1
    assert laterality.slot_flags("Sagittal", iop=None, first_ipp=first_ipp_inc, last_ipp=last_ipp_inc, side="L") == 1


def test_coronal_axial_orientation():
    """Coronal with r[0] < 0 gets flipped; LEFT inverts it; U keeps level 1 only."""
    # Coronal with r[0] < 0
    iop_neg = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    # Level 1: r[0] < 0 -> flip=1
    assert laterality.slot_flags("Coronal", iop=iop_neg, first_ipp=None, last_ipp=None, side="R") == 1
    assert laterality.slot_flags("Coronal", iop=iop_neg, first_ipp=None, last_ipp=None, side="U") == 1
    # Left inverts level 1 -> 1 - 1 = 0
    assert laterality.slot_flags("Coronal", iop=iop_neg, first_ipp=None, last_ipp=None, side="L") == 0

    # Coronal with r[0] > 0
    iop_pos = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    # Level 1: r[0] > 0 -> flip=0
    assert laterality.slot_flags("Coronal", iop=iop_pos, first_ipp=None, last_ipp=None, side="R") == 0
    assert laterality.slot_flags("Coronal", iop=iop_pos, first_ipp=None, last_ipp=None, side="U") == 0
    # Left inverts level 1 -> 1 - 0 = 1
    assert laterality.slot_flags("Coronal", iop=iop_pos, first_ipp=None, last_ipp=None, side="L") == 1

    # Axial follows the same coronal/axial logic
    assert laterality.slot_flags("Axial", iop=iop_neg, first_ipp=None, last_ipp=None, side="R") == 1
    assert laterality.slot_flags("Axial", iop=iop_neg, first_ipp=None, last_ipp=None, side="U") == 1
    assert laterality.slot_flags("Axial", iop=iop_neg, first_ipp=None, last_ipp=None, side="L") == 0


def test_padded_slices_remain_at_end():
    """Padded slices stay at the end after a reverse (mask with 20 real + 6 padded in slot 0)."""
    h, w = 16, 16
    vol = np.zeros((96, h, w), dtype=np.uint8)
    mask = np.zeros(96, dtype=np.uint8)

    # Slot 0 has 26 slices total: 20 real, 6 padded
    mask[:20] = 1
    mask[20:26] = 0

    # Fill real slices with distinct markers: slice i has value (i + 1)
    for i in range(20):
        vol[i] = (i + 1)
    # Fill padded slices with marker 99
    vol[20:26] = 99

    flags = np.array([1, 0, 0, 0, 0], dtype=np.int32)
    out = laterality.apply_flags(vol, mask, flags)

    # Real slices 0..19 must be reversed: out[0] == 20, out[1] == 19, ..., out[19] == 1
    for i in range(20):
        expected_val = 20 - i
        assert np.all(out[i] == expected_val), f"Slice {i} expected {expected_val}, got {out[i, 0, 0]}"

    # Padded slices 20..25 must remain untouched at the end
    assert np.all(out[20:26] == 99), "Padded slices did not remain at the end!"

    # Unmodified slots (26..96) must be identical
    assert np.array_equal(out[26:], vol[26:]), "Other slots modified unexpectedly!"


def test_all_zero_flags_returns_identical():
    """apply_flags with all-zero flags returns an identical array."""
    np.random.seed(42)
    vol = np.random.randint(0, 255, (96, 32, 32), dtype=np.uint8)
    mask = np.random.randint(0, 2, (96,), dtype=np.uint8)

    flags_zeros = np.zeros(5, dtype=np.int32)
    out = laterality.apply_flags(vol, mask, flags_zeros)
    assert np.array_equal(out, vol), "All-zero flags did not return an identical array!"

    # List of zeros as well
    out_list = laterality.apply_flags(vol, mask, [0, 0, 0, 0, 0])
    assert np.array_equal(out_list, vol), "List of zeros did not return an identical array!"


def test_side_voting():
    """Side vote: {L, L, R} -> U (conflict); {L, L} -> L; no tags -> U."""
    # Conflict across series -> 'U'
    assert laterality.side_from_headers([
        {"ImageLaterality": "L"},
        {"ImageLaterality": "L"},
        {"ImageLaterality": "R"},
    ]) == "U"

    # All agreeing 'L' -> 'L'
    assert laterality.side_from_headers([
        {"ImageLaterality": "L"},
        {"ImageLaterality": "L"},
    ]) == "L"

    # All agreeing 'R' -> 'R'
    assert laterality.side_from_headers([
        {"ImageLaterality": "R"},
        {"ImageLaterality": "R"},
    ]) == "R"

    # No tags -> 'U'
    assert laterality.side_from_headers([]) == "U"
    assert laterality.side_from_headers([{}]) == "U"
    assert laterality.side_from_headers([{"ImageLaterality": None, "Laterality": "", "SeriesDescription": ""}]) == "U"

    # Fallback to Laterality when ImageLaterality is absent
    assert laterality.side_from_headers([
        {"ImageLaterality": None, "Laterality": "L"},
        {"ImageLaterality": None, "Laterality": "L"},
    ]) == "L"

    # Fallback to SeriesDescription tokens
    assert laterality.side_from_headers([
        {"SeriesDescription": "KNEE LT SAG PD"},
        {"SeriesDescription": "KNEE LEFT COR FS"},
    ]) == "L"

    # SeriesDescription conflict
    assert laterality.side_from_headers([
        {"SeriesDescription": "KNEE LEFT SAG"},
        {"SeriesDescription": "KNEE RIGHT COR"},
    ]) == "U"

    # Conflicting ImageLaterality and Laterality within series
    assert laterality.side_from_headers([
        {"ImageLaterality": "L"},
        {"Laterality": "R"},
    ]) == "U"


def test_fleet_parity():
    """fleet_v6 copies identical to laterality.py character-for-character."""
    functions_to_check = ["side_from_headers", "slot_flags", "apply_flags"]
    for fn in functions_to_check:
        src_lat = inspect.getsource(getattr(laterality, fn))
        src_fleet = inspect.getsource(getattr(fleet_v6, fn))
        assert src_lat == src_fleet, f"Source mismatch in function '{fn}' between laterality.py and fleet_v6.py"


def test_coronal_axial_horizontal_flip():
    """Horizontal flip on width axis for coronal and axial slots."""
    vol = np.zeros((96, 4, 4), dtype=np.uint8)
    mask = np.ones(96, dtype=np.uint8)

    # Fill coronal slot 2 (indices 48..66) with asymmetric pattern: left col=1, right col=9
    for s in range(48, 66):
        vol[s, :, 0] = 1
        vol[s, :, 3] = 9

    flags = np.array([0, 0, 1, 0, 0], dtype=np.int32)
    out = laterality.apply_flags(vol, mask, flags)

    for s in range(48, 66):
        assert np.all(out[s, :, 0] == 9), f"Coronal slice {s} left col should be 9 after flip"
        assert np.all(out[s, :, 3] == 1), f"Coronal slice {s} right col should be 1 after flip"


def main():
    tests = [
        test_sagittal_orientation,
        test_coronal_axial_orientation,
        test_padded_slices_remain_at_end,
        test_all_zero_flags_returns_identical,
        test_side_voting,
        test_fleet_parity,
        test_coronal_axial_horizontal_flip,
    ]

    print("=" * 70)
    print("RUNNING LATERALITY CANONICALISATION UNIT TESTS")
    print("=" * 70)
    passed = 0
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {name} -> {e}")
            failed += 1

    print("-" * 70)
    print(f"Total: {len(tests)} | Passed: {passed} | Failed: {failed}")
    print("=" * 70)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
