"""Tests for Task T05: Data Recipe (Clean windows, mirror aug, contrast aug, slot emb).

Run with:
    .venv/bin/python tests/test_data_recipe.py
"""
import inspect
import os
import random
import sys
import unittest.mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Mock pydicom and cv2 for local environment without full DICOM toolchain installed
if "pydicom" not in sys.modules:
    sys.modules["pydicom"] = unittest.mock.MagicMock()
    sys.modules["pydicom.pixel_data_handlers"] = unittest.mock.MagicMock()
    sys.modules["pydicom.pixel_data_handlers.util"] = unittest.mock.MagicMock()
if "cv2" not in sys.modules:
    sys.modules["cv2"] = unittest.mock.MagicMock()

import numpy as np
import torch
import torch.nn as nn

import train_knee
from kaggle_base954_fleet import fleet_v6


def test_clean_centers_no_straddle_and_no_mask():
    """Verify clean centers never straddle a slot boundary and never touch a masked slice.

    Uses synthetic mask with slot 3 empty (Coronal non-fluid slices 66..77).
    """
    # 96 slices total: Slot 0 (0..26), Slot 1 (26..48), Slot 2 (48..66), Slot 3 (66..78), Slot 4 (78..96)
    mask = np.ones(96, dtype=np.uint8)
    mask[66:78] = 0  # Slot 3 is empty
    mask[22:26] = 0  # Padding at end of slot 0
    mask[44:48] = 0  # Padding at end of slot 1

    valid = np.where(mask > 0)[0]
    count = 30

    # Test eval mode (deterministic evenly spaced)
    centers_eval = train_knee.pick_centers(valid, count=count, train=False, clean_windows=True, mask=mask, max_d=96)
    assert len(centers_eval) == count, f"Expected {count} centers, got {len(centers_eval)}"

    for c in centers_eval:
        assert 1 <= c <= 94, f"Center {c} out of range [1, 94]"
        # Must not touch any masked slice
        assert mask[c - 1] > 0, f"Window center {c}: slice {c-1} is masked"
        assert mask[c] > 0, f"Window center {c}: slice {c} is masked"
        assert mask[c + 1] > 0, f"Window center {c}: slice {c+1} is masked"

        # Must be in the exact same slot
        s_prev = train_knee.slot_of(c - 1)
        s_curr = train_knee.slot_of(c)
        s_next = train_knee.slot_of(c + 1)
        assert s_prev == s_curr == s_next, (
            f"Center {c} straddles slots: c-1={c-1} (slot {s_prev}), "
            f"c={c} (slot {s_curr}), c+1={c+1} (slot {s_next})"
        )
        assert s_curr != 3, f"Center {c} belongs to empty slot 3"

    # Test train mode (random sampled pool)
    centers_train = train_knee.pick_centers(valid, count=50, train=True, clean_windows=True, mask=mask, max_d=96)
    assert len(centers_train) == 50, f"Expected 50 centers, got {len(centers_train)}"

    for c in centers_train:
        assert mask[c - 1] > 0 and mask[c] > 0 and mask[c + 1] > 0
        assert train_knee.slot_of(c - 1) == train_knee.slot_of(c) == train_knee.slot_of(c + 1)
        assert train_knee.slot_of(c) != 3

    # Verify parity with fleet_v6 copy
    centers_fleet = fleet_v6.pick_centers(valid, count=count, train=False, clean_windows=True, mask=mask, max_d=96)
    assert centers_eval == centers_fleet, "fleet_v6.pick_centers output diverged from train_knee.pick_centers"

    print("PASS: test_clean_centers_no_straddle_and_no_mask")


def test_legacy_centers_unchanged_without_flag():
    """Verify without clean_windows the center list is 100% unchanged from legacy code."""
    def legacy_centers(valid, count, train=False, D=96):
        lo, hi = int(valid.min()), int(valid.max())
        cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
        if not cs:
            cs = [max(1, min((lo + hi) // 2, D - 2))]
        if train:
            reps = count // len(cs) + 1
            pool = cs * reps
            random.shuffle(pool)
            return pool[:count]
        idx = np.linspace(0, len(cs) - 1, count).round().astype(int)
        return [cs[i] for i in idx]

    test_cases = [
        np.arange(0, 96),
        np.arange(5, 75),
        np.arange(10, 30),
        np.array([20, 21, 22]),
        np.array([1, 2, 3, 4, 5]),
        np.arange(0, 50),
    ]

    for valid in test_cases:
        for count in [12, 24, 42, 94]:
            legacy = legacy_centers(valid, count, train=False)
            new_res = train_knee.pick_centers(valid, count, train=False, clean_windows=False)
            assert legacy == new_res, f"Mismatch for valid range [{valid.min()}, {valid.max()}] with count {count}"

            # Even if mask is passed, clean_windows=False must preserve legacy output
            mask = np.ones(96, dtype=np.uint8)
            new_with_mask = train_knee.pick_centers(valid, count, train=False, clean_windows=False, mask=mask)
            assert legacy == new_with_mask, "Passing mask with clean_windows=False changed center selection"

            # Check fleet_v6 copy matches legacy
            fleet_res = fleet_v6.pick_centers(valid, count, train=False, clean_windows=False)
            assert legacy == fleet_res, "fleet_v6 legacy centers diverged from train_knee legacy"

    print("PASS: test_legacy_centers_unchanged_without_flag")


def test_mirror_volume_real_reversed_and_identity_twice():
    """Verify sagittal real slices are reversed, padded slices untouched, coronal/axial flipped.

    Also verify mirroring twice is strictly the identity operation.
    """
    H, W = 16, 16
    vol = np.zeros((96, H, W), dtype=np.uint8)
    # Fill each slice with synthetic recognizable non-zero gradient pattern
    for i in range(96):
        row = np.arange(W, dtype=np.uint8).reshape(1, W)
        vol[i] = (row + (i * 2 + 1)) % 255

    # Synthetic mask with realistic layout
    mask = np.zeros(96, dtype=np.uint8)
    # Slot 0 (0..26, Sagittal fluid): 20 real slices, 6 padded
    mask[0:20] = 1
    # Slot 1 (26..48, Sagittal non-fluid): 18 real slices, 4 padded
    mask[26:44] = 1
    # Slot 2 (48..66, Coronal fluid): 18 real slices
    mask[48:66] = 1
    # Slot 3 (66..78, Coronal non-fluid): 0 real slices (empty)
    mask[66:78] = 0
    # Slot 4 (78..96, Axial): 18 real slices
    mask[78:96] = 1

    mirrored = train_knee.mirror_volume(vol, mask)

    # 1. Sagittal slot 0: real slices reversed in order
    assert np.array_equal(mirrored[0:20], vol[0:20][::-1]), "Slot 0 real slices were not reversed"
    # Padded slices untouched
    assert np.array_equal(mirrored[20:26], vol[20:26]), "Slot 0 padded slices were modified"

    # 2. Sagittal slot 1: real slices reversed in order
    assert np.array_equal(mirrored[26:44], vol[26:44][::-1]), "Slot 1 real slices were not reversed"
    # Padded slices untouched
    assert np.array_equal(mirrored[44:48], vol[44:48]), "Slot 1 padded slices were modified"

    # 3. Coronal slot 2: horizontally flipped (axis=-1)
    assert np.array_equal(mirrored[48:66], np.flip(vol[48:66], axis=-1)), "Slot 2 slices were not horizontally flipped"

    # 4. Coronal slot 3: horizontally flipped
    assert np.array_equal(mirrored[66:78], np.flip(vol[66:78], axis=-1)), "Slot 3 slices were not horizontally flipped"

    # 5. Axial slot 4: horizontally flipped
    assert np.array_equal(mirrored[78:96], np.flip(vol[78:96], axis=-1)), "Slot 4 slices were not horizontally flipped"

    # 6. Double mirror is identity
    mirrored_twice = train_knee.mirror_volume(mirrored, mask)
    assert np.array_equal(mirrored_twice, vol), "Double mirror did not reproduce the original volume"

    print("PASS: test_mirror_volume_real_reversed_and_identity_twice")


def test_slot_of_matches_center():
    """Verify slot_of correctly maps all 96 slice indices and matches center slot IDs."""
    expected_ranges = [
        (0, 26, 0),    # Sagittal fluid (26 slices)
        (26, 48, 1),   # Sagittal non-fluid (22 slices)
        (48, 66, 2),   # Coronal fluid (18 slices)
        (66, 78, 3),   # Coronal non-fluid (12 slices)
        (78, 96, 4),   # Axial (18 slices)
    ]

    for start, end, expected_slot in expected_ranges:
        for idx in range(start, end):
            s_train = train_knee.slot_of(idx)
            s_fleet = fleet_v6.slot_of(idx)
            assert s_train == expected_slot, f"Slice {idx} mapped to slot {s_train}, expected {expected_slot}"
            assert s_fleet == expected_slot, f"fleet_v6 slice {idx} mapped to {s_fleet}, expected {expected_slot}"

    # Out of range clamped fallbacks
    assert train_knee.slot_of(-10) == 0
    assert train_knee.slot_of(96) == 4
    assert train_knee.slot_of(200) == 4

    print("PASS: test_slot_of_matches_center")


def test_forward_signature_and_zero_init_identity():
    """Verify forward signature backward compatibility and zero-init identity of slot embedding."""
    class DummyBackbone(nn.Module):
        def __init__(self, f_dim=32):
            super().__init__()
            self.num_features = f_dim
            self.linear = nn.Linear(3 * 8 * 8, f_dim)

        def forward(self, x):
            # x: (N, 3, 8, 8)
            flat = x.flatten(1)
            return self.linear(flat)

    torch.manual_seed(123)
    bb1 = DummyBackbone(32)
    bb2 = DummyBackbone(32)
    bb2.load_state_dict(bb1.state_dict())

    # Model 1 without slot_emb
    m_no_emb = train_knee.RaptorClassifier(bb1, F_dim=32, slot_emb=False)
    # Model 2 with slot_emb
    m_with_emb = train_knee.RaptorClassifier(bb2, F_dim=32, slot_emb=True)

    # Copy head weights
    m_with_emb.norm.load_state_dict(m_no_emb.norm.state_dict())
    m_with_emb.att.load_state_dict(m_no_emb.att.state_dict())
    m_with_emb.clsW.data.copy_(m_no_emb.clsW.data)
    m_with_emb.clsb.data.copy_(m_no_emb.clsb.data)

    m_no_emb.eval()
    m_with_emb.eval()

    # Input: B=2, K=6, 3 channels, 8x8
    x = torch.randn(2, 6, 3, 8, 8)
    slots = torch.randint(0, 5, (2, 6), dtype=torch.long)

    # 1. Signature backward compatibility
    out1 = m_no_emb(x)
    out2 = m_no_emb(x, slots=None)
    assert torch.equal(out1, out2), "m_no_emb(x) != m_no_emb(x, slots=None)"

    # 2. Embedding init is exactly zeros
    assert m_with_emb.slot_emb is not None
    assert torch.all(m_with_emb.slot_emb.weight == 0.0), "slot_emb.weight was not initialized to zeros"

    # 3. Output with slots=None matches no_emb
    out3 = m_with_emb(x, slots=None)
    assert torch.equal(out1, out3), "m_with_emb(x, slots=None) != m_no_emb(x)"

    # 4. Output with slots provided matches no_emb because embedding is initialized to zero
    out4 = m_with_emb(x, slots=slots)
    assert torch.allclose(out1, out4, atol=1e-6), "Zero-init slot_emb did not produce identical logits to baseline"

    # 5. Check fleet_v6 RaptorClassifier behaves identically
    bb_fleet = DummyBackbone(32)
    bb_fleet.load_state_dict(bb1.state_dict())
    m_fleet = fleet_v6.RaptorClassifier(bb_fleet, F_dim=32, slot_emb=True)
    m_fleet.norm.load_state_dict(m_no_emb.norm.state_dict())
    m_fleet.att.load_state_dict(m_no_emb.att.state_dict())
    m_fleet.clsW.data.copy_(m_no_emb.clsW.data)
    m_fleet.clsb.data.copy_(m_no_emb.clsb.data)
    m_fleet.eval()

    out_fleet = m_fleet(x, slots=slots)
    assert torch.allclose(out1, out_fleet, atol=1e-6), "fleet_v6.RaptorClassifier output diverged from train_knee"

    print("PASS: test_forward_signature_and_zero_init_identity")


def test_source_parity_slot_of_and_pick_centers():
    """Verify train_knee and fleet_v6 source codes are 100% identical for parity functions."""
    assert inspect.getsource(train_knee.slot_of) == inspect.getsource(fleet_v6.slot_of), (
        "inspect.getsource(train_knee.slot_of) != inspect.getsource(fleet_v6.slot_of)"
    )
    assert inspect.getsource(train_knee.pick_centers) == inspect.getsource(fleet_v6.pick_centers), (
        "inspect.getsource(train_knee.pick_centers) != inspect.getsource(fleet_v6.pick_centers)"
    )
    assert inspect.getsource(train_knee.check_fallback_guard) == inspect.getsource(fleet_v6.check_fallback_guard), (
        "inspect.getsource(train_knee.check_fallback_guard) != inspect.getsource(fleet_v6.check_fallback_guard)"
    )
    assert train_knee.SLOT_BOUNDS == fleet_v6.SLOT_BOUNDS, (
        "SLOT_BOUNDS differ between train_knee and fleet_v6"
    )

    print("PASS: test_source_parity_slot_of_and_pick_centers")


def test_fallback_guard_raises_at_4pct_passes_at_2pct():
    """Verify fallback guard raises RuntimeError at 4% and passes cleanly at 2% and 3%."""
    # 4% fallback rate: 4 out of 100 studies > 3% threshold -> MUST raise RuntimeError
    raised = False
    try:
        train_knee.check_fallback_guard(fallback_count=4, total_studies=100, max_rate=0.03)
    except RuntimeError:
        raised = True
    assert raised, "train_knee.check_fallback_guard did NOT raise at 4% fallback rate"

    raised_fleet = False
    try:
        fleet_v6.check_fallback_guard(fallback_count=4, total_studies=100, max_rate=0.03)
    except RuntimeError:
        raised_fleet = True
    assert raised_fleet, "fleet_v6.check_fallback_guard did NOT raise at 4% fallback rate"

    # 2% fallback rate: 2 out of 100 studies <= 3% threshold -> MUST NOT raise
    train_knee.check_fallback_guard(fallback_count=2, total_studies=100, max_rate=0.03)
    fleet_v6.check_fallback_guard(fallback_count=2, total_studies=100, max_rate=0.03)

    # 3% exact boundary: 3 out of 100 studies == 3% threshold -> MUST NOT raise
    train_knee.check_fallback_guard(fallback_count=3, total_studies=100, max_rate=0.03)
    fleet_v6.check_fallback_guard(fallback_count=3, total_studies=100, max_rate=0.03)

    # Edge case: 0 total studies -> MUST NOT raise
    train_knee.check_fallback_guard(fallback_count=0, total_studies=0, max_rate=0.03)
    fleet_v6.check_fallback_guard(fallback_count=0, total_studies=0, max_rate=0.03)

    print("PASS: test_fallback_guard_raises_at_4pct_passes_at_2pct")


def main():
    print("=" * 70)
    print("Running Task T05 Test Suite (Data Recipe)")
    print("=" * 70)
    test_clean_centers_no_straddle_and_no_mask()
    test_legacy_centers_unchanged_without_flag()
    test_mirror_volume_real_reversed_and_identity_twice()
    test_slot_of_matches_center()
    test_forward_signature_and_zero_init_identity()
    test_source_parity_slot_of_and_pick_centers()
    test_fallback_guard_raises_at_4pct_passes_at_2pct()
    print("=" * 70)
    print("ALL 7 TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


def test_fleet_cap_never_repeats_centers():
    import importlib.util, numpy as np
    spec = importlib.util.spec_from_file_location("tk", "train_knee.py")
    src = open("kaggle_base954_fleet/fleet_v6.py").read()
    ns = {"np": np, "random": __import__("random")}
    for name in ("SLOT_BOUNDS", "def slot_of", "def pick_centers"):
        i = src.index(name); exec(src[i:src.index("\n\n\n", i)], ns)
    valid = np.arange(10, 50)                      # 38 legal centers < K_EVAL 94
    c = ns["pick_centers"](valid, 94, train=False, cap=True, max_d=96)
    assert len(c) == len(set(c)) == 38, len(c)


if __name__ == "__main__":
    main()
