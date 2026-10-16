#!/usr/bin/env python3
"""tests/test_fleet_multi.py - CPU checks for multi-checkpoint fleet inference (RAPTOR_CKPTS).
    .venv/bin/python -I tests/test_fleet_multi.py
"""
import importlib.util
import os
import sys
import time
import types

import numpy as np
import torch

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _n in ("pydicom", "pydicom.pixel_data_handlers", "pydicom.pixel_data_handlers.util", "cv2"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["pydicom.pixel_data_handlers.util"].apply_modality_lut = lambda *a, **k: None
spec = importlib.util.spec_from_file_location("wt_fleet_v6", os.path.join(WT, "kaggle_base954_fleet", "fleet_v6.py"))
fleet = importlib.util.module_from_spec(spec)
sys.modules["wt_fleet_v6"] = fleet
spec.loader.exec_module(fleet)


class Const(torch.nn.Module):
    """Stub model: constant logit c for every finding (like test_roi_inference's stub)."""
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.clean_windows, self.has_slot_emb, self.slots, self.input_norm, self.roi = False, False, None, "none", False

    def forward(self, x, slots=None):
        return torch.zeros(1, 12) + self.c


def _study():
    rng = np.random.RandomState(0)
    return rng.randint(0, 256, size=(fleet.MAXS, 64, 64)).astype(np.uint8), np.ones(fleet.MAXS, np.uint8)


def test_layout_mismatch_raises():
    base = {k: None for k in fleet.LAYOUT_KEYS}
    base.update(arch="convnext_tiny.in12k_ft_in1k", res=256, norm="imagenet", slot_sizes="40,40,36,36,40")
    fleet.check_same_layout([base, dict(base)])
    try:
        fleet.check_same_layout([base, dict(base, res=384)])
    except ValueError as e:
        assert "'res'" in str(e)
        return "same layout accepted; res 256 vs 384 refused"
    raise AssertionError("mixed res accepted")


def test_single_equals_predict_study():
    vol, mask = _study()
    for tta in (False, True):
        p1, _ = fleet.predict_study(Const(0.13), vol, mask, 64, "cpu", use_tta=tta)
        pm, _ = fleet.predict_study_multi([Const(0.13)], vol, mask, 64, "cpu", use_tta=tta)
        assert np.abs(p1 - pm).max() < 1e-7
    return "one-model multi == predict_study (tta off/on)"


def test_multi_is_mean():
    vol, mask = _study()
    pm, _ = fleet.predict_study_multi([Const(0.13), Const(-0.4)], vol, mask, 64, "cpu", use_tta=False)
    want = 0.5 * (torch.sigmoid(torch.tensor(0.13)) + torch.sigmoid(torch.tensor(-0.4))).item()
    assert np.abs(pm - want).max() < 1e-6
    return "two models -> mean probability"


def test_find_checkpoints_env():
    os.environ["RAPTOR_CKPTS"] = "/a/x.pt,/b/y.pt,"
    try:
        assert fleet.find_checkpoints() == ["/a/x.pt", "/b/y.pt"]
    finally:
        del os.environ["RAPTOR_CKPTS"]
    return "RAPTOR_CKPTS parsed"


TESTS = [("layout_mismatch_raises", test_layout_mismatch_raises),
         ("single_equals_predict_study", test_single_equals_predict_study),
         ("multi_is_mean", test_multi_is_mean), ("find_checkpoints_env", test_find_checkpoints_env)]


def main():
    fails = 0
    t0 = time.time()
    for name, fn in TESTS:
        t = time.time()
        try:
            print(f"PASS {name} {fn()} ({time.time()-t:.1f}s)", flush=True)
        except Exception as e:
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e} ({time.time()-t:.1f}s)", flush=True)
    print(f"done {len(TESTS)-fails}/{len(TESTS)} pass ({time.time()-t0:.1f}s)", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
