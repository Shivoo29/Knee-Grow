#!/usr/bin/env python3
"""tests/test_zoom.py - CPU checks for two-scale tiles (--zoom_mm): full field + centred close-up.
    .venv/bin/python -I tests/test_zoom.py
"""
import importlib.util
import inspect
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
for _n in ("pydicom", "pydicom.pixel_data_handlers", "pydicom.pixel_data_handlers.util", "cv2"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["pydicom.pixel_data_handlers.util"].apply_modality_lut = lambda *a, **k: None


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tk = _load("wt_train_knee", os.path.join(WT, "train_knee.py"))
fleet = _load("wt_fleet_v6", os.path.join(WT, "kaggle_base954_fleet", "fleet_v6.py"))


def _vol():
    rng = np.random.RandomState(0)
    return rng.randint(0, 256, size=(96, 384, 384)).astype(np.uint8), np.ones(96, np.uint8)


def test_zoom_px():
    assert tk.zoom_px(100) == 274 and tk.zoom_px(0) == 0 and tk.zoom_px(140) == 384
    assert inspect.getsource(tk.zoom_px) == inspect.getsource(fleet.zoom_px) and tk.FIELD_MM == fleet.FIELD_MM
    return "100 mm of the 140 mm / 384 px field = 274 px; train and fleet share source"


def test_fleet_tiles():
    vol, mask = _vol()
    x0, s0, c0 = fleet._build_windows(vol, mask, 64, k=10)
    x1, s1, c1 = fleet._build_windows(vol, mask, 64, k=10, zoom_px=274)
    assert x1.shape[0] == 2 * x0.shape[0] == 20
    assert torch.equal(x1[0::2], x0)
    assert list(c1[0::2]) == list(c0) and list(c1[1::2]) == list(c0) and list(s1[1::2]) == list(s0)
    c, o = c0[0], (384 - 274) // 2
    z = torch.from_numpy(vol[c - 1:c + 2, o:o + 274, o:o + 274].astype(np.float32) / 255.0)[None]
    assert torch.allclose(x1[1], F.interpolate(z, size=(64, 64), mode="bilinear", align_corners=False)[0])
    return "full tiles unchanged; each centre adds its centred 100 mm crop"


def test_train_matches_fleet():
    vol, mask = _vol()
    uid = "row0"
    ds = tk.StudyWindows(WT, [uid], {uid: 0}, {uid: np.zeros(12, np.float32)}, 64, 20, train=False, zoom_px=274)
    ds.vols, ds.masks = vol[None], mask[None]
    x_ref = ds[0][0]
    x_new, _, _ = fleet._build_windows(vol, mask, 64, k=10, zoom_px=274)
    assert tuple(x_ref.shape) == tuple(x_new.shape) == (20, 3, 64, 64)
    d = float((x_ref - x_new).abs().max())
    assert d < 1e-6, d
    return f"StudyWindows(train=False) == fleet tiles, maxdiff {d:.1e}"


def test_zoom_with_roi_refused():
    try:
        tk.StudyWindows(WT, [], {}, {}, 64, 20, train=False, roi=True,
                        roi_sbox=np.zeros((1, 96, 4)), roi_cen=np.zeros((1, 96, 2)), zoom_px=274)
    except ValueError:
        return "zoom + roi refused"
    raise AssertionError("zoom with roi accepted")


TESTS = [("zoom_px", test_zoom_px), ("fleet_tiles", test_fleet_tiles),
         ("train_matches_fleet", test_train_matches_fleet), ("zoom_with_roi_refused", test_zoom_with_roi_refused)]


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
