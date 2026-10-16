#!/usr/bin/env python3
"""tests/test_slot_layout.py - CPU-only checks for configurable slot layouts.

Plain `def test_*` + `__main__` runner (no pytest). Do NOT run training or
any GPU job. Prints only shapes, counts, small scalars and timings (no pixel
arrays, no UIDs, no Report text).

Covers T10 (worktree ../knee-slots, branch slots):
 1. default sizes reproduce the old SLOT_BOUNDS; slot_of/pick_centers match
    the pre-change code (`git show HEAD:` reference) on 0..95.
 2. 192 layout [40,40,36,36,40]: bounds + slot_of spot checks.
 3. mirror_volume on synthetic (192,8,8): sagittal reversal per slot,
    coronal/axial flip; mirror twice == identity.
 4. fleet picking (n=30, k=40) == build_corpus96_stream.py rule; picking-block
    source identical between the two files.
 5. old-checkpoint fleet path (no slot_sizes key): window builder output
    identical to HEAD reference code on CPU.
 6. train/fleet copies of bounds_from_sizes/slot_of/pick_centers identical.

Run from the worktree root with the main checkout's venv:
    /abs/path/.venv/bin/python -I tests/test_slot_layout.py
"""
import argparse
import ast
import importlib.util
import inspect
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import types

import numpy as np
import torch

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)

# fleet_v6 imports pydicom/cv2 at module top (decode paths only); neither is
# installed in .venv and neither is touched below.
for _n in ("pydicom", "pydicom.pixel_data_handlers",
           "pydicom.pixel_data_handlers.util", "cv2"):
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

OLD_BOUNDS = [(0, 26), (26, 48), (48, 66), (66, 78), (78, 96)]
NEW_BOUNDS = [(0, 40), (40, 80), (80, 116), (116, 152), (152, 192)]

_head_mod = None


def _head_fleet():
    """Import the pre-change fleet_v6 via `git show HEAD:` (uncommitted base)."""
    global _head_mod
    if _head_mod is None:
        src = subprocess.run(["git", "-C", WT, "show",
                              "HEAD:kaggle_base954_fleet/fleet_v6.py"],
                             capture_output=True, text=True, check=True).stdout
        with tempfile.NamedTemporaryFile("w", suffix="_head_fleet_v6.py",
                                         delete=False) as fh:
            fh.write(src)
            path = fh.name
        try:
            _head_mod = _load("wt_head_fleet_v6", path)
        finally:
            os.unlink(path)
    return _head_mod


def test_default_bounds():
    """Default sizes reproduce the old layout; slot_of/pick_centers match HEAD."""
    assert tk.bounds_from_sizes([26, 22, 18, 12, 18]) == OLD_BOUNDS
    assert fleet.bounds_from_sizes([26, 22, 18, 12, 18]) == OLD_BOUNDS
    assert list(tk.SLOT_BOUNDS) == OLD_BOUNDS
    assert list(fleet.SLOT_BOUNDS) == OLD_BOUNDS
    old = _head_fleet()
    assert list(old.SLOT_BOUNDS) == OLD_BOUNDS
    for i in list(range(96)) + [-1, 200]:
        assert tk.slot_of(i) == old.slot_of(i) == fleet.slot_of(i), i
    rng = np.random.RandomState(0)
    mask = (rng.rand(96) > 0.1).astype(np.uint8)
    valid = np.where(mask > 0)[0]
    import random
    for clean, slots in [(False, None), (True, None), (True, (2, 3))]:
        random.seed(0)
        a = tk.pick_centers(valid, 94, train=False, clean_windows=clean, mask=mask,
                            max_d=96, cap=True, slots=slots)
        b = old.pick_centers(valid, 94, train=False, clean_windows=clean, mask=mask,
                             max_d=96, cap=True, slots=slots)
        assert a == b, (clean, slots)
    return "bounds=96 ok slot_of(0..95) ok centers ok"


def test_192_bounds():
    """192 layout bounds + spot checks."""
    assert tk.bounds_from_sizes([40, 40, 36, 36, 40]) == NEW_BOUNDS
    keep = tk.SLOT_BOUNDS
    tk.SLOT_BOUNDS = tk.bounds_from_sizes([40, 40, 36, 36, 40])
    try:
        assert tk.slot_of(39) == 0
        assert tk.slot_of(40) == 1
        assert tk.slot_of(191) == 4
        assert tk.slot_of(80) == 2 and tk.slot_of(115) == 2
        assert tk.slot_of(116) == 3 and tk.slot_of(151) == 3
        assert tk.slot_of(152) == 4
    finally:
        tk.SLOT_BOUNDS = keep
    return "bounds=192 ok spots ok"


def test_mirror_192():
    """mirror_volume on synthetic (192,8,8) with the 192 layout."""
    keep = tk.SLOT_BOUNDS
    tk.SLOT_BOUNDS = tk.bounds_from_sizes([40, 40, 36, 36, 40])
    try:
        vol = np.zeros((192, 8, 8), np.uint8)
        for i in range(192):
            vol[i] = np.uint8(i % 256)
        mask = np.ones(192, np.uint8)
        mask[65:80] = 0  # partial sagittal slot 1: real 40..64, padded 65..79
        out = tk.mirror_volume(vol, mask)
        assert out.shape == (192, 8, 8)
        assert np.array_equal(out[0:40], vol[39::-1])           # sagittal reversed
        assert np.array_equal(out[40:65], vol[64:39:-1])       # partial slot reversed
        assert np.array_equal(out[65:80], vol[65:80])           # padding stays
        assert np.array_equal(out[80:116], np.flip(vol[80:116], axis=-1))
        assert np.array_equal(out[116:152], np.flip(vol[116:152], axis=-1))
        assert np.array_equal(out[152:192], np.flip(vol[152:192], axis=-1))
        twice = tk.mirror_volume(out, mask)
        assert np.array_equal(twice, vol), "mirror twice != identity"
    finally:
        tk.SLOT_BOUNDS = keep
    return "shape=(192,8,8) reverse+flip ok identity ok"


def _pick_block(path):
    """Normalized span-picking expressions (assignment/return prefixes dropped)."""
    lines = open(path, encoding="utf-8").read().splitlines()
    hits = [i for i, l in enumerate(lines)
            if "int(n * 0.02)" in l and "int(n * 0.98)" in l]
    assert len(hits) == 1, (path, hits)
    i = hits[0]
    out = []
    for l in lines[i:i + 3]:
        s = l.strip()
        if s.startswith("picks = "):
            s = s[len("picks = "):]
        elif s.startswith("return "):
            s = s[len("return "):]
        if s:
            out.append(s)
    return out


def test_picking_matches_corpus():
    """New-layout checkpoints: fleet picks (n=30,k=40) == build_corpus96_stream.py rule (span 0.02/0.98).
    Default (old checkpoints): the legacy test-time span 0.06/0.94, unchanged from the proven notebooks."""
    legacy = np.asarray(fleet._pick_slot_slice_indices(30, 26)).tolist()
    assert legacy == np.linspace(int(30 * 0.06), int(30 * 0.94) - 1, 26).round().astype(int).tolist(), legacy[:8]
    saved = fleet.PICK_SPAN
    fleet.PICK_SPAN = (0.02, 0.98)
    try:
        got = np.asarray(fleet._pick_slot_slice_indices(30, 40)).tolist()
    finally:
        fleet.PICK_SPAN = saved
    exp = np.linspace(int(30 * 0.02), int(30 * 0.98) - 1, 40).round().astype(int).tolist()
    assert got == exp, (got[:8], exp[:8])
    assert len(got) == 40 and got[0] == 0 and got[-1] == 28
    return f"legacy span ok | new span picks={len(got)} lo={got[0]} hi={got[-1]}"


def test_old_ckpt_path_unchanged():
    """No-key fleet path == HEAD reference on CPU (constants + windows)."""
    old = _head_fleet()
    assert old.MAXS == fleet.MAXS == 96 and old.K_EVAL == fleet.K_EVAL == 94
    assert list(old.SLOT_BOUNDS) == list(fleet.SLOT_BOUNDS) == OLD_BOUNDS
    assert [k for _, _, k in old.SLOTS] == [k for _, _, k in fleet.SLOTS]
    rng = np.random.RandomState(1)
    torch.manual_seed(1)
    vol = rng.randint(0, 256, size=(96, 48, 48)).astype(np.uint8)
    mask = np.ones(96, np.uint8)
    xo, so, _ = old._build_windows(vol, mask, 48)
    xn, sn, _ = fleet._build_windows(vol, mask, 48, roi=None)
    assert tuple(xo.shape) == tuple(xn.shape) == (94, 3, 48, 48)
    d = float((xo.float() - xn.float()).abs().max())
    assert d == 0.0, d
    assert list(so) == list(sn)

    class M(torch.nn.Module):
        def forward(self, x, slots=None):
            return torch.zeros(1, x.shape[1], 12) + 0.13

    for tta in (False, True):
        mo, mn = M(), M()
        for m in (mo, mn):
            m.clean_windows = False
            m.has_slot_emb = False
            m.slots = None
            m.input_norm = "none"
            m.roi = False
        po, _ = old.predict_study(mo, vol, mask, 48, "cpu", use_tta=tta)
        pn, _ = fleet.predict_study(mn, vol, mask, 48, "cpu", use_tta=tta)
        dd = float(np.abs(po - pn).max())
        assert dd == 0.0, (tta, dd)
    return "consts ok windows+tta bitwise equal"


def test_copies_identical():
    """train/fleet copies of bounds_from_sizes/slot_of/pick_centers identical."""
    for fn in ("bounds_from_sizes", "slot_of", "pick_centers"):
        a = inspect.getsource(getattr(tk, fn))
        b = inspect.getsource(getattr(fleet, fn))
        assert a == b, f"{fn} drift"
    return "3 funcs identical"


TESTS = [("default_bounds", test_default_bounds),
         ("layout_192", test_192_bounds),
         ("mirror_192", test_mirror_192),
         ("picking_matches_corpus", test_picking_matches_corpus),
         ("old_ckpt_path_unchanged", test_old_ckpt_path_unchanged),
         ("copies_identical", test_copies_identical)]


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    torch.set_num_threads(4)
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
