#!/usr/bin/env python3
"""tests/test_roi_inference.py - CPU-only checks for ROI inference support.

Plain functions + __main__ runner (no pytest). Do NOT run any GPU job.
Prints only shapes, counts, small scalars and timings (no pixel arrays,
no UIDs, no Report text).

Covers T08:
 1. fleet_v6 ROI helpers are source-identical to roi_localize / build_roi_boxes.
 2. Fleet-computed boxes equal roi_boxes.npz rows on 3 real corpus studies.
 3. Fleet _build_windows (ROI, compartment) equals StudyWindows(train=False).
 4. Legacy (no-roi) window path unchanged vs main-branch code.
 5. tools/add_roi_keys.py backfills keys on a tiny fake checkpoint.

Run from the worktree root:
    /path/to/.venv/bin/python -I tests/test_roi_inference.py [--data DIR]
--data defaults to the main checkout (the worktree lacks the 59 GB volume
file and roi_boxes.npz; both are read read-only via mmap).
"""
import argparse
import ast
import copy
import importlib.util
import inspect
import os
import random
import subprocess
import sys
import tempfile
import textwrap
import time
import types

import numpy as np
import torch

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = "/home/codemaster29/Documents/still_learning/rsna-knee-abnormality-detection"
sys.path.insert(0, WT)

# fleet_v6 imports pydicom at module top (decode paths only); it is not
# installed in .venv, so stub it before import. Window/ROI code never touches it.
_pyd = types.ModuleType("pydicom")
_pyd_util = types.ModuleType("pydicom.pixel_data_handlers.util")
_pyd_util.apply_modality_lut = lambda *a, **k: None
_pyd_handlers = types.ModuleType("pydicom.pixel_data_handlers")
_pyd.dcmread = None
sys.modules.setdefault("pydicom", _pyd)
sys.modules.setdefault("pydicom.pixel_data_handlers", _pyd_handlers)
sys.modules.setdefault("pydicom.pixel_data_handlers.util", _pyd_util)
# cv2 is decode-path only (fleet_v6.py:195); not installed in .venv, never
# touched by window/ROI code.
sys.modules.setdefault("cv2", types.ModuleType("cv2"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fleet = _load("wt_fleet_v6", os.path.join(WT, "kaggle_base954_fleet", "fleet_v6.py"))
roi_loc = _load("wt_roi_localize", os.path.join(WT, "roi_localize.py"))
# train_knee.py does `from roi_localize import ...`; point that name at the
# already-loaded module so object identity below is meaningful.
sys.modules.setdefault("roi_localize", roi_loc)
broi = _load("wt_build_roi_boxes", os.path.join(WT, "build_roi_boxes.py"))
tk = _load("wt_train_knee", os.path.join(WT, "train_knee.py"))

DATA = MAIN


def t1_source_identity():
    """ROI helper sources in fleet_v6 match roi_localize / build_roi_boxes."""
    a = inspect.getsource(fleet._roi_square_box).replace(
        "def _roi_square_box(", "def square_box(", 1)
    b = inspect.getsource(roi_loc.square_box)
    assert a == b, "square_box text drift"
    a = inspect.getsource(fleet._roi_compartments).replace(
        "def _roi_compartments(", "def compartments(", 1)
    b = inspect.getsource(roi_loc.compartments)
    assert a == b, "compartments text drift"
    # train_knee uses these same objects (import alias chain).
    tk_text = open(os.path.join(WT, "train_knee.py"), encoding="utf-8").read()
    assert "from roi_localize import square_box as _roi_square_box" in tk_text, "import line drift"
    assert tk._roi_square_box is roi_loc.square_box, "square_box object drift"
    assert tk._roi_compartments is roi_loc.compartments, "compartments object drift"
    # Per-slice body: fleet helper vs _process_chunk `for c in range(D)` loop,
    # normalizing bare `continue` (loop) to bare `return` (helper).
    f_body = ast.parse(textwrap.dedent(inspect.getsource(fleet._roi_slice_box))).body[0].body
    if (isinstance(f_body[0], ast.Expr) and isinstance(f_body[0].value, ast.Constant)
            and isinstance(f_body[0].value.value, str)):
        f_body = f_body[1:]  # helper docstring
    b_tree = ast.parse(textwrap.dedent(inspect.getsource(broi._process_chunk)))
    loop = None
    for node in ast.walk(b_tree):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
            fn = node.iter.func
            if isinstance(fn, ast.Name) and fn.id == "range":
                args = node.iter.args
                if len(args) == 1 and isinstance(args[0], ast.Name) and args[0].id == "D":
                    loop = node
    assert loop is not None, "range(D) loop not found"

    class C2R(ast.NodeTransformer):
        def visit_Continue(self, node):
            return ast.copy_location(ast.Return(value=None), node)

    normed = [C2R().visit(copy.deepcopy(n)) for n in loop.body]
    df = ast.dump(ast.Module(body=list(f_body), type_ignores=[]), annotate_fields=False,
                  include_attributes=False)
    db = ast.dump(ast.Module(body=normed, type_ignores=[]), annotate_fields=False,
                  include_attributes=False)
    assert df == db, "slice-box body drift"
    return f"funcs=3 ok (square {len(a)}ch)"


def _open_data():
    vols = np.load(os.path.join(DATA, "ds", "corpus96", "all_vols.npy"), mmap_mode="r")
    masks = np.load(os.path.join(DATA, "ds", "corpus96", "all_masks.npy"))
    rb = np.load(os.path.join(DATA, "roi_boxes.npz"), allow_pickle=True)
    return vols, masks, rb


def t2_boxes_equal_npz():
    """Fleet slice boxes on 3 real studies equal roi_boxes.npz rows."""
    vols, masks, rb = _open_data()
    D = vols.shape[1]
    H, W = vols.shape[-2], vols.shape[-1]
    assert rb["sbox"].shape == (vols.shape[0], D, 4), rb["sbox"].shape
    assert rb["cen"].shape == (vols.shape[0], D, 2), rb["cen"].shape
    y_grid, x_grid = np.indices((H, W), dtype=np.float32)
    rows = [0, 1000, vols.shape[0] - 1]
    bad = 0
    for r in rows:
        sb = np.zeros((1, D, 4), dtype=np.int16)
        ce = np.zeros((1, D, 2), dtype=np.int16)
        for c in range(D):
            fleet._roi_slice_box(vols[r], masks[r], c, sb, ce, 0, W, H, x_grid, y_grid)
        bad += int((sb[0] != rb["sbox"][r]).sum() + (ce[0] != rb["cen"][r]).sum())
    assert bad == 0, f"{bad} box ints differ"
    return f"rows={rows} shape=({D},4)/({D},2) mismatches=0"


def _study_windows(row, uid, res, k, norm, rb):
    vols, masks, _ = _open_data()
    labels = {uid: np.zeros(12, np.float32)}
    ds = tk.StudyWindows(WT, [uid], {uid: row}, labels, res, k, train=False,
                         norm=norm, roi=True, roi_mode="compartment",
                         roi_pad=0.06, roi_overlap=0.12,
                         roi_sbox=rb["sbox"], roi_cen=rb["cen"],
                         clean_windows=True, slot_emb=True, slots=(2, 3))
    ds.vols, ds.masks = vols, masks  # bypass corpus discovery; same arrays
    return ds, vols, masks


def t3_windows_equal_study(norm="none"):
    """Fleet ROI window builder equals StudyWindows(train=False), same k."""
    res, k, row = 384, 94, 0
    vols, masks, rb = _open_data()
    uid = f"row{row}"
    ds, _, _ = _study_windows(row, uid, res, k, norm, rb)
    x_ref, _, s_ref = ds[0]
    roi = {"mode": "compartment", "pad": 0.06, "overlap": 0.12}
    x_new, slots_new, _ = fleet._build_windows(vols[row], masks[row], res,
                                               clean_win=True, slots=(2, 3),
                                               k=k, roi=roi, norm=norm)
    assert tuple(x_new.shape) == tuple(x_ref.shape), (x_new.shape, x_ref.shape)
    d = float((x_new.float() - x_ref.float()).abs().max())
    assert d < 1e-6, f"max abs diff {d}"
    assert list(slots_new) == list(s_ref.tolist()), "slot ids differ"
    assert len(slots_new) == k, len(slots_new)
    return f"norm={norm} shape={tuple(x_new.shape)} maxdiff={d:.2e} slots_ok"


def _legacy_ns():
    """Exec main-branch window code (via git show) in an isolated namespace."""
    src = subprocess.run(["git", "-C", WT, "show",
                          "c0f8758:kaggle_base954_fleet/fleet_v6.py"],   # last commit before ROI inference,
                         capture_output=True, text=True, check=True).stdout
    tree = ast.parse(src)
    want_fn = {"slot_of", "pick_centers", "apply_norm", "mirror_tri", "predict_study"}
    mod = ast.Module(body=[], type_ignores=[])
    consts = {}
    ev = {"np": np}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want_fn:
            mod.body.append(node)
        elif isinstance(node, ast.Assign):
            seg = ast.get_source_segment(src, node)
            if seg is None:
                continue
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if names and all(n == n.upper() or n.startswith("_") for n in names):
                try:
                    for n in names:
                        ev[n] = eval(compile(ast.Expression(node.value), "<c>", "eval"), ev)
                    consts.update({n: ev[n] for n in names})
                except Exception:
                    continue
    ns = {"np": np, "torch": torch, "F": torch.nn.functional,
          "time": time, "random": random}
    ns.update(consts)
    exec(compile(mod, "<legacy>", "exec"), ns)  # noqa: S102 (test-only, local git content)
    return ns, consts


def t4_legacy_path_unchanged():
    """No-roi builder + predict_study match main-branch behavior on CPU."""
    ns, consts = _legacy_ns()
    for key in ("SLOT_BOUNDS", "MAXS", "IMG", "K_EVAL"):
        assert getattr(fleet, key) == consts[key] or list(getattr(fleet, key)) == list(consts[key]), key
    rng = np.random.RandomState(0)
    torch.manual_seed(0)
    dev = "cpu"
    diffs = []
    for res in (64, int(consts["IMG"])):
        vol = rng.randint(0, 256, size=(96, 64, 64)).astype(np.uint8)
        mask = np.ones(96, dtype=np.uint8)

        class M(torch.nn.Module):
            def forward(self, x, slots=None):
                return torch.zeros(1, x.shape[1], 12) + 0.13

        for tta in (False, True):
            m_old, m_new = M(), M()
            for m in (m_old, m_new):
                m.clean_windows = False
                m.has_slot_emb = False
                m.slots = None
                m.input_norm = "none"
            p_old, _ = ns["predict_study"](m_old, vol, mask, res, dev, use_tta=tta)
            m_new.roi = False
            x_new, _, _ = fleet._build_windows(vol, mask, res, k=int(consts["K_EVAL"]))
            p_new, _ = fleet.predict_study(m_new, vol, mask, res, dev, use_tta=tta)
            d = float(np.abs(p_old - p_new).max())
            diffs.append(d)
            assert d < 1e-6, (res, tta, d)
    return f"res=[64,{consts['IMG']}] tta=[F,T] maxdiff={max(diffs):.2e}"


def t5_add_roi_keys():
    """tools/add_roi_keys.py backfills the 4 keys, keeps the rest."""
    import torch as _t
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "raptor_ft_test.pt")
        ck = {"model": {"w": _t.zeros(2, 2)}, "res": 96, "norm": "none",
              "slots": "2,3", "arch": "x", "lab": ["a"]}
        _t.save(ck, src)
        r = subprocess.run([sys.executable, "-I",
                            os.path.join(WT, "tools", "add_roi_keys.py"),
                            src, "--roi_mode", "compartment"],
                           capture_output=True, text=True, check=True)
        out = src[:-3] + "_roi.pt"
        assert os.path.isfile(out), r.stdout + r.stderr
        back = _t.load(out, map_location="cpu", weights_only=False)
        assert back["roi"] is True and back["roi_mode"] == "compartment"
        assert back["roi_pad"] == 0.06 and back["roi_overlap"] == 0.12
        assert back["res"] == 96 and list(back["model"].keys()) == ["w"]
        assert len(back) == len(ck) + 4 - 4 + 4, (len(back), len(ck))
    return "keys=4 kept_rest out_suffix=_roi.pt"


TESTS = [("t1_source_identity", t1_source_identity),
         ("t2_boxes_equal_npz", t2_boxes_equal_npz),
         ("t3_windows_none", lambda: t3_windows_equal_study("none")),
         ("t3_windows_imagenet", lambda: t3_windows_equal_study("imagenet")),
         ("t4_legacy_path_unchanged", t4_legacy_path_unchanged),
         ("t5_add_roi_keys", t5_add_roi_keys)]


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=MAIN)
    a = ap.parse_args()
    DATA = a.data
    torch.set_num_threads(4)
    fails = 0
    t0 = time.time()
    for name, fn in TESTS:
        t = time.time()
        try:
            info = fn()
            print(f"PASS {name} {info} ({time.time()-t:.1f}s)", flush=True)
        except Exception as e:
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e} ({time.time()-t:.1f}s)", flush=True)
    print(f"done {len(TESTS)-fails}/{len(TESTS)} pass ({time.time()-t0:.1f}s)", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
