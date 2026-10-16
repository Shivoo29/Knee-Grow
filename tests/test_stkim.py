#!/usr/bin/env python3
"""tests/test_stkim.py - CPU-only checks for ACMIL stochastic top-K window masking (--stkim_k).

Plain `def test_*` + `__main__` runner (no pytest). No training, no GPU, no data.
    .venv/bin/python -I tests/test_stkim.py
"""
import importlib.util
import inspect
import os
import sys
import time
import types

import torch

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
F, K, N = 16, 10, 12


def _model(k, p):
    torch.manual_seed(0)
    m = tk.RaptorClassifier(torch.nn.Identity(), F_dim=F, n=N, drop=0.0)
    m.stkim_k, m.stkim_p = k, p
    return m


def test_eval_unchanged():
    feats = torch.randn(2, K, F)
    a, b = _model(0, 0.7).eval(), _model(3, 0.7).eval()
    with torch.no_grad():
        assert torch.equal(a.head(feats), b.head(feats))
    return "eval logits identical with and without stkim"


def test_train_masks_topk():
    m = _model(2, 1.0).train()
    feats = torch.randn(2, K, F)
    with torch.no_grad():
        h = m.norm(feats)
        top = m.att(h).topk(2, dim=1).indices
        w = m._attend(h)
    assert torch.all(w.gather(1, top) == 0)
    assert torch.allclose(w.sum(1), torch.ones(2, N))
    return "p=1: each finding's top-2 windows get weight 0, rows sum to 1"


def test_p_zero_is_noop():
    feats = torch.randn(2, K, F)
    m0, m1 = _model(0, 0.7).train(), _model(3, 0.0).train()
    with torch.no_grad():
        assert torch.allclose(m0.head(feats), m1.head(feats))
    return "p=0 identical to off"


def test_fleet_copy_identical():
    assert inspect.getsource(tk.RaptorClassifier) == inspect.getsource(fleet.RaptorClassifier)
    return "train and fleet RaptorClassifier share source"


TESTS = [("eval_unchanged", test_eval_unchanged), ("train_masks_topk", test_train_masks_topk),
         ("p_zero_is_noop", test_p_zero_is_noop), ("fleet_copy_identical", test_fleet_copy_identical)]


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
