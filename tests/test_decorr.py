#!/usr/bin/env python3
"""tests/test_decorr.py - CPU checks for tools/decorr.py on synthetic predictions (no data, no ids printed).
    .venv/bin/python -I tests/test_decorr.py
"""
import importlib.util
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("decorr", os.path.join(WT, "tools", "decorr.py"))
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)
LAB = dc.LAB


def _frames(n=2000, seed=0, noise_b=None):
    rng = np.random.RandomState(seed)
    ids = [f"s{i}" for i in range(n)]
    y = (rng.rand(n, 12) < 0.3).astype(float)
    a = y + rng.randn(n, 12)
    b = a.copy() if noise_b is None else y + noise_b * rng.randn(n, 12)
    mk = lambda v: pd.DataFrame(v, index=ids, columns=LAB)
    return mk(a), mk(b), mk(y)


def test_identical():
    A, B, T = _frames()
    r, n = dc.compare(A, B, T)
    assert n == 2000
    assert np.allclose(r["rho_pos"], 1) and np.allclose(r["rho_neg"], 1)
    assert np.allclose(r["auc_blend"], r["auc_a"]) and np.allclose(r["fix"], 0) and np.allclose(r["brk"], 0)
    return "identical members: rho=1, no fixes, blend AUC = AUC"


def test_independent():
    A, B, T = _frames(noise_b=1.0)
    r, _ = dc.compare(A, B, T)
    assert r["rho_pos"].abs().max() < 0.15 and r["rho_neg"].abs().max() < 0.15
    assert (r["auc_blend"] > r[["auc_a", "auc_b"]].max(axis=1)).all()
    return f"independent: |rho|<0.15, blend beats both (mean +{(r['auc_blend']-r['auc_a']).mean():.3f})"


def test_alignment_and_disjoint():
    A, B, T = _frames(noise_b=1.0)
    r1, _ = dc.compare(A, B, T)
    r2, _ = dc.compare(A, B.sample(frac=1.0, random_state=1), T)
    assert np.allclose(r1.values, r2.values)
    B2 = B.copy()
    B2.index = [f"x{i}" for i in range(len(B2))]
    try:
        dc.compare(A, B2, T)
    except SystemExit as e:
        assert "no common study ids" in str(e)
        return "shuffled B gives identical numbers; disjoint ids refused"
    raise AssertionError("disjoint ids were accepted")


def test_npz_loader():
    A, _, _ = _frames(n=50)
    with tempfile.TemporaryDirectory() as td:
        np.savez(os.path.join(td, "dc_t_fold0.npz"), pred=A.values[:25], ids=np.array(A.index[:25]))
        np.savez(os.path.join(td, "dc_t_fold1.npz"), pred=A.values[25:], ids=np.array(A.index[25:]))
        df = dc.load_preds(os.path.join(td, "dc_t_fold*.npz"))
    assert df.shape == (50, 12) and np.allclose(df.loc[A.index].values, A.values)
    return "two fold npz files concatenate by id"


TESTS = [("identical", test_identical), ("independent", test_independent),
         ("alignment_and_disjoint", test_alignment_and_disjoint), ("npz_loader", test_npz_loader)]


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
