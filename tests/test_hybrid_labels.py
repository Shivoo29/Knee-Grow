#!/usr/bin/env python3
"""tests/test_hybrid_labels.py - synthetic checks for tools/build_hybrid_labels.py.
    .venv/bin/python -I tests/test_hybrid_labels.py
"""
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("bhl", os.path.join(WT, "tools", "build_hybrid_labels.py"))
bhl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bhl)
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
       'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def _t(seed, n=20, prefix="w_"):
    rng = np.random.RandomState(seed)
    d = pd.DataFrame(rng.rand(n, 12).astype("float32"), columns=[prefix + c if prefix else c for c in LAB])
    d.insert(0, "StudyInstanceUID", [f"s{i}" for i in range(n)])
    return d


def test_override_and_weights():
    base, over, w = _t(0, prefix=""), _t(1, prefix=""), _t(2)
    w["w_MCL"] = 0.0
    out, ow = bhl.build(base, over.sample(frac=1.0, random_state=3), ["MCL"], w)
    assert np.allclose(out["MCL"].values, over["MCL"].values)
    keep = [c for c in LAB if c != "MCL"]
    assert np.allclose(out[keep].values, base[keep].values)
    assert (ow["w_MCL"] == 1.0).all() and np.allclose(ow["w_ACL"].values, w["w_ACL"].values)
    return "MCL replaced (aligned by id), others untouched, w_MCL reset to 1"


def test_missing_studies_refused():
    base, over = _t(0, prefix=""), _t(1, n=10, prefix="")
    try:
        bhl.build(base, over, ["MCL"])
    except SystemExit as e:
        assert "lacks 10" in str(e)
        return "override missing base studies refused"
    raise AssertionError("missing studies accepted")


TESTS = [("override_and_weights", test_override_and_weights), ("missing_studies_refused", test_missing_studies_refused)]


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
