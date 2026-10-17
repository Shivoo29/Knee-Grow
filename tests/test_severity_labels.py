#!/usr/bin/env python3
"""tests/test_severity_labels.py - synthetic checks for tools/severity_labels.py.
    .venv/bin/python -I tests/test_severity_labels.py
"""
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("sl", os.path.join(WT, "tools", "severity_labels.py"))
sl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sl)


def _grades(n=5, seed=0):
    rng = np.random.RandomState(seed)
    d = {"StudyInstanceUID": [f"s{i}" for i in range(n)]}
    for f, opts in sl.V.items():
        p = rng.dirichlet(np.ones(len(opts)), size=n)
        for j, o in enumerate(sorted(opts)):   # Clef orders choice options alphabetically
            d[f"{f}|{o}"] = p[:, j]
    d["n_tokens"] = [3000] * n
    return pd.DataFrame(d)


def test_expected_value():
    g = _grades()
    sev = sl.to_labels(g, sl.V)
    f, o = "Effusion", ["large", "moderate", "none_or_physiologic", "not_mentioned", "trace_or_small", "unspecified"]
    want = sum(g[f"{f}|{k}"] * sl.V[f][k] for k in o)
    assert np.allclose(sev[f].values, want.values, atol=1e-6) and sev["StudyInstanceUID"].tolist() == g["StudyInstanceUID"].tolist()
    one = g.copy()
    one[[c for c in g.columns if c.startswith("Effusion|")]] = 0.0
    one["Effusion|moderate"] = 1.0
    assert np.allclose(sl.to_labels(one, sl.V)["Effusion"], 0.85)
    return "label = sum P(grade) * V[grade]; a certain 'moderate' effusion maps to 0.85"


def test_presence():
    g = _grades()
    anyp = sl.to_labels(g, sl.PRESENCE)
    want = 1 - g["ACL|not_mentioned"] - g["ACL|intact"]
    assert np.allclose(anyp["ACL"].values, want.values, atol=1e-6)
    return "presence = 1 - P(not mentioned) - P(normal)"


def test_option_mismatch_refused():
    g = _grades().rename(columns={"Fracture|possible": "Fracture|maybe"})
    try:
        sl.to_labels(g, sl.V)
    except SystemExit as e:
        assert "Fracture" in str(e)
        return "renamed option refused"
    raise AssertionError("mismatched options accepted")


def test_notebook_options_match():
    nb = open(os.path.join(WT, "molab", "clef_grades.py")).read()
    for f, opts in sl.V.items():
        for o in opts:
            assert f'"{o}"' in nb, (f, o)
    return "every V option name appears in the molab notebook"


TESTS = [("expected_value", test_expected_value), ("presence", test_presence),
         ("option_mismatch_refused", test_option_mismatch_refused), ("notebook_options_match", test_notebook_options_match)]


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
