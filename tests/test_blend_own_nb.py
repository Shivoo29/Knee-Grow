#!/usr/bin/env python3
"""tests/test_blend_own_nb.py - the builder emits valid notebooks; the stage keeps the base on failure.
    .venv/bin/python -I tests/test_blend_own_nb.py
"""
import json
import os
import subprocess
import sys
import tempfile
import time

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
GROUP = "cnxt=raptor_ft_cnxt_f0_swa.pt,raptor_ft_cnxt_f1_swa.pt"


def _build(td, base):
    subprocess.run([PY, "-I", os.path.join(WT, "kaggle_blend_own", "build_nb.py"), "--base", base,
                    "--slug", "rsna-knee-test", "--out", td, "--group", GROUP,
                    "--datasets", "shivoo29/knee-cnxt-folds", "--w", "0.3"], check=True, capture_output=True)
    nb = json.load(open(os.path.join(td, "rsna-knee-test.ipynb")))
    meta = json.load(open(os.path.join(td, "kernel-metadata.json")))
    return nb, meta


def _src(c):
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


def test_blendmine_mode():
    with tempfile.TemporaryDirectory() as td:
        nb, meta = _build(td, "blendmine")
        fleet_cell, stage = _src(nb["cells"][-2]), _src(nb["cells"][-1])
        ns = {}
        exec(fleet_cell, ns)
        assert ns["FLEET_V6"] == open(os.path.join(WT, "kaggle_base954_fleet", "fleet_v6.py")).read()
        for s in ("W_OWN = 0.3", "HAS_BASE = True", "STOP_AFTER_H = 7.0", "raptor_ft_cnxt_f1_swa.pt"):
            assert s in stage, s
        compile(stage, "<stage>", "exec")
        assert meta["id"] == "shivoo29/rsna-knee-test" and "shivoo29/knee-cnxt-folds" in meta["dataset_sources"]
        assert "heliosli/rsna-knee-r6-frozen-fe055406-w1" in meta["dataset_sources"]
    return "fleet source embedded verbatim, stage compiles, datasets merged"


def test_none_mode():
    with tempfile.TemporaryDirectory() as td:
        nb, meta = _build(td, "none")
        assert len(nb["cells"]) == 3 and "NB_T0 = time.time()" in _src(nb["cells"][0])
        assert "HAS_BASE = False" in _src(nb["cells"][2])
        assert meta["dataset_sources"] == ["shivoo29/knee-cnxt-folds"]
    return "standalone notebook: setup + fleet + stage"


def test_stage_keeps_base_on_failure():
    with tempfile.TemporaryDirectory() as td:
        nb, _ = _build(td, "blendmine")
        stage = _src(nb["cells"][-1])
        assert "OWN_FAILED" in stage and "continue" in stage.split("OWN_FAILED")[1][:200]
    return "a failed group is logged and skipped while a base exists"


TESTS = [("blendmine_mode", test_blendmine_mode), ("none_mode", test_none_mode),
         ("stage_keeps_base_on_failure", test_stage_keeps_base_on_failure)]


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
