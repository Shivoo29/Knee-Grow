#!/usr/bin/env python3
"""tests/test_gemma_labels.py - synthetic checks for tools/gemma_labels.py (no server, no real reports).
    .venv/bin/python -I tests/test_gemma_labels.py
"""
import csv
import importlib.util
import json
import os
import sys
import tempfile
import time

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("gl", os.path.join(WT, "tools", "gemma_labels.py"))
gl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gl)


def _csv(path, n=30, n_gold=10):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["StudyInstanceUID", "Report"] + gl.LAB)
        for i in range(n):
            lab = [str(float((i + j) % 2)) for j in range(len(gl.LAB))] if i < n_gold else [""] * len(gl.LAB)
            w.writerow([f"1.2.{i}", f"FINDINGS line one of study {i}.\nIMPRESSION: line two, \"quoted\"." if i != 17 else "", *lab])


def _fake(calls):
    def ask(base_url, model, system, report, timeout=600):
        calls.append((system, report))
        if len(calls) % 5 == 0:
            return "sorry"                                   # unparsable once, then the retry parses
        return "ok " + json.dumps({k: (len(calls) * 7 + j) % 101 for j, k in enumerate(gl.LAB)})
    return ask


def test_two_fold_no_leak():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "train.csv")
        _csv(p)
        jobs, systems = gl.plan(gl.read_rows(p))
        assert [s for _, s in jobs[:10]].count("B") == 5 and len(jobs) == 29   # 10 gold first; empty report skipped
        for r, s in jobs:
            own = f"study {r['StudyInstanceUID'].split('.')[-1]}."
            if s != "all":
                assert own not in systems[s], "a gold report appears among its own examples"
        assert systems["all"].count("Example ") == 10
    return "gold rows first, each sees only the other half; non-gold rows see all 10 examples"


def test_run_parse_resume():
    with tempfile.TemporaryDirectory() as td:
        p, out = os.path.join(td, "train.csv"), os.path.join(td, "out.csv")
        _csv(p)
        calls = []
        gl.main(["--train", p, "--model", "m", "--out", out, "--limit", "12"], ask_fn=_fake(calls))
        gl.main(["--train", p, "--model", "m", "--out", out], ask_fn=_fake(calls))
        rows = gl.read_rows(out)
        assert len(rows) == 29 and len({r["StudyInstanceUID"] for r in rows}) == 29
        assert list(rows[0]) == ["StudyInstanceUID"] + gl.LAB + ["shots", "ok"]
        assert all(r["ok"] == "1" and 0 <= float(r["ACL"]) <= 1 for r in rows)
    return "29 rows over two runs (resume), header exact, every row parsed after retry"


def test_parse_scales():
    j = lambda v: json.dumps({k: v for k in gl.LAB})
    assert gl.parse("x " + j(35)) == [0.35] * 12 and gl.parse(j(0.35)) == [0.35] * 12
    assert gl.parse(j(1)) == [0.01] * 12 and gl.parse('{"ACL": 3}') is None
    return "percent and fraction answers both map to 0-1; incomplete JSON rejected"


TESTS = [("two_fold_no_leak", test_two_fold_no_leak), ("run_parse_resume", test_run_parse_resume),
         ("parse_scales", test_parse_scales)]


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
