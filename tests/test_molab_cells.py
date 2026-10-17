#!/usr/bin/env python3
"""tests/test_molab_cells.py - cloud/molab_cells.py cells run outside molab: the form, the launcher (secrets reach
the run's environment and the 600 links file, never the cell output) and the log view.
    uv run -q --with marimo==0.25.1 python -I tests/test_molab_cells.py
"""
import ast
import contextlib
import io
import os
import sys
import tempfile
import time
import types

import marimo as mo

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WT, "cloud"))
import molab_cells as mc  # noqa: E402

SECRET = {"kaggle_key": " kSECRETk ", "batch1": "https://s3/knee_DICOMs_batch1.tar.xz?sig=S1",
          "batch2": "https://s3/knee_DICOMs_batch2.tar.xz?sig=S2"}
FAKE_RUN = 'echo "FOLD=$FOLD user=$KAGGLE_USERNAME key=${#KAGGLE_KEY} links=$(wc -l < "$FASTMRI_URLS")"; sleep 2\n'


def _run(code, ns):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(code, ns)
    return out.getvalue()


def test_cells():
    with tempfile.TemporaryDirectory() as td:
        work = os.path.join(td, "knee")
        setup, launch, tick, log = mc.cells("2", work)
        ns = {"mo": mo}
        _run(setup, ns)
        assert ns["setup"].value is None and ns["FOLD"] == "2"
        swapped = dict(SECRET, batch1=SECRET["batch2"], batch2=SECRET["batch1"])
        msg = ns["_check"](swapped) + ns["_check"](dict(SECRET, kaggle_key=" "))
        assert "sig=" not in msg and "SECRET" not in msg and ns["_check"](SECRET) is None
        assert "not running" in _run(launch, dict(ns))          # nothing submitted yet
        open(os.path.join(work, "run_fold.sh"), "w").write(FAKE_RUN)
        ns["setup"] = types.SimpleNamespace(value=SECRET)       # the user pressed Start
        out = _run(launch, dict(ns)) + _run(launch, dict(ns))   # second run while alive: no second process
        assert out.count("running") == 2 and "SECRET" not in out and "sig=" not in out, out
        urls = os.path.join(work, "fastmri_urls.txt")
        assert os.stat(urls).st_mode & 0o777 == 0o600
        assert open(urls).read() == "".join(f"knee_DICOMs_{b} {SECRET[b]}\n" for b in ("batch1", "batch2"))
        time.sleep(3)
        assert open(os.path.join(work, "run_fold.log")).read() == "FOLD=2 user=shivoo29 key=8 links=2\n"
        ns.update(tick=None)
        _run(log, ns)
        compile(mc.payload("2"), "payload", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    return "form validates without echoing, launcher starts once, secrets only in env + 600 file, payload compiles"


def main():
    t = time.time()
    try:
        print(f"PASS cells {test_cells()} ({time.time()-t:.1f}s)", flush=True)
        return 0
    except Exception as e:
        print(f"FAIL cells: {type(e).__name__}: {e} ({time.time()-t:.1f}s)", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
