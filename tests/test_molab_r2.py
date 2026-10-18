#!/usr/bin/env python3
"""tests/test_molab_r2.py - cloud/molab_r2.py cells outside molab: the launcher starts run_r2.sh once with the key in
its environment only, ignores a stale pid file, and the payload compiles.
    uv run -q --with marimo==0.25.1 python -I tests/test_molab_r2.py
"""
import ast
import contextlib
import io
import os
import pathlib
import sys
import time
import types

import marimo as mo

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cloud"))
import molab_r2 as m  # noqa: E402


def main():
    title, form, launch, tick, log = m.cells("1")
    w = pathlib.Path("/tmp/knee")
    w.mkdir(exist_ok=True)
    (w / "run_r2.sh").write_text('echo "FOLD=$FOLD key=${#KAGGLE_KEY}"; sleep 3\n')
    (w / "run_r2.pid").write_text(str(os.getpid()))  # stale pid naming a live, unrelated process
    (w / "run_r2.log").unlink(missing_ok=True)
    ns = {"mo": mo, "r2_key": types.SimpleNamespace(value={"key": " kSECRETk "})}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(launch, dict(ns))
        exec(launch, dict(ns))  # second run while alive: no second process
    assert out.getvalue().count("running") == 2 and "SECRET" not in out.getvalue(), out.getvalue()
    time.sleep(1)
    assert (w / "run_r2.log").read_text() == "FOLD=1 key=8\n", (w / "run_r2.log").read_text()
    ns["r2_tick"] = None
    exec(log, ns)
    exec(form, {"mo": mo})
    compile(m.payload("1"), "payload", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    print("PASS molab_r2: launcher starts once, ignores a stale pid, key only in env, payload compiles")


if __name__ == "__main__":
    main()
