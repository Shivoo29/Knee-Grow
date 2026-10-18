#!/usr/bin/env python3
"""cloud/molab_r2.py - marimo-pair payload for one round-2 fold on molab. No secrets in it.

Writes run_r2.sh into the kernel's /tmp/knee and adds: a title, a password form (Kaggle key only), a launcher that
starts run_r2.sh as its own process with the key in its environment only, and a log view refreshed every minute.
A pid file left by a sandbox restart can name an unrelated live process, so the launcher also checks the command line.

    python3 -I cloud/molab_r2.py 1 | MARIMO_TOKEN=<pairing token> \
        bash .agents/skills/marimo-pair/scripts/execute-code.sh --url <pairing url> -
"""
import base64
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

TITLE = 'mo.md("# ROUND-2 fold __FOLD__\\n**Needs you:** paste the Kaggle key and press **Start**. It then runs on its own (~5-6 h) and uploads the result itself.")'

FORM = '''r2_key = mo.ui.dictionary({"key": mo.ui.text(kind="password", label="Kaggle API key of shivoo29", full_width=True)}).form(
    submit_button_label="Start round-2 fold __FOLD__", clear_on_submit=True)
r2_key'''

LAUNCH = '''import os as _o, subprocess as _s, sys as _y, pathlib as _p
_w = _p.Path("/tmp/knee"); _pid = _w / "run_r2.pid"


def _r2_alive():
    try:
        return "run_r2.sh" in open(f"/proc/{_pid.read_text().strip()}/cmdline").read()
    except OSError:
        return False


if r2_key.value and r2_key.value["key"].strip() and not _r2_alive():
    _k = r2_key.value["key"].strip()
    _env = {**_o.environ, "KAGGLE_USERNAME": "shivoo29", ("KAGGLE_API_TOKEN" if _k.startswith("KGAT_") else "KAGGLE_KEY"): _k,
            "FOLD": "__FOLD__", "WORK": str(_w), "PY": _y.executable}
    _proc = _s.Popen(["bash", str(_w / "run_r2.sh")], cwd=_w, env=_env, start_new_session=True,
                     stdout=open(_w / "run_r2.log", "ab"), stderr=_s.STDOUT)
    _pid.write_text(str(_proc.pid))
    del _k, _env
print("round-2 fold __FOLD__:", "running" if _r2_alive() else "not running: paste the key above and press Start")'''

TICK = 'r2_tick = mo.ui.refresh(options=["1m"], default_interval="1m")\nr2_tick'

LOG = '''r2_tick
import pathlib as _pl
_a = _pl.Path("/tmp/knee/run_r2.log"); _b = _pl.Path("/tmp/knee/run/train_cnxt_r2_f__FOLD__.log")
_lines = (_a.read_text().splitlines()[-4:] if _a.exists() else []) + \\
         ([x for x in _b.read_text().splitlines() if x.startswith("ep")][-6:] if _b.exists() else [])
mo.md("**log**\\n\\n```\\n" + "\\n".join(_lines or ["nothing yet"]) + "\\n```")'''


def cells(fold):
    return [c.replace("__FOLD__", fold) for c in (TITLE, FORM, LAUNCH, TICK, LOG)]


def payload(fold):
    rs = base64.b64encode(open(os.path.join(HERE, "run_r2.sh"), "rb").read()).decode()
    return f'''import base64, pathlib
import marimo._code_mode as cm
w = pathlib.Path("/tmp/knee"); w.mkdir(exist_ok=True)
(w / "run_r2.sh").write_bytes(base64.b64decode("{rs}"))
async with cm.get_context() as ctx:
    if "r2_key" in ctx.globals:
        print("cells already there; run_r2.sh refreshed")
    else:
        if "mo" not in ctx.globals:
            ctx.run_cell(ctx.create_cell("import marimo as mo", hide_code=False))
        for code in {cells(fold)!r}:
            ctx.run_cell(ctx.create_cell(code, hide_code=False))
        print("round-2 fold {fold}: cells added")
'''


if __name__ == "__main__":
    if sys.argv[1:] not in (["0"], ["1"], ["2"], ["3"]):
        sys.exit("usage: molab_r2.py <fold 0-3>")
    sys.stdout.write(payload(sys.argv[1]))
