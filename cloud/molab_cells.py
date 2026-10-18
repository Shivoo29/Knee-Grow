#!/usr/bin/env python3
"""cloud/molab_cells.py - marimo-pair payload that sets up one student fold in a molab notebook. No secrets in it.

molab notebooks are public but undiscoverable: anyone with the link sees the cell code and outputs and can fork
the notebook. So the Kaggle key and the fastMRI links never go in a cell, a file molab keeps, or an output. The
payload writes run_fold.sh into the kernel's /tmp/knee and adds four cells:
  1. a password form (Kaggle key + the two fastMRI links) that the user fills in in the browser
  2. a launcher: on submit, starts run_fold.sh as a background process with the values in its environment only
  3. + 4. a log view refreshed every minute (also keeps the session from looking idle)
The run is its own process (new session), so an interrupted cell or a dropped pairing call never stops it.

    python3 -I cloud/molab_cells.py 1 | MARIMO_TOKEN=<pairing token> \
        bash .agents/skills/marimo-pair/scripts/execute-code.sh --url <pairing url> -
"""
import base64
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

SETUP = '''import os, subprocess, sys
from pathlib import Path

FOLD = "__FOLD__"  # one student fold per instance
work = Path("__WORK__")
work.mkdir(parents=True, exist_ok=True)


def _check(v):
    if not v or not all(x.strip() for x in v.values()):
        return "Fill in all three fields."
    if "knee_DICOMs_batch1" not in v["batch1"] or "knee_DICOMs_batch2" not in v["batch2"]:
        return "Links look wrong or swapped: the batch1 link goes in the batch1 field."


setup = mo.ui.dictionary({
    "kaggle_key": mo.ui.text(kind="password", label="Kaggle API key or token of shivoo29", full_width=True),
    "batch1": mo.ui.text(kind="password", label="fastMRI knee_DICOMs_batch1 link", full_width=True),
    "batch2": mo.ui.text(kind="password", label="fastMRI knee_DICOMs_batch2 link", full_width=True),
}).form(submit_button_label=f"Start fold {FOLD}", clear_on_submit=True, validate=_check)
setup'''

LAUNCH = '''def _alive(pid_file):
    try:  # a finished run stays a zombie until reaped, so read its state
        return open(f"/proc/{pid_file.read_text().strip()}/stat").read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


_pid = work / "run_fold.pid"
if setup.value and not _alive(_pid):
    _urls = work / "fastmri_urls.txt"
    _urls.touch(mode=0o600)
    _urls.write_text("".join(f"knee_DICOMs_{b} {setup.value[b].strip()}\\n" for b in ("batch1", "batch2")))
    _key = setup.value["kaggle_key"].strip()  # new-style tokens (KGAT_...) go in KAGGLE_API_TOKEN
    _env = {**os.environ, "KAGGLE_USERNAME": "shivoo29", ("KAGGLE_API_TOKEN" if _key.startswith("KGAT_") else "KAGGLE_KEY"): _key,
            "FOLD": FOLD, "WORK": str(work), "FASTMRI_URLS": str(_urls), "PY": sys.executable,
            "WORKERS": "3", "BS": "2", "ACCUM": "8"}  # molab: files and loader RAM share 160 GiB; bs 8 x 8 workers ran out
    (work / "fm").mkdir(exist_ok=True)
    (work / "fm" / ".done_knee_DICOMs_batch1").touch()  # fastMRI batch2 only: ~30 GB in memory, not ~86 GB
    _proc = subprocess.Popen(["bash", str(work / "run_fold.sh")], cwd=work, env=_env, start_new_session=True,
                             stdout=open(work / "run_fold.log", "ab"), stderr=subprocess.STDOUT)
    _pid.write_text(str(_proc.pid))
print(f"fold {FOLD}:", "running" if _alive(_pid) else "not running: fill in the form above and press Start")'''

TICK = '''tick = mo.ui.refresh(options=["1m"], default_interval="1m", label="log refresh")
tick'''

LOG = '''tick
_lines = []
for _f, _n in ((work / "run_fold.log", 20), (work / "run" / f"train_cnxt_fm_f{FOLD}.log", 8)):
    if _f.exists():
        _lines += [f"== {_f.name}"] + _f.read_text(errors="replace").splitlines()[-_n:]
mo.plain_text("\\n".join(_lines) or "no log yet")'''


def cells(fold, work="/tmp/knee"):
    return [SETUP.replace("__FOLD__", fold).replace("__WORK__", work), LAUNCH, TICK, LOG]


def payload(fold):
    rf = base64.b64encode(open(os.path.join(HERE, "run_fold.sh"), "rb").read()).decode()
    return f'''import base64, pathlib
import marimo._code_mode as cm
w = pathlib.Path("/tmp/knee")
w.mkdir(exist_ok=True)
(w / "run_fold.sh").write_bytes(base64.b64decode("{rf}"))
async with cm.get_context() as ctx:
    if "setup" in ctx.globals:
        print("cells already there; run_fold.sh refreshed")
    else:
        if "mo" not in ctx.globals:
            ctx.run_cell(ctx.create_cell("import marimo as mo", hide_code=False))
        for code in {cells(fold)!r}:
            ctx.run_cell(ctx.create_cell(code, hide_code=False))
        print("fold {fold}: cells added, run_fold.sh written")
'''


if __name__ == "__main__":
    if sys.argv[1:] not in (["0"], ["1"], ["2"], ["3"]):
        sys.exit("usage: molab_cells.py <fold 0-3>")
    sys.stdout.write(payload(sys.argv[1]))
