#!/usr/bin/env python3
"""cloud/molab_r2.py - marimo-pair payload for one round-2 fold on molab. No secrets in it.

Writes run_r2.sh into the kernel's /tmp/knee and adds: a title, a password form (Kaggle key only), a launcher that
starts run_r2.sh as its own process with the key in its environment only, and a log view refreshed every 5s.
A pid file left by a sandbox restart can name an unrelated live process, so the launcher also checks the command line.

    python3 -I cloud/molab_r2.py 1 | MARIMO_TOKEN=<pairing token> \
        bash .agents/skills/marimo-pair/scripts/execute-code.sh --url <pairing url> -
"""
import base64
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

TITLE = 'mo.md("# ROUND-2 fold __FOLD__ (__TAG__)\\n**Needs you:** paste the Kaggle key and press **Start**. It then runs on its own (~5-6 h) and uploads the result itself.")'

FORM = '''r2_key = mo.ui.dictionary({"key": mo.ui.text(kind="password", label="Kaggle API key of shivoo29", full_width=True)}).form(
    submit_button_label="Start __TAG__", clear_on_submit=True)
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
            "FOLD": "__FOLD__", "WORK": str(_w), "PY": _y.executable,
            "LABELS": "__LABELS__", "TAG": "__TAG__", "DATASET_NAME": "__DATASET_NAME__"}
    _proc = _s.Popen(["bash", str(_w / "run_r2.sh")], cwd=_w, env=_env, start_new_session=True,
                     stdout=open(_w / "run_r2.log", "ab"), stderr=_s.STDOUT)
    _pid.write_text(str(_proc.pid))
    del _k, _env
print("job __TAG__:", "running" if _r2_alive() else "not running: paste the key above and press Start")'''

TICK = 'r2_tick = mo.ui.refresh(options=["5s", "30s"], default_interval="5s", label="live log")\nr2_tick'

LOG = '''r2_tick
import pathlib as _pl, subprocess as _sp
_k = _pl.Path("/tmp/knee")
def _tail(p, n):
    return _pl.Path(p).read_text(errors="replace").splitlines()[-n:] if _pl.Path(p).exists() else []
_parts = []
for _name, _f, _n in [("job", _k / "run_r2.log", 4), ("train __TAG__", _k / "run/train___TAG__.log", 6),
                      ("parallel", _k / "run_r2_f2.log", 2), ("", _k / "run/train_cnxt_r2_f2.log", 6)]:
    _t = _tail(_f, _n)
    if _t:
        _parts += ([f"== {_name}"] if _name else []) + _t
_gpu = _sp.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
# molab kills the sandbox near 160 GB of files (kept in memory) + process RAM; guard.log shows any job the guard stopped
_files = int(_sp.run(["df", "-B1", "--output=used", "/"], capture_output=True, text=True).stdout.split()[-1]) / 2**30
try:
    _ram = int(open("/sys/fs/cgroup/memory/memory.usage_in_bytes").read()) / 2**30
except OSError:
    _ram = 0.0
_tot = _files + _ram
_parts += [f"== memory: files {_files:.0f} GB + RAM {_ram:.0f} GB = {_tot:.0f} of ~160 GB"
           + ("  !! DANGER" if _tot > 145 else "  (warning)" if _tot > 130 else "  ok")] + _tail(_k / "guard.log", 3)
mo.md("**live log** (GPU " + _gpu + ")\\n\\n```\\n" + "\\n".join(_parts or ["nothing yet"]) + "\\n```")'''


def cells(fold, labels="labels_r2.parquet", tag=None, dataset_name=None):
    tag = tag or f"cnxt_r2_f{fold}"
    dataset_name = dataset_name or f"knee-cnxt-r2-f{fold}"
    replacements = {
        "__FOLD__": str(fold),
        "__LABELS__": str(labels),
        "__TAG__": str(tag),
        "__DATASET_NAME__": str(dataset_name),
    }

    def _sub(s):
        for k, v in replacements.items():
            s = s.replace(k, v)
        return s

    return [_sub(c) for c in (TITLE, FORM, LAUNCH, TICK, LOG)]


def payload(fold, labels="labels_r2.parquet", tag=None, dataset_name=None):
    tag = tag or f"cnxt_r2_f{fold}"
    dataset_name = dataset_name or f"knee-cnxt-r2-f{fold}"
    rs = base64.b64encode(open(os.path.join(HERE, "run_r2.sh"), "rb").read()).decode()
    cell_list = cells(fold, labels=labels, tag=tag, dataset_name=dataset_name)
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
        for code in {cell_list!r}:
            ctx.run_cell(ctx.create_cell(code, hide_code=False))
        print("round-2 fold {fold} ({tag}): cells added")
'''


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="molab marimo payload generator")
    parser.add_argument("fold", choices=["0", "1", "2", "3"])
    parser.add_argument("--labels", default="labels_r2.parquet")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--dataset_name", default=None)
    args = parser.parse_args()
    tag = args.tag or (f"cnxt_r2l_f{args.fold}" if "r2l" in args.labels else f"cnxt_r2_f{args.fold}")
    ds_name = args.dataset_name or (f"knee-cnxt-r2l-f{args.fold}" if "r2l" in args.labels else f"knee-cnxt-r2-f{args.fold}")
    sys.stdout.write(payload(args.fold, labels=args.labels, tag=tag, dataset_name=ds_name))
