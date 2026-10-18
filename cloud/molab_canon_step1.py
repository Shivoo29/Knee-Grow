#!/usr/bin/env python3
"""cloud/molab_canon_step1.py - marimo-pair payload for Knee-Side Canonicalization Step 1 on Molab.

Uploads run_canon_step1.sh, predict_corpus.py, and eval_canon_step1.py into the kernel's /tmp/knee.
Adds:
  1. Title with task description & safety reminders.
  2. Password form for Kaggle API key (zero secrets stored in code or files).
  3. Detached process launcher with PID / cmdline guard.
  4. Auto-refreshing status and log view (5s interval).
"""
import base64
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TITLE = '''mo.md("""# 🦵 Knee-Side Canonicalization — Step 1: Inference-Only Test
**Hypothesis:** Mirroring left knees into canonical right knees eliminates the left-knee deficit on the 5 side-specific findings.
- **Scope:** Fold 0 held-out studies (~1,088 studies: 518 Left, 570 Right).
- **Model:** `raptor_ft_cnxt_sev_f0_swa.pt` from `shivoo29/knee-cnxt-sev-folds`.
- **Variants:** Left Plain vs Left Mirrored, Right Plain vs Right Mirrored (control), and Flip-TTA.
- **Success Gate:** Left-mirrored beats Left-plain on side-specific findings by $\\\\ge +0.005$ while Right-mirrored is not better than Right-plain.

**Action needed:** Paste Kaggle API key below and click **Start Step 1**.""")'''

FORM = '''canon_key = mo.ui.dictionary({
    "key": mo.ui.text(kind="password", label="Kaggle API key (username: shivoo29)", full_width=True)
}).form(submit_button_label="Start Step 1 Inference Test", clear_on_submit=True)
canon_key'''

LAUNCH = '''import os as _o, subprocess as _s, sys as _y, pathlib as _p
_w = _p.Path("/tmp/knee"); _pid = _w / "canon_step1.pid"


def _is_alive():
    try:
        return "run_canon_step1.sh" in open(f"/proc/{_pid.read_text().strip()}/cmdline").read()
    except OSError:
        return False


if canon_key.value and canon_key.value["key"].strip() and not _is_alive():
    _k = canon_key.value["key"].strip()
    _env = {
        **_o.environ,
        "KAGGLE_USERNAME": "shivoo29",
        ("KAGGLE_API_TOKEN" if _k.startswith("KGAT_") else "KAGGLE_KEY"): _k,
        "WORK": str(_w),
        "PY": _y.executable
    }
    _proc = _s.Popen(
        ["bash", str(_w / "run_canon_step1.sh")],
        cwd=_w,
        env=_env,
        start_new_session=True,
        stdout=open(_w / "run_canon_step1.log", "ab"),
        stderr=_s.STDOUT
    )
    _pid.write_text(str(_proc.pid))
    del _k, _env

print("Step 1 Job Status:", "RUNNING (PID " + _pid.read_text().strip() + ")" if _is_alive() else "NOT RUNNING (Enter Kaggle key above and click Start)")
'''

TICK = 'canon_tick = mo.ui.refresh(options=["5s", "15s", "30s"], default_interval="5s", label="live log")\ncanon_tick'

LOG = '''canon_tick
import pathlib as _pl, subprocess as _sp
_k = _pl.Path("/tmp/knee")


def _tail(p, n):
    return _pl.Path(p).read_text(errors="replace").splitlines()[-n:] if _pl.Path(p).exists() else []


_parts = []
# 1. Main runner log
_t = _tail(_k / "run_canon_step1.log", 6)
if _t:
    _parts += ["== run_canon_step1.log =="] + _t

# 2. Plain inference log
_tp = _tail(_k / "run/oof_plain.log", 4)
if _tp:
    _parts += ["== oof_plain.log =="] + _tp

# 3. Mirrored inference log
_tm = _tail(_k / "run/oof_mirrored.log", 4)
if _tm:
    _parts += ["== oof_mirrored.log =="] + _tm

# 4. Final report if available
_rep = _tail(_k / "run/step1_report.txt", 40)
if _rep:
    _parts += ["== step1_report.txt =="] + _rep

_gpu = _sp.run(
    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
    capture_output=True, text=True
).stdout.strip()

_files = int(_sp.run(["df", "-B1", "--output=used", "/"], capture_output=True, text=True).stdout.split()[-1]) / 2**30
try:
    _ram = int(open("/sys/fs/cgroup/memory/memory.usage_in_bytes").read()) / 2**30
except OSError:
    _ram = 0.0
_tot = _files + _ram
_parts += [
    f"== Memory: files {_files:.1f} GB + RAM {_ram:.1f} GB = {_tot:.1f} / ~160 GB"
    + (" [!! DANGER]" if _tot > 145 else " [WARNING]" if _tot > 130 else " [OK]")
]

mo.md(f"### Live Execution Telemetry (GPU: {_gpu})\\n```\\n" + "\\n".join(_parts or ["Waiting to start..."]) + "\\n```")
'''


def cells():
    return [TITLE, FORM, LAUNCH, TICK, LOG]


def payload():
    sh_b64 = base64.b64encode(open(os.path.join(HERE, "run_canon_step1.sh"), "rb").read()).decode()
    pred_b64 = base64.b64encode(open(os.path.join(ROOT, "tools/predict_corpus.py"), "rb").read()).decode()
    eval_b64 = base64.b64encode(open(os.path.join(ROOT, "tools/eval_canon_step1.py"), "rb").read()).decode()

    return f'''import base64, pathlib
import marimo._code_mode as cm
w = pathlib.Path("/tmp/knee"); w.mkdir(exist_ok=True)
run_dir = w / "run"; run_dir.mkdir(exist_ok=True)

(w / "run_canon_step1.sh").write_bytes(base64.b64decode("{sh_b64}"))
(run_dir / "predict_corpus.py").write_bytes(base64.b64decode("{pred_b64}"))
(run_dir / "eval_canon_step1.py").write_bytes(base64.b64decode("{eval_b64}"))

async with cm.get_context() as ctx:
    if "canon_key" in ctx.globals:
        print("Cells already injected; runner scripts refreshed")
    else:
        if "mo" not in ctx.globals:
            ctx.run_cell(ctx.create_cell("import marimo as mo", hide_code=False))
        for code in {cells()!r}:
            ctx.run_cell(ctx.create_cell(code, hide_code=False))
        print("Knee Canonicalization Step 1: cells successfully injected into notebook")
'''


if __name__ == "__main__":
    sys.stdout.write(payload())
