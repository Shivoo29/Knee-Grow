#!/usr/bin/env python3
"""cloud/molab_watchdog.py - Automated Watchdog & Auto-Healer for Molab CoAtNet Training.

Monitors the Molab runner. If the training process died or stopped unexpectedly,
it automatically re-launches the runner. Because every epoch is backed up to Kaggle,
molab_train.sh will automatically resume from the last completed epoch.
"""
import os
import sys
import time
import subprocess
import pathlib

HERE = pathlib.Path(__file__).parent.resolve()
REPO_ROOT = HERE.parent
SCRIPT = REPO_ROOT / ".agents" / "skills" / "marimo-pair" / "scripts" / "execute-code.sh"
URL = "https://sb-062136a720965abc.sb.molab.run/"
TOKEN = "REDACTED_MOLAB_TOKEN"
LOG_FILE = HERE / "watchdog.log"


def log(msg):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_code_on_molab(py_code, timeout=60):
    cmd = [
        "bash", str(SCRIPT),
        "--url", URL,
        "--token", TOKEN,
        "-c", py_code
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return res.returncode == 0, res.stdout + res.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


CHECK_CODE = """
import os, pathlib, psutil

work = pathlib.Path("/tmp/knee")
pid_file = work / "runner.pid"

is_alive = False
if pid_file.exists():
    try:
        pid = int(pid_file.read_text().strip())
        is_alive = psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:
        pass

# Check master log
master_log = ""
if (work / "molab_cnxt410.log").exists():
    master_log = (work / "molab_cnxt410.log").read_text(errors="replace")[-1000:]

# Check fold logs
fold_tails = {}
for f in range(5):
    f_log = ""
    lp = work / "run" / f"train_cnxt_r3_s410_f{f}.log"
    wp = work / f"log_cnxt_r3_s410_f{f}.txt"
    if lp.exists():
        f_log = lp.read_text(errors="replace")[-1000:]
    elif wp.exists():
        f_log = wp.read_text(errors="replace")[-1000:]
    lines = [l for l in f_log.splitlines() if "GOLD macro-AUC" in l or "DONE" in l or "loss" in l]
    if lines:
        fold_tails[f] = lines[-2:]

all_done = "ALL_DONE" in master_log

# GPU & RAM
smi = ""
try:
    import subprocess
    smi = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
except Exception:
    pass

import json
print("###JSON_START###")
print(json.dumps({
    "alive": is_alive,
    "all_done": all_done,
    "gpu": smi,
    "folds": fold_tails,
    "master_tail": master_log.splitlines()[-4:] if master_log else []
}))
print("###JSON_END###")
"""

RESTART_CODE = """
import os, pathlib, subprocess

work = pathlib.Path("/tmp/knee")
runner_script = work / "run_sequential_coat.sh"
pid_file = work / "runner.pid"

if runner_script.exists():
    master_log = open(work / "molab_master.log", "ab")
    proc = subprocess.Popen(["bash", str(runner_script)], cwd=work, start_new_session=True, stdout=master_log, stderr=subprocess.STDOUT)
    pid_file.write_text(str(proc.pid))
    print(f"RESTARTED_PID_{proc.pid}")
else:
    print("RUNNER_SCRIPT_NOT_FOUND")
"""


def check_once():
    ok, out = run_code_on_molab(CHECK_CODE)
    if not ok:
        log(f"Connection error to Molab: {out[:200]}")
        return False, "connection_failed"

    if "###JSON_START###" not in out:
        log(f"Malformed response from Molab: {out[:200]}")
        return False, "malformed_response"

    import json
    data_str = out.split("###JSON_START###")[1].split("###JSON_END###")[0].strip()
    try:
        data = json.loads(data_str)
    except Exception as e:
        log(f"JSON parse error: {e} | Raw data: {data_str[:150]}")
        return False, "parse_error"

    alive = data.get("alive", False)
    all_done = data.get("all_done", False)
    gpu = data.get("gpu", "N/A")
    folds = data.get("folds", {})
    master_tail = " | ".join(data.get("master_tail", []))

    log(f"Molab Status: Alive={alive}, Done={all_done}, GPU={gpu}")
    if master_tail: log(f"Master: {master_tail}")
    for f, lines in folds.items():
        if lines: log(f"  Fold {f}: {' | '.join(lines)}")

    if all_done:
        log("🎉 All ConvNeXt s410 folds finished successfully!")
        return True, "completed"

    if not alive:
        log("⚠️ WARNING: Runner is dead! Triggering automatic healing / restart...")
        rok, rout = run_code_on_molab(RESTART_CODE)
        log(f"Restart response: {rout}")
        return False, "restarted"

    return True, "running"


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--loop":
        log("Starting continuous watchdog loop (every 60s)...")
        while True:
            try:
                check_once()
            except Exception as e:
                log(f"Watchdog loop exception: {e}")
            time.sleep(60)
    else:
        check_once()
