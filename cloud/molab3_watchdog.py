#!/usr/bin/env python3
"""cloud/molab3_watchdog.py - Automated Watchdog & OOM Handler for ConvNeXt s310 (Instance 3).

Monitors Instance 3 (sb-5db0f9a383670956):
- Checks runner liveness, GPU util, memory, and per-fold training progress.
- Detects OOM in epoch 0 and auto-relaunches with PARALLEL=2 if necessary.
- Logs telemetry and macro-AUC updates to cloud/watchdog3.log.
"""
import os
import sys
import time
import subprocess
import pathlib
import json

HERE = pathlib.Path(__file__).parent.resolve()
REPO_ROOT = HERE.parent
SCRIPT = REPO_ROOT / ".agents" / "skills" / "marimo-pair" / "scripts" / "execute-code.sh"
URL = "https://sb-5db0f9a383670956.sb.molab.run/"
TOKEN = "REDACTED_MOLAB_TOKEN"
LOG_FILE = HERE / "watchdog3.log"


def log(msg):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_code_on_molab3(py_code, timeout=60):
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
import os, pathlib, psutil, json, subprocess

work = pathlib.Path("/tmp/knee")
pid_file = work / "runner.pid"
master_log = work / "molab_cnxt310.log"

is_alive = False
if pid_file.exists():
    try:
        pid = int(pid_file.read_text().strip())
        is_alive = psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:
        pass

# Check fold logs
fold_tails = {}
has_oom = False
for f in range(5):
    lpath = work / "run" / f"train_cnxt_r3_s310_f{f}.log"
    txt_path = work / f"log_cnxt_r3_s310_f{f}.txt"
    content = ""
    if lpath.exists():
        content = lpath.read_text(errors="replace")[-1500:]
    elif txt_path.exists():
        content = txt_path.read_text(errors="replace")[-1500:]
    
    if "out of memory" in content.lower() or "outofmemoryerror" in content.lower():
        has_oom = True
    
    lines = [l for l in content.splitlines() if "GOLD macro-AUC" in l or "DONE" in l or "loss" in l]
    fold_tails[f] = lines[-2:] if lines else []

# GPU
smi = ""
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
except Exception:
    pass

mlog_tail = ""
if master_log.exists():
    mlog_tail = master_log.read_text(errors="replace")[-1000:]

all_done = "ALL_DONE" in mlog_tail

print("###JSON_START###")
print(json.dumps({
    "alive": is_alive,
    "all_done": all_done,
    "has_oom": has_oom,
    "gpu": smi,
    "folds": fold_tails,
    "master_tail": mlog_tail.splitlines()[-4:] if mlog_tail else []
}))
print("###JSON_END###")
"""

RESTART_PARALLEL2_CODE = """
import os, pathlib, subprocess, signal, psutil

work = pathlib.Path("/tmp/knee")
pid_file = work / "runner.pid"

# Kill old runner and python children
if pid_file.exists():
    try:
        mpid = int(pid_file.read_text().strip())
        parent = psutil.Process(mpid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except Exception:
        pass

# Also pkill train_knee if any orphan
subprocess.run(["pkill", "-9", "-f", "train_knee.py"])

# Launch with PARALLEL=2
# Note: KAGGLE_USERNAME and credentials are kept in the environment of the parent or marimo kernel
env = os.environ.copy()
env["PRESET"] = "cnxt"
env["SEED_BASE"] = "310"
env["FOLDS"] = "0 1 2 3 4"
env["PARALLEL"] = "2"
env["WORK"] = "/tmp/knee"

mlog = open(work / "molab_cnxt310.log", "ab")
mlog.write(b"\\n=== OOM Detected: Auto-Restarting with PARALLEL=2 ===\\n")
mlog.flush()

proc = subprocess.Popen(
    ["bash", "-c", "cd /tmp/knee && bash molab_train.sh"],
    cwd=work,
    env=env,
    start_new_session=True,
    stdout=mlog,
    stderr=subprocess.STDOUT
)
pid_file.write_text(str(proc.pid))
print(f"RELAUNCHED_PARALLEL2_PID_{proc.pid}")
"""


def check_once():
    ok, out = run_code_on_molab3(CHECK_CODE, timeout=30)
    if not ok:
        log(f"Molab3 notice: {out[:120]}")
        return False, "conn_fail"

    if "###JSON_START###" not in out:
        log(f"Molab3 output: {out[:120]}")
        return False, "no_json"

    data_str = out.split("###JSON_START###")[1].split("###JSON_END###")[0].strip()
    try:
        data = json.loads(data_str)
    except Exception as e:
        log(f"Parse error: {e}")
        return False, "json_err"

    alive = data.get("alive", False)
    all_done = data.get("all_done", False)
    has_oom = data.get("has_oom", False)
    gpu = data.get("gpu", "N/A")
    folds = data.get("folds", {})
    master_tail = " | ".join(data.get("master_tail", []))

    log(f"Molab3 Status: Alive={alive}, Done={all_done}, GPU={gpu}")
    if master_tail:
        log(f"Master Log: {master_tail[-200:]}")
    
    for f, lines in folds.items():
        if lines:
            log(f"  Fold {f}: {' | '.join(lines)}")

    if all_done:
        log("🎉 All 5 ConvNeXt folds completed on Molab3!")
        return True, "all_done"

    if has_oom:
        log("⚠️ OOM detected in fold logs! Relaunching with PARALLEL=2...")
        rok, rout = run_code_on_molab3(RESTART_PARALLEL2_CODE, timeout=30)
        log(f"Relaunch result: {rout.strip()}")
        return False, "oom_relaunch"

    return True, "running"


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--loop":
        log("Starting Molab3 60-second watchdog loop...")
        while True:
            try:
                check_once()
            except Exception as e:
                log(f"Watchdog exception: {e}")
            time.sleep(60)
    else:
        check_once()
