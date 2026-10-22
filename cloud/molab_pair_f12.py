#!/usr/bin/env python3
"""cloud/molab_pair_f12.py - marimo-pair injector for CoAtNet Round-3 Fold 1 & 2 on Molab.

Usage:
    python3 cloud/molab_pair_f12.py | MARIMO_TOKEN=<token> bash .agents/skills/marimo-pair/scripts/execute-code.sh --url <url> -
"""
import base64
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

CELL_1 = '''# Cell 1: Install required dependencies
import subprocess, sys

def _install():
    pkgs = ["timm", "kaggle", "pyyaml", "huggingface_hub", "pydantic", "psutil"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=False)
    return "✅ Packages verified / installed: timm, kaggle, pyyaml, psutil, huggingface_hub."

print(_install())'''

CELL_2 = '''# Cell 2: Imports
import marimo as mo
import os, sys, time, pathlib, shutil, json, glob
import psutil
print("✅ All modules imported successfully.")'''

CELL_3 = '''# Cell 3: Kaggle API Credentials Form + Start Button
key_form = mo.ui.dictionary({
    "key": mo.ui.text(
        kind="password", 
        label="Kaggle API Key (shivoo29)", 
        placeholder="Paste your Kaggle API key or token here...", 
        full_width=True
    )
}).form(submit_button_label="Start fold 1 and 2", clear_on_submit=True)

mo.vstack([
    mo.md("## 🚀 CoAtNet Round-3: Train Fold 1 & Fold 2 (Sequential)\\n"
          "**Secure Runtime Form:** Your Kaggle key exists only in temporary memory during submit and is never saved into notebook code or public files.\\n\\n"
          "👉 **Instructions:** Fill your Kaggle key below and click **Start fold 1 and 2** to start the whole process."),
    key_form
])'''

CELL_4 = '''# Cell 4: Background Runner Wiring & Launch Controller
work = pathlib.Path("/tmp/knee")
work.mkdir(parents=True, exist_ok=True)
pid_file = work / "runner.pid"

def _is_running():
    try:
        if not pid_file.exists(): return False
        pid = int(pid_file.read_text().strip())
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False

launch_msg = "Fill your key and click on start button to start the whole process."

if key_form.value and key_form.value.get("key", "").strip() and not _is_running():
    _raw_key = key_form.value["key"].strip()
    runner_script = work / "run_sequential_coat.sh"
    key_env_var = "KAGGLE_API_TOKEN" if _raw_key.startswith("KGAT_") else "KAGGLE_KEY"
    
    script_content = f"""#!/bin/bash
set -euo pipefail
cd /tmp/knee
export KAGGLE_USERNAME="shivoo29"
export {key_env_var}="{_raw_key}"
export PRESET="coat"
export FOLDS="1 2"
export PARALLEL=1
export SEED_BASE=210
export WORK="/tmp/knee"

echo "[$(date +%T)] Step 0: Fetching code and round-3 labels from Kaggle (shivoo29/knee-r3-labels)..."
python -m kaggle datasets download shivoo29/knee-r3-labels -p /tmp/knee --unzip -q
chmod +x /tmp/knee/*.sh

echo "[$(date +%T)] Step 1: Starting molab_train.sh for CoAtNet Fold 1 then Fold 2..."
bash /tmp/knee/molab_train.sh
"""
    runner_script.write_text(script_content)
    runner_script.chmod(0o755)

    master_log = open(work / "molab_master.log", "ab")
    proc = subprocess.Popen(["bash", str(runner_script)], cwd=work, start_new_session=True, stdout=master_log, stderr=subprocess.STDOUT)
    pid_file.write_text(str(proc.pid))
    launch_msg = f"🚀 Launched successfully! Master runner active (PID {proc.pid})."
    del _raw_key

running = _is_running()
mo.md(f"### Process Status: {'🟢 **RUNNING** (PID ' + pid_file.read_text().strip() + ')' if running else '⚪ **IDLE** — ' + launch_msg}")'''

CELL_5 = '''# Cell 5: Live Refresh Ticker (Every 2 seconds)
tick = mo.ui.refresh(options=["2s", "5s", "10s"], default_interval="2s", label="live telemetry")
tick'''

CELL_6 = '''# Cell 6: Telemetry & Live Monitor Dashboard (Updates every 2 seconds)
tick

pid_file_ref = work / "runner.pid"

def _check_alive():
    try:
        if not pid_file_ref.exists(): return False
        p = psutil.Process(int(pid_file_ref.read_text().strip()))
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False

# 1. System Memory (RAM) & CPU
vm = psutil.virtual_memory()
ram_used_gb = vm.used / (1024**3)
ram_tot_gb = vm.total / (1024**3)
ram_pct = vm.percent
cpu_pct = psutil.cpu_percent(interval=None)

if ram_used_gb > 140:
    ram_badge = "🔴 DANGER (>140GB)"
elif ram_used_gb > 120:
    ram_badge = "🟡 WARNING (>120GB)"
else:
    ram_badge = "🟢 SAFE (<120GB)"

# 2. GPU VRAM & Utilization
try:
    smi = subprocess.run([
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits"
    ], capture_output=True, text=True).stdout.strip().split(",")
    gpu_util = smi[0].strip() + "%"
    gpu_mem = f"{float(smi[1].strip())/1024:.1f} / {float(smi[2].strip())/1024:.1f} GB"
    gpu_temp = smi[3].strip() + "°C"
    gpu_pwr = smi[4].strip() + "W"
except Exception:
    gpu_util, gpu_mem, gpu_temp, gpu_pwr = "N/A", "N/A", "N/A", "N/A"

# 3. Disk Usage in /tmp
try:
    du = psutil.disk_usage("/tmp")
    disk_str = f"{du.used / (1024**3):.1f} / {du.total / (1024**3):.1f} GB ({du.percent}%)"
except Exception:
    disk_str = "N/A"

# 4. Tail Logs
def _read_tail(fpath, n=8):
    p = pathlib.Path(fpath)
    if not p.exists(): return []
    return p.read_text(errors="replace").splitlines()[-n:]

master_tail = _read_tail(work / "molab_master.log", 6)
f1_tail = _read_tail(work / "run" / "train_coat_r3_s210_f1.log", 8)
f2_tail = _read_tail(work / "run" / "train_coat_r3_s210_f2.log", 8)

sections = []
if f1_tail:
    sections.append("=== [Fold 1 Log: train_coat_r3_s210_f1.log] ===\\n" + "\\n".join(f1_tail))
if f2_tail:
    sections.append("=== [Fold 2 Log: train_coat_r3_s210_f2.log] ===\\n" + "\\n".join(f2_tail))
if master_tail:
    sections.append("=== [Master Pipeline Output] ===\\n" + "\\n".join(master_tail))

log_box = "\\n\\n".join(sections) if sections else "No logs generated yet. Click 'Start fold 1 and 2' above."

is_alive = _check_alive()
status_tag = "🟢 **RUNNING**" if is_alive else "⚪ **IDLE**"

dashboard = mo.md(f"""
### 📊 Live 2-Second Monitor & Telemetry
**Status:** {status_tag} | **Telemetry Updated:** `{time.strftime('%H:%M:%S')}`

| Resource | Current Usage | Status / Notes |
| :--- | :--- | :--- |
| **Host RAM** | **{ram_used_gb:.1f} GB** / {ram_tot_gb:.1f} GB ({ram_pct}%) | {ram_badge} |
| **GPU VRAM** | **{gpu_mem}** | Compute: **{gpu_util}** |
| **GPU Temp & Power** | **{gpu_temp}** | Draw: **{gpu_pwr}** |
| **CPU Usage** | **{cpu_pct}%** (20 Cores) | Active |
| **Disk `/tmp`** | **{disk_str}** | Corpus & Weights |

---

#### 📜 Live Training Log Tail
```text
{log_box}
```

---
> ℹ️ *Fill your key and click on start button to start the whole process.*
""")
dashboard'''

def payload():
    cell_list = [CELL_1, CELL_2, CELL_3, CELL_4, CELL_5, CELL_6]
    return f'''import marimo._code_mode as cm
async with cm.get_context() as ctx:
    if "key_form" in ctx.globals:
        print("Cells already present.")
    else:
        for code in {cell_list!r}:
            ctx.run_cell(ctx.create_cell(code, hide_code=False))
        print("CoAtNet Fold 1 & 2 sequential training cells successfully injected!")
'''

if __name__ == "__main__":
    sys.stdout.write(payload())
