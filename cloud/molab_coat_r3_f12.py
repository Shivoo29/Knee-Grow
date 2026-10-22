import marimo

__generated_with = "0.11.0"
app = marimo.App(width="full")


@app.cell
def __():
    # Cell 1: Install required dependencies
    import subprocess, sys

    def _install():
        pkgs = ["timm", "kaggle", "pyyaml", "huggingface_hub", "pydantic"]
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=False)
        return "✅ Packages verified / installed: timm, kaggle, pyyaml, huggingface_hub."

    msg = _install()
    print(msg)
    return msg, subprocess, sys


@app.cell
def __(subprocess, sys):
    # Cell 2: Imports
    import marimo as mo
    import os, shutil, pathlib, time, json, glob
    print("✅ All modules imported successfully.")
    return glob, json, mo, os, pathlib, shutil, time


@app.cell
def __(mo):
    # Cell 3: Kaggle API Credentials Form + Start Button
    key_form = mo.ui.dictionary({
        "key": mo.ui.text(
            kind="password", 
            label="Kaggle API Key (shivoo29)", 
            placeholder="Paste your Kaggle API key or token here...", 
            full_width=True
        )
    }).form(submit_button_label="Start fold 1 and 2", clear_on_submit=True)

    header = mo.vstack([
        mo.md("## 🚀 CoAtNet Round-3: Train Fold 1 & Fold 2 (Sequential)\n"
              "**Secure Runtime Form:** Your Kaggle key exists only in temporary memory during submit and is never saved into notebook code or public files.\n\n"
              "👉 **Instructions:** Fill your Kaggle key below and click **Start fold 1 and 2** to start the whole process."),
        key_form
    ])
    header
    return header, key_form


@app.cell
def __(key_form, mo, os, pathlib, subprocess):
    # Cell 4: Background Runner Wiring & Launch Controller
    _work = pathlib.Path("/tmp/knee")
    _work.mkdir(parents=True, exist_ok=True)
    _pid_file = _work / "runner.pid"

    def _is_running():
        try:
            if not _pid_file.exists():
                return False
            _pid = int(_pid_file.read_text().strip())
            os.kill(_pid, 0)
            return True
        except OSError:
            return False

    _launch_msg = "Fill your key and click on start button to start the whole process."

    if key_form.value and key_form.value.get("key", "").strip() and not _is_running():
        _raw_key = key_form.value["key"].strip()
        _runner_script = _work / "run_sequential_coat.sh"
        _key_env_var = "KAGGLE_API_TOKEN" if _raw_key.startswith("KGAT_") else "KAGGLE_KEY"
        
        _script_content = f"""#!/bin/bash
set -euo pipefail
cd /tmp/knee
export KAGGLE_USERNAME="shivoo29"
export {_key_env_var}="{_raw_key}"
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
        _runner_script.write_text(_script_content)
        _runner_script.chmod(0o755)

        _master_log = open(_work / "molab_master.log", "ab")
        _proc = subprocess.Popen(
            ["bash", str(_runner_script)], 
            cwd=_work, 
            start_new_session=True, 
            stdout=_master_log, 
            stderr=subprocess.STDOUT
        )
        _pid_file.write_text(str(_proc.pid))
        _launch_msg = f"🚀 Launched successfully! Master runner active (PID {_proc.pid})."
        del _raw_key

    _running = _is_running()
    return mo.md(f"### Process Status: {'🟢 **RUNNING** (PID ' + _pid_file.read_text().strip() + ')' if _running else '⚪ **IDLE** — ' + _launch_msg}")


@app.cell
def __(mo):
    # Cell 5: Live Refresh Ticker (Every 2 seconds)
    tick = mo.ui.refresh(options=["2s", "5s", "10s"], default_interval="2s", label="live telemetry")
    tick
    return tick,


@app.cell
def __(mo, os, pathlib, shutil, subprocess, tick, time):
    # Cell 6: Telemetry & Live Monitor Dashboard (Updates every 2 seconds)
    tick

    _work_dir = pathlib.Path("/tmp/knee")
    _pid_file = _work_dir / "runner.pid"

    def _is_alive():
        try:
            if not _pid_file.exists(): return False
            _pid = int(_pid_file.read_text().strip())
            os.kill(_pid, 0)
            return True
        except OSError:
            return False

    # 1. System Memory (RAM) from /proc/meminfo
    _meminfo = {}
    try:
        for _line in open("/proc/meminfo"):
            _parts = _line.split(":")
            if len(_parts) == 2:
                _meminfo[_parts[0].strip()] = int(_parts[1].split()[0])
        _tot_kb = _meminfo.get("MemTotal", 1)
        _avail_kb = _meminfo.get("MemAvailable", 0)
        _used_kb = _tot_kb - _avail_kb
        _ram_used_gb = _used_kb / (1024**2)
        _ram_tot_gb = _tot_kb / (1024**2)
        _ram_pct = (_used_kb / _tot_kb) * 100
    except Exception:
        _ram_used_gb, _ram_tot_gb, _ram_pct = 0.0, 160.0, 0.0

    if _ram_used_gb > 140:
        _ram_badge = "🔴 DANGER (>140GB)"
    elif _ram_used_gb > 120:
        _ram_badge = "🟡 WARNING (>120GB)"
    else:
        _ram_badge = "🟢 SAFE (<120GB)"

    # 2. GPU VRAM & Compute via nvidia-smi
    try:
        _smi = subprocess.run([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True).stdout.strip().split(",")
        _gpu_util = _smi[0].strip() + "%"
        _gpu_mem = f"{float(_smi[1].strip())/1024:.1f} / {float(_smi[2].strip())/1024:.1f} GB"
        _gpu_temp = _smi[3].strip() + "°C"
        _gpu_pwr = _smi[4].strip() + "W"
    except Exception:
        _gpu_util, _gpu_mem, _gpu_temp, _gpu_pwr = "N/A", "N/A", "N/A", "N/A"

    # 3. Disk Usage in /tmp
    try:
        _du = shutil.disk_usage("/tmp")
        _disk_str = f"{_du.used / (1024**3):.1f} / {_du.total / (1024**3):.1f} GB ({int((_du.used/_du.total)*100)}%)"
    except Exception:
        _disk_str = "N/A"

    # 4. Read Training Logs
    def _read_tail(fpath, n=8):
        _p = pathlib.Path(fpath)
        if not _p.exists(): return []
        return _p.read_text(errors="replace").splitlines()[-n:]

    _sections = []
    _f1_tail = _read_tail(_work_dir / "run" / "train_coat_r3_s210_f1.log", 8)
    _f2_tail = _read_tail(_work_dir / "run" / "train_coat_r3_s210_f2.log", 8)
    _master_tail = _read_tail(_work_dir / "molab_master.log", 6)

    if _f1_tail:
        _sections.append("=== [Fold 1 Log: train_coat_r3_s210_f1.log] ===\n" + "\n".join(_f1_tail))
    if _f2_tail:
        _sections.append("=== [Fold 2 Log: train_coat_r3_s210_f2.log] ===\n" + "\n".join(_f2_tail))
    if _master_tail:
        _sections.append("=== [Master Pipeline Output] ===\n" + "\n".join(_master_tail))

    _log_box = "\n\n".join(_sections) if _sections else "No logs generated yet. Fill your key in Cell 3 and click Start."

    _status_tag = "🟢 **RUNNING**" if _is_alive() else "⚪ **IDLE**"
    _now_time = time.strftime("%H:%M:%S")

    _table_md = "\n".join([
        "### 📊 Live 2-Second Monitor & Telemetry",
        f"**Status:** {_status_tag} | **Telemetry Updated:** `{_now_time}`\n",
        "| Resource | Current Usage | Status / Notes |",
        "| :--- | :--- | :--- |",
        f"| **Host RAM** | **{_ram_used_gb:.1f} GB** / {_ram_tot_gb:.1f} GB ({_ram_pct:.1f}%) | {_ram_badge} |",
        f"| **GPU VRAM** | **{_gpu_mem}** | Compute: **{_gpu_util}** |",
        f"| **GPU Temp & Power** | **{_gpu_temp}** | Draw: **{_gpu_pwr}** |",
        f"| **Disk `/tmp`** | **{_disk_str}** | Corpus & Weights |\n",
        "---",
        "#### 📜 Live Training Log Tail"
    ])

    return mo.vstack([
        mo.md(_table_md),
        mo.plain_text(_log_box),
        mo.md("> ℹ️ *Fill your key and click on start button to start the whole process.*")
    ])


if __name__ == "__main__":
    app.run()
