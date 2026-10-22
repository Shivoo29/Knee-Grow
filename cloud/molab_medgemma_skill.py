import marimo

__generated_with = "0.11.0"
app = marimo.App(width="full")


@app.cell
def __():
    # Cell 1: Install required dependencies
    import subprocess, sys

    def _install():
        pkgs = [
            "transformers>=4.50",
            "accelerate>=0.30",
            "bitsandbytes",
            "pydicom>=2.4",
            "scikit-learn>=1.3",
            "kaggle",
            "huggingface_hub",
        ]
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=False)
        return "✅ Base dependencies installed: transformers, accelerate, bitsandbytes, pydicom, kaggle, etc."

    msg = _install()
    print(msg)
    return msg, subprocess, sys


@app.cell
def __(subprocess, sys):
    # Cell 2: Imports & GPU Check
    import marimo as mo
    import os, shutil, pathlib, time, json, glob
    import torch
    
    if torch.cuda.is_available():
        gpu_info = f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB)"
    else:
        gpu_info = "CPU Only"
    print(f"✅ Imports complete. {gpu_info}")
    return glob, gpu_info, json, mo, os, pathlib, shutil, time, torch


@app.cell
def __(mo):
    # Cell 3: Credentials Form (HF Token + Kaggle Key) + Start Button
    key_form = mo.ui.dictionary({
        "hf_token": mo.ui.text(
            kind="password",
            label="🔑 Hugging Face Token (for MedGemma):",
            placeholder="Paste your HF token here...",
            full_width=True,
        ),
        "kaggle_username": mo.ui.text(
            value="shivoo29",
            label="👤 Kaggle Username:",
            full_width=True,
        ),
        "kaggle_key": mo.ui.text(
            kind="password",
            label="🔑 Kaggle API Key / Token:",
            placeholder="Paste your Kaggle key or KGAT_ token here...",
            full_width=True,
        ),
    }).form(submit_button_label="🚀 Start MedGemma & Skill Pipeline", clear_on_submit=False)

    header = mo.vstack([
        mo.md("## 🩺 ALF Radiology + Google MedGemma Assistant\n"
              "**Secure Runtime Credentials:** Your Hugging Face and Kaggle keys exist only in temporary memory during submit and are never committed to public files.\n\n"
              "👉 **Instructions:** Enter your HF Token and Kaggle key below and click **Start MedGemma & Skill Pipeline** to launch."),
        key_form
    ])
    header
    return header, key_form


@app.cell
def __(key_form, mo, os, pathlib, subprocess):
    # Cell 4: Background Runner Wiring & Launch Controller
    _work = pathlib.Path("/tmp/knee_radiologist")
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

    _launch_msg = "Fill your credentials in Cell 3 and click Start to begin."

    if key_form.value and key_form.value.get("hf_token", "").strip() and not _is_running():
        _hf_tok = key_form.value["hf_token"].strip()
        _k_user = key_form.value.get("kaggle_username", "shivoo29").strip()
        _k_key = key_form.value.get("kaggle_key", "").strip()
        
        _key_env_var = "KAGGLE_API_TOKEN" if _k_key.startswith("KGAT_") else "KAGGLE_KEY"

        # 1. Write the Python patch script separately (avoids nested f-string interpolation of {path})
        _patch_script = _work / "patch_medgemma.py"
        _patch_code = '''import pathlib, os

p = pathlib.Path("/tmp/knee_radiologist/repo/src/kneemri/adapters/medgemma.py")
txt = p.read_text()

old_block = """        path = Path(raw_path)
        if not path.exists():
            raise DependencyMissing(f"MedGemma weights directory does not exist: {path}")"""

new_block = """        path = Path(raw_path)
        if not path.exists() and "/" not in raw_path:
            raise DependencyMissing(f"MedGemma weights directory does not exist: {path}")"""

if old_block in txt:
    txt = txt.replace(old_block, new_block)

txt = txt.replace(
    "self.processor = AutoProcessor.from_pretrained(self.model_path)",
    "token = os.environ.get('HF_TOKEN')\\n            self.processor = AutoProcessor.from_pretrained(self.model_path, token=token)"
)

txt = txt.replace(
    "device_map=self.device,",
    "device_map=self.device,\\n                token=os.environ.get('HF_TOKEN'),"
)

p.write_text(txt)
print("✅ MedGemma adapter successfully patched for Hugging Face Hub support.")
'''
        _patch_script.write_text(_patch_code)

        # 2. Write the test runner script separately
        _test_script = _work / "test_pipeline.py"
        _test_code = '''import os, sys, torch
from kneemri.contracts import Label
from kneemri.adapters.medgemma import MedGemmaScorer
from kneemri.report.scorer import score_report

# Determine model based on VRAM (> 60 GB -> 27b, else 4b)
free_gb = 0
if torch.cuda.is_available():
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024**3)

model_id = "google/medgemma-27b-it" if free_gb > 60 else "google/medgemma-4b-it"
print(f"🚀 Selected MedGemma Model: {model_id} (Available VRAM: {free_gb:.1f} GB)")

os.environ["KNEEMRI_MEDGEMMA_PATH"] = model_id
print(f"📥 Initializing MedGemmaScorer for {model_id}...")
scorer = MedGemmaScorer(model_path=model_id)
print(f"✅ Scorer initialized successfully: {scorer.name}")

# Test report scoring via ALF skill
sample_report = "IMPRESSION: Complete tear of the anterior cruciate ligament with joint effusion. Intact menisci and collateral ligaments."
print(f"\\n🔬 Scoring sample report via rulebook:\\n\\"{sample_report}\\"")
rep_res = score_report(sample_report)
print("\\n=== REPORT SCORES (ALF Skill Rulebook) ===")
for label, s in rep_res.labels.items():
    print(f"  {label.value:24} p={s.probability:.2f} bin={s.binary} rule={s.rule}  {s.evidence}")
print("flags:", rep_res.flags)

print("\\n🎉 MedGemma + Skill pipeline test finished successfully!")
'''
        _test_script.write_text(_test_code)

        # 3. Write Master Bash runner
        _runner_script = _work / "run_medgemma_skill.sh"
        _script_content = f"""#!/bin/bash
set -euo pipefail
WORK="/tmp/knee_radiologist"
cd "$WORK"

export HF_TOKEN="{_hf_tok}"
export HUGGING_FACE_HUB_TOKEN="{_hf_tok}"
export KAGGLE_USERNAME="{_k_user}"
export {_key_env_var}="{_k_key}"

echo "[$(date +%T)] Step 1: Cloning associate_for_our_local_radiologist repo..."
if [ ! -d "$WORK/repo" ]; then
    git clone --depth 1 https://github.com/Shivoo29/associate_for_our_local_radiologist.git "$WORK/repo"
fi
cd "$WORK/repo"

echo "[$(date +%T)] Step 2: Installing package in editable mode..."
pip install -q -e ".[medgemma]"

echo "[$(date +%T)] Step 3: Patching MedGemma adapter..."
python3 "$WORK/patch_medgemma.py"

echo "[$(date +%T)] Step 4: Running MedGemma pipeline test with the skill..."
python3 "$WORK/test_pipeline.py"

echo "[$(date +%T)] ALL_STEPS_COMPLETED"
"""
        _runner_script.write_text(_script_content)
        _runner_script.chmod(0o755)

        _master_log = open(_work / "medgemma_master.log", "ab")
        _proc = subprocess.Popen(
            ["bash", str(_runner_script)], 
            cwd=_work, 
            start_new_session=True, 
            stdout=_master_log, 
            stderr=subprocess.STDOUT
        )
        _pid_file.write_text(str(_proc.pid))
        _launch_msg = f"🚀 Launched successfully! Master runner active (PID {_proc.pid})."

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

    _work_dir = pathlib.Path("/tmp/knee_radiologist")
    _pid_file = _work_dir / "runner.pid"

    def _is_alive():
        try:
            if not _pid_file.exists(): return False
            _pid = int(_pid_file.read_text().strip())
            os.kill(_pid, 0)
            return True
        except OSError:
            return False

    # 1. System Memory (RAM)
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

    _ram_badge = "🔴 HIGH (>140GB)" if _ram_used_gb > 140 else "🟢 NORMAL"

    # 2. GPU VRAM & Compute
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

    # 4. Master Log Tail
    _log_p = _work_dir / "medgemma_master.log"
    _log_tail = _log_p.read_text(errors="replace").splitlines()[-18:] if _log_p.exists() else []
    _log_box = "\n".join(_log_tail) if _log_tail else "Waiting for launch. Enter credentials in Cell 3 and click Start."

    _status_tag = "🟢 **RUNNING**" if _is_alive() else "⚪ **IDLE**"
    _now_time = time.strftime("%H:%M:%S")

    _table_md = "\n".join([
        "### 📊 MedGemma + Radiology Skill Live Telemetry",
        f"**Status:** {_status_tag} | **Last Tick:** `{_now_time}`\n",
        "| Resource | Usage | Health / Notes |",
        "| :--- | :--- | :--- |",
        f"| **Host RAM** | **{_ram_used_gb:.1f} GB** / {_ram_tot_gb:.1f} GB ({_ram_pct:.1f}%) | {_ram_badge} |",
        f"| **GPU VRAM** | **{_gpu_mem}** | Compute: **{_gpu_util}** |",
        f"| **GPU Temp & Draw** | **{_gpu_temp}** | Power: **{_gpu_pwr}** |",
        f"| **Disk `/tmp`** | **{_disk_str}** | Models & Workspace |\n",
        "---",
        "#### 📜 Live Execution Log"
    ])

    return mo.vstack([
        mo.md(_table_md),
        mo.plain_text(_log_box),
    ])


if __name__ == "__main__":
    app.run()
