import marimo._code_mode as cm

cell_vblA = """key_form = mo.ui.dictionary({
    "key": mo.ui.text(
        kind="password", 
        label="Kaggle API Key (shivoo29)", 
        placeholder="Paste your Kaggle API key or token here...", 
        full_width=True
    )
}).form(submit_button_label="Start ConvNeXt s410 (All 5 Folds)")

header = mo.vstack([
    mo.md(\"\"\"## 🚀 ConvNeXt Round-3: Train All 5 Folds (s410)
**Host Environment:** corpus192 already present in `/tmp/knee`! Instant training start.
**Secure Runtime Form:** Kaggle credentials exist only in runtime memory and are passed strictly as environment variables to the subprocess. Never saved to disk, cells, or logs.

👉 **Status:** Ready to train Folds 0, 1, 2, 3, 4 sequentially.\"\"\"),
    key_form
])
header"""

cell_bkHC = """work = pathlib.Path("/tmp/knee")
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

launch_msg = "Enter key in form and click start to begin ConvNeXt s410."

if key_form.value and key_form.value.get("key", "").strip() and not _is_running():
    _raw_key = key_form.value["key"].strip()
    _env = os.environ.copy()
    _env["KAGGLE_USERNAME"] = "shivoo29"
    if _raw_key.startswith("KGAT_"):
        _env["KAGGLE_API_TOKEN"] = _raw_key
    else:
        _env["KAGGLE_KEY"] = _raw_key
    _env["PRESET"] = "cnxt"
    _env["SEED_BASE"] = "410"
    _env["FOLDS"] = "0 1 2 3 4"
    _env["PARALLEL"] = "1"
    _env["WORKERS"] = "4"
    _env["WORK"] = "/tmp/knee"

    master_log = open(work / "molab_cnxt410.log", "ab")
    proc = subprocess.Popen(
        ["bash", "-c", "cd /tmp/knee && bash molab_train.sh"],
        cwd=work,
        env=_env,
        start_new_session=True,
        stdout=master_log,
        stderr=subprocess.STDOUT
    )
    pid_file.write_text(str(proc.pid))
    launch_msg = f"🚀 Launched successfully! Master runner active (PID {proc.pid}). Logs: /tmp/knee/molab_cnxt410.log"
    del _raw_key
    del _env

running = _is_running()
mo.md(f"### Process Status: {'🟢 **RUNNING** (PID ' + pid_file.read_text().strip() + ')' if running else '⚪ **IDLE** — ' + launch_msg}")"""

cell_PKri = """tick

_work_dir = pathlib.Path("/tmp/knee")
_pid_file = _work_dir / "runner.pid"

def _is_alive():
    try:
        if not _pid_file.exists(): return False
        _pid = int(_pid_file.read_text().strip())
        return psutil.pid_exists(_pid) and psutil.Process(_pid).status() != psutil.STATUS_ZOMBIE
    except Exception:
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

# 4. Check folds 0 to 4
_fold_rows = []
for _f in range(5):
    _log_p = _work_dir / "run" / f"train_cnxt_r3_s410_f{_f}.log"
    _wrap_log = _work_dir / f"log_cnxt_r3_s410_f{_f}.txt"
    _status = "Pending"
    _last_auc = "No metrics yet"
    
    if _wrap_log.exists():
        for _line in _wrap_log.read_text(errors="replace").splitlines():
            if _line.startswith("DONE"):
                _status = "✅ Finished (DONE)"
                break
                
    if _log_p.exists():
        _lines = _log_p.read_text(errors="replace").splitlines()
        for _line in reversed(_lines):
            if "GOLD macro-AUC" in _line:
                _last_auc = _line.strip()
                break
        if _status != "✅ Finished (DONE)":
            _status = "🔄 Training"
            
    _fold_rows.append(f"- **Fold {_f}:** {_status} | `{_last_auc}`")

_mlog = _work_dir / "molab_cnxt410.log"
_mlog_tail = _mlog.read_text(errors="replace").splitlines()[-8:] if _mlog.exists() else []

_status_tag = "🟢 **RUNNING**" if _is_alive() else "⚪ **IDLE**"
_now_time = time.strftime("%H:%M:%S")

_table_md = "\\n".join([
    "### 📊 ConvNeXt s410 Live Monitor & Telemetry",
    f"**Status:** {_status_tag} | **Telemetry Updated:** `{_now_time}`\\n",
    "| Resource | Current Usage | Status / Notes |",
    "| :--- | :--- | :--- |",
    f"| **Host RAM** | **{_ram_used_gb:.1f} GB** / {_ram_tot_gb:.1f} GB ({_ram_pct:.1f}%) | {_ram_badge} |",
    f"| **GPU VRAM** | **{_gpu_mem}** | Compute: **{_gpu_util}** |",
    f"| **GPU Temp & Power** | **{_gpu_temp}** | Draw: **{_gpu_pwr}** |",
    f"| **Disk `/tmp`** | **{_disk_str}** | Corpus & Weights |\\n",
    "---",
    "#### 🎯 Folds Progress (s410):\\n" + "\\n".join(_fold_rows),
    "\\n#### 📜 Master Log Tail (`molab_cnxt410.log`):"
])

mo.vstack([
    mo.md(_table_md),
    mo.plain_text("\\n".join(_mlog_tail) if _mlog_tail else "Waiting for start..."),
    mo.md("> ℹ️ *Job running sequentially (Folds 0-4). Each completed fold auto-publishes to Kaggle.*")
])"""

async def apply():
    async with cm.get_context(skip_staleness_check=True) as ctx:
        ctx.edit_cell('vblA', cell_vblA, hide_code=False)
        ctx.edit_cell('bkHC', cell_bkHC, hide_code=False)
        ctx.edit_cell('PKri', cell_PKri, hide_code=False)
        ctx.run_cell('vblA')
        ctx.run_cell('bkHC')
        ctx.run_cell('PKri')
        print("Updated and executed cells vblA, bkHC, PKri on Molab 1!")

await apply()
