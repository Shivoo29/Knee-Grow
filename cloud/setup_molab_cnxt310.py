#!/usr/bin/env python3
"""cloud/setup_molab_cnxt310.py - Configure and launch ConvNeXt s310 training on Molab instance."""
import subprocess
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
SCRIPT = REPO_ROOT / ".agents" / "skills" / "marimo-pair" / "scripts" / "execute-code.sh"
URL = "https://sb-5db0f9a383670956.sb.molab.run/"
TOKEN = "fe36f22213bd628cee59a1cedfbd24866f10c3c1dfb6a0fa5fedbce6cd19df83"

REMOTE_SETUP_CODE = r'''
import marimo._code_mode as cm

cell1_code = """import os, sys, subprocess, pathlib, time, shutil
import marimo as mo"""

cell2_code = '''key_form = mo.ui.dictionary({
    'key': mo.ui.text(
        kind='password', 
        label='Kaggle API Key / Token (shivoo29)', 
        placeholder='Paste your Kaggle API key or token here...', 
        full_width=True
    )
}).form(submit_button_label='Start ConvNeXt s310', clear_on_submit=False)

header = mo.vstack([
    mo.md("""## 🚀 ConvNeXt Round-3: Train All 5 Folds (s310)
**Secure Runtime Form:** Your Kaggle credentials exist only in runtime memory and are passed strictly as environment variables to the subprocess (`KAGGLE_USERNAME=shivoo29`, `KAGGLE_API_TOKEN` / `KAGGLE_KEY`). Never saved to disk, cells, or logs.

👉 **Instructions:** Enter your Kaggle API key or token and click **Start ConvNeXt s310**."""),
    key_form
])
header'''

cell3_code = '''# Launch controller for ConvNeXt round-3
_work = pathlib.Path('/tmp/knee')
_pid_file = _work / 'runner.pid'

def is_running():
    try:
        if not _pid_file.exists():
            return False
        _pid = int(_pid_file.read_text().strip())
        os.kill(_pid, 0)
        return True
    except OSError:
        return False

_launch_msg = 'Enter Kaggle key in form and click Start to launch the background runner.'

if key_form.value and key_form.value.get('key', '').strip() and not is_running():
    _raw_key = key_form.value['key'].strip()
    _env = os.environ.copy()
    _env['KAGGLE_USERNAME'] = 'shivoo29'
    if _raw_key.startswith('KGAT_'):
        _env['KAGGLE_API_TOKEN'] = _raw_key
    else:
        _env['KAGGLE_KEY'] = _raw_key
    _env['PRESET'] = 'cnxt'
    _env['SEED_BASE'] = '310'
    _env['FOLDS'] = '0 1 2 3 4'
    _env['PARALLEL'] = '3'
    _env['WORK'] = '/tmp/knee'
    
    _log_file = open(_work / 'molab_cnxt310.log', 'ab')
    _proc = subprocess.Popen(
        ['bash', '-c', 'cd /tmp/knee && bash molab_train.sh'],
        cwd=_work,
        env=_env,
        start_new_session=True,
        stdout=_log_file,
        stderr=subprocess.STDOUT,
    )
    _pid_file.write_text(str(_proc.pid))
    _launch_msg = f'🚀 Launched background runner (PID {_proc.pid})! Logs: /tmp/knee/molab_cnxt310.log'
    del _raw_key
    del _env

_status = f'🟢 **RUNNING** (PID {_pid_file.read_text().strip()})' if is_running() else f'⚪ **IDLE** — {_launch_msg}'
mo.md(f'### Process Status: {_status}')'''

cell4_code = '''refresh = mo.ui.refresh(default_interval='60s')
refresh

_smi_str = 'N/A'
try:
    _smi = subprocess.run([
        'nvidia-smi',
        '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
        '--format=csv,noheader,nounits'
    ], capture_output=True, text=True).stdout.strip()
    _parts = [p.strip() for p in _smi.split(',')]
    _smi_str = f'GPU: {_parts[0]}% | VRAM: {float(_parts[1])/1024:.1f}/{float(_parts[2])/1024:.1f} GB | Temp: {_parts[3]}°C | Power: {_parts[4]}W'
except Exception as _e:
    _smi_str = f'GPU: unavailable ({_e})'

_fold_rows = []
_run_dir = pathlib.Path('/tmp/knee/run')

for _f in range(5):
    _log_p = _run_dir / f'train_cnxt_r3_s310_f{_f}.log'
    _wrap_log = pathlib.Path(f'/tmp/knee/log_cnxt_r3_s310_f{_f}.txt')
    
    _status = 'Pending'
    _last_auc = 'No metrics yet'
    
    if _wrap_log.exists():
        for _line in _wrap_log.read_text(errors='replace').splitlines():
            if _line.startswith('DONE'):
                _status = '✅ Finished (DONE)'
                break
                
    if _log_p.exists():
        _lines = _log_p.read_text(errors='replace').splitlines()
        for _line in reversed(_lines):
            if 'GOLD macro-AUC' in _line:
                _last_auc = _line.strip()
                break
        if _status != '✅ Finished (DONE)':
            _status = '🔄 Training'
            
    _fold_rows.append(f'- **Fold {_f}:** {_status} | `{_last_auc}`')

_mlog = pathlib.Path('/tmp/knee/molab_cnxt310.log')
_mlog_tail = _mlog.read_text(errors='replace').splitlines()[-10:] if _mlog.exists() else []

_folds_text = '\\n'.join(_fold_rows)
_tail_text = '\\n'.join(_mlog_tail) if _mlog_tail else 'Waiting for start...'

mo.vstack([
    mo.md(f"""### 📊 Live Status (Refreshed every 60s)
**NVIDIA-SMI:** `{_smi_str}`

#### Folds Progress:
{_folds_text}"""),
    mo.md('#### 📜 molab_cnxt310.log (tail)'),
    mo.plain_text(_tail_text),
])'''

async with cm.get_context(skip_staleness_check=True) as ctx:
    cids = list(ctx.cells.keys())
    print('Existing cells:', cids)
    # Delete extra cells if any
    for cid in cids[1:]:
        ctx.delete_cell(cid)
    
    # Edit cell 0
    first_cid = cids[0]
    ctx.edit_cell(first_cid, cell1_code)
    c2 = ctx.create_cell(cell2_code, after=first_cid)
    c3 = ctx.create_cell(cell3_code, after=c2)
    c4 = ctx.create_cell(cell4_code, after=c3)
    ctx.run_cell(first_cid)
    ctx.run_cell(c2)
    ctx.run_cell(c3)
    ctx.run_cell(c4)
    print('Updated notebook with cells:', [first_cid, c2, c3, c4])
'''

def main():
    cmd = ["bash", str(SCRIPT), "--url", URL, "--token", TOKEN, "-c", REMOTE_SETUP_CODE]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)

if __name__ == "__main__":
    main()
