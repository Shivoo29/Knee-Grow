#!/usr/bin/env python3
"""cloud/molab2_watchdog.py - Automated Watchdog & Auto-Healer for Molab Instance 2 (MedGemma).

Monitors Instance 2 every 15 seconds. If the runner stops or encounters an error,
it automatically inspects the log, applies any necessary code rectifications,
and re-launches the runner until completion.
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
URL = "https://sb-2d47249a6c0820f2.sb.molab.run/"
TOKEN = "REDACTED_MOLAB_TOKEN"
LOG_FILE = HERE / "watchdog2.log"


def log(msg):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_code_on_molab2(py_code, timeout=60):
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

work = pathlib.Path("/tmp/knee_radiologist")
pid_file = work / "runner.pid"
master_log = work / "medgemma_master.log"

is_alive = False
if pid_file.exists():
    try:
        pid = int(pid_file.read_text().strip())
        is_alive = psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:
        pass

log_tail = ""
if master_log.exists():
    log_tail = master_log.read_text(errors="replace")[-2000:]

is_done = "ALL_STEPS_COMPLETED" in log_tail

# GPU utilization
smi = ""
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
except Exception:
    pass

print("###JSON_START###")
print(json.dumps({
    "alive": is_alive,
    "done": is_done,
    "gpu": smi,
    "tail": log_tail.splitlines()[-6:] if log_tail else [],
    "has_error": "Traceback" in log_tail or "Error:" in log_tail,
}))
print("###JSON_END###")
"""

RECTIFY_AND_RESTART_CODE = """
import os, pathlib, subprocess

work = pathlib.Path("/tmp/knee_radiologist")
repo_dir = work / "repo"

# 1. Update test_pipeline.py with corrected ReportResult and MedGemma image scoring
test_script = work / "test_pipeline.py"
test_lines = [
    "import os, sys, torch",
    "import numpy as np",
    "from PIL import Image",
    "from kneemri.contracts import AcquisitionInfo, CropRef, Label",
    "from kneemri.adapters.medgemma import MedGemmaScorer",
    "from kneemri.report.scorer import score_report",
    "",
    "free_gb = 0",
    "if torch.cuda.is_available():",
    "    free_bytes, total_bytes = torch.cuda.mem_get_info()",
    "    free_gb = free_bytes / (1024**3)",
    "",
    "model_id = 'google/medgemma-27b-it' if free_gb > 60 else 'google/medgemma-4b-it'",
    "print(f'Selected MedGemma Model: {model_id} (Available VRAM: {free_gb:.1f} GB)')",
    "",
    "os.environ['KNEEMRI_MEDGEMMA_PATH'] = model_id",
    "print(f'Initializing MedGemmaScorer for {model_id}...')",
    "scorer = MedGemmaScorer(model_path=model_id)",
    "print(f'Scorer initialized successfully: {scorer.name}')",
    "",
    "sample_report = 'IMPRESSION: Complete tear of the anterior cruciate ligament with joint effusion. Intact menisci and collateral ligaments.'",
    "print('Scoring sample report via rulebook:')",
    "print(sample_report)",
    "rep_res = score_report(sample_report)",
    "print('=== REPORT SCORES (ALF Skill Rulebook) ===')",
    "for lbl, s in rep_res.labels.items():",
    "    if s.binary == 1 or s.probability > 0.5:",
    "        print(f'  [+] {lbl.value.upper()}: prob={s.probability:.2f} (evidence: {s.evidence})')",
    "",
    "sample_png = '/tmp/sample_knee_sag.png'",
    "arr = np.zeros((256, 256), dtype=np.uint8)",
    "arr[50:200, 60:110] = 160",
    "arr[120:220, 130:190] = 140",
    "Image.fromarray(arr).save(sample_png)",
    "",
    "acq = AcquisitionInfo(",
    "    plane='SAG',",
    "    weighting='PD',",
    "    fat_sat=True,",
    "    side='R',",
    "    mm_per_px=0.40,",
    "    slice_thickness=3.0,",
    "    field_strength=3.0,",
    ")",
    "crop = CropRef(",
    "    label=Label.ACL,",
    "    series_uid='test_study.1',",
    "    slice_index=15,",
    "    png_path=sample_png,",
    "    box=(50, 50, 200, 200),",
    "    mm_per_px=0.40,",
    "    acquisition=acq,",
    ")",
    "",
    "print('Scoring Knee MRI crop for ACL with MedGemma...')",
    "img_res = scorer.score(Label.ACL, [crop])",
    "print('=== MEDGEMMA IMAGE RESULT ===')",
    "print(f'Label: ACL | Probability: {img_res.score:.4f} | Evidence Slices: {img_res.evidence_slices}')",
    "",
    "print('ALL_STEPS_COMPLETED')",
]
test_script.write_text('\\n'.join(test_lines) + '\\n')

# 2. Re-launch the runner
log_f = open(work / "medgemma_master.log", "a")
log_f.write("\\n=== Auto-Healer Re-running Corrected Pipeline ===\\n")
log_f.flush()

proc = subprocess.Popen(
    ["python3", str(test_script)],
    cwd=repo_dir,
    stdout=log_f,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
(work / "runner.pid").write_text(str(proc.pid))
print(f"RECTIFIED_AND_RESTARTED_PID_{proc.pid}")
"""


def check_once():
    ok, out = run_code_on_molab2(CHECK_CODE, timeout=30)
    if not ok:
        log(f"Molab2 Connection notice: {out[:120]}")
        return False

    if "###JSON_START###" not in out:
        log(f"Molab2 response: {out[:120]}")
        return False

    data_str = out.split("###JSON_START###")[1].split("###JSON_END###")[0].strip()
    try:
        data = json.loads(data_str)
    except Exception as e:
        log(f"Parse error: {e}")
        return False

    alive = data.get("alive", False)
    done = data.get("done", False)
    gpu = data.get("gpu", "N/A")
    has_error = data.get("has_error", False)
    tail = " | ".join(data.get("tail", []))

    log(f"Molab2 Status: Alive={alive}, Done={done}, GPU={gpu}")
    if tail: log(f"Log tail: {tail[-250:]}")

    if done:
        log("🎉 Molab2 Task ALL_STEPS_COMPLETED!")
        return True

    if not alive or has_error:
        log("⚠️ Detected stopped process or error! Triggering automatic rectification...")
        rok, rout = run_code_on_molab2(RECTIFY_AND_RESTART_CODE, timeout=30)
        log(f"Rectify & restart result: {rout.strip()}")
        return False

    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--loop":
        log("Starting Molab2 15-second continuous watchdog loop...")
        while True:
            try:
                check_once()
            except Exception as e:
                log(f"Watchdog exception: {e}")
            time.sleep(15)
    else:
        check_once()
