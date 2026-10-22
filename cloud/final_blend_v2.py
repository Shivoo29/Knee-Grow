# RSNA Knee Final Blend v2
# Public Raptor (384 + 224 mirror) + R6 flip TTA + Our Round-2 4-fold
# Scanner prior for Lateral OA | Flip TTA on all | Rank blend per-finding

# ─── CONFIG ──────────────────────────────────────────────────────────────
RUN_INFERENCE = True  # Set False for static analysis
DEADLINE_HOURS = 8.3
OUR_WEIGHT = 0.20  # CHANGE: 0.20 (safe) or 0.35 (bold)

import os, sys, time, json, subprocess, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

if not RUN_INFERENCE:
    raise SystemExit("Static notebook. Set RUN_INFERENCE=True, attach datasets, enable T4 GPU.")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA required")

NB_T0 = time.time()
DEADLINE = NB_T0 + 8.3 * 3600
LABELS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
OUR_WEIGHT = 0.20  # CHANGE THIS PER SUBMISSION

def pct(x):
    return rankdata(x, method="average") / (len(x) + 1)

def run(cmd, name, timeout=2*3600):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    rc = subprocess.run(cmd, timeout=timeout).returncode
    if rc: raise RuntimeError(f"{name} failed: {rc}")
    print(f"{name} done ({time.time()-time.time():.0f}s)")

# ─── STAGE 1: RAPTOR 384 ────────────────────────────────────────────────
print("STAGE 1: Raptor 384")
subprocess.run([sys.executable, "-u", "/kaggle/input/rsna-knee-publication-swa-weights-20261007/raptor_run.py"], timeout=2*3600, check=True)
raptor_384 = pd.read_csv("/kaggle/working/raptor_raw.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")

# ─── STAGE 1b: RAPTOR 224 ──────────────────────────────────────────────
print("STAGE 1b: Raptor 224")
subprocess.run([sys.executable, "-u", "/kaggle/input/rsna-knee-publication-swa-weights-20261007/raptor224_run.py"], timeout=2*3600, check=True)
raptor_224 = pd.read_csv("/kaggle/working/raptor224_raw.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")

# ─── STAGE 2: R6 FLIP TTA ──────────────────────────────────────────────
print("STAGE 2: R6 Flip TTA")
subprocess.run([
    sys.executable, "-u", "/kaggle/input/rsna-orthofoundation-fold0-epoch7-20261004/probe_dual.py",
    "--data", "/kaggle/input/rsna-knee-abnormality-detection",
    "--split", "test", "--code", "/kaggle/input/rsna-orthofoundation-fold0-epoch7-20261004",
    "--bundle", "/kaggle/input/rsna-knee-r6-frozen-fe055406",
    "--arm", "E_lrprior", "--output", "/kaggle/working/r6_pred",
    "--amp", "auto", "--loader-workers", "2",
    "--deadline", str(time.time() + 8.3*3600),
    "--flip-tta",
], env={**os.environ, "PYTHONPATH": "/kaggle/input/rsna-orthofoundation-fold0-epoch7-20261004:"+os.environ.get("PYTHONPATH","")}, check=True)
r6 = pd.read_csv("/kaggle/working/r6_pred/worker0/raw.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")

# ─── STAGE 3: OUR 4-FOLD ROUND-2 FLIP TTA ──────────────────────────────
print("STAGE 3: Our Round-2 4-fold Flip TTA")
# Run inference on our 4 folds
# This would use our train_knee.py with --eval flag and the 4 checkpoints
# For now, load pre-computed predictions (to be uploaded as dataset)
our_path = "/kaggle/input/knee-cnxt-r2-folds/mean_flip_tta.csv"
if os.path.exists(our_path):
    our = pd.read_csv(our_path, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
else:
    raise FileNotFoundError("Our model predictions not found. Upload knee-cnxt-r2-folds with mean_flip_tta.csv")

# ─── ALIGN ──────────────────────────────────────────────────────────────
test_ids = raptor_384.index.tolist()
n = len(test_ids)
for df in [raptor_384, raptor_224, r6, our]:
    df = df.loc[test_ids, LABELS]

# ─── RANK BLEND ────────────────────────────────────────────────────────
def pct(x): return rankdata(x, method="average") / (len(x) + 1)

# Raptor half = 0.65*rank(384) + 0.35*rank(224)
r384 = raptor_384[LABELS].apply(pct)
r224 = raptor_224[LABELS].apply(pct)
raptor_half = 0.65 * r384 + 0.35 * r224

# R6
r6_r = r6[LABELS].apply(pct)

# Public half (Raptor + R6) with heliosli V2 per-finding weights
W_RAPTOR_C = {"Lateral OA": 0.15, "PF OA": 0.65, "Synovitis": 0.85, "Baker's": 0.30}
public = pd.DataFrame(index=test_ids, columns=LABELS, dtype=float)
for c in LABELS:
    w_r = W_RAPTOR_C.get(c, 0.5)
    public[c] = w_r * raptor_half[c] + (1 - w_r) * r6_r[c]

# Our model
our_r = our.loc[test_ids, LABELS].apply(pct)

# Final blend: (1 - OUR_WEIGHT) * public + OUR_WEIGHT * our_r
final = (1 - OUR_WEIGHT) * public + OUR_WEIGHT * our_r[LABELS].apply(pct)

# ─── SCANNER PRIOR FOR LATERAL OA ──────────────────────────────────────
# Load scanner prior (Manufacturer|Model -> Lateral OA prior)
# In practice: read Manufacturer+Model from each test study's DICOM
# For now, apply constant adjustment based on training data analysis
# Lateral OA: model + scanner prior blended 0.7/0.3
# This is a placeholder - full impl reads DICOM tags
lateral_idx = LABELS.index("Lateral OA")
# final.iloc[:, lateral_idx] = 0.7 * final.iloc[:, lateral_idx] + 0.3 * 0.235  # global prior

# ─── SUBMISSION ────────────────────────────────────────────────────────
sub = pd.DataFrame(final, columns=LABELS)
sub.insert(0, "StudyInstanceUID", sub.index)
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(f"Submission written: {sub.shape}")
print("NOTEBOOK_COMPLETE")