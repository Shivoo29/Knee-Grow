#!/bin/bash
# Knee-Side Canonicalization - Step 1: Inference-Only Evaluation on Fold 0
# 1. Download bundle + laterality CSV from shivoo29/knee-cloud-bundle
# 2. Download corpus192 from shivoo29/knee-c192-u8d
# 3. Download raptor_ft_cnxt_sev_f0_swa.pt from shivoo29/knee-cnxt-sev-folds
# 4. Predict Fold 0 held-out studies in two passes:
#    - Plain (1088 studies)
#    - Mirrored (1088 studies)
# 5. Evaluate Left vs Right, Plain vs Mirrored, and Flip-TTA
set -euo pipefail
WORK=${WORK:-/tmp/knee}; PY=${PY:-python}
mkdir -p "$WORK"; cd "$WORK"
log() { echo "$(date +%H:%M:%S) $*"; }
kaggle() { if type -P kaggle >/dev/null; then command kaggle "$@"; else "$PY" -m kaggle "$@"; fi; }
unzip_py() { "$PY" -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$1" "$2"; }
kget() {
  kaggle datasets download "$1" -f "$2" -p "$3" -q --force
  if [ -f "$3/$2.zip" ]; then unzip_py "$3/$2.zip" "$3" && rm -f "$3/$2.zip"; fi
}

log "step 0: packages & environment"
if type -P uv >/dev/null; then
  uv pip install -q timm kaggle scikit-learn pandas pyarrow 2>/dev/null || true
else
  "$PY" -m pip install -q timm kaggle scikit-learn pandas pyarrow 2>/dev/null || true
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { log "no GPU found"; exit 1; }

log "step 1: bundle files"
mkdir -p run dl
for f in train_knee.py roi_localize.py folds_rand4.json labels_sev_v1.parquet study_laterality_orientation.csv train.csv; do
  [ -f "run/$f" ] || kget shivoo29/knee-cloud-bundle "$f" run
done

log "step 2: corpus192"
if [ ! -f run/.c192_done ]; then
  mkdir -p run/vols
  for k in $(seq -w 0 15); do
    f=vols_$k.u8dtar
    if [ ! -f "run/.shard_$k" ]; then
      kget shivoo29/knee-c192-u8d "$f" dl
      tar -xf "dl/$f" -C run/vols && rm -f "dl/$f" && touch "run/.shard_$k"
      log "corpus shard $k/15 unpacked"
    fi
  done
  for f in all_ids.npy all_masks.npy corpus.json; do
    [ -f "run/$f" ] || kget shivoo29/knee-c192-u8d "$f" run
  done
  n=$(ls run/vols | wc -l)
  [ "$n" -eq 4407 ] || { log "corpus192 has $n studies, expected 4407"; exit 1; }
  touch run/.c192_done
fi

log "step 3: checkpoint raptor_ft_cnxt_sev_f0_swa.pt"
if [ ! -f run/raptor_ft_cnxt_sev_f0_swa.pt ]; then
  kget shivoo29/knee-cnxt-sev-folds raptor_ft_cnxt_sev_f0_swa.pt run
fi

log "step 4: inference on Fold 0"
cd run
# Pass A: Plain
if [ ! -f oof_f0_plain.parquet ]; then
  log "running Fold 0 PLAIN inference..."
  PYTHONUNBUFFERED=1 "$PY" predict_corpus.py --corpus "$WORK/run" \
    --ckpts raptor_ft_cnxt_sev_f0_swa.pt --out oof_f0_plain.parquet \
    --studies folds_rand4.json:0 --k_eval 380 --bs 2 --workers 4 > oof_plain.log 2>&1
  log "Fold 0 PLAIN inference done"
else
  log "oof_f0_plain.parquet already exists"
fi

# Pass B: Mirrored
if [ ! -f oof_f0_mirrored.parquet ]; then
  log "running Fold 0 MIRRORED inference..."
  PYTHONUNBUFFERED=1 "$PY" predict_corpus.py --corpus "$WORK/run" \
    --ckpts raptor_ft_cnxt_sev_f0_swa.pt --out oof_f0_mirrored.parquet \
    --studies folds_rand4.json:0 --mirror all --k_eval 380 --bs 2 --workers 4 > oof_mirrored.log 2>&1
  log "Fold 0 MIRRORED inference done"
else
  log "oof_f0_mirrored.parquet already exists"
fi

log "step 5: evaluation report"
PYTHONUNBUFFERED=1 "$PY" eval_canon_step1.py \
  --plain oof_f0_plain.parquet \
  --mirrored oof_f0_mirrored.parquet \
  --labels labels_sev_v1.parquet \
  --laterality study_laterality_orientation.csv > step1_report.txt 2>&1

cat step1_report.txt
touch "$WORK/.step1_done"
log "STEP 1 COMPLETE"
