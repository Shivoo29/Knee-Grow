#!/bin/bash
# One ConvNeXt student fold on a big rented/free GPU (RTX Pro 6000 on molab, a Kaggle notebook, any Linux box):
#   1. code + labels from the private Kaggle dataset shivoo29/knee-cloud-bundle
#   2. fastMRI knee DICOMs straight from fastMRI's S3 into a u8d corpus (same studies and ids as node1's teacher;
#      fastMRI data is never uploaded anywhere, as its data use agreement asks)
#   3. our competition corpus192 (16 compressed shards) from shivoo29/knee-c192-u8d, unpacked shard by shard
#   4. wait until the bundle carries the teacher labels, train fold FOLD, publish the result as the private
#      dataset <your kaggle user>/knee-cnxt-fm-f<FOLD> (checkpoint, OOF, gold predictions, log)
# Needs: NVIDIA GPU, ~150 GB free under WORK (peak while batch1 is built), a Kaggle API token with access to
# shivoo29's two datasets (~/.kaggle/kaggle.json or KAGGLE_USERNAME + KAGGLE_KEY), and FASTMRI_URLS = a file of
# '<archive name> <url>' lines from your fastMRI email (knee_DICOMs_batch1, knee_DICOMs_batch2). Rerun-safe.
#
#   FOLD=2 WORK=/data FASTMRI_URLS=~/fastmri_urls.txt bash run_fold.sh
set -euo pipefail
: "${FOLD:?set FOLD=0..3}"
: "${FASTMRI_URLS:?set FASTMRI_URLS=<file of fastMRI links>}"
WORK=${WORK:-/data}; FM_GB=${FM_GB:-64}; PY=${PY:-python}; NP=$(nproc)
mkdir -p "$WORK"; cd "$WORK"
log() { echo "$(date +%H:%M:%S) $*"; }
kget() {  # kget <dataset> <file> <dir>: one file of a Kaggle dataset, unzipped if Kaggle served it zipped
  kaggle datasets download "$1" -f "$2" -p "$3" -q --force
  if [ -f "$3/$2.zip" ]; then (cd "$3" && unzip -q -o "$2.zip" && rm -f "$2.zip"); fi
}

log "step 0: packages"
$PY -c "import torch, timm, pydicom, cv2, pyarrow, sklearn" 2>/dev/null || \
  $PY -m pip install -q timm pydicom python-gdcm pylibjpeg pylibjpeg-libjpeg opencv-python-headless pyarrow scikit-learn kaggle
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || log "nvidia-smi not found (CPU only?)"
df -h "$WORK" | tail -n 1

log "step 1: bundle"
if [ ! -f bundle/train_knee.py ]; then kaggle datasets download shivoo29/knee-cloud-bundle -p bundle --unzip -q; fi
mkdir -p run code/tools
cp bundle/train_knee.py bundle/roi_localize.py bundle/run_cnxt_fm_folds.sh bundle/labels_sev_v1.parquet \
   bundle/label_weights_sev_v1.parquet bundle/folds_rand4.json bundle/train.csv run/
cp bundle/build_corpus96_stream.py code/
cp bundle/fastmri_corpus.py bundle/fetch_ranges.py code/tools/
chmod +x run/run_cnxt_fm_folds.sh

log "step 2: fastMRI corpus (S3 -> u8d), batch2 whole + first $FM_GB GB of batch1"
mkdir -p dl fm
for spec in knee_DICOMs_batch2:0 "knee_DICOMs_batch1:$FM_GB"; do
  name=${spec%%:*}; gb=${spec##*:}
  [ -f "fm/.done_$name" ] && continue
  $PY -I code/tools/fetch_ranges.py --urls "$FASTMRI_URLS" --archive "$name" --out "dl/$name.tar.xz" --max_gb "$gb" --threads 16
  $PY -I code/tools/fastmri_corpus.py --tar "dl/$name.tar.xz" --out fm --workers "$NP" --format u8d
  rm -f "dl/$name.tar.xz" "dl/$name.tar.xz.done"; touch "fm/.done_$name"
done
log "fastMRI corpus: $(ls fm/vols | wc -l) studies"

log "step 3: corpus192 (competition, 16 shards)"
if [ ! -f run/.c192_done ]; then
  mkdir -p run/vols
  for k in $(seq -w 0 15); do
    f=vols_$k.u8dtar
    [ -f "run/.shard_$k" ] && continue
    kget shivoo29/knee-c192-u8d "$f" dl
    tar -xf "dl/$f" -C run/vols && rm -f "dl/$f" && touch "run/.shard_$k"
  done
  for f in all_ids.npy all_masks.npy corpus.json; do kget shivoo29/knee-c192-u8d "$f" run; done
  n=$(ls run/vols | wc -l)
  [ "$n" -eq 4407 ] || { log "corpus192 has $n studies, expected 4407"; exit 1; }
  touch run/.c192_done
fi

log "step 4: teacher labels"
for i in $(seq 1 72); do
  [ -f bundle/fastmri_teacher_sev.parquet ] && break
  log "teacher labels not in the bundle yet; checking again in 10 min"
  sleep 600
  kaggle datasets download shivoo29/knee-cloud-bundle -f fastmri_teacher_sev.parquet -p bundle -q --force || true
  if [ -f bundle/fastmri_teacher_sev.parquet.zip ]; then (cd bundle && unzip -q -o fastmri_teacher_sev.parquet.zip); fi
done
cp bundle/fastmri_teacher_sev.parquet run/

log "step 5: train fold $FOLD"
cd run
RAPTOR_INPUT="$WORK/run" RAPTOR_EXTRA_CORPUS="$WORK/fm" TEACHER=fastmri_teacher_sev.parquet PY=$PY \
  BS=${BS:-8} ACCUM=${ACCUM:-2} WORKERS=${WORKERS:-$NP} ./run_cnxt_fm_folds.sh "$FOLD"
grep -E "^ep[0-9]+ |TOPK|Traceback" "train_cnxt_fm_f$FOLD.log" | tail -n 4

log "step 6: publish"
KUSER=$($PY -c "import json, os; print(os.environ.get('KAGGLE_USERNAME') or json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
OUT="$WORK/out_f$FOLD"; mkdir -p "$OUT"
cp raptor_ft_cnxt_fm_f${FOLD}_swa.pt raptor_ft_cnxt_fm_f${FOLD}.json "train_cnxt_fm_f$FOLD.log" "$OUT/"
cp raptor_oof_cnxt_fm_f${FOLD}*.npz raptor_gold_cnxt_fm_f${FOLD}*.npz "$OUT/" 2>/dev/null || true
echo "{\"title\": \"knee-cnxt-fm-f$FOLD\", \"id\": \"$KUSER/knee-cnxt-fm-f$FOLD\", \"licenses\": [{\"name\": \"CC0-1.0\"}]}" > "$OUT/dataset-metadata.json"
kaggle datasets create -p "$OUT" -t -q || kaggle datasets version -p "$OUT" -m "fold $FOLD" -t -q
log "DONE fold $FOLD -> $KUSER/knee-cnxt-fm-f$FOLD"
