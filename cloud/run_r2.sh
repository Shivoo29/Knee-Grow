#!/bin/bash
# One round-2 ConvNeXt fold on molab (or any Linux GPU box), competition data only (no fastMRI):
#   1. code + round-2 labels from shivoo29/knee-cloud-bundle
#   2. corpus192 (16 shards) from shivoo29/knee-c192-u8d, skipped if already unpacked
#   3. train fold FOLD on labels_r2.parquet (bs 2, accum 8, 5 loader workers; molab: 20 CPUs, files + RAM share 160 GB)
#   4. publish checkpoint, OOF, gold predictions and log as the private dataset <user>/knee-cnxt-r2-f<FOLD>
# Needs KAGGLE_USERNAME + KAGGLE_KEY (or KAGGLE_API_TOKEN) in the environment. Rerun-safe.
#   FOLD=1 WORK=/tmp/knee PY=python bash run_r2.sh
set -euo pipefail
: "${FOLD:?set FOLD=0..3}"
WORK=${WORK:-/tmp/knee}; PY=${PY:-python}
mkdir -p "$WORK"; cd "$WORK"
log() { echo "$(date +%H:%M:%S) $*"; }
kaggle() { if type -P kaggle >/dev/null; then command kaggle "$@"; else "$PY" -m kaggle "$@"; fi; }
unzip_py() { "$PY" -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$1" "$2"; }
kget() {
  kaggle datasets download "$1" -f "$2" -p "$3" -q --force
  if [ -f "$3/$2.zip" ]; then unzip_py "$3/$2.zip" "$3" && rm -f "$3/$2.zip"; fi
}

log "step 0: packages"
"$PY" -c "import torch, timm, pyarrow, sklearn, kaggle" 2>/dev/null || "$PY" -m pip install -q torch timm pyarrow pandas scikit-learn kaggle
nvidia-smi --query-gpu=name --format=csv,noheader || { log "no GPU: switch this notebook to a GPU machine"; exit 1; }
if [ -f before_r2.sh ]; then log "before_r2.sh (waits for the job already on this GPU)"; bash before_r2.sh; fi

log "step 1: bundle (code + round-2 labels)"
mkdir -p run dl
for f in train_knee.py roi_localize.py labels_r2.parquet label_weights_sev_v1.parquet folds_rand4.json train.csv; do
  kget shivoo29/knee-cloud-bundle "$f" run
done

log "step 2: corpus192"
if [ ! -f run/.c192_done ]; then
  mkdir -p run/vols
  for k in $(seq -w 0 15); do
    f=vols_$k.u8dtar
    [ -f "run/.shard_$k" ] && continue
    kget shivoo29/knee-c192-u8d "$f" dl
    tar -xf "dl/$f" -C run/vols && rm -f "dl/$f" && touch "run/.shard_$k"
    log "corpus part $k of 15"
  done
  for f in all_ids.npy all_masks.npy corpus.json; do kget shivoo29/knee-c192-u8d "$f" run; done
  n=$(ls run/vols | wc -l)
  [ "$n" -eq 4407 ] || { log "corpus192 has $n studies, expected 4407"; exit 1; }
  touch run/.c192_done
fi

log "step 3: train round-2 fold $FOLD"
cd run
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RAPTOR_INPUT="$WORK/run" \
"$PY" train_knee.py --ckpt timm --arch convnext_tiny.in12k_ft_in1k --res 256 --norm imagenet --zoom_mm 100 \
  --bs 2 --accum 8 --k 48 --k_eval 380 --grad_ckpt --epochs 15 --patience 5 --bb_lr 1e-4 --head_lr 1e-3 --workers ${WORKERS:-5} \
  --clean_windows --mirror_p 0.5 --contrast_aug --slot_emb --slot_sizes 40,40,36,36,40 \
  --labels labels_r2.parquet --label_weights label_weights_sev_v1.parquet \
  --stkim_k 3 --stkim_p 0.7 --oof_every 300 \
  --folds 4 --fold "$FOLD" --fold_file folds_rand4.json --seed $((110 + FOLD)) --tag cnxt_r2_f$FOLD \
  > "train_cnxt_r2_f$FOLD.log" 2>&1
grep -E "^ep[0-9]+ loss|TOPK" "train_cnxt_r2_f$FOLD.log" | tail -n 3

log "step 4: publish"
KUSER=${KAGGLE_USERNAME:-shivoo29}
OUT="$WORK/out_r2_f$FOLD"; mkdir -p "$OUT"
cp raptor_ft_cnxt_r2_f${FOLD}_swa.pt raptor_ft_cnxt_r2_f$FOLD.json "train_cnxt_r2_f$FOLD.log" "$OUT/"
cp raptor_oof_cnxt_r2_f${FOLD}*.npz raptor_gold_cnxt_r2_f${FOLD}*.npz "$OUT/" 2>/dev/null || true
echo "{\"title\": \"knee-cnxt-r2-f$FOLD\", \"id\": \"$KUSER/knee-cnxt-r2-f$FOLD\", \"licenses\": [{\"name\": \"CC0-1.0\"}]}" > "$OUT/dataset-metadata.json"
kaggle datasets create -p "$OUT" -q || kaggle datasets version -p "$OUT" -m "fold $FOLD" -q
log "DONE round-2 fold $FOLD -> $KUSER/knee-cnxt-r2-f$FOLD"
