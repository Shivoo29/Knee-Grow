#!/bin/bash
# Final round-3 training on one molab GPU (RTX Pro 6000, 96 GB). Everything comes from Kaggle; nothing is local.
#   PRESET=cnxt|coat  FOLDS="0 1 2 3 4"  PARALLEL=2 (folds trained at once on the one GPU)  SEED_BASE=110 (backup: 210)
# Needs secrets KAGGLE_API_TOKEN (or KAGGLE_KEY) + KAGGLE_USERNAME. Each finished fold is published as the private
# dataset <KAGGLE_USERNAME>/knee-r3-<preset>-s<seed>-f<fold>. Logs: log_<preset>_r3_s<seed>_f<fold>.txt next to this file.
set -uo pipefail
cd "$(dirname "$0")"
PRESET=${PRESET:?set PRESET=cnxt or coat}; FOLDS=${FOLDS:-0 1 2 3 4}; PARALLEL=${PARALLEL:-2}
export WORK=${WORK:-/tmp/knee} NFOLDS=5 FOLD_FILE=folds_rand5.json LABELS=labels_r3.parquet LW=label_weights_pu.parquet \
       LABEL_DS=${LABEL_DS:-shivoo29/knee-r3-labels} SEED_BASE=${SEED_BASE:-110}
case $PRESET in
  cnxt) export ARCH=convnext_tiny.in12k_ft_in1k RES=256 EXTRA="" ;;
  # ponytail: k 12 (24 took >1 h/epoch) / k_eval 128 keeps 5 CoAtNet folds inside Kaggle's 9 h; raise k only if epoch 0 is fast
  coat) export ARCH=coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k RES=384 EXTRA="--k 12 --k_eval 128 --bb_lr 4e-5 --head_lr 2e-3" ;;
  *) echo "PRESET must be cnxt or coat"; exit 1 ;;
esac
KUSER=${KAGGLE_USERNAME:?set KAGGLE_USERNAME}; RDS=knee-r3-resume-$PRESET-s$SEED_BASE; SAVE=$WORK/resume_up
kg() { python -m kaggle "$@"; }
mkdir -p "$WORK/run" "$SAVE"
# a disconnect can wipe this machine: restore the per-epoch resume states backed up to Kaggle (none on a first start)
kg datasets download "$KUSER/$RDS" -p "$WORK/run" --unzip -q >/dev/null 2>&1 && echo "$(date +%T) restored resume states: $(ls "$WORK"/run/raptor_resume_* 2>/dev/null | wc -l)"
todo=""
for f in $FOLDS; do
  if kg datasets files "$KUSER/knee-r3-$PRESET-s$SEED_BASE-f$f" >/dev/null 2>&1; then echo "fold $f already on Kaggle, skipped"; else todo="$todo $f"; fi
done
FOLDS=$todo
# after every epoch (checked each minute): back up the resume states of unfinished folds + logs to $KUSER/$RDS
( last=""; while sleep 60; do
    now=$(stat -c %Y "$WORK"/run/raptor_resume_${PRESET}_r3_s${SEED_BASE}_f*.pt 2>/dev/null | tr '\n' ' ')
    [ -z "$now" ] || [ "$now" = "$last" ] && continue
    rm -f "$SAVE"/*.pt
    for r in "$WORK"/run/raptor_resume_${PRESET}_r3_s${SEED_BASE}_f*.pt; do
      t=$(basename "$r" .pt); t=${t#raptor_resume_}
      [ -f "$r" ] && [ ! -f "$WORK/run/raptor_ft_${t}_swa.pt" ] && cp "$r" "$SAVE/"
    done
    ls "$SAVE"/*.pt >/dev/null 2>&1 || continue
    cp "$WORK"/run/train_${PRESET}_r3_s${SEED_BASE}_f*.log "$SAVE"/ 2>/dev/null
    echo "{\"title\": \"$RDS\", \"id\": \"$KUSER/$RDS\", \"licenses\": [{\"name\": \"CC0-1.0\"}]}" > "$SAVE/dataset-metadata.json"
    { kg datasets version -p "$SAVE" -m "$(date +%T)" -q || kg datasets create -p "$SAVE" -q; } >/dev/null 2>&1 \
      && { last=$now; echo "$(date +%T) epoch state backed up to Kaggle"; }
  done ) &
saver=$!; disown $saver   # not a fold: kept out of the PARALLEL count and the final wait
first=1
for f in $FOLDS; do
  tag=${PRESET}_r3_s${SEED_BASE}_f$f
  TAG=$tag DATASET_NAME=knee-r3-$PRESET-s$SEED_BASE-f$f FOLD=$f bash run_r2.sh > "log_$tag.txt" 2>&1 &
  pid=$!; echo "$(date +%T) started fold $f (log_$tag.txt)"
  if [ $first = 1 ]; then  # the first fold downloads the data; the others wait for it instead of racing
    until [ -f "$WORK/run/.c192_done" ] && [ -f "$WORK/run/$LW" ]; do
      kill -0 $pid 2>/dev/null || { echo "fold $f stopped during download, see log_$tag.txt"; exit 1; }
      sleep 30
    done
    first=0
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
done
wait
kill $saver 2>/dev/null
echo "$(date +%T) ALL_DONE"
