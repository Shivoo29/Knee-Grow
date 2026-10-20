#!/bin/bash
# Final round-3 folds on a local GPU (PC / node1), one fold after another; same recipe as cloud/molab_train.sh.
#   PRESET=cnxt|coat FOLDS="0 1 2 3 4" SEED_BASE=210 PY=<python> RAPTOR_INPUT=<corpus192 dir> ./run_r3.sh
# Needs beside it: train_knee.py, roi_localize.py, labels_r3.parquet, label_weights_pu.parquet, folds_rand5.json.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python}; FOLDS=${FOLDS:-0 1 2 3 4}; SEED_BASE=${SEED_BASE:-210}
case ${PRESET:?set PRESET=cnxt or coat} in
  cnxt) A="--arch convnext_tiny.in12k_ft_in1k --res 256" ;;
  coat) A="--arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --k 24 --k_eval 128 --bb_lr 4e-5 --head_lr 2e-3" ;;
esac
for f in $FOLDS; do
  tag=${PRESET}_r3_s${SEED_BASE}_f$f
  [ -f "raptor_ft_${tag}_swa.pt" ] && { echo "skip $tag (done)"; continue; }
  echo "$(date +%T) fold $f -> train_$tag.log"
  $PY train_knee.py --ckpt timm --arch convnext_tiny.in12k_ft_in1k --res 256 --norm imagenet --zoom_mm 100 \
    --bs 2 --accum 8 --k 48 --k_eval 380 --grad_ckpt --epochs 15 --patience 5 --bb_lr 1e-4 --head_lr 1e-3 --workers ${WORKERS:-6} \
    --clean_windows --mirror_p 0.5 --contrast_aug --slot_emb --slot_sizes 40,40,36,36,40 \
    --labels labels_r3.parquet --label_weights label_weights_pu.parquet \
    --stkim_k 3 --stkim_p 0.7 --oof_every 300 \
    --folds 5 --fold "$f" --fold_file folds_rand5.json --seed $((SEED_BASE + f)) --tag "$tag" $A \
    > "train_$tag.log" 2>&1
done
echo "$(date +%T) ALL_DONE"
