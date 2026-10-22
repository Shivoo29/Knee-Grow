#!/bin/bash
# Final recipe: round-2 ConvNeXt on ALL labelled studies (no held-out fold; gold-58 picks epochs / SWA). Resumable.
#   PY=<python> RAPTOR_INPUT=<ds dir> ./run_cnxt_r2all.sh <seed> [--smoke]
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python}
s=$1; shift
$PY train_knee.py --ckpt timm --arch convnext_tiny.in12k_ft_in1k --res 256 --norm imagenet --zoom_mm 100 \
  --bs 2 --accum 8 --k 48 --k_eval 380 --grad_ckpt --epochs 15 --patience 5 --bb_lr 1e-4 --head_lr 1e-3 --workers ${WORKERS:-8} \
  --clean_windows --mirror_p 0.5 --contrast_aug --slot_emb --slot_sizes 40,40,36,36,40 \
  --labels labels_r2.parquet --label_weights label_weights_sev_v1.parquet \
  --stkim_k 3 --stkim_p 0.7 \
  --folds 0 --seed "$s" --tag cnxt_r2_all_s$s${1:+_smoke} --resume raptor_resume_cnxt_r2_all_s$s.pt "$@" \
  >> "train_cnxt_r2_all_s$s${1:+_smoke}.log" 2>&1
