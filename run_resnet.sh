#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export RAPTOR_INPUT="$(pwd)/ds"
export PYTHONUNBUFFERED=1

echo "Starting ResNet-50 training on local RTX 4070 SUPER..."
.venv/bin/python train_knee.py \
  --ckpt timm \
  --arch resnet50d \
  --norm imagenet \
  --res 336 \
  --epochs 16 \
  --bs 4 \
  --accum 4 \
  --k 12 \
  --k_eval 42 \
  --bb_lr 5e-5 \
  --head_lr 1e-3 \
  --wd 0.05 \
  --tag r50_v1 \
  --labels labels_avg.parquet \
  2>&1 | tee -a r50_v1.log
