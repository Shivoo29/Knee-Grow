#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
export RAPTOR_INPUT="$(pwd)/ds"
export PYTHONUNBUFFERED=1

echo "Starting ConvNeXt-Small (384px) training on local RTX 4070 SUPER..."
systemd-inhibit --what=idle:sleep --why="RSNA Knee Training convnext" \
  python train_knee.py \
    --ckpt timm \
    --arch convnext_small.fb_in22k_ft_in1k_384 \
    --norm imagenet \
    --res 384 \
    --epochs 16 \
    --patience 5 \
    --bs 2 \
    --accum 8 \
    --k 12 \
    --k_eval 42 \
    --bb_lr 3e-5 \
    --head_lr 1e-3 \
    --wd 0.05 \
    --tag convnext_s1 \
    --labels labels_avg.parquet \
    2>&1 | tee -a convnext_s1.log
