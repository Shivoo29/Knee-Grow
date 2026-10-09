#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export RAPTOR_INPUT="$(pwd)/ds"
export PYTHONUNBUFFERED=1

echo "Starting CoAtNet-224 training on local RTX 4070 SUPER..."
systemd-inhibit --what=idle:sleep --why="RSNA Knee Training coat224" \
  .venv/bin/python train_knee.py \
    --ckpt timm \
    --arch coatnet_rmlp_1_rw_224 \
    --res 224 \
    --epochs 16 \
    --bs 4 \
    --accum 4 \
    --k 12 \
    --k_eval 42 \
    --bb_lr 4e-5 \
    --head_lr 2e-3 \
    --wd 0.05 \
    --tag coat224_v1 \
    --labels labels_avg.parquet \
    2>&1 | tee -a coat224_v1.log
