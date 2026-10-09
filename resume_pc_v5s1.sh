#!/bin/bash
set -e

cd /home/codemaster29/Documents/still_learning/rsna-knee-abnormality-detection
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export RAPTOR_INPUT=/home/codemaster29/Documents/still_learning/rsna-knee-abnormality-detection/ds

echo "Resuming v5s1 on local RTX 4070S from raptor_resume_v5s1.pt..."

exec systemd-inhibit --what=sleep:idle --why="knee v5s1 training" \
    python train_knee.py \
    --ckpt timm \
    --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
    --res 384 \
    --epochs 16 \
    --patience 5 \
    --bs 1 \
    --accum 16 \
    --k 12 \
    --k_eval 42 \
    --win_chunk 8 \
    --grad_ckpt \
    --bb_lr 4e-5 \
    --head_lr 2e-3 \
    --wd 0.05 \
    --seed 1 \
    --tag v5s1 \
    --labels labels_avg.parquet \
    --resume raptor_resume_v5s1.pt
