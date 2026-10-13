#!/bin/bash
# Distilled student, DINOv2 ViT-Base (first 6 blocks frozen), seed 11, PC (RTX 4070 SUPER 12 GB). Same effective batch as the CoAtNet students.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RAPTOR_INPUT="$PWD/ds"
exec systemd-inhibit --what=sleep:idle --why="knee dinov2 s11 training" \
  .venv/bin/python train_knee.py --ckpt timm --arch vit_base_patch14_reg4_dinov2.lvd142m \
    --res 378 --norm imagenet --bs 1 --accum 16 --k 12 --k_eval 94 --epochs 16 --patience 5 \
    --win_chunk 8 --grad_ckpt --freeze_blocks 6 --bb_lr 2e-5 \
    --labels labels_distil_v1.parquet --seed 11 --tag dinov2_s11 "$@"
