#!/bin/bash
# Distilled student (CoAtNet version 6, seed 8) on the PC (RTX 4070 SUPER 12 GB). Same recipe as node1 v6 seeds; bs 1 x accum 16 = same effective batch.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RAPTOR_INPUT="$PWD/ds"
exec systemd-inhibit --what=sleep:idle --why="knee v6d_s8 training" \
  .venv/bin/python train_knee.py --ckpt timm --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
    --res 384 --bs 1 --accum 16 --k 12 --k_eval 94 --epochs 16 --patience 5 --win_chunk 8 --grad_ckpt \
    --labels labels_distil_v1.parquet --seed 8 --tag v6d_s8
