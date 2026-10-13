#!/bin/bash
set -e
cd /home/s05814802723/projects/knee
source ~/miniconda3/etc/profile.d/conda.sh
conda activate knee

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home/s05814802723/miniconda3/envs/knee/bin/python train_knee.py     --ckpt timm     --arch maxvit_base_tf_384.in1k     --res 384     --epochs 16     --patience 5     --topk 3     --bs 2     --accum 8     --grad_ckpt     --win_chunk 6     --k 12     --k_eval 94     --bb_lr 3e-5     --head_lr 1e-3     --wd 0.05     --time_budget_h 12.5     --tag maxvit_b_96     --labels labels_avg.parquet     2>&1 | tee train_maxvit_b_96.log
