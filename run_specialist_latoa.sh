#!/bin/bash
# run_specialist_latoa.sh - Train Compartment-ROI Specialist for Lateral OA & Meniscus
set -e
cd /home/s05814802723/projects/knee
source ~/miniconda3/etc/profile.d/conda.sh
conda activate knee

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Compartment mode crops the knee joint and splits each slice into separate
# Medial and Lateral compartment tiles (2 tiles/center, 6 centers -> 12 windows).
# Directly resolves lateral joint space narrowing and eliminates left/right knee ambiguity.
/home/s05814802723/miniconda3/envs/knee/bin/python train_knee.py \
    --ckpt timm \
    --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
    --res 384 \
    --epochs 16 \
    --patience 5 \
    --topk 3 \
    --bs 2 \
    --accum 8 \
    --grad_ckpt \
    --roi \
    --roi_mode compartment \
    --roi_overlap 0.12 \
    --roi_boxes /home/s05814802723/projects/knee/roi_boxes.npz \
    --k 12 \
    --k_eval 48 \
    --bb_lr 3.5e-5 \
    --head_lr 1.5e-3 \
    --wd 0.05 \
    --time_budget_h 12.0 \
    --tag coat_latoa_roi \
    --labels labels_avg.parquet \
    2>&1 | tee train_coat_latoa_roi.log
