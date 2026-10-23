#!/bin/bash
# train_specialist_f0.sh - Production Compartment-ROI Specialist (Fold 0, Site-Grouped)
set -e
cd /home/s05814802723/projects/knee
source ~/miniconda3/etc/profile.d/conda.sh
conda activate knee

export PYTHONUNBUFFERED=1
export RAPTOR_INPUT=/home/s05814802723/projects/knee/ds
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================================================="
echo "STARTING COMPARTMENT-ROI COATNET SPECIALIST (FOLD 0, SITE-GROUPED)"
echo "Target: Lateral OA, PF OA, Synovitis Breakthrough via Left/Right Separation"
echo "=============================================================================="

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
    --tag coat_latoa_roi_f0 \
    --labels /home/s05814802723/projects/knee/labels_avg.parquet \
    --fold_file /home/s05814802723/projects/knee/ds/folds_site.json \
    --folds 5 \
    --fold 0 \
    2>&1 | tee train_coat_latoa_roi_f0.log

echo "=============================================================================="
echo "TRAINING COMPLETE: coat_latoa_roi_f0"
echo "=============================================================================="
