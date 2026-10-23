#!/bin/bash
set -e
cd /home/s05814802723/projects/knee
source ~/miniconda3/etc/profile.d/conda.sh
conda activate knee

export PYTHONUNBUFFERED=1
export RAPTOR_INPUT=/home/s05814802723/projects/knee/ds

/home/s05814802723/miniconda3/envs/knee/bin/python train_knee.py \
    --ckpt timm \
    --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
    --res 384 \
    --epochs 1 \
    --bs 2 \
    --accum 2 \
    --grad_ckpt \
    --roi \
    --roi_mode compartment \
    --roi_overlap 0.12 \
    --roi_boxes /home/s05814802723/projects/knee/roi_boxes.npz \
    --k 12 \
    --k_eval 24 \
    --bb_lr 3.5e-5 \
    --head_lr 1.5e-3 \
    --wd 0.05 \
    --tag test_smoke_compartment \
    --labels /home/s05814802723/projects/knee/labels_avg.parquet \
    --fold_file /home/s05814802723/projects/knee/ds/folds_site.json \
    --folds 5 \
    --fold 0 \
    --smoke
