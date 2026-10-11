#!/bin/bash
# run_local_pfoa_specialist.sh - Train Axial Patellofemoral OA Specialist on local RTX 4070 SUPER
set -e
cd /home/codemaster29/Documents/still_learning/rsna-knee-abnormality-detection
source .venv/bin/activate

export PYTHONUNBUFFERED=1

# Uses bs 4 with chunked forward pass, keeping VRAM ~1.8 GB on RTX 4070S.
.venv/bin/python train_specialist_pfoa.py \
    --data_dir ds/axial_pfoa \
    --labels labels_avg.parquet \
    --gold_csv data/train.csv \
    --arch coatnet_rmlp_1_rw_224 \
    --epochs 16 \
    --bs 4 \
    --lr 2e-4 \
    --wd 0.02 \
    --out_dir specialist_pfoa \
    2>&1 | tee train_specialist_pfoa.log
