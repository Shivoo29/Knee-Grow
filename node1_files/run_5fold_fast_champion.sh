#!/bin/bash
# run_5fold_fast_champion.sh - High-Throughput 5-Fold Champion Training Pipeline
# Fully saturates NVIDIA L4 24GB VRAM and Xeon 64 cores on node1
set -e

cd /home/s05814802723/projects/knee
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAPTOR_INPUT=$HOME/projects/knee/ds
export RAPTOR_PRELOAD=1

PYTHON=~/miniconda3/envs/knee/bin/python
TAG_BASE="coat_roi_5fold"
ROI_BOXES="/home/s05814802723/projects/knee/roi_boxes.npz"
FOLDS_FILE="/home/s05814802723/projects/knee/ds/folds_site.json"
LABELS="/home/s05814802723/projects/knee/labels_avg.parquet"

echo "=============================================================================="
echo "LAUNCHING HIGH-THROUGHPUT 5-FOLD CHAMPION PIPELINE (BS=4, 16 WORKERS, L4 GPU)"
echo "Config: CoAtNet-384 | Compartment-ROI | 5 Folds | SWA Top-3 | RAPTOR_PRELOAD"
echo "=============================================================================="

for FOLD in 0 1 2 3 4; do
    TAG="${TAG_BASE}_f${FOLD}"
    LOG="${TAG}.log"
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] >>> STARTING FOLD ${FOLD}/5 (Tag: ${TAG}) <<<"

    $PYTHON train_knee.py \
        --ckpt timm \
        --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
        --res 384 \
        --epochs 12 \
        --patience 4 \
        --topk 3 \
        --bs 4 \
        --accum 4 \
        --workers 16 \
        --grad_ckpt \
        --roi \
        --roi_mode compartment \
        --roi_overlap 0.12 \
        --roi_boxes "$ROI_BOXES" \
        --k 12 \
        --k_eval 48 \
        --bb_lr 3.5e-5 \
        --head_lr 1.5e-3 \
        --wd 0.05 \
        --tag "$TAG" \
        --labels "$LABELS" \
        --fold_file "$FOLDS_FILE" \
        --folds 5 \
        --fold "$FOLD" \
        2>&1 | tee "$LOG"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] >>> FOLD ${FOLD} COMPLETED SUCCESSFULLY! <<<"
done

echo ""
echo "=============================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL 5 FOLDS COMPLETED! AGGREGATING OOFS..."
echo "=============================================================================="
$PYTHON train_5fold_knee.py --tag "${TAG_BASE}" --oof_dir . --labels "$LABELS"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Master 5-Fold Pipeline Finished!"
