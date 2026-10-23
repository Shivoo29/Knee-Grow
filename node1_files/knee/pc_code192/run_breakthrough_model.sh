#!/bin/bash
# Breakthrough Model: ConvNeXt-Small (ImageNet-22k pretrained) on 192-slice corpus with 100mm zoom crops,
# trained on labels_hybrid_v1.parquet with effusion-conditional synovitis loss weighting (label_weights_hybrid_v1.parquet).
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/s05814802723/miniconda3/envs/knee/bin/python

echo "================================================================================"
echo "LAUNCHING BREAKTHROUGH MODEL: convnext_small.fb_in22k_ft_in1k (Two-Scale Zoom)"
echo "Started at $(date)"
echo "================================================================================"

$PY train_knee.py --ckpt timm --arch convnext_small.fb_in22k_ft_in1k --res 256 --norm imagenet --zoom_mm 100 \
  --bs 2 --accum 8 --k 48 --k_eval 380 --grad_ckpt --epochs 15 --patience 5 --bb_lr 8e-5 --head_lr 1e-3 --workers 8 \
  --clean_windows --mirror_p 0.5 --contrast_aug --slot_emb --slot_sizes 40,40,36,36,40 \
  --labels labels_hybrid_v1.parquet --label_weights label_weights_hybrid_v1.parquet \
  --stkim_k 3 --stkim_p 0.7 --oof_every 300 \
  --folds 4 --fold 0 --fold_file folds_rand4.json --seed 77 --tag cnxt_s_breakthrough \
  > train_breakthrough.log 2>&1

echo "Breakthrough run completed at $(date) with exit code $?"
