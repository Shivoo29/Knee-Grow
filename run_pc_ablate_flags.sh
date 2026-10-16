#!/bin/bash
# Ablation: do the T05 recipe flags help? Same as the distilled 96-slice student (labels_distil_v1; seeds 7/8/10 scored
# gold 0.928/0.930/0.927 without flags) but WITH clean windows + mirror + contrast + slot embedding.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RAPTOR_INPUT="$PWD/ds"
exec systemd-inhibit --what=sleep:idle --why="knee ablation flags" \
  .venv/bin/python train_knee.py --ckpt timm --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
    --res 384 --bs 1 --accum 16 --k 12 --k_eval 94 --epochs 16 --patience 5 --win_chunk 8 --grad_ckpt \
    --clean_windows --mirror_p 0.5 --contrast_aug --slot_emb \
    --labels labels_distil_v1.parquet --seed 42 --tag v6dflags_s42
