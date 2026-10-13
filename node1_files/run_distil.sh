#!/bin/bash
# Distilled student: v6 recipe (96-slice CoAtNet) on labels_distil_v1 (0.6 teacher / 0.4 labels; 0.3/0.7 for Lateral OA, PF OA).
cd ~/projects/knee
export RAPTOR_INPUT=$HOME/projects/knee/ds RAPTOR_PRELOAD=1
~/miniconda3/envs/knee/bin/python train_knee.py --ckpt timm --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
  --res 384 --bs 2 --accum 8 --k 12 --k_eval 94 --epochs 16 --patience 5 --win_chunk 14 \
  --labels labels_distil_v1.parquet --seed 7 --tag v6d_s7 2>&1 | tee train_coat_v6d_s7.log
