#!/bin/bash
# v6 recipe (96-slice CoAtNet, labels_avg), extra seeds for the blend. Same args as v6/v6_s3, only --seed/--tag differ.
cd ~/projects/knee
export RAPTOR_INPUT=$HOME/projects/knee/ds RAPTOR_PRELOAD=1
for s in 4 5; do
  ~/miniconda3/envs/knee/bin/python train_knee.py --ckpt timm --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
    --res 384 --bs 2 --accum 8 --k 12 --k_eval 94 --epochs 16 --patience 5 --win_chunk 14 \
    --labels labels_avg.parquet --seed $s --tag v6_s$s 2>&1 | tee train_coat_v6_s$s.log
done
