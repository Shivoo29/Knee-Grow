#!/bin/bash
# node1: 24 windows/step on the 192-slice corpus + T05 flags; starts after d192_s50 finishes (one job per GPU).
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 RAPTOR_INPUT=$HOME/projects/knee/ds RAPTOR_PRELOAD=1
while pgrep -f "tag d192_s50" >/dev/null; do sleep 300; done
S=${1:-51}
~/miniconda3/envs/knee/bin/python train_knee.py --ckpt timm --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
    --res 384 --bs 2 --accum 8 --k 24 --k_eval 190 --epochs 20 --patience 6 --win_chunk 14 \
    --clean_windows --mirror_p 0.5 --contrast_aug --slot_emb \
    --slot_sizes 40,40,36,36,40 --labels labels_distil_v1.parquet \
    --seed $S --tag d192k24_s$S 2>&1 | tee "train_coat_d192k24_s$S.log"
