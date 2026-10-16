#!/bin/bash
# node1: CoAtNet on the 192-slice corpus (layout 40,40,36,36,40: ~90% of native slices kept, truly adjacent windows)
# with the strongest known labels (labels_distil_v1: base954 teacher + report labels; 96-slice version: gold 0.931, LB 0.948).
# Same recipe as the 96-slice 24-epoch student, no T05 flags (their effect is unproven).
# Run from ~/projects/knee/pc_code192:  tmux new -d -s d192 'bash run_node1_192.sh 50'
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 RAPTOR_INPUT=$HOME/projects/knee/ds RAPTOR_PRELOAD=1
S=${1:-50}
~/miniconda3/envs/knee/bin/python train_knee.py --ckpt timm --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
    --res 384 --bs 2 --accum 8 --k 12 --k_eval 190 --epochs 20 --patience 6 --win_chunk 14 \
    --slot_sizes 40,40,36,36,40 --labels labels_distil_v1.parquet \
    --seed $S --tag d192_s$S 2>&1 | tee "train_coat_d192_s$S.log"
