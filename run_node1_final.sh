#!/bin/bash
# node1 (L4 24 GB): OOF relabelling stage 3: FINAL model on labels_oofmix_v1.parquet, node1 batch layout (bs 2 x accum 8).
# Run from ~/projects/knee/pc_code inside tmux:  tmux new -s final 'bash run_node1_final.sh 41'
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 RAPTOR_INPUT=$HOME/projects/knee/ds RAPTOR_PRELOAD=1
S=${1:-41}
~/miniconda3/envs/knee/bin/python train_knee.py --ckpt timm --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
    --res 384 --bs 2 --accum 8 --k 12 --k_eval 94 --epochs 14 --patience 5 --win_chunk 14 \
    --clean_windows --mirror_p 0.5 --contrast_aug --slot_emb \
    --labels labels_oofmix_v1.parquet \
    --seed $S --tag final_s$S 2>&1 | tee "train_coat_final_s$S.log"
