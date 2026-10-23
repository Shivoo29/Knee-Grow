#!/bin/bash
# node1 (L4 24 GB): OOF relabelling folds, same recipe as run_pc_oof.sh, node1 batch layout (bs 2 x accum 8).
# Run from ~/projects/knee/pc_code inside tmux:  tmux new -s oof 'bash run_node1_oof.sh 2 3'
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 RAPTOR_INPUT=$HOME/projects/knee/ds RAPTOR_PRELOAD=1
for f in "$@"; do
  ~/miniconda3/envs/knee/bin/python train_knee.py --ckpt timm --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
    --res 384 --bs 2 --accum 8 --k 12 --k_eval 94 --epochs 12 --patience 4 --win_chunk 14 \
    --clean_windows --mirror_p 0.5 --contrast_aug --slot_emb \
    --labels labels_avg.parquet --folds 4 --fold "$f" --fold_file folds_rand4.json \
    --seed $((30 + f)) --tag oof4_f$f 2>&1 | tee "train_coat_oof4_f$f.log"
done
