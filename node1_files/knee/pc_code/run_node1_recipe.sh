#!/bin/bash
# Full recipe + bi-GRU sequence head on report labels (decides whether final models use --seq_head). Small batch to share the GPU with OOF folds.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 RAPTOR_INPUT=$HOME/projects/knee/ds
~/miniconda3/envs/knee/bin/python train_knee.py --ckpt timm --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
  --res 384 --bs 1 --accum 16 --k 12 --k_eval 94 --epochs 14 --patience 5 --win_chunk 8 --grad_ckpt \
  --clean_windows --mirror_p 0.5 --contrast_aug --slot_emb --seq_head --aux_w 0.3 \
  --label_weights label_weights_v1.parquet --labels labels_avg.parquet --seed 22 --tag recipe_s22 2>&1 | tee train_coat_recipe_s22.log
