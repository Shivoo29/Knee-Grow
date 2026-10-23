#!/bin/bash
# AXFS distilled student: v6 recipe with T05 clean windows and slot embeddings.
set -euo pipefail
cd "$HOME/projects/knee"
export RAPTOR_INPUT="$HOME/projects/knee/ds_axfs" RAPTOR_PRELOAD=1
nice -n 19 "$HOME/miniconda3/envs/knee/bin/python" "$HOME/projects/knee/train_axfs/train_knee.py" --ckpt timm --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
  --res 384 --bs 2 --accum 8 --k 12 --k_eval 94 --epochs 16 --patience 5 --win_chunk 14 \
  --clean_windows --slot_emb \
  --labels labels_distil_v1.parquet --seed 11 --tag v6d_axfs_s11 2>&1 | tee train_coat_v6d_axfs_s11.log
