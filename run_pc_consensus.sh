#!/bin/bash
# Consensus-mask test: best 96-slice student recipe (distilled labels + T05 flags, gold 0.9299 at seed 42)
# plus --label_weights label_weights_consensus_v1 (ignores the 12% of cells where the 31 public LLM tables are
# uncertain, mean 0.3-0.7; forum: Prateek Grover, "improved CV and public LB"). Compare with 0.9299.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RAPTOR_INPUT="$PWD/ds"
exec systemd-inhibit --what=sleep:idle --why="knee consensus mask" \
  .venv/bin/python train_knee.py --ckpt timm --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
    --res 384 --bs 1 --accum 16 --k 12 --k_eval 94 --epochs 16 --patience 5 --win_chunk 8 --grad_ckpt \
    --clean_windows --mirror_p 0.5 --contrast_aug --slot_emb \
    --labels labels_distil_v1.parquet --label_weights label_weights_consensus_v1.parquet --seed 42 --tag v6dmask_s42
