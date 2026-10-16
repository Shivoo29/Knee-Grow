#!/bin/bash
# PC: CoAtNet on the 192-slice corpus with 24 windows per training step (vs 12 on node1's d192_s50),
# distilled labels. Question: does denser window sampling help once the input has 2x the slices?
# Runs inside pc192/ (own all_*.npy symlinks -> ds/corpus192; the project root ones point at corpus96).
cd "$(dirname "$0")/pc192"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RAPTOR_INPUT="$(cd .. && pwd)/ds"
exec systemd-inhibit --what=sleep:idle --why="knee 192 k24" \
  ../.venv/bin/python train_knee.py --ckpt timm --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
    --res 384 --bs 1 --accum 16 --k 24 --k_eval 190 --epochs 14 --patience 5 --win_chunk 8 --grad_ckpt \
    --slot_sizes 40,40,36,36,40 --labels labels_distil_v1.parquet --seed 51 --tag d192k24_s51
