#!/bin/bash
# Full-recipe CoAtNet version 6 on REPORT labels (not the teacher): clean windows, per-study mirror + contrast aug,
# slot embedding, bi-GRU sequence head + aux loss (lumbar 2024 1st place), label down-weighting where teacher and
# report labels disagree by > 0.7 (lumbar 2024 2nd place). Baseline to beat: CoAtNet version 6 label-trained, gold ~0.925.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RAPTOR_INPUT="$PWD/ds"
exec systemd-inhibit --what=sleep:idle --why="knee recipe run" \
  .venv/bin/python train_knee.py --ckpt timm --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
    --res 384 --bs 1 --accum 16 --k 12 --k_eval 94 --epochs 14 --patience 5 --win_chunk 8 --grad_ckpt \
    --clean_windows --mirror_p 0.5 --contrast_aug --slot_emb --seq_head --aux_w 0.3 \
    --label_weights label_weights_v1.parquet --labels labels_avg.parquet --seed 22 --tag recipe_s22 "$@"
