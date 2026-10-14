#!/bin/bash
# Specialist 2 (PF OA / Synovitis / Effusion): CoAtNet version 6 recipe on SAGITTAL-FLUID + AXIAL windows only (slots 0,4), whole image
# (no ROI), clean windows + contrast aug, REPORT labels (labels_avg, not the teacher)
# so its errors differ from Raptor/R7. Mirror aug on (no ROI boxes to keep in sync).
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RAPTOR_INPUT="$PWD/ds"
exec systemd-inhibit --what=sleep:idle --why="knee PF specialist" \
  .venv/bin/python train_knee.py --ckpt timm --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
    --res 384 --bs 1 --accum 16 --k 12 --k_eval 60 --epochs 16 --patience 5 --win_chunk 8 --grad_ckpt \
    --slots 0,4 --mirror_p 0.5 --clean_windows --contrast_aug \
    --labels labels_avg.parquet --seed 23 --tag spec_pf_s23 "$@"
