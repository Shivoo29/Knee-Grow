#!/bin/bash
# Specialist 1 (OA / compartments): CoAtNet version 6 recipe on CORONAL windows only (slots 2,3), compartment ROI crops
# (medial/lateral halves of the joint box), clean windows + contrast aug, REPORT labels (labels_avg, not the teacher)
# so its errors differ from Raptor/R7. Mirror aug is off: roi_boxes are not mirrored with the image.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RAPTOR_INPUT="$PWD/ds"
exec systemd-inhibit --what=sleep:idle --why="knee OA specialist" \
  .venv/bin/python train_knee.py --ckpt timm --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
    --res 384 --bs 1 --accum 16 --k 12 --k_eval 60 --epochs 16 --patience 5 --win_chunk 8 --grad_ckpt \
    --roi --roi_mode compartment --slots 2,3 --clean_windows --contrast_aug \
    --labels labels_avg.parquet --seed 21 --tag spec_oa_s21 "$@"
