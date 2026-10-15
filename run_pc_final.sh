#!/bin/bash
# OOF relabelling, stage 3 (PC): final CoAtNet version 6 on labels_oofmix_v1.parquet (OOF image predictions mixed
# with report labels; build with build_targets_v2.py once all 4 folds exist), T05 recipe, all 4,349 training studies.
# Usage: ./run_pc_final.sh 40
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RAPTOR_INPUT="$PWD/ds"
S=${1:-40}
exec systemd-inhibit --what=sleep:idle --why="knee final s$S" \
  .venv/bin/python train_knee.py --ckpt timm --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
    --res 384 --bs 1 --accum 16 --k 12 --k_eval 94 --epochs 14 --patience 5 --win_chunk 8 --grad_ckpt \
    --clean_windows --mirror_p 0.5 --contrast_aug --slot_emb \
    --labels labels_oofmix_v1.parquet --seed "$S" --tag final_s$S
