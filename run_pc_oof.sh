#!/bin/bash
# OOF relabelling, stage 1: CoAtNet version 6 on REPORT labels only (labels_avg; no teacher anywhere, so the OOF
# predictions are independent of Raptor/R7), T05 recipe (clean windows, mirror, contrast, slot embedding).
# Each fold model writes raptor_oof_oof4_fold<k>.npz = its predictions on the held-out quarter of the train set.
# Usage: ./run_pc_oof.sh 0 1     (runs the given folds one after another)
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RAPTOR_INPUT="$PWD/ds"
for f in "$@"; do
  systemd-inhibit --what=sleep:idle --why="knee OOF fold $f" \
    .venv/bin/python train_knee.py --ckpt timm --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
      --res 384 --bs 1 --accum 16 --k 12 --k_eval 94 --epochs 12 --patience 4 --win_chunk 8 --grad_ckpt \
      --clean_windows --mirror_p 0.5 --contrast_aug --slot_emb \
      --labels labels_avg.parquet --folds 4 --fold "$f" --fold_file folds_rand4.json \
      --seed $((30 + f)) --tag oof4_f$f > "train_coat_oof4_f$f.log" 2>&1
done
