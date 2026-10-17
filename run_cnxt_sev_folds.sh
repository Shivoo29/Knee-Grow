#!/bin/bash
# ConvNeXt-T 2.5D attention-MIL on the 192-slice corpus: a second model family whose errors should differ from
# the public blend (CoAtNet/Raptor-like). 24 window centres per step, each as two tiles (140 mm field + centred
# 100 mm close-up, 0.39 mm/px at 256 px), Clef severity labels (labels_sev_v1: 11 findings from Clef grades, Lateral OA from labels_hybrid_v1), ACMIL,
# epoch selection on a 300-study held-out subset instead of gold-58. --k/--k_eval count tiles (2 per centre).
# Run from the directory holding train_knee.py and the corpus192 all_*.npy files:
#   PY=<python> RAPTOR_INPUT=<ds dir> [RAPTOR_PRELOAD=1] ./run_cnxt_folds.sh 0 1
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python}
for f in "$@"; do
  $PY train_knee.py --ckpt timm --arch convnext_tiny.in12k_ft_in1k --res 256 --norm imagenet --zoom_mm 100 \
    --bs 2 --accum 8 --k 48 --k_eval 380 --grad_ckpt --epochs 15 --patience 5 --bb_lr 1e-4 --head_lr 1e-3 --workers 8 \
    --clean_windows --mirror_p 0.5 --contrast_aug --slot_emb --slot_sizes 40,40,36,36,40 \
    --labels labels_sev_v1.parquet --label_weights label_weights_sev_v1.parquet \
    --stkim_k 3 --stkim_p 0.7 --oof_every 300 \
    --folds 4 --fold "$f" --fold_file folds_rand4.json --seed $((80 + f)) --tag cnxt_sev_f$f \
    > "train_cnxt_sev_f$f.log" 2>&1
done
