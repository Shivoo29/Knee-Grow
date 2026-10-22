#!/bin/bash
# BiGRU sequence-head test: ConvNeXt severity fold 1 recipe (seed 81) + --seq_head --aux_w 0.3. Field and close-up
# tiles run as two separate slice sequences (seq_tracks=2). Compare with pc192 cnxt_sev_f1 on the 1,087 OOF studies.
#   PY=<python> RAPTOR_INPUT=<ds dir> ./run_cnxt_seq.sh 1 [--smoke]
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python}
f=$1; shift
$PY train_knee.py --ckpt timm --arch convnext_tiny.in12k_ft_in1k --res 256 --norm imagenet --zoom_mm 100 \
  --bs 2 --accum 8 --k 48 --k_eval 380 --grad_ckpt --epochs 15 --patience 5 --bb_lr 1e-4 --head_lr 1e-3 --workers 8 \
  --clean_windows --mirror_p 0.5 --contrast_aug --slot_emb --slot_sizes 40,40,36,36,40 \
  --labels labels_sev_v1.parquet --label_weights label_weights_sev_v1.parquet \
  --stkim_k 3 --stkim_p 0.7 --oof_every 300 --seq_head --aux_w 0.3 \
  --folds 4 --fold "$f" --fold_file folds_rand4.json --seed $((80 + f)) --tag cnxt_seq_f$f "$@" \
  > "train_cnxt_seq_f$f${1:+_smoke}.log" 2>&1
