#!/bin/bash
# RadImageNet test: round-2 fold recipe (seed 110+f) with the RadImageNet ResNet50 (MIT, public weights; CT/MRI/US
# pretraining incl. musculoskeletal MRI) instead of ImageNet ConvNeXt-T. Compare with cnxt_r2_f<f> (fold 0 gold 0.9296).
#   PY=<python> RAPTOR_INPUT=<ds dir> ./run_rin_r2.sh 0 [--smoke]
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python}
f=$1; shift
$PY train_knee.py --ckpt radimagenet_resnet50.pt --arch resnet50 --res 256 --norm imagenet --zoom_mm 100 \
  --bs 2 --accum 8 --k 48 --k_eval 380 --grad_ckpt --epochs 15 --patience 5 --bb_lr 1e-4 --head_lr 1e-3 --workers ${WORKERS:-8} \
  --clean_windows --mirror_p 0.5 --contrast_aug --slot_emb --slot_sizes 40,40,36,36,40 \
  --labels labels_r2.parquet --label_weights label_weights_sev_v1.parquet \
  --stkim_k 3 --stkim_p 0.7 --oof_every 300 \
  --folds 4 --fold "$f" --fold_file folds_rand4.json --seed $((110 + f)) --tag rin_r2_f$f "$@" \
  > "train_rin_r2_f$f${1:+_smoke}.log" 2>&1
