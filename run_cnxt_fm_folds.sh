#!/bin/bash
# ConvNeXt-T students: competition studies (Clef severity labels) + fastMRI knee studies labelled by our own
# ConvNeXt teacher (fastmri/run_teacher.sh). fastMRI rows always train, never validate. Needs RAPTOR_EXTRA_CORPUS = the
# fastMRI corpus folder and TEACHER = the teacher parquet (default fastmri_teacher_cnxt.parquet). 12 epochs (2.4x data).
# epoch selection on a 300-study held-out subset instead of gold-58. --k/--k_eval count tiles (2 per centre).
# Run from the directory holding train_knee.py and the corpus192 all_*.npy files:
#   PY=<python> RAPTOR_INPUT=<ds dir> RAPTOR_EXTRA_CORPUS=<fastmri corpus> [TEACHER=<parquet>] [BS=2 ACCUM=8 WORKERS=8] ./run_cnxt_fm_folds.sh 0 1
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python}
: "${RAPTOR_EXTRA_CORPUS:?set RAPTOR_EXTRA_CORPUS to the fastMRI corpus folder}"
for f in "$@"; do
  $PY train_knee.py --ckpt timm --arch convnext_tiny.in12k_ft_in1k --res 256 --norm imagenet --zoom_mm 100 \
    --bs ${BS:-2} --accum ${ACCUM:-8} --k 48 --k_eval 380 --grad_ckpt --epochs 12 --patience 5 --bb_lr 1e-4 --head_lr 1e-3 --workers ${WORKERS:-8} \
    --clean_windows --mirror_p 0.5 --contrast_aug --slot_emb --slot_sizes 40,40,36,36,40 \
    --labels labels_sev_v1.parquet --label_weights label_weights_sev_v1.parquet \
    --stkim_k 3 --stkim_p 0.7 --oof_every 300 \
    --folds 4 --fold "$f" --fold_file folds_rand4.json --seed $((100 + f)) --tag cnxt_fm_f$f \
    --extra_labels "${TEACHER:-fastmri_teacher_cnxt.parquet}" --extra_weight 1.0 ${EXTRA:-} \
    > "train_cnxt_fm_f$f.log" 2>&1
done
