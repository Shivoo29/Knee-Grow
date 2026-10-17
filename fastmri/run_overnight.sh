#!/bin/bash
# node1 side: after the fastMRI corpus build ends and the ConvNeXt seed-2 run finishes its fold 1, stop that
# run (frees the GPU), then label the corpus with the Clef-severity ConvNeXt folds as they arrive from the PC.
# Output fastmri_teacher_sev.parquet: probabilities mapped back to the labels_sev_v1 scale (--calibrate).
cd "$(dirname "$0")"
PY=~/miniconda3/envs/knee/bin/python
CK=../pc_code192
while tmux has-session -t fastmri 2>/dev/null; do sleep 300; done
echo "$(date +%H:%M) corpus build finished"
while ! grep -q "wrote SWA OOF" $CK/train_cnxt_sev2_f1.log 2>/dev/null; do sleep 300; done
tmux kill-session -t cnxt_sev2 2>/dev/null && echo "$(date +%H:%M) stopped cnxt_sev2 after fold 1"
sleep 60
for f in 0 1 2 3; do
  while [ ! -f $CK/raptor_ft_cnxt_sev_f${f}_swa.pt ]; do sleep 300; done
  CKS=$(for g in $(seq 0 $f); do printf "%s," $CK/raptor_ft_cnxt_sev_f${g}_swa.pt; done)
  $PY -I code/tools/predict_corpus.py --corpus corpus --k_eval 190 --bs 2 --workers 8 --ckpts "${CKS%,}" \
      --calibrate $CK/labels_sev_v1.parquet --out fastmri_teacher_sev.parquet
  echo "$(date +%H:%M) teacher has folds 0-$f"
done
echo TEACHER_DONE
