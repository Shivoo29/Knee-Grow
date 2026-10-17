#!/bin/bash
# Teacher pass on node1: our ConvNeXt-T 4-fold family (hybrid labels, OAI-free) labels every fastMRI study.
# Mean sigmoid over the 4 checkpoints, no flip TTA, every 2nd window (--k_eval 190 tiles). Per-checkpoint
# results are cached next to the output, so a rerun (or adding checkpoints) only computes what is new.
# Run from ~/projects/knee/fastmri (holds code/, corpus/); checkpoints in ../pc_code192.
cd "$(dirname "$0")"
PY=${PY:-~/miniconda3/envs/knee/bin/python}
CK=${CK:-../pc_code192}
$PY -I code/tools/predict_corpus.py --corpus corpus --k_eval 190 --bs 2 --workers 8 --out fastmri_teacher_cnxt.parquet \
  --ckpts $CK/raptor_ft_cnxt_f0_swa.pt,$CK/raptor_ft_cnxt_f1_swa.pt,$CK/raptor_ft_cnxt_f2_swa.pt,$CK/raptor_ft_cnxt_f3_swa.pt
