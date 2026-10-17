#!/bin/bash
# node1: fastMRI student fold 0 (ConvNeXt-T), started by hand once fastmri_teacher_sev.parquet is complete.
# corpus192 is preloaded into RAM (117 GB); fastMRI studies are read one by one from the NFS corpus folder.
cd ~/projects/knee/fm_student
RAPTOR_INPUT=$HOME/projects/knee/ds RAPTOR_PRELOAD=1 RAPTOR_EXTRA_CORPUS=$HOME/projects/knee/fastmri/corpus \
TEACHER=$HOME/projects/knee/fastmri/fastmri_teacher_sev.parquet PY=~/miniconda3/envs/knee/bin/python \
  ./run_cnxt_fm_folds.sh ${1:-0}
