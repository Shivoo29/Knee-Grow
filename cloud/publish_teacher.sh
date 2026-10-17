#!/bin/bash
# PC: when node1's teacher pass has all four cnxt_sev folds (TEACHER_DONE in its log), copy the labels here and
# add them to the cloud bundle as a new version. The cloud runs wait for exactly this file (run_fold.sh step 4).
cd "$(dirname "$0")/.."
T=fastmri_teacher_sev.parquet
until ssh -o BatchMode=yes node1 "grep -q TEACHER_DONE ~/projects/knee/fastmri/overnight.log" 2>/dev/null; do sleep 600; done
scp -q node1:projects/knee/fastmri/$T $T
.venv/bin/python -I -c "import pandas as pd; d = pd.read_parquet('$T'); print(len(d), 'studies,', int(d.n_ckpts.iloc[0]), 'checkpoints')"
MSG="fastMRI teacher labels (cnxt_sev, 4 folds)" bash cloud/make_bundle.sh $T
