#!/bin/bash
# node1 queue: finish CoAtNet s210 fold 4, then ConvNeXt s210 folds 2-4 (fast here: 251 GB RAM), CoAtNet 3 2, extra seeds
cd ~/projects/knee/r3
until grep -q "^DONE" train_coat_r3_s210_f4.log 2>/dev/null; do sleep 60; done
tmux kill-session -t r3 2>/dev/null; sleep 3
pgrep -f "^$HOME/miniconda3/envs/knee/bin/python train_knee.py" | xargs -r kill; sleep 10
for job in "cnxt 210 2 3 4" "coat 210 3 2" "cnxt 310 0 1 2 3 4" "coat 310 0 1 2 3 4"; do
  set -- $job; m=$1 s=$2; shift 2
  PRESET=$m SEED_BASE=$s FOLDS="$*" PY=$HOME/miniconda3/envs/knee/bin/python RAPTOR_INPUT=$HOME/projects/knee ./run_r3b.sh
done
