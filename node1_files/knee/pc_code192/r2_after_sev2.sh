#!/bin/bash
# node1: after ConvNeXt severity seed 2 fold 2 finishes, stop the sev2 loop (skip its fold 3) and train round-2 fold 3
cd "$(dirname "$0")"
while ! grep -q "wrote SWA OOF" train_cnxt_sev2_f2.log 2>/dev/null; do sleep 300; done
sleep 120; tmux kill-session -t cnxt_sev2 2>/dev/null && echo "$(date +%H:%M) stopped cnxt_sev2 after fold 2"
sleep 60
RAPTOR_INPUT=$HOME/projects/knee/ds RAPTOR_PRELOAD=1 PY=$HOME/miniconda3/envs/knee/bin/python ./run_cnxt_r2.sh 3
echo "$(date +%H:%M) r2 fold 3 done"
