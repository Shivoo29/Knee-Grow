#!/bin/bash
# Waits for node1's last OOF fold (SWA OOF file is written last), builds labels_oofmix_v1, starts final models:
# node1 seed 41 (tmux "final"), PC seed 40. File checks only (no pgrep: it matched its own command line before).
cd "$(dirname "$0")"
R=projects/knee/pc_code
echo "$(date) waiting for fold 3"
until ssh -o ConnectTimeout=10 node1 "test -f $R/raptor_oof_oof4_f3_swa_fold3.npz" 2>/dev/null; do sleep 300; done
sleep 120
echo "$(date) fold 3 done; copying"
scp -q "node1:$R/raptor_oof_oof4_f3_*fold3.npz" "node1:$R/raptor_gold_oof4_f3*.npz" "node1:$R/train_coat_oof4_f*.log" "node1:$R/raptor_ft_oof4_f*.json" . || exit 1
.venv/bin/python build_targets_v2.py --alpha 0.6 --out labels_oofmix_v1.parquet || exit 1
scp -q labels_oofmix_v1.parquet run_node1_final.sh "node1:$R/" || exit 1
ssh node1 "cd $R && tmux new -d -s final 'bash run_node1_final.sh 41'" && echo "$(date) node1 final_s41 started"
echo "$(date) PC final_s40 starting"
exec ./run_pc_final.sh 40 > train_coat_final_s40.log 2>&1
