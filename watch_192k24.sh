#!/bin/bash
# Copies node1's 192-slice corpus, waits for the PC flag ablation to finish, then starts run_pc_192k24.sh.
# The PC project root symlinks all_*.npy to corpus96, so this run reads corpus192 through a separate root dir.
cd "$(dirname "$0")"
echo "$(date) copying corpus192"
for f in all_ids.npy all_masks.npy slots.csv all_vols.npy; do
  scp -q "node1:projects/knee/ds/corpus192/$f" ds/corpus192/ || { echo "$(date) COPY FAILED $f"; exit 1; }
done
echo "$(date) copy done: $(du -sh ds/corpus192 | cut -f1)"
until grep -q "^TOPK epochs" train_coat_v6dflags_s42.log 2>/dev/null; do sleep 300; done
sleep 60
echo "$(date) ablation finished; starting 192 k24"
exec ./run_pc_192k24.sh > pc192/train_coat_d192k24_s51.log 2>&1
