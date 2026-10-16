#!/bin/bash
# Copies node1's 192-slice corpus, waits for the PC flag ablation to finish, then starts run_pc_192k24.sh.
# The PC project root symlinks all_*.npy to corpus96, so this run reads corpus192 through a separate root dir.
cd "$(dirname "$0")"
echo "$(date) copying corpus192"
for f in all_ids.npy all_masks.npy slots.csv; do
  scp -q "node1:projects/knee/ds/corpus192/$f" ds/corpus192/ || { echo "$(date) COPY FAILED $f"; exit 1; }
done
# all_vols.npy (116 GB): gzip-1 stream (~1.4x smaller over the ~5 MB/s link), size checked after
ssh node1 "pigz -1 -p 16 -c projects/knee/ds/corpus192/all_vols.npy" | gzip -dc > ds/corpus192/all_vols.npy || { echo "$(date) COPY FAILED all_vols"; exit 1; }
[ "$(stat -c %s ds/corpus192/all_vols.npy)" = "124769009792" ] || { echo "$(date) SIZE MISMATCH all_vols"; exit 1; }
echo "$(date) copy done: $(du -sh ds/corpus192 | cut -f1)"
until grep -q "^TOPK epochs" train_coat_v6dflags_s42.log 2>/dev/null; do sleep 300; done
sleep 60
echo "$(date) ablation finished; starting 192 k24"
exec ./run_pc_192k24.sh > pc192/train_coat_d192k24_s51.log 2>&1
