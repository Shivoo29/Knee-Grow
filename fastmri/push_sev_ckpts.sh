#!/bin/bash
# PC side: copy each finished cnxt_sev fold checkpoint to node1 (the fastMRI teacher runs there).
# A fold is finished once its SWA OOF file exists; the copy lands under .part and is renamed when complete.
cd "$(dirname "$0")/../pc192"
for f in 0 1 2 3; do
  while [ ! -f raptor_oof_cnxt_sev_f${f}_swa_fold${f}.npz ]; do sleep 300; done
  sleep 60
  scp -q raptor_ft_cnxt_sev_f${f}_swa.pt node1:projects/knee/pc_code192/raptor_ft_cnxt_sev_f${f}_swa.pt.part \
    && ssh node1 "mv ~/projects/knee/pc_code192/raptor_ft_cnxt_sev_f${f}_swa.pt.part ~/projects/knee/pc_code192/raptor_ft_cnxt_sev_f${f}_swa.pt" \
    && echo "$(date +%H:%M) pushed fold $f"
done
