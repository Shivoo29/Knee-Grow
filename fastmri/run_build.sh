#!/bin/bash
# node1: fastMRI knee DICOMs -> 192-slice corpus. Each archive is first fetched to local disk by resumable
# range requests (tools/fetch_ranges.py; the signed URL never reaches a command line), then built from the
# file (CPU-bound, minutes) and deleted. batch2 whole (~1,900 studies), then the first 64 GB of batch1
# (~3,900 studies; the builder keeps every complete study of a cut archive). Rerun-safe at every step.
cd "$(dirname "$0")"
PY=~/miniconda3/envs/knee/bin/python
DL=/tmp/fastmri_dl; mkdir -p $DL; chmod 700 $DL
for spec in knee_DICOMs_batch2:0 knee_DICOMs_batch1:64; do
  name=${spec%%:*}; gb=${spec##*:}
  $PY -I code/tools/fetch_ranges.py --urls .urls --archive $name --out $DL/$name.tar.xz --max_gb $gb > fetch_$name.log 2>&1 \
    && $PY -I code/tools/fastmri_corpus.py --tar $DL/$name.tar.xz --out corpus --workers 24 > build_$name.log 2>&1 \
    && rm -f $DL/$name.tar.xz $DL/$name.tar.xz.done
done
