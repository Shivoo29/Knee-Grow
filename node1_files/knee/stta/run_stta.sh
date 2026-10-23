#!/bin/bash
# Series TTA pilot on node1: the round-2 fold-0 SWA model reads the 510 fold-0 + gold studies that have an unused
# series of the same plane and contrast, once with the corpus192 picks (base) and once with the alternates (alt).
cd "$(dirname "$0")"
PY=$HOME/miniconda3/envs/knee/bin/python
for v in base:../ds/corpus192 alt:../ds/alt_f0; do
  n=${v%%:*}; c=${v#*:}
  echo "$(date +%H:%M:%S) $n"
  PYTHONUNBUFFERED=1 $PY tools/predict_corpus.py --corpus "$c" --ckpts raptor_ft_cnxt_r2_f0_swa.pt --studies changed_ids.txt \
    --out stta_$n.parquet --workers 8 --bs 1 --win_chunk 64 2>&1 | grep -E "studies\]|done\]|out\]|Error|Traceback"
done
echo "$(date +%H:%M:%S) DONE stta"
