#!/bin/bash
# Idea 10 on the PC: round-2 fold-0 SWA model reads only some planes (inference only)
cd "$(dirname "$0")"
PY=$HOME/Documents/still_learning/rsna-knee-abnormality-detection/.venv/bin/python
for v in all: cor:2,3 axsag:0,1,4 sag:0,1 corax:2,3,4; do
  name=${v%%:*}; sl=${v#*:}
  for part in oof:folds_rand4.json:0 gold:idea10/gold_ids.txt; do
    pn=${part%%:*}; src=${part#*:}
    echo "$(date +%H:%M:%S) variant $name ($pn)"
    PYTHONUNBUFFERED=1 $PY tools/predict_corpus.py --corpus . --ckpts raptor_ft_cnxt_r2_f0_swa.pt --studies "$src" \
      --out idea10/${name}_${pn}.parquet --workers 6 --bs 1 --win_chunk 64 ${sl:+--slots $sl} 2>&1 | grep -E "done|out\]|Error|Traceback"
  done
done
echo "$(date +%H:%M:%S) DONE idea 10"
