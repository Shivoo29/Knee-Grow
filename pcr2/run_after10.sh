#!/bin/bash
# after idea 10: (1) existing fold-0 model on axial-FS-swapped volumes, (2) contrast-only variants
cd "$(dirname "$0")"
PY=$HOME/Documents/still_learning/rsna-knee-abnormality-detection/.venv/bin/python
until grep -q "DONE idea 10" idea10/run.log; do sleep 30; done
run() { echo "$(date +%H:%M:%S) $1"; PYTHONUNBUFFERED=1 $PY tools/predict_corpus.py --ckpts raptor_ft_cnxt_r2_f0_swa.pt --workers 6 --bs 1 --win_chunk 64 "${@:2}" 2>&1 | grep -E "done|out\]|Error|Traceback"; }
run "axial-FS swap" --corpus axfix --out idea10/axfix.parquet
for v in fs:0,2 plain:1,3; do
  n=${v%%:*}; s=${v#*:}
  run "variant $n (oof)" --corpus . --studies folds_rand4.json:0 --slots $s --out idea10/${n}_oof.parquet
  run "variant $n (gold)" --corpus . --studies idea10/gold_ids.txt --slots $s --out idea10/${n}_gold.parquet
done
echo "$(date +%H:%M:%S) DONE after10"
