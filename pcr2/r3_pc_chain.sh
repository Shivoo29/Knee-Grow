#!/bin/bash
# PC queue (slow disk: ConvNeXt fold 1 only, then CoAtNet 0-1, then extra seeds)
while kill -0 2296477 2>/dev/null; do sleep 20; done   # let the interrupted fold-1 run finish writing its resume state
for job in 'cnxt 210 1' 'coat 210 0 1' 'cnxt 410 0 1 2 3 4' 'coat 410 0 1 2 3 4'; do
  set -- $job; m=$1 s=$2; shift 2
  PRESET=$m SEED_BASE=$s FOLDS="$*" PY=/home/codemaster29/Documents/still_learning/rsna-knee-abnormality-detection/.venv/bin/python RAPTOR_INPUT=/home/codemaster29/Documents/still_learning/rsna-knee-abnormality-detection/data /home/codemaster29/Documents/still_learning/rsna-knee-abnormality-detection/pcr2/run_r3.sh
done
