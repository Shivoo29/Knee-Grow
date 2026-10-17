#!/bin/bash
# PC: pack ds/corpus192 (117 GB) into 16 compressed shards (~61 GB) and upload them as the private Kaggle
# dataset shivoo29/knee-c192-u8d, which the RTX Pro 6000 runs download (cloud/run_fold.sh). Rerun-safe:
# finished shards are kept.
set -e
cd "$(dirname "$0")/.."
OUT=kaggle_ds/knee-c192-u8d
nice -n 19 .venv/bin/python -I tools/export_corpus.py --src ds/corpus192 --out $OUT --shards 16 --workers 4
echo '{"title": "knee-c192-u8d", "id": "shivoo29/knee-c192-u8d", "licenses": [{"name": "CC0-1.0"}]}' > $OUT/dataset-metadata.json
kaggle datasets create -p $OUT -r skip
echo UPLOAD_DONE
