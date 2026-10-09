#!/usr/bin/env bash
# One-shot status of everything running for the competition.
cd "$(dirname "$0")"
C=rsna-knee-abnormality-detection

echo "== Kaggle submissions (latest 3)"
kaggle competitions submissions $C -v 2>/dev/null | .venv/bin/python -c "
import csv,sys
for r in list(csv.DictReader(sys.stdin))[:3]:
    print(r['date'][:16], r['status'].split('.')[-1], r['publicScore'] or '-', '|', r['description'][:50])"

echo "== Kaggle notebooks"
for k in shivoo29/rsna-knee-v5-fast-efficiency shivoo29/rsna-knee-v5-full-tta shivoo29/rsna-knee-v5-submission; do kaggle kernels status $k 2>/dev/null | sed 's/has status//'; done


echo "== node1 training"
ssh -o ConnectTimeout=5 node1 'grep -E "^ep|BEST" ~/projects/knee/v5.log 2>/dev/null | tail -3 | cut -c1-75 || grep -E "^ep|BEST" ~/projects/knee/v4base.log | tail -3 | cut -c1-75;
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader' 2>/dev/null || echo "node1 unreachable"

grep -E "^ep|BEST" coat224_v1.log 2>/dev/null | tail -3 | cut -c1-75 || grep -E "^ep|BEST" r50_v1.log 2>/dev/null | tail -3 | cut -c1-75 || echo "no local log yet"

nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null

