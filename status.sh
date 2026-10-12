#!/usr/bin/env bash
# One-shot status: Kaggle submissions + kernels, node1 training, PC training.
cd "$(dirname "$0")"
C=rsna-knee-abnormality-detection

echo "== Kaggle submissions (latest 4)"
kaggle competitions submissions $C -v 2>/dev/null | .venv/bin/python -c "
import csv,sys
for r in list(csv.DictReader(sys.stdin))[:4]:
    print(' ', r['date'][:16], r['status'].split('.')[-1], r['publicScore'] or '-', '|', r['description'][:55])"

echo "== Kaggle notebooks"
for k in rsna-knee-base954-fleet rsna-knee-teacher954-c0 rsna-knee-teacher954-c1 rsna-knee-teacher954-c2 rsna-knee-teacher954-c3; do
  kaggle kernels status shivoo29/$k 2>/dev/null | sed 's/has status "KernelWorkerStatus./: /; s/"$//; s/^/  /'
done

echo "== node1 (newest training log)"
ssh -o ConnectTimeout=5 node1 'cd ~/projects/knee; f=$(ls -t train_coat_*.log | head -1); echo "  $f"; grep -E "^ep" $f | tail -2 | cut -c1-90 | sed "s/^/  /";
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed "s/^/  GPU /"' 2>/dev/null || echo "  node1 unreachable"

echo "== PC"
f=$(ls -t train_coat_*.log 2>/dev/null | head -1); echo "  $f"
grep -E "^ep" "$f" 2>/dev/null | tail -2 | cut -c1-90 | sed 's/^/  /'
pgrep -f "train_knee.py" >/dev/null && echo "  training process: running" || echo "  training process: NOT running"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed 's/^/  GPU /'
