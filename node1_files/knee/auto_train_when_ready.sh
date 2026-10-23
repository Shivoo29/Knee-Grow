#!/bin/bash
EXTRACT_PID=3650436
echo "[watcher] Waiting for extraction process $EXTRACT_PID to finish..."
while kill -0 $EXTRACT_PID 2>/dev/null; do
    sleep 15
done
echo "[watcher] Extraction finished! Launching Bi-LSTM head training now..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate knee
cd ~/projects/knee
python train_bilstm.py 2>&1 | tee train_bilstm.log
echo "[watcher] Bi-LSTM head training complete!"
