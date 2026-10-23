#!/bin/bash
EXTRACT_PID=3756272
echo "[watcher_c96] Waiting for extraction process $EXTRACT_PID to finish..."
while kill -0 $EXTRACT_PID 2>/dev/null; do
    sleep 20
done
echo "[watcher_c96] Extraction finished! Verifying embeddings_c96.npy..."
if [ -f ~/projects/knee/embeddings_c96.npy ]; then
    echo "[watcher_c96] Launching 96-slice Bi-LSTM head training now..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate knee
    cd ~/projects/knee
    python train_bilstm_c96.py 2>&1 | tee train_bilstm_c96.log
    echo "[watcher_c96] 96-slice Bi-LSTM head training complete!"
else
    echo "[watcher_c96] Error: embeddings_c96.npy not found!"
fi
