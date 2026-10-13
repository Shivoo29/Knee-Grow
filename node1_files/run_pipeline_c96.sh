#!/bin/bash
set -e

cd /home/s05814802723/projects/knee
rm -f extract_c96_progress.txt

# Kill previous c96 sessions if any
tmux kill-session -t extract_c96 2>/dev/null || true
tmux kill-session -t auto_bilstm_c96 2>/dev/null || true

# Start 96-slice extraction in tmux
tmux new-session -d -s extract_c96 "source ~/miniconda3/etc/profile.d/conda.sh; conda activate knee; cd ~/projects/knee; python extract_c96_gpu.py 2>&1 | tee extract_c96.log"

sleep 3
EXTRACT_PID=$(pgrep -f "python extract_c96_gpu.py" | head -n 1)
echo "96-slice extraction process started with PID: $EXTRACT_PID"

# Create watcher script for c96
cat > auto_train_c96.sh << WATCHER_EOF
#!/bin/bash
EXTRACT_PID=$EXTRACT_PID
echo "[watcher_c96] Waiting for extraction process \$EXTRACT_PID to finish..."
while kill -0 \$EXTRACT_PID 2>/dev/null; do
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
WATCHER_EOF
chmod +x auto_train_c96.sh

# Start watcher in tmux
tmux new-session -d -s auto_bilstm_c96 "bash ~/projects/knee/auto_train_c96.sh 2>&1 | tee ~/projects/knee/watcher_c96.log"
echo "Parallel 96-slice pipeline successfully launched!"
