#!/bin/bash
set -e

cd /home/s05814802723/projects/knee
rm -f extract_progress.txt

# Kill previous sessions if any
tmux kill-session -t extract_gpu 2>/dev/null || true
tmux kill-session -t auto_bilstm 2>/dev/null || true

# Start bfloat16 extraction in tmux
tmux new-session -d -s extract_gpu "source ~/miniconda3/etc/profile.d/conda.sh; conda activate knee; cd ~/projects/knee; python extract_embeddings_gpu.py 2>&1 | tee extract_gpu.log"

sleep 3
EXTRACT_PID=$(pgrep -f "python extract_embeddings_gpu.py" | head -n 1)
echo "Extraction process started with PID: $EXTRACT_PID"

# Update watcher script with new PID
cat > auto_train_when_ready.sh << WATCHER_EOF
#!/bin/bash
EXTRACT_PID=$EXTRACT_PID
echo "[watcher] Waiting for extraction process \$EXTRACT_PID to finish..."
while kill -0 \$EXTRACT_PID 2>/dev/null; do
    sleep 15
done
echo "[watcher] Extraction finished! Launching Bi-LSTM head training now..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate knee
cd ~/projects/knee
python train_bilstm.py 2>&1 | tee train_bilstm.log
echo "[watcher] Bi-LSTM head training complete!"
WATCHER_EOF
chmod +x auto_train_when_ready.sh

# Start watcher in tmux
tmux new-session -d -s auto_bilstm "bash ~/projects/knee/auto_train_when_ready.sh 2>&1 | tee ~/projects/knee/watcher.log"
echo "Auto-training pipeline successfully launched!"
