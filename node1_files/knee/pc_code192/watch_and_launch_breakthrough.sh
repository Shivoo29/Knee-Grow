#!/bin/bash
# Multi-Tier Queue Watcher for node1
# Guarantees that the breakthrough training ONLY starts when ALL ongoing processes AND queued jobs are 100% finished.

MONITOR_PID=3574168
DIR="/home/s05814802723/projects/knee/pc_code192"
cd "$DIR"

echo "================================================================================"
echo "MULTI-TIER QUEUE WATCHER ACTIVE"
echo "Target 1: Monitor active parent runner PID $MONITOR_PID (run_cnxt_folds.sh 0 1)"
echo "Target 2: Monitor all train_knee.py processes"
echo "Target 3: Monitor NVIDIA L4 GPU compute occupancy (nvidia-smi)"
echo "Started at: $(date)"
echo "================================================================================"

# Tier 1: Wait for specific parent runner PID 3574168 (Fold 0 and Fold 1) to exit
while kill -0 "$MONITOR_PID" 2>/dev/null; do
    ACTIVE_COUNT=$(pgrep -f "train_knee.py" | wc -l)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
    echo "[$(date '+%H:%M:%S')] Tier 1: Waiting for PID $MONITOR_PID... Active train_knee processes: $ACTIVE_COUNT | GPU VRAM: ${GPU_MEM} MiB"
    sleep 45
done

echo ""
echo "[$(date '+%H:%M:%S')] Tier 1 Complete: PID $MONITOR_PID has terminated!"

# Tier 2: Wait for ANY ongoing train_knee.py processes to finish
echo "[$(date '+%H:%M:%S')] Tier 2: Checking for any remaining train_knee.py processes..."
while true; do
    TRAIN_PIDS=$(pgrep -f "train_knee.py" || true)
    if [ -z "$TRAIN_PIDS" ]; then
        echo "[$(date '+%H:%M:%S')] Tier 2 Complete: Zero train_knee.py processes detected."
        break
    else
        COUNT=$(echo "$TRAIN_PIDS" | wc -l)
        echo "[$(date '+%H:%M:%S')] Tier 2: $COUNT train_knee.py processes still running. Waiting..."
        sleep 30
    fi
done

# Tier 3: Verify GPU is completely idle (< 500 MiB VRAM and 0 compute apps) for 60 consecutive seconds
echo "[$(date '+%H:%M:%S')] Tier 3: Verifying GPU compute idle state..."
IDLE_STREAK=0
while [ "$IDLE_STREAK" -lt 2 ]; do
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
    GPU_APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    
    if [ "$GPU_MEM" -lt 500 ] && [ "$GPU_APPS" -eq 0 ]; then
        IDLE_STREAK=$((IDLE_STREAK + 1))
        echo "[$(date '+%H:%M:%S')] Tier 3: GPU verified idle ($IDLE_STREAK/2 checks, VRAM: ${GPU_MEM} MiB, Compute apps: 0)."
    else
        IDLE_STREAK=0
        echo "[$(date '+%H:%M:%S')] Tier 3: GPU still occupied (VRAM: ${GPU_MEM} MiB, Compute apps: $GPU_APPS). Waiting..."
    fi
    sleep 30
done

echo ""
echo "================================================================================"
echo "ALL ONGOING PROCESSES AND QUEUED JOBS ARE 100% COMPLETE!"
echo "GPU is verified idle and ready."
echo "Timestamp: $(date)"
echo "Verifying Fold 0 and Fold 1 artifacts:"
ls -lh "$DIR"/raptor_ft_cnxt_*.pt 2>/dev/null
echo "================================================================================"

echo "Starting Breakthrough Model in 10 seconds..."
sleep 10
bash "$DIR/run_breakthrough_model.sh"
