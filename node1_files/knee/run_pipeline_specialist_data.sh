#!/bin/bash
# run_pipeline_specialist_data.sh - Sequentially extract both specialist datasets on Node-1 CPU
set -e
cd /home/s05814802723/projects/knee
source ~/miniconda3/etc/profile.d/conda.sh
conda activate knee

export PYTHONUNBUFFERED=1

echo "[watcher] Waiting for build_roi_boxes.py to complete..."
while pgrep -f "build_roi_boxes.py" > /dev/null; do
    sleep 20
done

echo "[watcher] Checking roi_boxes.npz..."
if [ -f roi_boxes.npz ]; then
    echo "[watcher] [SUCCESS] roi_boxes.npz verified: $(ls -lh roi_boxes.npz | awk '{print $5}')"
else
    echo "[watcher] [ERROR] roi_boxes.npz missing!"
    exit 1
fi

echo "[watcher] Starting dedicated Axial PF OA extraction (18 slices @ 224px, ~3.9 GB)..."
/home/s05814802723/miniconda3/envs/knee/bin/python extract_axial_pfoa.py \
    --workers 16 \
    --res 224 \
    --k 18 \
    2>&1 | tee extract_axial.log

echo "[watcher] All specialist datasets generated successfully!"
