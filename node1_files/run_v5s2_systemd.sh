#!/bin/bash
set -e

cd /home/s05814802723/projects/knee
export RAPTOR_INPUT=/home/s05814802723/projects/knee/ds
export RAPTOR_PRELOAD=1
export PYTHONUNBUFFERED=1

echo "shivoo v5s2 (systemd)" > /tmp/knee_gpu.lock
cleanup() {
    rm -f /tmp/knee_gpu.lock
}
trap cleanup EXIT INT TERM

exec /home/s05814802723/miniconda3/envs/knee/bin/python train_knee.py     --ckpt timm     --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k     --res 384     --epochs 20     --patience 5     --bs 2     --accum 8     --k 12     --k_eval 42     --bb_lr 4e-5     --head_lr 2e-3     --wd 0.05     --seed 2     --tag v5s2     --labels labels_avg.parquet
