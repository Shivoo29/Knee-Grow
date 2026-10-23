cd ~/projects/knee/spike_kneeseg
rm -rf vols1 pred; venv/bin/python resamp.py || exit 1; rm -rf vols; mv vols1 vols
export OMP_NUM_THREADS=4 LOKY_MAX_CPU_COUNT=32 CUDA_VISIBLE_DEVICES=
taskset -c 0-31 venv/bin/python seg.py > seg.log 2>&1
