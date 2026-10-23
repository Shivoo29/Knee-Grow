#!/usr/bin/env python3
"""build_roi_boxes.py - Extract per-slice tissue bounding boxes and centroids for 96-slice corpus.

Reads corpus96/all_vols.npy (4407, 96, 384, 384) directly via memory-map.
Extracts tissue bounding box `sbox` and center-of-mass `cen` per slice using parallel workers.
Saves to `roi_boxes.npz` to unlock `--roi --roi_mode compartment` in train_knee.py.
"""

import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def _process_chunk(chunk_args):
    """Worker function to process a batch of studies."""
    vols_path, masks_path, start_idx, end_idx, W, H = chunk_args
    # Open mmap per worker process
    vols = np.load(vols_path, mmap_mode="r")
    masks = np.load(masks_path)

    n_studies = end_idx - start_idx
    D = vols.shape[1]

    sboxes = np.zeros((n_studies, D, 4), dtype=np.int16)
    cens = np.zeros((n_studies, D, 2), dtype=np.int16)

    y_grid, x_grid = np.indices((H, W), dtype=np.float32)

    for i_local, i_global in enumerate(range(start_idx, end_idx)):
        vol = vols[i_global]      # (96, 384, 384) uint8
        mask = masks[i_global]    # (96,) uint8

        for c in range(D):
            if mask[c] == 0:
                sboxes[i_local, c] = [0, 0, W, H]
                cens[i_local, c] = [W // 2, H // 2]
                continue

            img = vol[c]
            max_val = img.max()
            if max_val < 15:
                sboxes[i_local, c] = [0, 0, W, H]
                cens[i_local, c] = [W // 2, H // 2]
                continue

            # Threshold tissue foreground
            thresh = max(15, int(max_val * 0.12))
            fg = img >= thresh

            if fg.sum() < 40:
                thresh = 10
                fg = img >= thresh

            if fg.sum() < 40:
                sboxes[i_local, c] = [0, 0, W, H]
                cens[i_local, c] = [W // 2, H // 2]
                continue

            ys = np.where(fg.any(axis=1))[0]
            xs = np.where(fg.any(axis=0))[0]

            y0, y1 = int(ys[0]), int(ys[-1]) + 1
            x0, x1 = int(xs[0]), int(xs[-1]) + 1

            # Weighted centroid
            w = img[fg].astype(np.float32)
            w_sum = w.sum()
            if w_sum > 0:
                cx = float((x_grid[fg] * w).sum() / w_sum)
                cy = float((y_grid[fg] * w).sum() / w_sum)
            else:
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0

            sboxes[i_local, c] = [x0, y0, x1, y1]
            cens[i_local, c] = [int(round(cx)), int(round(cy))]

    return start_idx, end_idx, sboxes, cens


def main():
    parser = argparse.ArgumentParser(description="Extract ROI boxes for knee corpus")
    parser.add_argument("--corpus_dir", default=os.path.expanduser("~/projects/knee/ds/corpus96"),
                        help="Path containing all_vols.npy, all_masks.npy, all_ids.npy")
    parser.add_argument("--output", default=os.path.expanduser("~/projects/knee/roi_boxes.npz"),
                        help="Target output path for roi_boxes.npz")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--chunk_size", type=int, default=64, help="Studies per worker task")
    args = parser.parse_args()

    vols_path = os.path.join(args.corpus_dir, "all_vols.npy")
    masks_path = os.path.join(args.corpus_dir, "all_masks.npy")
    ids_path = os.path.join(args.corpus_dir, "all_ids.npy")

    if not os.path.exists(vols_path):
        raise FileNotFoundError(f"Missing {vols_path}")

    ids = np.load(ids_path, allow_pickle=True)
    masks = np.load(masks_path)
    vols_shape = np.load(vols_path, mmap_mode="r").shape

    N, D, H, W = vols_shape
    print(f"[build_roi_boxes] Corpus: {N} studies, {D} slices, {H}x{W} resolution")
    print(f"[build_roi_boxes] Spawning {args.workers} workers, chunk_size={args.chunk_size}")

    t0 = time.time()
    chunks = []
    for start in range(0, N, args.chunk_size):
        end = min(start + args.chunk_size, N)
        chunks.append((vols_path, masks_path, start, end, W, H))

    all_sboxes = np.zeros((N, D, 4), dtype=np.int16)
    all_cens = np.zeros((N, D, 2), dtype=np.int16)

    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for start_idx, end_idx, sboxes, cens in executor.map(_process_chunk, chunks):
            all_sboxes[start_idx:end_idx] = sboxes
            all_cens[start_idx:end_idx] = cens
            completed += (end_idx - start_idx)
            if completed % 512 == 0 or completed == N:
                elapsed = time.time() - t0
                rate = completed / elapsed
                print(f"[build_roi_boxes] Processed {completed}/{N} studies ({completed/N*100:.1f}%) "
                      f"in {elapsed:.1f}s ({rate:.1f} studies/s)", flush=True)

    elapsed = time.time() - t0
    print(f"[build_roi_boxes] Extraction finished in {elapsed:.1f}s. Saving to {args.output}...")

    np.savez_compressed(args.output, ids=ids, sbox=all_sboxes, cen=all_cens)
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"[build_roi_boxes] [SUCCESS] Saved {args.output} ({file_size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
