#!/usr/bin/env python3
"""
GPU-accelerated extraction of per-slice backbone embeddings from trained CoAtNet v6.
Safely runs with ~1.5 GB peak VRAM in fp16 alongside ConvNeXt training.

Saves:
  embeddings.npy  : (4407, 44, 1024) float16
  masks.npy       : (4407, 44) uint8 (1=valid scan slice, 0=padded)
  uids.npy        : (4407,) str (StudyInstanceUIDs)
  gold_labels.npy : (4407, 12) float32 (0/1 for 58 gold studies, NaN for others)
  soft_labels.npy : (4407, 12) float32 (LLM soft labels from labels_avg.parquet)
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm

HERE   = os.path.dirname(os.path.abspath(__file__))
CKPT   = os.path.join(HERE, 'raptor_ft_v6.pt')
CORP1  = os.path.join(HERE, 'ds', 'corpus1')
CORP2  = os.path.join(HERE, 'ds', 'corpus2')
LAB_PAR = os.path.join(HERE, 'labels_avg.parquet')
TRAIN_CSV = os.path.join(HERE, 'train.csv')

OUT_EMB   = os.path.join(HERE, 'embeddings.npy')
OUT_MASK  = os.path.join(HERE, 'masks.npy')
OUT_UID   = os.path.join(HERE, 'uids.npy')
OUT_GOLD  = os.path.join(HERE, 'gold_labels.npy')
OUT_SOFT  = os.path.join(HERE, 'soft_labels.npy')
PROGRESS_FILE = os.path.join(HERE, 'extract_progress.txt')

LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[info] Using device: {device}', flush=True)

    # 1. Load Corpus
    print('[1/5] Loading corpus indexes...', flush=True)
    ids1 = np.load(os.path.join(CORP1, 'all_ids.npy'))
    ids2 = np.load(os.path.join(CORP2, 'extra_ids.npy'))
    m1   = np.load(os.path.join(CORP1, 'all_masks.npy'))
    m2   = np.load(os.path.join(CORP2, 'extra_masks.npy'))
    vols1 = np.load(os.path.join(CORP1, 'all_vols.npy'), mmap_mode='r')
    vols2 = np.load(os.path.join(CORP2, 'extra_vols.npy'), mmap_mode='r')

    all_uids = np.concatenate([ids1, ids2])
    all_masks = np.concatenate([m1, m2], axis=0)
    N = len(all_uids)
    print(f'      Total studies: {N}', flush=True)

    uid_to_source = {}
    for i, u in enumerate(ids1):
        uid_to_source[u] = (1, i)
    for i, u in enumerate(ids2):
        uid_to_source[u] = (2, i)

    # 2. Load Labels
    print('[2/5] Loading labels...', flush=True)
    soft_df = pd.read_parquet(LAB_PAR)
    soft_map = {
        row['StudyInstanceUID']: np.array([row[c] for c in LABELS], dtype=np.float32)
        for _, row in soft_df.iterrows()
    }

    train_df = pd.read_csv(TRAIN_CSV)
    gold_subset = train_df.dropna(subset=LABELS)
    gold_map = {
        row['StudyInstanceUID']: np.array([row[c] for c in LABELS], dtype=np.float32)
        for _, row in gold_subset.iterrows()
    }
    print(f'      Soft labels: {len(soft_map)} | Gold labels: {len(gold_map)}', flush=True)

    # Prepare labels arrays
    gold_arr = np.full((N, 12), np.nan, dtype=np.float32)
    soft_arr = np.full((N, 12), np.nan, dtype=np.float32)
    for i, u in enumerate(all_uids):
        if u in gold_map:
            gold_arr[i] = gold_map[u]
        if u in soft_map:
            soft_arr[i] = soft_map[u]

    # Save static metadata
    np.save(OUT_MASK, all_masks)
    np.save(OUT_UID, all_uids)
    np.save(OUT_GOLD, gold_arr)
    np.save(OUT_SOFT, soft_arr)
    print('      Saved masks.npy, uids.npy, gold_labels.npy, soft_labels.npy', flush=True)

    # 3. Load Backbone Model
    print('[3/5] Loading CoAtNet v6 backbone...', flush=True)
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    arch = ck['arch']
    bb = timm.create_model(arch, pretrained=False, num_classes=0, in_chans=3, global_pool='avg')
    st = {k[len('backbone.'):]: v for k, v in ck['model'].items() if k.startswith('backbone.')}
    bb.load_state_dict(st, strict=True)
    bb.to(device).eval()
    print(f'      Backbone {arch} loaded on GPU.', flush=True)

    # Normalization constants (broadcastable to [11, 3, 1, 1])
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # 4. Prepare Memory-Mapped Embeddings Storage
    print('[4/5] Initializing embeddings storage...', flush=True)
    if os.path.exists(OUT_EMB):
        emb_mmap = np.lib.format.open_memmap(OUT_EMB, mode='r+')
    else:
        emb_mmap = np.lib.format.open_memmap(OUT_EMB, mode='w+', dtype=np.float16, shape=(N, 44, 1024))

    start_idx = 0
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as pf:
                start_idx = int(pf.read().strip())
            print(f'      Resuming from study {start_idx}/{N}', flush=True)
        except Exception:
            start_idx = 0

    # 5. Extraction Loop
    print(f'[5/5] Extracting embeddings from study {start_idx} to {N}...', flush=True)
    t0 = time.time()
    last_print = t0

    for i in range(start_idx, N):
        u = all_uids[i]
        src, row_idx = uid_to_source[u]
        vol = vols1[row_idx] if src == 1 else vols2[row_idx] # (44, 336, 336) uint8

        # Build 44 three-slice windows
        D = vol.shape[0] # 44
        windows = np.empty((D, 3, 336, 336), dtype=np.float32)
        for c in range(D):
            c_lo = max(0, c - 1)
            c_hi = min(D - 1, c + 1)
            windows[c, 0] = vol[c_lo]
            windows[c, 1] = vol[c]
            windows[c, 2] = vol[c_hi]
        windows /= 255.0

        # Move to GPU
        x = torch.from_numpy(windows).to(device, non_blocking=True)

        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                # Bilinear resize to 384x384
                x_384 = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)
                # Normalize
                x_norm = (x_384 - mean) / std
                # Process in 4 chunks of 11 to keep VRAM overhead < 1.5 GB
                chunk_feats = []
                for chunk in x_norm.split(11, dim=0):
                    chunk_feats.append(bb(chunk))
                study_feats = torch.cat(chunk_feats, dim=0) # (44, 1024)

        # Write to memmap in float16 (cpu float first)
        emb_mmap[i] = study_feats.float().cpu().numpy().astype(np.float16)

        # Checkpoint progress
        if (i + 1) % 50 == 0 or (i + 1) == N:
            emb_mmap.flush()
            with open(PROGRESS_FILE, 'w') as pf:
                pf.write(str(i + 1))
            now = time.time()
            done = (i + 1) - start_idx
            rate = done / (now - t0 + 1e-5)
            rem = (N - (i + 1)) / (rate + 1e-5)
            allocated = torch.cuda.memory_allocated(device) / (1024**2)
            reserved = torch.cuda.memory_reserved(device) / (1024**2)
            print(f'[{i+1}/{N}] {rate:.2f} studies/s | ETA: {rem/60:.1f}m | GPU alloc: {allocated:.0f}MB, res: {reserved:.0f}MB', flush=True)

    emb_mmap.flush()
    print(f'[done] Extracted all {N} studies in {(time.time() - t0)/60:.1f} minutes!', flush=True)

if __name__ == '__main__':
    main()
