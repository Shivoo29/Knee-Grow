#!/usr/bin/env python3
"""
Stage 1: Extract per-slice backbone embeddings from the trained v6 CoAtNet.
Runs fully on CPU (so it does not touch the GPU running ConvNeXt).

Outputs (all in ~/projects/knee/):
  embeddings.npy   shape=(N, 44, 1024) float16   ~400 MB
  emb_ids.npy      shape=(N,)          str        study UIDs
  emb_labels.npy   shape=(N, 12)       float32    soft labels (NaN for unlabelled)
"""
import os, time
import numpy as np
import torch
import timm
import pandas as pd

HERE  = os.path.dirname(os.path.abspath(__file__))
CKPT  = os.path.join(HERE, 'raptor_ft_v6.pt')
OUT_E = os.path.join(HERE, 'embeddings.npy')
OUT_I = os.path.join(HERE, 'emb_ids.npy')
OUT_L = os.path.join(HERE, 'emb_labels.npy')
CORP1 = os.path.join(HERE, 'ds', 'corpus1')
CORP2 = os.path.join(HERE, 'ds', 'corpus2')
LABELS= os.path.join(HERE, 'labels_avg.parquet')
GOLD  = os.path.join(HERE, 'train.csv')

LAB = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
       'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

# ── load corpus ───────────────────────────────────────────────────────────────
print('[1/4] loading corpus id lists...', flush=True)
ids1  = np.load(os.path.join(CORP1, 'all_ids.npy'))
ids2  = np.load(os.path.join(CORP2, 'extra_ids.npy'))
vols1 = np.load(os.path.join(CORP1, 'all_vols.npy'),   mmap_mode='r')  # (3200,44,336,336)
vols2 = np.load(os.path.join(CORP2, 'extra_vols.npy'), mmap_mode='r')  # (1207,44,336,336)
all_ids = np.concatenate([ids1, ids2])  # (4407,)
uid2part = {}
for i, uid in enumerate(ids1): uid2part[uid] = ('1', i)
for i, uid in enumerate(ids2): uid2part[uid] = ('2', i)
print(f'    corpus: {len(all_ids)} studies', flush=True)

# ── load labels ───────────────────────────────────────────────────────────────
print('[2/4] loading labels...', flush=True)
ldf = pd.read_parquet(LABELS)
uid2label = {row['StudyInstanceUID']: np.array([row[c] for c in LAB], dtype=np.float32)
             for _, row in ldf.iterrows()}
# identify gold-58 studies (held out for validation)
gold_df = pd.read_csv(GOLD)
if all(c in gold_df.columns for c in LAB):
    gold_ids = set(gold_df.dropna(subset=LAB)['StudyInstanceUID'].tolist())
else:
    gold_ids = set()
print(f'    {len(uid2label)} LLM-labelled, {len(gold_ids)} gold-58 studies', flush=True)

# ── load backbone ──────────────────────────────────────────────────────────────
print('[3/4] loading CoAtNet v6 backbone on CPU...', flush=True)
ck   = torch.load(CKPT, map_location='cpu', weights_only=False)
arch = ck['arch']
bb   = timm.create_model(arch, pretrained=False, num_classes=0, in_chans=3, global_pool='avg')
# strip 'backbone.' prefix from saved state dict
st   = {k[len('backbone.'):]: v for k, v in ck['model'].items() if k.startswith('backbone.')}
bb.load_state_dict(st, strict=True)
bb.eval()
# ImageNet normalisation used at training time (--norm imagenet)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
print(f'    loaded: {arch}  F_dim=1024', flush=True)

# ── extract ────────────────────────────────────────────────────────────────────
print('[4/4] extracting embeddings (CPU only)...', flush=True)
t0 = time.time()
uids_out, embs_out, labs_out = [], [], []

for idx, uid in enumerate(all_ids):
    part, row = uid2part[uid]
    vol = vols1[row] if part == '1' else vols2[row]   # (44,336,336) uint8
    D   = vol.shape[0]

    # build 3-slice windows: [c-1, c, c+1], clamped at edges
    wins = []
    for c in range(D):
        lo  = max(0, c - 1)
        hi  = min(D - 1, c + 1)
        tri = np.stack([vol[lo], vol[c], vol[hi]], 0).astype(np.float32) / 255.0
        wins.append(tri)
    x = torch.from_numpy(np.stack(wins, 0))   # (44,3,336,336)
    x = (x - MEAN) / STD

    with torch.no_grad():
        # chunk of 8 keeps peak RAM reasonable
        feat = torch.cat([bb(chunk) for chunk in x.split(8, dim=0)], 0)  # (44,1024)

    lab = uid2label.get(uid, np.full(12, np.nan, dtype=np.float32))
    uids_out.append(uid)
    embs_out.append(feat.half().numpy())   # store as float16 to save disk
    labs_out.append(lab)

    if (idx + 1) % 100 == 0:
        elapsed = time.time() - t0
        eta     = elapsed / (idx + 1) * (len(all_ids) - idx - 1)
        print(f'    {idx+1}/{len(all_ids)} | {elapsed/60:.1f} min | ETA {eta/60:.1f} min', flush=True)

embs_arr = np.stack(embs_out, 0)   # (N,44,1024) float16
labs_arr = np.stack(labs_out, 0)   # (N,12)      float32
ids_arr  = np.array(uids_out)      # (N,)

np.save(OUT_E, embs_arr)
np.save(OUT_I, ids_arr)
np.save(OUT_L, labs_arr)
print(f'\nDone in {(time.time()-t0)/60:.1f} min', flush=True)
print(f'embeddings.npy  {embs_arr.shape}  {embs_arr.nbytes/1e6:.0f} MB', flush=True)
print(f'emb_ids.npy     {ids_arr.shape}', flush=True)
print(f'emb_labels.npy  {labs_arr.shape}', flush=True)
