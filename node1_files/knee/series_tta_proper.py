#!/usr/bin/env python3
"""Series TTA test using the exact StudyWindows dataset from training."""
import sys
sys.path.insert(0, '/home/s05814802723/projects/knee')

import torch
import timm
import numpy as np
import pandas as pd
import json
import random
from torch.utils.data import DataLoader

from train_knee_ours import RaptorClassifier, StudyWindows, _open_corpus

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load checkpoint
ckpt_path = '/home/s05814802723/projects/knee/raptor_ft_cnxt_r2_f0_swa.pt'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
model_state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

# Create backbone
bb = timm.create_model('convnext_tiny.in12k_ft_in1k', pretrained=False, num_classes=0)
print("Backbone created")

# Build model with backbone - matching training config
model = RaptorClassifier(
    bb,
    F_dim=768,
    n=12,
    drop=0.2,
    slot_emb=True,
    seq_head=False
)
model.load_state_dict(model_state)
model.to(device)
model.eval()
model.win_chunk = 0
model.stkim_k = 0
model.stkim_p = 0.7
model.seq_tracks = 1
print("Model loaded")

# Load data
series_df = pd.read_csv('/home/s05814802723/projects/knee/cpu_ideas/train_series.csv')

with open('/home/s05814802723/projects/knee/folds_rand4.json') as f:
    data = json.load(f)
fold_map = data['folds'] if 'folds' in data else data

fold0_heldout = [uid for uid, fold in fold_map.items() if fold == 0]
print(f"Fold 0 held-out: {len(fold0_heldout)} studies")

# Load corpora
alt_ids = np.load('/home/s05814802723/projects/knee/ds/alt_f0/all_ids.npy', allow_pickle=True)
alt_vols = np.load('/home/s05814802723/projects/knee/ds/alt_f0/all_vols.npy', mmap_mode='r')
alt_masks = np.load('/home/s05814802723/projects/knee/ds/alt_f0/all_masks.npy', mmap_mode='r')

prim_ids = np.load('/home/s05814802723/projects/knee/ds/corpus192/all_ids.npy', allow_pickle=True)

# Build id2row for primary corpus
prim_id2row = {uid: i for i, uid in enumerate(prim_ids)}
alt_id2row = {uid: i for i, uid in enumerate(alt_ids)}

# Get fold 0 studies that have alternates
alt_slots = pd.read_csv('/home/s05814802723/projects/knee/ds/alt_f0/slots.csv')
fold0_with_alts = []
for uid in fold0_heldout:
    if uid in prim_id2row and uid in alt_id2row:
        alt_rows = alt_slots[alt_slots.StudyInstanceUID == uid]
        if alt_rows.filled.any():
            fold0_with_alts.append(uid)

print(f"Fold 0 studies with alts in both corpora: {len(fold0_with_alts)}")

labels = pd.read_parquet('/home/s05814802723/projects/knee/labels_sev_v1.parquet').set_index('StudyInstanceUID')
findings = [c for c in labels.columns if c != 'StudyInstanceUID']

# Use the actual StudyWindows dataset for both corpora
# Primary corpus
prim_ds = StudyWindows(
    root='/home/s05814802723/projects/knee/ds/corpus192',
    ids=fold0_with_alts,
    id2row=prim_id2row,
    labels={uid: labels.loc[uid].values.astype(np.float32) for uid in fold0_with_alts},
    res=256,
    k=380,  # k_eval from training
    train=False,
    aug=False,
    norm="none",
    slot_emb=True,
    slots=None,
    seq_head=False,
)

# Alt corpus - need to create a modified version that uses alt_vols
# For now, let's just test on a small subset using the primary corpus
# and manually swap in alt volumes for the specific series

print("Testing approach: we need to create a custom dataset that can use alt volumes")
print("This is complex. Let's use a simpler approach: run inference on the PC where")
print("the inference pipeline is already set up correctly.")

# Actually, let's just run the test on the PC where the model and data are properly set up
# Copy the alt_f0 corpus to PC and run there.

