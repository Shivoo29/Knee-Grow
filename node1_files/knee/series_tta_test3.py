#!/usr/bin/env python3
"""Series TTA test: compare first-pick vs mean vs max over alternate series on fold 0 OOF."""
import sys
sys.path.insert(0, '/home/s05814802723/projects/knee')

import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Load the model from our training code
from train_knee_ours import RaptorClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load model
ckpt_path = '/home/s05814802723/projects/knee/raptor_ft_cnxt_r2_f0_swa.pt'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
print(f"Checkpoint type: {type(ckpt)}")
if isinstance(ckpt, dict):
    print(f"Checkpoint keys: {list(ckpt.keys())}")

model = RaptorClassifier(backbone='convnext_tiny.in12k_ft_in1k', pretrained=False)
if isinstance(ckpt, dict) and 'model' in ckpt:
    model.load_state_dict(ckpt['model'])
else:
    model.load_state_dict(ckpt)
model.to(device)
model.eval()
print("Model loaded")

# Load data
from build_corpus96_stream import build_study_table, SLOTS
import json

# Load series info
series_df = pd.read_csv('/home/s05814802723/projects/knee/cpu_ideas/train_series.csv')
train_df = pd.read_csv('/home/s05814802723/projects/knee/train.csv')

# Load folds
with open('/home/s05814802723/projects/knee/folds_rand4.json') as f:
    folds = json.load(f)
fold0_heldout = folds[0]['val']
print(f"Fold 0 held-out: {len(fold0_heldout)} studies")

# Load alt corpus
alt_ids = np.load('/home/s05814802723/projects/knee/ds/alt_f0/all_ids.npy')
alt_vols = np.load('/home/s05814802723/projects/knee/ds/alt_f0/all_vols.npy', mmap_mode='r')
alt_masks = np.load('/home/s05814802723/projects/knee/ds/alt_f0/all_masks.npy', mmap_mode='r')
alt_slots = pd.read_csv('/home/s05814802723/projects/knee/ds/alt_f0/slots.csv')

# Load primary corpus
prim_ids = np.load('/home/s05814802723/projects/knee/ds/corpus192/all_ids.npy')
prim_vols = np.load('/home/s05814802723/projects/knee/ds/corpus192/all_vols.npy', mmap_mode='r')
prim_masks = np.load('/home/s05814802723/projects/knee/ds/corpus192/all_masks.npy', mmap_mode='r')

# Map study -> index in corpus
prim_idx = {uid: i for i, uid in enumerate(prim_ids)}
alt_idx = {uid: i for i, uid in enumerate(alt_ids)}

# Get fold 0 studies that have alternates
fold0_with_alts = []
for uid in fold0_heldout:
    if uid in prim_idx and uid in alt_idx:
        alt_rows = alt_slots[alt_slots.StudyInstanceUID == uid]
        if alt_rows.filled.any():
            fold0_with_alts.append(uid)

print(f"Fold 0 studies with alts in both corpora: {len(fold0_with_alts)}")

# Load labels
labels = pd.read_parquet('/home/s05814802723/projects/knee/labels_sev_v1.parquet').set_index('StudyInstanceUID')
findings = [c for c in labels.columns if c != 'StudyInstanceUID']

# Inference function
def predict_study(vols, masks, idx):
    v = torch.from_numpy(vols[idx:idx+1]).float().to(device) / 255.0
    m = torch.from_numpy(masks[idx:idx+1]).bool().to(device)
    with torch.no_grad():
        logits = model(v, m)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return probs

# Test on all studies with alts
test_uids = fold0_with_alts
print(f"\nTesting on {len(test_uids)} studies...")

results = {'first': [], 'mean': [], 'max': []}

for i, uid in enumerate(test_uids):
    p_idx = prim_idx[uid]
    a_idx = alt_idx[uid]
    
    p_pred = predict_study(prim_vols, prim_masks, p_idx)
    a_pred = predict_study(alt_vols, alt_masks, a_idx)
    
    results['first'].append(p_pred)
    results['mean'].append((p_pred + a_pred) / 2)
    results['max'].append(np.maximum(p_pred, a_pred))
    
    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(test_uids)}")

# Evaluate
from sklearn.metrics import roc_auc_score

for name, preds in results.items():
    preds_arr = np.array(preds)
    y_true = labels.loc[test_uids, findings].values > 0.5
    aucs = []
    for j, f in enumerate(findings):
        if len(np.unique(y_true[:, j])) > 1:
            aucs.append(roc_auc_score(y_true[:, j], preds_arr[:, j]))
    macro = np.mean(aucs)
    print(f"\n{name:6s}: Macro AUC = {macro:.4f}")
    for j, f in enumerate(findings):
        if len(np.unique(y_true[:, j])) > 1:
            print(f"  {f:18s}: {aucs[j]:.4f}")

# Also compute per-finding delta
print("\n=== DELTA (mean - first) ===")
first_arr = np.array(results['first'])
mean_arr = np.array(results['mean'])
max_arr = np.array(results['max'])
y_true = labels.loc[test_uids, findings].values > 0.5
for j, f in enumerate(findings):
    if len(np.unique(y_true[:, j])) > 1:
        auc_f = roc_auc_score(y_true[:, j], first_arr[:, j])
        auc_m = roc_auc_score(y_true[:, j], mean_arr[:, j])
        auc_x = roc_auc_score(y_true[:, j], max_arr[:, j])
        print(f"  {f:18s}: first={auc_f:.4f} mean={auc_m:.4f} ({auc_m-auc_f:+.4f}) max={auc_x:.4f} ({auc_x-auc_f:+.4f})")

