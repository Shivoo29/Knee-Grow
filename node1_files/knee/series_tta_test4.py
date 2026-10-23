#!/usr/bin/env python3
"""Series TTA test: compare first-pick vs mean vs max over alternate series on fold 0 OOF."""
import sys
sys.path.insert(0, '/home/s05814802723/projects/knee')

import torch
import numpy as np
import pandas as pd

# Recreate model with exact checkpoint parameters
from train_knee_ours import RaptorClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load checkpoint first to get params
ckpt_path = '/home/s05814802723/projects/knee/raptor_ft_cnxt_r2_f0_swa.pt'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
print(f"Checkpoint keys: {list(ckpt.keys())}")

# Extract model params from checkpoint
model_state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

# Build model with same architecture
model = RaptorClassifier(
    backbone='convnext_tiny.in12k_ft_in1k',
    F_dim=384,
    n=12,
    drop=0.2,
    slot_emb=True,
    seq_head=False  # round-2 models don't use seq_head
)
model.load_state_dict(model_state)
model.to(device)
model.eval()
print("Model loaded")

# Load data
from build_corpus96_stream import build_study_table, SLOTS
import json

series_df = pd.read_csv('/home/s05814802723/projects/knee/cpu_ideas/train_series.csv')

with open('/home/s05814802723/projects/knee/folds_rand4.json') as f:
    folds = json.load(f)
fold0_heldout = folds[0]['val']
print(f"Fold 0 held-out: {len(fold0_heldout)} studies")

# Load corpora
alt_ids = np.load('/home/s05814802723/projects/knee/ds/alt_f0/all_ids.npy')
alt_vols = np.load('/home/s05814802723/projects/knee/ds/alt_f0/all_vols.npy', mmap_mode='r')
alt_masks = np.load('/home/s05814802723/projects/knee/ds/alt_f0/all_masks.npy', mmap_mode='r')
alt_slots = pd.read_csv('/home/s05814802723/projects/knee/ds/alt_f0/slots.csv')

prim_ids = np.load('/home/s05814802723/projects/knee/ds/corpus192/all_ids.npy')
prim_vols = np.load('/home/s05814802723/projects/knee/ds/corpus192/all_vols.npy', mmap_mode='r')
prim_masks = np.load('/home/s05814802723/projects/knee/ds/corpus192/all_masks.npy', mmap_mode='r')

prim_idx = {uid: i for i, uid in enumerate(prim_ids)}
alt_idx = {uid: i for i, uid in enumerate(alt_ids)}

fold0_with_alts = []
for uid in fold0_heldout:
    if uid in prim_idx and uid in alt_idx:
        alt_rows = alt_slots[alt_slots.StudyInstanceUID == uid]
        if alt_rows.filled.any():
            fold0_with_alts.append(uid)

print(f"Fold 0 studies with alts in both corpora: {len(fold0_with_alts)}")

labels = pd.read_parquet('/home/s05814802723/projects/knee/labels_sev_v1.parquet').set_index('StudyInstanceUID')
findings = [c for c in labels.columns if c != 'StudyInstanceUID']

# Slot mapping function (copied from train_knee)
def slot_of(plane, fat_sup):
    PLANES = ["Sagittal", "Coronal", "Axial"]
    return PLANES.index(plane) + 3 * (1 - int(fat_sup))

# Inference
def predict_study(vols, masks, idx, study_uid, series_df, corpus_slots=None):
    # Get slots for this study
    study_series = series_df[series_df.StudyInstanceUID == study_uid]
    slot_indices = []
    for _, row in study_series.iterrows():
        s = slot_of(row['Anatomical_Plane'], row['Fat_Suppression'])
        slot_indices.append(s)
    
    # Map to window positions (k windows per study)
    # In our model k=48 for eval
    # For simplicity, just use the volume as-is (model expects (B, K, H, W, C) or similar)
    v = torch.from_numpy(vols[idx:idx+1]).float().to(device) / 255.0
    m = torch.from_numpy(masks[idx:idx+1]).bool().to(device)
    
    # Create slot tensor (same for all windows of this study)
    K = v.shape[1]  # number of windows
    slots = torch.tensor(slot_indices[:K]).unsqueeze(0).to(device) if len(slot_indices) >= K else torch.zeros(1, K, dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits = model(v, slots=slots)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return probs

# Actually the corpus already has the right structure. Let's just use the volumes directly.
# The volumes are already windowed and arranged.
# Need to check the shape
print(f"Primary vols shape: {prim_vols.shape}")
print(f"Primary masks shape: {prim_masks.shape}")
print(f"Alt vols shape: {alt_vols.shape}")

# Test on subset first
test_uids = fold0_with_alts[:100]
print(f"\nTesting on {len(test_uids)} studies...")

results = {'first': [], 'mean': [], 'max': []}

for i, uid in enumerate(test_uids):
    p_idx = prim_idx[uid]
    a_idx = alt_idx[uid]
    
    p_pred = predict_study(prim_vols, prim_masks, p_idx, uid, series_df)
    a_pred = predict_study(alt_vols, alt_masks, a_idx, uid, series_df)
    
    results['first'].append(p_pred)
    results['mean'].append((p_pred + a_pred) / 2)
    results['max'].append(np.maximum(p_pred, a_pred))
    
    if (i + 1) % 20 == 0:
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

# Per-finding delta
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

