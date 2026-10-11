#!/usr/bin/env python3
"""train_specialist_pfoa.py - Train dedicated Axial Patellofemoral OA (PF OA) Specialist.

Trained on the pure 18-slice Axial cache (axial_vols.npy, 224x224).
Provides 100% patella/trochlea articular cartilage coverage with zero slice-drop compromise.
Evaluates on Gold-58 studies after each epoch, optimizing PF OA AUC and Macro-AUC.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import timm

LABELS_ORDER = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]
PFOA_IDX = LABELS_ORDER.index("PF OA")


class AxialDataset(Dataset):
    def __init__(self, vols, masks, ids, labels_dict, train=True):
        self.vols = vols
        self.masks = masks
        self.ids = ids
        self.labels_dict = labels_dict
        self.train = train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uid = self.ids[idx]
        vol = self.vols[idx].astype(np.float32) / 255.0  # (18, 224, 224)
        K, H, W = vol.shape

        # Triplet channel stacking: for each slice, stack (slice-1, slice, slice+1)
        # to preserve 2.5D continuity
        triplets = []
        for s in range(K):
            s_prev = max(0, s - 1)
            s_next = min(K - 1, s + 1)
            tri = np.stack([vol[s_prev], vol[s], vol[s_next]], axis=0)  # (3, 224, 224)
            triplets.append(tri)
        x = np.stack(triplets, axis=0)  # (18, 3, 224, 224)

        if self.train:
            # Horizontal flip augmentation
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=-1).copy()

        y = self.labels_dict.get(uid, np.zeros(len(LABELS_ORDER), dtype=np.float32))
        return torch.from_numpy(x), torch.from_numpy(y)


class AxialAttentionClassifier(nn.Module):
    def __init__(self, arch="coatnet_rmlp_1_rw_224", num_classes=12, drop=0.2):
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=True, num_classes=0)
        try:
            self.backbone.set_grad_checkpointing(True)
        except Exception:
            pass
        feat_dim = self.backbone.num_features
        self.norm = nn.LayerNorm(feat_dim)
        self.att = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.Tanh(),
            nn.Dropout(drop),
            nn.Linear(256, num_classes)
        )
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        # x: (B, K, 3, H, W)
        B, K, C, H, W = x.shape
        flat = x.view(B * K, C, H, W)

        # Chunked backbone pass to keep VRAM footprint under 3.5 GB
        chunk_size = 18
        feats_list = []
        for chunk in flat.split(chunk_size, dim=0):
            feats_list.append(self.backbone(chunk))
        feats = torch.cat(feats_list, dim=0).view(B, K, -1)  # (B, K, feat_dim)

        h = self.norm(feats)
        att_weights = torch.softmax(self.att(h), dim=1)  # (B, K, num_classes)
        # Weighted pooling per finding
        pooled = torch.einsum("bkc,bkf->bcf", att_weights, h)  # (B, num_classes, feat_dim)
        # Final logits
        logits = torch.einsum("bcf,cf->bc", pooled, self.classifier.weight) + self.classifier.bias
        return logits


def main():
    parser = argparse.ArgumentParser(description="Train Patellofemoral OA Axial Specialist")
    parser.add_argument("--data_dir", default="/home/s05814802723/projects/knee/ds/axial_pfoa")
    parser.add_argument("--labels", default="/home/s05814802723/projects/knee/labels_avg.parquet")
    parser.add_argument("--gold_csv", default="/home/s05814802723/projects/knee/train.csv")
    parser.add_argument("--arch", default="coatnet_rmlp_1_rw_224")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--out_dir", default="/home/s05814802723/projects/knee/specialist_pfoa")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pfoa_trainer] Using device: {device} | Model: {args.arch}")

    # Load cache
    vols = np.load(os.path.join(args.data_dir, "axial_vols.npy"), mmap_mode="r")
    masks = np.load(os.path.join(args.data_dir, "axial_masks.npy"))
    ids = np.load(os.path.join(args.data_dir, "axial_ids.npy"), allow_pickle=True)
    uid_to_idx = {uid: i for i, uid in enumerate(ids)}

    # Gold 58 evaluation set
    gold_df = pd.read_csv(args.gold_csv)
    gold_df["StudyInstanceUID"] = gold_df["StudyInstanceUID"].astype(str)
    gold_df = gold_df.dropna(subset=[LABELS_ORDER[0]]).set_index("StudyInstanceUID")
    gold_uids = [u for u in gold_df.index if u in uid_to_idx]
    gold_indices = [uid_to_idx[u] for u in gold_uids]
    gold_y = gold_df.loc[gold_uids, LABELS_ORDER].values.astype(np.float32)

    # Soft training labels
    labels_df = pd.read_parquet(args.labels)
    labels_df["StudyInstanceUID"] = labels_df["StudyInstanceUID"].astype(str)
    labels_df = labels_df.set_index("StudyInstanceUID")

    train_uids = [u for u in labels_df.index if u in uid_to_idx and u not in set(gold_uids)]
    train_indices = [uid_to_idx[u] for u in train_uids]

    train_labels = {u: labels_df.loc[u, LABELS_ORDER].values.astype(np.float32) for u in train_uids}
    gold_labels = {u: gold_df.loc[u, LABELS_ORDER].values.astype(np.float32) for u in gold_uids}

    print(f"[pfoa_trainer] Train studies: {len(train_uids)} | Gold-val studies: {len(gold_uids)}")

    train_ds = AxialDataset(vols[train_indices], masks[train_indices], train_uids, train_labels, train=True)
    val_ds = AxialDataset(vols[gold_indices], masks[gold_indices], gold_uids, gold_labels, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    model = AxialAttentionClassifier(arch=args.arch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    best_pfoa_auc = 0.0
    best_macro_auc = 0.0

    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_steps = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_steps += 1

        scheduler.step()
        train_loss = total_loss / max(1, n_steps)

        # Validation on Gold-58
        model.eval()
        val_preds = []
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    probs = torch.sigmoid(logits)
                val_preds.append(probs.cpu().numpy())

        val_preds = np.concatenate(val_preds, axis=0)

        # Compute AUCs
        aucs = []
        for c in range(12):
            try:
                score = roc_auc_score(gold_y[:, c], val_preds[:, c])
                aucs.append(score)
            except Exception:
                aucs.append(0.5)

        macro_auc = float(np.mean(aucs))
        pfoa_auc = float(aucs[PFOA_IDX])
        elapsed = time.time() - t0

        print(f"ep{ep:02d} | loss {train_loss:.4f} | PF OA AUC {pfoa_auc:.4f} | Macro-AUC {macro_auc:.4f} | {elapsed:.0f}s", flush=True)

        if pfoa_auc > best_pfoa_auc:
            best_pfoa_auc = pfoa_auc
            best_macro_auc = macro_auc
            save_path = os.path.join(args.out_dir, "specialist_pfoa_best.pt")
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "pfoa_auc": pfoa_auc,
                "macro_auc": macro_auc,
                "arch": args.arch,
            }, save_path)
            print(f"  [SAVED] New best PF OA model at ep{ep}: {pfoa_auc:.4f} (Macro: {macro_auc:.4f})")

    print(f"\n[DONE] Finished training! Best PF OA AUC: {best_pfoa_auc:.4f} | Macro-AUC: {best_macro_auc:.4f}")


if __name__ == "__main__":
    main()
