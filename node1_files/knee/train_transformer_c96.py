#!/usr/bin/env python3
"""
Multi-Head Attention / Transformer Sequence Pooler on pre-extracted 96-slice corpus.
Features:
- Pre-extracted embeddings (4407, 96, 1024) float16
- Per-Finding Learned Query Tokens (12 specialized queries attending to 96 slices)
- 2-layer Cross-Attention Transformer Decoder
- Native bfloat16 mixed-precision training
- Evaluates on Gold-58, logs macro-AUC, saves multi-seed checkpoints.
"""
import os, sys, time, random, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_EMB   = os.path.join(HERE, 'embeddings_c96.npy')
OUT_MASK  = os.path.join(HERE, 'masks_c96.npy')
OUT_UID   = os.path.join(HERE, 'uids_c96.npy')
OUT_GOLD  = os.path.join(HERE, 'gold_labels_c96.npy')
OUT_SOFT  = os.path.join(HERE, 'soft_labels_c96.npy')
V6_NPZ    = os.path.join(HERE, 'raptor_gold_v6_swa.npz')

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Knee96Dataset(Dataset):
    def __init__(self, emb_mmap, masks, labels, indices, augment=False):
        self.emb_mmap = emb_mmap
        self.masks = masks
        self.labels = labels
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        feat = self.emb_mmap[idx].astype(np.float32)
        if np.isnan(feat).any():
            feat = np.nan_to_num(feat, nan=0.0)
        mask = self.masks[idx].astype(np.uint8)
        label = self.labels[idx].astype(np.float32)

        x = torch.from_numpy(feat)
        m = torch.from_numpy(mask)
        y = torch.from_numpy(label)

        if self.augment:
            if random.random() < 0.5:
                x = x * (1.0 + (torch.rand(1, x.shape[-1]) - 0.5) * 0.1)
            if random.random() < 0.5:
                valid_idx = torch.where(m > 0)[0]
                if len(valid_idx) > 10:
                    n_drop = random.randint(1, 4)
                    perm = torch.randperm(len(valid_idx))[:n_drop]
                    x[valid_idx[perm]] = 0.0

        return x, m, y

class PerFindingAttentionPooler(nn.Module):
    """12 learned query tokens cross-attending to the 96 anatomical slices."""
    def __init__(self, f_dim=1024, n_heads=8, n_layers=2, n_classes=12, drop=0.2):
        super().__init__()
        self.f_dim = f_dim
        self.n_classes = n_classes
        
        self.in_proj = nn.Linear(f_dim, f_dim)
        self.norm = nn.LayerNorm(f_dim)
        
        self.pos_emb = nn.Parameter(torch.zeros(1, 96, f_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        
        self.finding_queries = nn.Parameter(torch.zeros(1, n_classes, f_dim))
        nn.init.trunc_normal_(self.finding_queries, std=0.02)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=f_dim, nhead=n_heads, dim_feedforward=f_dim * 2,
            dropout=drop, activation='gelu', batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(f_dim, 256),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(256, 1)
            ) for _ in range(n_classes)
        ])

    def forward(self, x, mask=None):
        B, K, F = x.shape
        h = self.norm(self.in_proj(x) + self.pos_emb[:, :K])
        q = self.finding_queries.expand(B, -1, -1)
        
        pad_mask = None
        if mask is not None:
            pad_mask = (mask < 0.5)
            
        out = self.decoder(tgt=q, memory=h, memory_key_padding_mask=pad_mask)
        
        logits = []
        for j in range(self.n_classes):
            logits.append(self.heads[j](out[:, j]))
        return torch.cat(logits, dim=-1)

def calc_macro_auc(preds, targets):
    aucs = {}
    vals = []
    for j, name in enumerate(LABELS):
        y_t = targets[:, j]
        y_p = preds[:, j]
        valid = ~np.isnan(y_t)
        if len(np.unique(y_t[valid])) > 1:
            score = roc_auc_score(y_t[valid], y_p[valid])
            aucs[name] = score
            vals.append(score)
        else:
            aucs[name] = np.nan
    return float(np.nanmean(vals)), aucs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--drop', type=float, default=0.2)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--tag', default='xfmr_c96')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[info] Using device: {device}', flush=True)

    print('[1/4] Loading pre-extracted 96-slice corpus...', flush=True)
    emb_mmap = np.load(OUT_EMB, mmap_mode='r')
    masks = np.load(OUT_MASK)
    uids = np.load(OUT_UID, allow_pickle=True)
    gold_arr = np.load(OUT_GOLD)
    soft_arr = np.load(OUT_SOFT)

    N = len(uids)
    gold_idx = np.where(~np.isnan(gold_arr[:, 0]))[0]
    train_idx = np.where(np.isnan(gold_arr[:, 0]) & ~np.isnan(soft_arr[:, 0]))[0]

    print(f'      Total studies: {N} | Feature shape: {emb_mmap.shape}', flush=True)
    print(f'      Training studies: {len(train_idx)} | Gold validation studies: {len(gold_idx)}', flush=True)

    seed_preds = []
    seed_bests = []

    for seed in range(args.seeds):
        print(f'\n{"="*60}', flush=True)
        print(f'Training 96-Slice Transformer Pooler — Seed {seed}', flush=True)
        print(f'{"="*60}', flush=True)
        set_seed(42 + seed * 100)

        tr_ds = Knee96Dataset(emb_mmap, masks, soft_arr, train_idx, augment=True)
        va_ds = Knee96Dataset(emb_mmap, masks, gold_arr, gold_idx, augment=False)

        tr_dl = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        va_dl = DataLoader(va_ds, batch_size=args.bs, shuffle=False, num_workers=2, pin_memory=True)

        model = PerFindingAttentionPooler(
            f_dim=emb_mmap.shape[-1],
            n_heads=args.heads,
            n_layers=args.layers,
            n_classes=12,
            drop=args.drop
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
        criterion = nn.BCEWithLogitsLoss()

        best_gold_auc = 0.0
        best_pred = None
        best_state = None

        for ep in range(args.epochs):
            model.train()
            total_loss = 0.0
            t0 = time.time()

            for x, m, y in tr_dl:
                x, m, y = x.to(device, non_blocking=True), m.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x, m)
                    loss = criterion(logits.float(), y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item() * len(x)

            sched.step()
            train_loss = total_loss / len(train_idx)

            model.eval()
            preds_list = []
            truths_list = []
            with torch.no_grad():
                for x, m, y in va_dl:
                    x, m = x.to(device), m.to(device)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        probs = torch.sigmoid(model(x, m).float()).cpu().numpy()
                    preds_list.append(probs)
                    truths_list.append(y.numpy())

            P = np.concatenate(preds_list, axis=0)
            Y = np.concatenate(truths_list, axis=0)
            gold_auc, _ = calc_macro_auc(P, Y)
            dt = time.time() - t0

            if gold_auc > best_gold_auc:
                best_gold_auc = gold_auc
                best_pred = P.copy()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                marker = ' ★ BEST'
            else:
                marker = ''

            print(f'ep {ep:02d}/{args.epochs} | loss: {train_loss:.4f} | gold macro-AUC: {gold_auc:.4f} ({dt:.1f}s){marker}', flush=True)

        seed_preds.append(best_pred)
        seed_bests.append(best_gold_auc)

        ckpt_path = os.path.join(HERE, f'raptor_ft_{args.tag}_s{seed}.pt')
        torch.save({
            'model': best_state,
            'arch': 'per_finding_attention_pooler',
            'f_dim': emb_mmap.shape[-1],
            'gold_auc': best_gold_auc,
            'tag': args.tag,
            'seed': seed,
            'lab': LABELS
        }, ckpt_path)
        print(f'Seed {seed} finished. Best Gold AUC: {best_gold_auc:.4f} (saved {os.path.basename(ckpt_path)})', flush=True)

    print(f'\n{"="*60}', flush=True)
    print('TRANSFORMER ATTENTION POOLER SUMMARY')
    print(f'{"="*60}', flush=True)
    for s, b in enumerate(seed_bests):
        print(f'  Seed {s} Gold Macro-AUC: {b:.4f}')

    xfmr_ens = np.mean(seed_preds, axis=0)
    ens_auc, per_class = calc_macro_auc(xfmr_ens, Y)
    print(f'\n>>> 3-Seed Transformer Attention Ensemble Gold Macro-AUC: {ens_auc:.4f} <<<')
    for name in LABELS:
        print(f'  {name:20s}: {per_class.get(name, 0.0):.4f}')

    np.savez(
        os.path.join(HERE, f'raptor_gold_{args.tag}_ens.npz'),
        pred=xfmr_ens, truth=Y, ids=uids[gold_idx]
    )

if __name__ == '__main__':
    main()
