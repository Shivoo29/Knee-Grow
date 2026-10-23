#!/usr/bin/env python3
"""
Train a Bi-LSTM sequential head with per-finding attention pooling
over pre-extracted slice embeddings from CoAtNet v6.

Key features:
- Bidirectional LSTM with hidden size 256 (512 total output dim)
- Attention pooling with slice-level mask support (ignores padded slices)
- Trains on soft targets from LLM reports
- Validates strictly on the 58 gold studies (calculates official competition metric: Macro ROC-AUC)
- Runs 3 random seeds
- Measures Pearson & Spearman rank correlation against existing v6 predictions
- Blends predictions with v6 to evaluate ensemble improvement
"""
import os, sys, time, random, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_EMB   = os.path.join(HERE, 'embeddings.npy')
OUT_MASK  = os.path.join(HERE, 'masks.npy')
OUT_UID   = os.path.join(HERE, 'uids.npy')
OUT_GOLD  = os.path.join(HERE, 'gold_labels.npy')
OUT_SOFT  = os.path.join(HERE, 'soft_labels.npy')
V6_GOLD_NPZ = os.path.join(HERE, 'raptor_gold_v6.npz')

LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# ──────────────────────── Dataset ────────────────────────
class KneeSliceDataset(Dataset):
    def __init__(self, emb_mmap, masks, labels, indices, augment=True):
        self.emb_mmap = emb_mmap
        self.masks = masks
        self.labels = labels
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        feat = self.emb_mmap[idx].astype(np.float32) # (44, 1024)
        if np.isnan(feat).any():
            feat = np.nan_to_num(feat, nan=0.0)
        mask = self.masks[idx].astype(np.uint8)     # (44,)
        label = self.labels[idx].astype(np.float32) # (12,)

        x = torch.from_numpy(feat)
        m = torch.from_numpy(mask)
        y = torch.from_numpy(label)

        if self.augment:
            # Mild feature jitter and random slice dropout
            if random.random() < 0.5:
                x = x * (1.0 + (torch.rand(1, x.shape[-1]) - 0.5) * 0.1)
            # Random slice masking (drop 1-3 valid slices)
            if random.random() < 0.5:
                valid_idx = torch.where(m > 0)[0]
                if len(valid_idx) > 6:
                    n_drop = random.randint(1, 3)
                    perm = torch.randperm(len(valid_idx))[:n_drop]
                    x[valid_idx[perm]] = 0.0

        return x, m, y

# ──────────────────────── Model ────────────────────────
class BiLSTMHeadModel(nn.Module):
    def __init__(self, f_dim=1024, lstm_hidden=256, n_layers=2, n_classes=12, drop=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(f_dim)
        self.lstm = nn.LSTM(
            input_size=f_dim,
            hidden_size=lstm_hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=drop if n_layers > 1 else 0.0
        )
        head_dim = lstm_hidden * 2 # 512
        self.norm_head = nn.LayerNorm(head_dim)
        self.att = nn.Sequential(
            nn.Linear(head_dim, 256),
            nn.Tanh(),
            nn.Dropout(drop),
            nn.Linear(256, n_classes)
        )
        self.clsW = nn.Parameter(torch.zeros(n_classes, head_dim))
        self.clsb = nn.Parameter(torch.zeros(n_classes))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n_classes = n_classes

    def forward(self, x, mask=None):
        # x: (B, K, f_dim)
        # mask: (B, K)
        h = self.norm(x)
        h, _ = self.lstm(h) # (B, K, head_dim)
        h = self.norm_head(h)
        att_logits = self.att(h) # (B, K, n_classes)

        if mask is not None:
            mask_exp = mask.unsqueeze(-1).bool()
            att_logits = att_logits.masked_fill(~mask_exp, -1e9)

        a = torch.softmax(att_logits, dim=1) # (B, K, n_classes)
        pooled = torch.einsum('bkn,bkf->bnf', a, h) # (B, n_classes, head_dim)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

def calc_macro_auc(preds, truths):
    aucs = {}
    for j, name in enumerate(LABELS):
        t = truths[:, j]
        p = preds[:, j]
        valid = ~np.isnan(t) & np.isfinite(t) & np.isfinite(p)
        if len(np.unique(t[valid])) > 1:
            try:
                aucs[name] = roc_auc_score(t[valid], p[valid])
            except Exception:
                pass
    macro = np.mean(list(aucs.values())) if aucs else 0.0
    return macro, aucs

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--drop', type=float, default=0.3)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--tag', default='bilstm_v1')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[info] Using device: {device}', flush=True)

    # Load arrays
    print('[1/4] Loading pre-extracted data...', flush=True)
    emb_mmap = np.load(OUT_EMB, mmap_mode='r')
    masks = np.load(OUT_MASK)
    uids = np.load(OUT_UID)
    gold_arr = np.load(OUT_GOLD)
    soft_arr = np.load(OUT_SOFT)

    N = len(uids)
    gold_idx = np.where(~np.isnan(gold_arr[:, 0]))[0]
    train_idx = np.where(np.isnan(gold_arr[:, 0]) & ~np.isnan(soft_arr[:, 0]))[0]

    print(f'      Total studies: {N}', flush=True)
    print(f'      Training studies (soft labels): {len(train_idx)}', flush=True)
    print(f'      Validation studies (gold-58): {len(gold_idx)}', flush=True)

    # Check v6 benchmark
    v6_preds = None
    if os.path.exists(V6_GOLD_NPZ):
        v6_npz = np.load(V6_GOLD_NPZ)
        v6_preds = v6_npz['pred']
        v6_truth = v6_npz['truth']
        v6_macro, _ = calc_macro_auc(v6_preds, v6_truth)
        print(f'      Reference CoAtNet v6 Gold Macro-AUC: {v6_macro:.4f}', flush=True)

    seed_preds = []
    seed_bests = []

    # 3-seed training loop
    for seed in range(args.seeds):
        print(f'\n{"="*60}', flush=True)
        print(f'Training Bi-LSTM Head — Seed {seed}', flush=True)
        print(f'{"="*60}', flush=True)
        set_seed(42 + seed * 100)

        tr_ds = KneeSliceDataset(emb_mmap, masks, soft_arr, train_idx, augment=True)
        va_ds = KneeSliceDataset(emb_mmap, masks, gold_arr, gold_idx, augment=False)

        tr_dl = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        va_dl = DataLoader(va_ds, batch_size=args.bs, shuffle=False, num_workers=2, pin_memory=True)

        model = BiLSTMHeadModel(
            f_dim=emb_mmap.shape[-1],
            lstm_hidden=args.hidden,
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
                logits = model(x, m)
                loss = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item() * len(x)

            sched.step()
            train_loss = total_loss / len(train_idx)

            # Validation on gold-58
            model.eval()
            preds_list = []
            truths_list = []
            with torch.no_grad():
                for x, m, y in va_dl:
                    x, m = x.to(device), m.to(device)
                    probs = torch.sigmoid(model(x, m)).cpu().numpy()
                    preds_list.append(probs)
                    truths_list.append(y.numpy())

            P = np.concatenate(preds_list, axis=0)
            Y = np.concatenate(truths_list, axis=0)
            gold_auc, _ = calc_macro_auc(P, Y)
            dt = time.time() - t0

            is_best = gold_auc > best_gold_auc
            if is_best:
                best_gold_auc = gold_auc
                best_pred = P.copy()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                marker = ' ★ BEST'
            else:
                marker = ''

            print(f'ep {ep:02d}/{args.epochs} | loss: {train_loss:.4f} | gold macro-AUC: {gold_auc:.4f} ({dt:.1f}s){marker}', flush=True)

        seed_preds.append(best_pred)
        seed_bests.append(best_gold_auc)
        # Save seed checkpoint
        ckpt_path = os.path.join(HERE, f'raptor_ft_{args.tag}_s{seed}.pt')
        torch.save({
            'model': best_state,
            'arch': 'bilstm_head',
            'hidden': args.hidden,
            'layers': args.layers,
            'f_dim': emb_mmap.shape[-1],
            'gold_auc': best_gold_auc,
            'tag': args.tag,
            'seed': seed,
            'lab': LABELS
        }, ckpt_path)
        print(f'Seed {seed} finished. Best Gold AUC: {best_gold_auc:.4f} (saved {os.path.basename(ckpt_path)})', flush=True)

    # ──────────────────────── Evaluation & Blend ────────────────────────
    print(f'\n{"="*60}', flush=True)
    print('SUMMARY & ENSEMBLE EVALUATION')
    print(f'{"="*60}', flush=True)
    for s, b in enumerate(seed_bests):
        print(f'  Seed {s} Gold Macro-AUC: {b:.4f}')

    # Ensemble of the 3 BiLSTM seeds
    bilstm_ens_preds = np.mean(seed_preds, axis=0)
    ens_auc, per_class_auc = calc_macro_auc(bilstm_ens_preds, Y)
    print(f'\n>>> 3-Seed Bi-LSTM Ensemble Gold Macro-AUC: {ens_auc:.4f} <<<')

    print('\nPer-class AUC:')
    for name in LABELS:
        print(f'  {name:20s}: {per_class_auc.get(name, 0.0):.4f}')

    # Save ensemble predictions
    np.savez(
        os.path.join(HERE, f'raptor_gold_{args.tag}_ens.npz'),
        pred=bilstm_ens_preds, truth=Y, ids=uids[gold_idx]
    )

    # Blend with CoAtNet v6 if available
    if v6_preds is not None:
        print(f'\n{"-"*60}', flush=True)
        print('CORRELATION & BLENDING WITH COATNET V6')
        print(f'{"-"*60}', flush=True)
        # Prediction correlation
        p_corrs = [pearsonr(bilstm_ens_preds[:, j], v6_preds[:, j])[0] for j in range(12)]
        s_corrs = [spearmanr(bilstm_ens_preds[:, j], v6_preds[:, j])[0] for j in range(12)]
        mean_p = np.mean(p_corrs)
        mean_s = np.mean(s_corrs)
        print(f'Mean Pearson correlation with v6:  {mean_p:.4f}')
        print(f'Mean Spearman rank correlation:     {mean_s:.4f}')
        print('(Lower correlation indicates architectural diversity that helps the ensemble)')

        # Rank Average Blend
        from scipy.stats import rankdata
        def rank_norm(arr):
            # Rank normalise each column to [0, 1]
            return np.apply_along_axis(lambda col: rankdata(col) / len(col), 0, arr)

        v6_rank = rank_norm(v6_preds)
        lstm_rank = rank_norm(bilstm_ens_preds)

        print('\nBlending weights exploration (Rank Average):')
        for w_lstm in [0.1, 0.2, 0.25, 0.3, 0.4]:
            w_v6 = 1.0 - w_lstm
            blend_rank = w_v6 * v6_rank + w_lstm * lstm_rank
            b_auc, _ = calc_macro_auc(blend_rank, Y)
            gain = b_auc - v6_macro
            print(f'  Weight {w_v6:.2f} v6 + {w_lstm:.2f} Bi-LSTM -> Blend AUC: {b_auc:.4f} (Δ {gain:+.4f})')

        # Also linear average
        print('\nBlending weights exploration (Linear Average):')
        for w_lstm in [0.1, 0.2, 0.25, 0.3, 0.4]:
            w_v6 = 1.0 - w_lstm
            blend_lin = w_v6 * v6_preds + w_lstm * bilstm_ens_preds
            b_auc, _ = calc_macro_auc(blend_lin, Y)
            gain = b_auc - v6_macro
            print(f'  Weight {w_v6:.2f} v6 + {w_lstm:.2f} Bi-LSTM -> Blend AUC: {b_auc:.4f} (Δ {gain:+.4f})')

if __name__ == '__main__':
    main()
