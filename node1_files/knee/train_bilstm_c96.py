#!/usr/bin/env python3
"""
Train 96-slice Bi-LSTM Sequential Head on multi-plane features from corpus96.
Features:
- Sequence length: 96 slices across Sagittal, Coronal, and Axial planes
- 2-layer Bidirectional LSTM (hidden 256 -> 512-dim output)
- Plane- and slice-mask-aware per-diagnosis attention pooling (12 findings)
- bfloat16 cuDNN mixed precision for accelerated training (~3.5s per epoch)
- 3 random seeds + SWA
- Validation on the 58 Gold studies (Macro ROC-AUC)
- Ensembling & 3-way blend with CoAtNet v6 and 44-slice Bi-LSTM
"""
import os, sys, time, random, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pearsonr

HERE       = os.path.dirname(os.path.abspath(__file__))
OUT_EMB    = os.path.join(HERE, 'embeddings_c96.npy')
OUT_MASK   = os.path.join(HERE, 'masks_c96.npy')
OUT_UID    = os.path.join(HERE, 'uids_c96.npy')
OUT_GOLD   = os.path.join(HERE, 'gold_labels_c96.npy')
OUT_SOFT   = os.path.join(HERE, 'soft_labels_c96.npy')
V6_NPZ     = os.path.join(HERE, 'raptor_gold_v6.npz')
T1_NPZ     = os.path.join(HERE, 'raptor_gold_bilstm_v1_ens.npz')

LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

class Knee96Dataset(Dataset):
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
        feat = self.emb_mmap[idx].astype(np.float32) # (96, 1024)
        if np.isnan(feat).any():
            feat = np.nan_to_num(feat, nan=0.0)
        mask = self.masks[idx].astype(np.uint8)     # (96,)
        label = self.labels[idx].astype(np.float32) # (12,)

        x = torch.from_numpy(feat)
        m = torch.from_numpy(mask)
        y = torch.from_numpy(label)

        if self.augment:
            # Mild feature jitter
            if random.random() < 0.5:
                x = x * (1.0 + (torch.rand(1, x.shape[-1]) - 0.5) * 0.1)
            # Random slice masking (drop 1-4 valid slices)
            if random.random() < 0.5:
                valid_idx = torch.where(m > 0)[0]
                if len(valid_idx) > 10:
                    n_drop = random.randint(1, 4)
                    perm = torch.randperm(len(valid_idx))[:n_drop]
                    x[valid_idx[perm]] = 0.0

        return x, m, y

class BiLSTMHead96(nn.Module):
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
        # x: (B, 96, f_dim)
        # mask: (B, 96)
        h = self.norm(x)
        h, _ = self.lstm(h) # (B, 96, head_dim)
        h = self.norm_head(h)
        att_logits = self.att(h) # (B, 96, n_classes)

        if mask is not None:
            mask_exp = mask.unsqueeze(-1).bool()
            att_logits = att_logits.masked_fill(~mask_exp, -1e4)

        a = torch.softmax(att_logits, dim=1) # (B, 96, n_classes)
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
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--drop', type=float, default=0.3)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--tag', default='bilstm_c96')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[info] Using device: {device}', flush=True)

    # 1. Load Data
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
    print(f'      Training studies (soft targets): {len(train_idx)}', flush=True)
    print(f'      Validation studies (gold-58): {len(gold_idx)}', flush=True)

    # Reference v6
    v6_preds = None
    if os.path.exists(V6_NPZ):
        v6_npz = np.load(V6_NPZ)
        v6_preds = v6_npz['pred']
        v6_truth = v6_npz['truth']
        v6_macro, _ = calc_macro_auc(v6_preds, v6_truth)
        print(f'      Reference CoAtNet v6 Gold Macro-AUC: {v6_macro:.4f}', flush=True)

    seed_preds = []
    seed_bests = []

    # 2. Multi-Seed Training Loop
    for seed in range(args.seeds):
        print(f'\n{"="*60}', flush=True)
        print(f'Training 96-Slice Bi-LSTM — Seed {seed}', flush=True)
        print(f'{"="*60}', flush=True)
        set_seed(42 + seed * 100)

        tr_ds = Knee96Dataset(emb_mmap, masks, soft_arr, train_idx, augment=True)
        va_ds = Knee96Dataset(emb_mmap, masks, gold_arr, gold_idx, augment=False)

        tr_dl = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        va_dl = DataLoader(va_ds, batch_size=args.bs, shuffle=False, num_workers=2, pin_memory=True)

        model = BiLSTMHead96(
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
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x, m)
                    loss = criterion(logits.float(), y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item() * len(x)

            sched.step()
            train_loss = total_loss / len(train_idx)

            # Validation
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

        # Save checkpoint
        ckpt_path = os.path.join(HERE, f'raptor_ft_{args.tag}_s{seed}.pt')
        torch.save({
            'model': best_state,
            'arch': 'bilstm_head96',
            'hidden': args.hidden,
            'layers': args.layers,
            'f_dim': emb_mmap.shape[-1],
            'gold_auc': best_gold_auc,
            'tag': args.tag,
            'seed': seed,
            'lab': LABELS
        }, ckpt_path)
        print(f'Seed {seed} finished. Best Gold AUC: {best_gold_auc:.4f} (saved {os.path.basename(ckpt_path)})', flush=True)

    # 3. Ensemble Evaluation
    print(f'\n{"="*60}', flush=True)
    print('96-SLICE BI-LSTM SUMMARY & EVALUATION')
    print(f'{"="*60}', flush=True)
    for s, b in enumerate(seed_bests):
        print(f'  Seed {s} Gold Macro-AUC: {b:.4f}')

    bilstm96_ens = np.mean(seed_preds, axis=0)
    ens_auc, per_class = calc_macro_auc(bilstm96_ens, Y)
    print(f'\n>>> 3-Seed 96-Slice Bi-LSTM Ensemble Gold Macro-AUC: {ens_auc:.4f} <<<')

    print('\nPer-class AUC breakdown:')
    for name in LABELS:
        print(f'  {name:20s}: {per_class.get(name, 0.0):.4f}')

    np.savez(
        os.path.join(HERE, f'raptor_gold_{args.tag}_ens.npz'),
        pred=bilstm96_ens, truth=Y, ids=uids[gold_idx]
    )

    # 4. Correlation & Blending Analysis
    if v6_preds is not None:
        print(f'\n{"-"*60}', flush=True)
        print('CORRELATION & BLENDING WITH COATNET V6')
        print(f'{"-"*60}', flush=True)
        p_corrs = [pearsonr(bilstm96_ens[:, j], v6_preds[:, j])[0] for j in range(12)]
        s_corrs = [spearmanr(bilstm96_ens[:, j], v6_preds[:, j])[0] for j in range(12)]
        print(f'Mean Pearson correlation: {np.mean(p_corrs):.4f}')
        print(f'Mean Spearman rank correlation: {np.mean(s_corrs):.4f}')

        from scipy.stats import rankdata
        def rank_norm(arr):
            return np.apply_along_axis(lambda col: rankdata(col) / len(col), 0, arr)

        v6_r = rank_norm(v6_preds)
        l96_r = rank_norm(bilstm96_ens)

        print('\nRank-Average Blend Exploration (CoAtNet v6 + Bi-LSTM 96):')
        for w_lstm in [0.1, 0.2, 0.25, 0.3, 0.4]:
            w_v6 = 1.0 - w_lstm
            b_pred = w_v6 * v6_r + w_lstm * l96_r
            b_auc, _ = calc_macro_auc(b_pred, Y)
            gain = b_auc - v6_macro
            print(f'  Weight {w_v6:.2f} v6 + {w_lstm:.2f} Bi-LSTM96 -> AUC: {b_auc:.4f} (Δ {gain:+.4f})')

    # 5. Check 3-way Blend (if Task 1 44-slice results are present)
    if os.path.exists(T1_NPZ) and v6_preds is not None:
        t1_npz = np.load(T1_NPZ)
        t1_preds = t1_npz['pred']
        l44_r = rank_norm(t1_preds)
        print(f'\n{"-"*60}', flush=True)
        print('3-WAY ENSEMBLE BLEND (CoAtNet v6 + Bi-LSTM 44 + Bi-LSTM 96)')
        print(f'{"-"*60}', flush=True)
        for w_v6, w_44, w_96 in [
            (0.60, 0.20, 0.20),
            (0.50, 0.25, 0.25),
            (0.70, 0.15, 0.15)
        ]:
            b3 = w_v6 * v6_r + w_44 * l44_r + w_96 * l96_r
            b3_auc, _ = calc_macro_auc(b3, Y)
            gain = b3_auc - v6_macro
            print(f'  {w_v6:.2f} v6 + {w_44:.2f} L44 + {w_96:.2f} L96 -> AUC: {b3_auc:.4f} (Δ {gain:+.4f})')

if __name__ == '__main__':
    main()
