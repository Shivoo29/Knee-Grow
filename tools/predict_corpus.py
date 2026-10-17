#!/usr/bin/env python3
"""Teacher predictions on a corpus folder: mean sigmoid over checkpoints, each run with the layout
saved in it (arch, res, norm, zoom, slots), the way train_knee.py validates (no flip TTA).

    python tools/predict_corpus.py --corpus ~/projects/knee/fastmri/corpus \
        --ckpts pc192/raptor_ft_cnxt_f0_swa.pt,pc192/raptor_ft_cnxt_f1_swa.pt --out fastmri_teacher.parquet

--corpus is a tools/fastmri_corpus.py folder (vols/*.u8z) or an all_vols.npy corpus. Each
checkpoint's predictions are cached as <out>.<checkpoint>.npz, so a rerun skips finished ones.
Output: StudyInstanceUID, the 12 findings (mean probability over checkpoints), n_ckpts.

--calibrate <the teachers' training labels parquet>: train_knee.py trains with BCE pos_weight
w = clip((1 - prev) / prev, 1, 10), which inflates its sigmoid outputs to p = w q / (w q + 1 - q).
This maps each output back to q = p / (p + w (1 - p)), the scale of the training labels, so the
teacher's labels can sit next to competition labels in one training set.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import train_knee as tk  # noqa: E402


def predict(ckpt, vols, masks, ids, k_eval=0, bs=2, workers=4, win_chunk=0, dev="cpu"):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    sizes = [int(v) for v in str(st.get("slot_sizes") or "26,22,18,12,18").split(",")]
    assert vols.shape[1] == sum(sizes), f"{ckpt}: corpus depth {vols.shape[1]} != slot_sizes {sizes}"
    tk.SLOT_BOUNDS = tk.bounds_from_sizes(sizes)     # read by slot_of / pick_centers, inherited by workers
    tk._CORPUS = (vols, masks)                       # StudyWindows opens the corpus through this cache
    zoom = tk.zoom_px(st.get("zoom_mm") or 0.0)
    k = k_eval or (vols.shape[1] - 2) * (2 if zoom else 1)
    slot_emb = bool(st.get("slot_emb"))
    ds = tk.StudyWindows(tk.HERE, list(ids), {u: i for i, u in enumerate(ids)},
                         {u: np.zeros(12, np.float32) for u in ids}, st["res"], k, train=False,
                         norm=st.get("norm", "none"), clean_windows=bool(st.get("clean_windows")),
                         slot_emb=slot_emb, seq_head=bool(st.get("seq_head")), zoom_px=zoom,
                         slots=tuple(int(v) for v in str(st["slots"]).split(",")) if st.get("slots") else None)
    bb = tk.build_backbone(st["arch"], pretrained=False)
    model = tk.RaptorClassifier(bb, F_dim=bb.num_features, slot_emb=slot_emb, seq_head=bool(st.get("seq_head")))
    model.load_state_dict(st["model"])
    model.win_chunk = win_chunk
    model.eval().to(dev)
    amp = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    out, t0 = [], time.time()
    with torch.no_grad():
        for i, batch in enumerate(DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, collate_fn=tk.collate)):
            slots = batch[2].to(dev) if slot_emb else None
            with torch.autocast("cuda", dtype=amp, enabled=dev == "cuda"):
                out.append(torch.sigmoid(model(batch[0].to(dev), slots=slots).float()).cpu().numpy())
            if (i + 1) % 200 == 0:
                print(f"  {os.path.basename(ckpt)}: {(i + 1) * bs}/{len(ids)} studies, "
                      f"{(time.time() - t0) / ((i + 1) * bs):.2f} s/study", flush=True)
    return np.concatenate(out)


def pos_weight(labels_path):
    """Per-finding BCE pos_weight exactly as train_knee.py derives it from its training labels."""
    prev = np.clip(pd.read_parquet(labels_path)[tk.LAB].values.astype(np.float64).mean(0), 0.03, 0.7)
    return np.clip((1 - prev) / prev, 1, 10)


def uninflate(p, w):
    return p / (p + w * (1 - p))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--ckpts", required=True, help="comma list of checkpoint paths")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k_eval", type=int, default=0, help="tiles per study; 0 = every window (x2 with zoom)")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--win_chunk", type=int, default=0, help="run the backbone over chunks of N tiles (small GPUs)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--calibrate", default=None, help="the teachers' training labels parquet: undo the pos_weight inflation")
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vols, masks, ids = tk._extra_part(a.corpus)
    if a.limit:
        ids = ids[:a.limit]
    preds = []
    for ck in [c for c in a.ckpts.split(",") if c]:
        cache = f"{a.out}.{os.path.splitext(os.path.basename(ck))[0]}.npz"
        if os.path.exists(cache) and list(np.load(cache, allow_pickle=True)["ids"]) == list(ids):
            preds.append(np.load(cache)["pred"])
            print(f"[cache] {os.path.basename(cache)}", flush=True)
            continue
        t = time.time()
        p = predict(ck, vols, masks, ids, a.k_eval, a.bs, a.workers, a.win_chunk, dev)
        np.savez(cache, pred=p, ids=np.array(ids))
        preds.append(p)
        print(f"[done] {os.path.basename(ck)}: {len(ids)} studies in {(time.time() - t) / 60:.1f} min", flush=True)
    if a.calibrate:
        w = pos_weight(a.calibrate)
        preds = [uninflate(p, w) for p in preds]
        print("[calibrate] pos_weight " + " ".join(f"{c[:6]} {v:.1f}" for c, v in zip(tk.LAB, w)), flush=True)
    df = pd.DataFrame(np.mean(preds, axis=0), columns=tk.LAB)
    df.insert(0, "StudyInstanceUID", list(ids))
    df["n_ckpts"] = len(preds)
    df.to_parquet(a.out, index=False)
    print(f"[out] {a.out}: {len(df)} studies x {len(preds)} checkpoints | mean prob "
          + " ".join(f"{c[:6]} {df[c].mean():.2f}" for c in tk.LAB), flush=True)


if __name__ == "__main__":
    main()
