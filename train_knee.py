# %% [code]
# %% [code]
#!/usr/bin/env python3
"""Knee MRI: training the twelve-finding model

This is the training half of the model behind the public 0.924 inference notebook. It reads the
precomputed slice stacks, trains a CoAtNet backbone with a per-finding attention pooling head, and
writes a checkpoint you can drop straight into that notebook.

WHAT YOU NEED ATTACHED

  1. The preprocessed corpus, both parts:
       kaggle.com/datasets/dreaddevelopment/knee-raptor-corpus          (3,200 studies)
       kaggle.com/datasets/dreaddevelopment/knee-raptor-corpus-ext      (1,207 studies)
     Every study is already reduced to a fixed 44 x 336 x 336 uint8 stack, so no DICOM reading
     happens here. The two parts concatenate in order.

  2. The competition data, for train.csv.

  3. Training labels, as a parquet with a StudyInstanceUID column and the twelve finding columns.
     THIS IS NOT PROVIDED, and it is the one thing you have to bring yourself. See below.

THE LABEL PROBLEM, WHICH IS THE REAL PROBLEM

The competition gives you 4,407 studies and structured labels for only 58 of them. Every other
study carries a free-text radiology report and nothing else. So before any of this trains, you need
to turn 4,349 reports into twelve numbers each.

The approach behind the published weights was to read each report with a language model and emit
twelve probabilities rather than twelve yes or no answers: a report that hedges, saying a tear is
suspected, becomes something near 0.8 rather than a 1. Soft targets are far more forgiving than
forcing every hedged sentence into a hard label, and the loss here expects them. The 58 studies
that come with real labels are held out and used only for validation, never trained on.

Point --labels at your own parquet built that way. The format is one row per study: a
StudyInstanceUID column plus the twelve finding columns, values between 0 and 1.

WHAT THE MODEL DOES

Three neighbouring slices are stacked into the three channels of one image, so the network sees a
little of what lies above and below the middle slice: most of the benefit of a 3D model at the cost
of a 2D one. Each of these three-slice windows goes through the backbone, and the windows are then
pooled by an attention layer that has separate weights for each of the twelve findings. That last
part matters more than anything else here. A cruciate tear may be visible on two slices while
osteoarthritis spreads across many, and one shared pooling weight forces those to compete; giving
each finding its own attention lets each draw on the slices that actually show it.

Training samples k windows per study at random and evaluates on k_eval windows spread evenly, so
each epoch sees a different view of the same study. Nothing else is augmented.

At the end it keeps the best epoch by validation macro-AUC, and also writes a checkpoint that
averages the weights of the best three epochs. Weight averaging costs nothing at inference, unlike
averaging predictions from three models, and it usually gives a small gain.

TYPICAL RUN

  python train_knee.py --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --epochs 16
      --bs 8 --k 12 --k_eval 24 --grad_ckpt --tag mymodel --labels /kaggle/input/YOURS/labels.parquet

About three hours on one 4090 for 16 epochs at 384. --grad_ckpt trades a little speed for a lot of
memory and is what makes bs 8 fit on a 24 GB card. Use --smoke for a fast wiring check.

MEMORY, MEASURED

The CoAtNet at 384 with --bs 8 and --grad_ckpt sits at about 23.7 GB, so it needs a 24 GB card.
On 16 GB use --bs 2 or 4. Separately from the GPU, this script keeps the best checkpoint plus the
top --topk epochs' weights in system RAM so it can average them at the end; that is about 87 MB a
copy for the small ViT and 293 MB for the CoAtNet, and the list fills over the first few epochs.
If you run out of host memory a few epochs in, set --topk 1 and lower --workers.

The checkpoint it writes is a dict with keys model, arch, res and lab, which is exactly what the
inference notebook expects.
"""
import os, sys, time, json, math, random, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import timm

# --- ROI localizer (anatomical joint crop). Optional so the no-ROI path is untouched. ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from roi_localize import square_box as _roi_square_box, compartments as _roi_compartments
except Exception:
    _roi_square_box = _roi_compartments = None

HERE = os.path.dirname(os.path.abspath(__file__))
RSNA = os.path.dirname(HERE)

# ---------------------------------------------------------------------------
# Input discovery. On Kaggle the corpus arrives as two read-only datasets and the
# competition data as a third, so nothing lives beside this script. Everything in
# this block is discovery only - the training code further down is unchanged.
# ---------------------------------------------------------------------------
def _find(*names, root=os.environ.get("RAPTOR_INPUT", "/kaggle/input")):
    """First path under root whose basename matches one of names."""
    for d, _, fs in os.walk(root):
        for n in names:
            if n in fs:
                return os.path.join(d, n)
    return None


class _TwoPartVols:
    """Presents the two published corpus parts as one array of shape (4407, 44, 336, 336).

    Both parts stay memory-mapped and are never concatenated on disk: copying 22 GB would be
    pointless when every read is a single study. Row order is part 1 then part 2, matching the
    order the id files concatenate in. That ordering is the contract between volumes, masks and
    ids, so do not sort any of them independently.
    """
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n_a = a.shape[0]
        self.shape = (a.shape[0] + b.shape[0],) + tuple(a.shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, row):
        return self.a[row] if row < self.n_a else self.b[row - self.n_a]


_CORPUS = None   # RAPTOR_PRELOAD=1: the whole corpus in RAM, loaded once in the parent so forked loader workers share it


def _open_corpus():
    """Return (vols, masks), from a local single-file corpus or the two public parts.
    RAPTOR_PRELOAD=1 reads the arrays fully into RAM (22 GB) instead of memory-mapping them: on a Kaggle TPU VM
    (377 GB RAM) the input mount serves random slice reads too slowly to keep one TPU core busy."""
    global _CORPUS
    if _CORPUS is not None:
        return _CORPUS
    mm = None if os.environ.get("RAPTOR_PRELOAD") == "1" else "r"
    local_v = os.path.join(HERE, "all_vols.npy")
    if os.path.exists(local_v):
        out = (np.load(local_v, mmap_mode=mm), np.load(os.path.join(HERE, "all_masks.npy")))
    else:
        av, bv = _find("all_vols.npy"), _find("extra_vols.npy")
        am, bm = _find("all_masks.npy"), _find("extra_masks.npy")
        if not all((av, bv, am, bm)):
            raise SystemExit("Could not find the corpus. Attach both parts: "
                             "dreaddevelopment/knee-raptor-corpus and "
                             "dreaddevelopment/knee-raptor-corpus-ext")
        t0 = time.time()
        vols = _TwoPartVols(np.load(av, mmap_mode=mm), np.load(bv, mmap_mode=mm))
        masks = np.concatenate([np.load(am), np.load(bm)], axis=0)
        if mm is None:
            print(f"[corpus] preloaded into RAM in {time.time()-t0:.0f}s", flush=True)
        out = (vols, masks)
    if mm is None:
        _CORPUS = out
    return out


def _open_ids():
    local = os.path.join(HERE, "all_ids.npy")
    if os.path.exists(local):
        return np.load(local, allow_pickle=True).astype(str)
    a, b = _find("all_ids.npy"), _find("extra_ids.npy")
    if not (a and b):
        raise SystemExit("Could not find all_ids.npy / extra_ids.npy - attach both corpus parts.")
    return np.concatenate([np.load(a, allow_pickle=True).astype(str),
                           np.load(b, allow_pickle=True).astype(str)])
LAB = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA",
       "Effusion","Synovitis","Baker's","Contusion","Fracture"]


_NO_LABELS = """
No training labels found, so there is nothing to train against.

The competition labels only 58 of the 4,407 studies. The other 4,349 carry a free-text
radiology report instead, so before this can train you have to turn those reports into
twelve probabilities per study and pass the result with --labels.

Expected format: a parquet with a StudyInstanceUID column plus the columns
  {cols}
with values between 0 and 1. Soft values work better than hard 0/1 here: the loss is built
for them, and hedged reports are common.

Everything else in this notebook is ready to run once that file exists.
"""


# ------------------------------- data ----------------------------------------
class StudyWindows(Dataset):
    """Per-study bag of 2.5D windows sampled from all_vols.npy (memmap).
    Each window = 3 physically-consecutive slices -> RGB, resized to `res`, in [0,1]
    (matches the SSL input pipeline: ToTensor, no ImageNet norm)."""
    def __init__(self, root, ids, id2row, labels, res, k, train, aug=True, norm="none",
                 roi=False, roi_mode="tight", roi_pad=0.06, roi_overlap=0.12,
                 roi_sbox=None, roi_cen=None):
        self.root = root
        self.ids = ids
        self.id2row = id2row
        self.labels = labels              # dict uid -> np.float32[12]
        self.res, self.k, self.train, self.aug = res, k, train, aug
        self.norm = norm                  # "none"=[0,1] (Raptor SSL); "imagenet"=DINOv2 stats
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.vols = None; self.masks = None
        # --- ROI anatomical joint-crop config ---
        self.roi = bool(roi)
        self.roi_mode, self.roi_pad, self.roi_overlap = roi_mode, roi_pad, roi_overlap
        self.roi_sbox = roi_sbox          # (N,D,4) int16 per-slice tissue bbox, or None
        self.roi_cen = roi_cen            # (N,D,2) int16 per-slice joint centroid, or None
        if self.roi and (_roi_square_box is None or roi_sbox is None):
            raise RuntimeError("roi=True but roi_localize or roi_boxes not available")

    def __len__(self): return len(self.ids)

    def _ensure(self):
        if self.vols is None:
            self.vols, self.masks = _open_corpus()   # (N,D,H,W) view, (N,D) u8

    def _centers(self, valid, count):
