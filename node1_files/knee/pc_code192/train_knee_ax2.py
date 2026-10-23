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
import os, sys, time, json, math, random, argparse, bisect, zlib
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


class _ZCorpus:
    """Compressed corpus folder: vols/<id>.<format> plus all_ids.npy, all_masks.npy and corpus.json.
    format u8z = zlib of the uint8 volume (tools/fastmri_corpus.py); u8d = zlib of its row-delta along
    the image width (tools/export_corpus.py, ~1.9x smaller than raw). Inflates one study per index."""
    def __init__(self, d):
        self.d = d
        self.ids = np.load(os.path.join(d, "all_ids.npy"), allow_pickle=True).astype(str)
        self.masks = np.load(os.path.join(d, "all_masks.npy"))
        meta = json.load(open(os.path.join(d, "corpus.json")))
        self.fmt = meta.get("format", "u8z")
        self.shape = (len(self.ids),) + tuple(meta["shape"])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, row):
        with open(os.path.join(self.d, "vols", f"{self.ids[row]}.{self.fmt}"), "rb") as f:
            a = np.frombuffer(zlib.decompress(f.read()), np.uint8).reshape(self.shape[1:])
        return np.cumsum(a, axis=-1, dtype=np.uint8) if self.fmt == "u8d" else a


class _DepthCat:
    """Two corpora with the same studies, joined along the slice axis (base slots, then add-on slots)."""
    def __init__(self, a, b):
        assert a.shape[0] == b.shape[0] and tuple(a.shape[2:]) == tuple(b.shape[2:]), (a.shape, b.shape)
        self.a, self.b = a, b
        self.shape = (a.shape[0], a.shape[1] + b.shape[1]) + tuple(a.shape[2:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, row):
        return np.concatenate([self.a[row], self.b[row]], axis=0)


class _MultiVols:
    """Corpus parts stacked by row, in the same order as their ids are concatenated."""
    def __init__(self, parts):
        assert len({tuple(p.shape[1:]) for p in parts}) == 1, [p.shape for p in parts]
        self.parts = parts
        self.starts = np.cumsum([0] + [p.shape[0] for p in parts]).tolist()
        self.shape = (self.starts[-1],) + tuple(parts[0].shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, row):
        i = bisect.bisect_right(self.starts, row) - 1
        return self.parts[i][row - self.starts[i]]


def _extra_dirs():
    """RAPTOR_EXTRA_CORPUS: comma list of extra corpus folders (tools/fastmri_corpus.py output or all_vols.npy)."""
    return [d for d in os.environ.get("RAPTOR_EXTRA_CORPUS", "").split(",") if d]


def _extra_part(d):
    """(vols, masks, ids) of one corpus folder; the volumes stay on disk."""
    if os.path.exists(os.path.join(d, "all_vols.npy")):
        return (np.load(os.path.join(d, "all_vols.npy"), mmap_mode="r"), np.load(os.path.join(d, "all_masks.npy")),
                np.load(os.path.join(d, "all_ids.npy"), allow_pickle=True).astype(str))
    z = _ZCorpus(d)
    return z, z.masks, z.ids


def _with_extras(vols, masks):
    parts = [_extra_part(d) for d in _extra_dirs()]
    if not parts:
        return vols, masks
    return _MultiVols([vols] + [p[0] for p in parts]), np.concatenate([masks] + [p[1] for p in parts])


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
    elif os.path.exists(os.path.join(HERE, "corpus.json")):   # compressed corpus folder beside the script
        z = _ZCorpus(HERE)
        out = (z, z.masks)
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
    out = _with_extras(*out)                          # RAPTOR_EXTRA_CORPUS parts stay on disk even with preload
    ax2 = os.environ.get("RAPTOR_AXIAL2")             # dir with an add-on slot (all_vols/all_masks/all_ids, same study order)
    if ax2:
        assert (np.load(os.path.join(ax2, "all_ids.npy"), allow_pickle=True).astype(str) == np.asarray(_open_ids()).astype(str)).all(), "axial2 study order differs"
        out = (_DepthCat(out[0], np.load(os.path.join(ax2, "all_vols.npy"), mmap_mode="r")),
               np.concatenate([out[1], np.load(os.path.join(ax2, "all_masks.npy"))], axis=1))
    if mm is None:
        _CORPUS = out
    return out


def _open_ids():
    local = os.path.join(HERE, "all_ids.npy")
    if os.path.exists(local):
        ids = np.load(local, allow_pickle=True).astype(str)
    else:
        a, b = _find("all_ids.npy"), _find("extra_ids.npy")
        if not (a and b):
            raise SystemExit("Could not find all_ids.npy / extra_ids.npy - attach both corpus parts.")
        ids = np.concatenate([np.load(a, allow_pickle=True).astype(str),
                              np.load(b, allow_pickle=True).astype(str)])
    extra = [np.load(os.path.join(d, "all_ids.npy"), allow_pickle=True).astype(str) for d in _extra_dirs()]
    return np.concatenate([ids] + extra) if extra else ids


def add_extra(train_ids, labels, lw_all, ids, comp_ids, labels_path, weight, limit=0):
    """Append extra-corpus studies that have a row in labels_path to the training set (never to the
    folds, OOF or gold). Their per-label loss weight is `weight`; competition rows keep theirs (1 if
    no --label_weights). Returns (train_ids, lw_all)."""
    ex = pd.read_parquet(labels_path)
    ex = ex.assign(StudyInstanceUID=ex["StudyInstanceUID"].astype(str)).set_index("StudyInstanceUID")
    extra = [u for u in ids if u not in comp_ids and u in ex.index]
    if limit:
        extra = extra[:limit]
    if not extra:
        raise SystemExit(f"--extra_labels {labels_path}: no study in RAPTOR_EXTRA_CORPUS has a label row")
    lw_all = dict(lw_all) if lw_all is not None else {u: np.ones(12, np.float32) for u in train_ids}
    vals = ex.loc[extra, LAB].values.astype(np.float32)
    for u, v in zip(extra, vals):
        labels[u] = v
        lw_all[u] = np.full(12, weight, np.float32)
    return list(train_ids) + extra, lw_all
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
def bounds_from_sizes(sizes):
    """Cumulative 5-slot bounds from per-slot slice counts.

    E.g. [26, 22, 18, 12, 18] -> [(0, 26), (26, 48), (48, 66), (66, 78), (78, 96)].
    Re-derived at startup from --slot_sizes; any 5-slot layout works.
    """
    bounds, start = [], 0
    for k in sizes:
        k = int(k)
        bounds.append((start, start + k))
        start += k
    return bounds


SLOT_BOUNDS = bounds_from_sizes([26, 22, 18, 12, 18])


def slot_of(idx: int) -> int:
    """Return slot index (0..4) for a slice index."""
    for s, (start, end) in enumerate(SLOT_BOUNDS):
        if start <= idx < end:
            return s
    if idx < 0:
        return 0
    return len(SLOT_BOUNDS) - 1


def window_pos(c: int, mask: np.ndarray, side_sign: float, mirrored: bool) -> float:
    """Anatomical position of window centre c in [-1, 1] along its slot's slice axis (--pos_emb).

    The corpus stores every series in true 3D order (sorted by IPP . slice normal; a 600-series DICOM sample had
    one direction per plane): sagittal patient-left -> patient-right, coronal front -> back, axial foot -> head.
    Sagittal becomes lateral (-1) -> medial (+1) via the knee side (+1 left, -1 right, 0 unknown -> 0).
    A training mirror (mirror_volume) reverses sagittal order, so it flips that sign; coronal/axial flips are
    in-plane and leave the through-plane position alone.
    """
    s = slot_of(c)
    start, _ = SLOT_BOUNDS[s]
    n = int(np.sum(mask[start:SLOT_BOUNDS[s][1]] > 0))
    t = min(max(2.0 * (c - start) / max(n - 1, 1) - 1.0, -1.0), 1.0)
    if s in (0, 1):
        t *= side_sign * (-1.0 if mirrored else 1.0)
    return float(t)


def pack_slot_pos(slot_ids, pos):
    """--pos_emb rides on the existing slot tensor: integer part = slot id, fraction = (pos + 1) / 2 * 0.999.
    ponytail: one float tensor instead of a new batch element threaded through every train/eval loop."""
    return torch.tensor([s + (p + 1.0) * 0.4995 for s, p in zip(slot_ids, pos)], dtype=torch.float32)


FIELD_MM = 140.0   # every corpus slice is a 140 mm square field at 384 px (build_corpus96_stream mm_crop_and_resize)


def zoom_px(zoom_mm, img=384):
    """Side in corpus pixels of a centred close-up of zoom_mm millimetres (0 = off)."""
    return int(round(img * float(zoom_mm) / FIELD_MM)) if zoom_mm else 0


def pick_centers(valid, count, train=False, clean_windows=False, mask=None, max_d=None, cap=False, slots=None):
    """Pick 3-slice window centers.

    If clean_windows is True, a center c is allowed only if c-1, c, c+1 are all real
    (mask > 0) AND in the same slot (slot_of(c-1) == slot_of(c) == slot_of(c+1)).
    Otherwise, picks from [lo+1, hi-1] matching legacy behavior.
    max_d defaults to the full layout depth (sum of slot sizes).
    """
    if max_d is None:
        max_d = sum(end - start for start, end in SLOT_BOUNDS)
    lo, hi = int(valid.min()), int(valid.max())
    if clean_windows and mask is not None:
        cs = [
            c for c in range(1, max_d - 1)
            if mask[c - 1] > 0 and mask[c] > 0 and mask[c + 1] > 0
            and slot_of(c - 1) == slot_of(c) == slot_of(c + 1)
        ]
    else:
        cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if slots is not None:   # specialist: only windows centred in these slots
        cs = [c for c in cs if slot_of(c) in slots]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, max_d - 2))]
    if train:
        reps = count // len(cs) + 1
        pool = cs * reps
        random.shuffle(pool)
        return pool[:count]
    # eval: evenly spaced deterministic
    if cap:
        count = min(count, len(cs))   # fleet inference: never repeat a center
    idx = np.linspace(0, len(cs) - 1, count).round().astype(int)
    return [cs[i] for i in idx]


# label order: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, ... (--swap_p swaps the two pairs)
SWAP_ML = [0, 1, 3, 2, 5, 4, 6, 7, 8, 9, 10, 11]


def mirror_volume(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Anatomical mirror of knee volume (D, H, W) over the current SLOT_BOUNDS.

    Sagittal slots (0, 1): reverse order of real slices (mask > 0); padded slices stay at end.
    Coronal/Axial slots (2, 3, 4): horizontal flip along image width (axis=-1).
    Mirroring twice is the identity operation.
    """
    out = vol.copy()
    for s, (start, end) in enumerate(SLOT_BOUNDS):
        if s in (0, 1):
            slot_m = mask[start:end]
            n_real = int(np.sum(slot_m > 0))
            if n_real > 1:
                out[start : start + n_real] = out[start : start + n_real][::-1]
        else:
            out[start:end] = np.flip(out[start:end], axis=-1)
    return out


def check_fallback_guard(fallback_count: int, total_studies: int, max_rate: float = 0.03):
    """Enforces strict fallback limit (< 3% of studies).

    Raises RuntimeError if fallbacks exceed 3% of studies.
    """
    if total_studies <= 0:
        return
    rate = fallback_count / float(total_studies)
    if rate > max_rate:
        raise RuntimeError(
            f"FATAL GUARD: fallback count {fallback_count}/{total_studies} "
            f"({rate:.2%}) exceeded allowed limit of {max_rate:.1%}!"
        )


class StudyWindows(Dataset):
    """Per-study bag of 2.5D windows sampled from all_vols.npy (memmap).
    Each window = 3 physically-consecutive slices -> RGB, resized to `res`, in [0,1]
    (matches the SSL input pipeline: ToTensor, no ImageNet norm)."""
    def __init__(self, root, ids, id2row, labels, res, k, train, aug=True, norm="none",
                 roi=False, roi_mode="tight", roi_pad=0.06, roi_overlap=0.12,
                 roi_sbox=None, roi_cen=None,
                 clean_windows=False, mirror_p=0.0, contrast_aug=False, slot_emb=False, slots=None,
                 seq_head=False, label_w=None, zoom_px=0, pos_side=None, canon_left=None, swap_p=0.0):
        self.canon_left = canon_left or set()   # --canon_side: these (left) knees are always mirrored to right-knee layout
        self.swap_p = float(swap_p)             # --swap_p: mirror + swap medial/lateral labels (train only)
        self.pos_side = pos_side          # dict uid -> +1 left / -1 right knee (--pos_emb), or None
        self.zoom_px = int(zoom_px)       # >0: each window also yields a centred close-up of this many corpus px
        if self.zoom_px and roi:
            raise ValueError("zoom_mm and roi are separate crop modes; use one")
        self.root = root
        self.ids = ids
        self.id2row = id2row
        self.labels = labels              # dict uid -> np.float32[12]
        self.res, self.k, self.train, self.aug = res, k, train, aug
        self.norm = norm                  # "none"=[0,1] (Raptor SSL); "imagenet"=DINOv2 stats
        self.clean_windows = bool(clean_windows)
        self.mirror_p = float(mirror_p)
        self.contrast_aug = bool(contrast_aug)
        self.slot_emb = bool(slot_emb)
        self.slots = slots
        self.seq_head = bool(seq_head)
        self.label_w = label_w            # dict uid -> np.float32[12] per-element weights, or None
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.vols = None; self.masks = None
        # --- ROI anatomical joint-crop config ---
        self.roi = bool(roi)
        self.roi_mode, self.roi_pad, self.roi_overlap = roi_mode, roi_pad, roi_overlap
        self.roi_sbox = roi_sbox          # (N,D,4) int16 per-slice tissue bbox, or None
        self.roi_cen = roi_cen            # (N,D,2) int16 per-slice joint centroid, or None
        if self.seq_head and self.roi and self.roi_mode == "compartment":
            raise ValueError("seq_head is incompatible with roi_mode compartment: 2 crops per center break ordering")
        if self.roi and (_roi_square_box is None or roi_sbox is None):
            raise RuntimeError("roi=True but roi_localize or roi_boxes not available")

    def __len__(self): return len(self.ids)

    def _ensure(self):
        if self.vols is None:
            self.vols, self.masks = _open_corpus()   # (N,D,H,W) view, (N,D) u8

    def _centers(self, valid, count, mask=None):
        return pick_centers(valid, count, train=self.train,
                            clean_windows=self.clean_windows,
                            mask=mask, max_d=self._D, slots=self.slots)

    def _resize(self, tri):
        """tri: (3,h,w) float32 [0,1] -> (3,res,res) float32."""
        t = torch.from_numpy(np.ascontiguousarray(tri))
        if t.shape[-1] != self.res or t.shape[-2] != self.res:
            t = F.interpolate(t[None], size=(self.res, self.res), mode="bilinear",
                              align_corners=False)[0]
        return t.numpy()

    def __getitem__(self, i):
        self._ensure()
        uid = self.ids[i]; row = self.id2row[uid]
        self._D = self.vols.shape[1]
        m = self.masks[row]
        valid = np.where(m > 0)[0]
        if len(valid) < 3: valid = np.arange(min(3, self._D))
        # compartment mode emits 2 crops/center -> sample ceil(k/2) centers to keep #windows==k
        compart = self.roi and self.roi_mode == "compartment"
        n_centers = (self.k + 1) // 2 if (compart or self.zoom_px) else self.k
        cs = self._centers(valid, n_centers, mask=m)
        if self.seq_head:
            cs = sorted(cs)               # GRU sees slices in anatomical order (train and eval)
        vol = self.vols[row]  # (D,H,W) u8  (single study read)
        mirrored = self.train and self.mirror_p > 0.0 and random.random() < self.mirror_p
        if mirrored:
            vol = mirror_volume(vol, m)
        if uid in self.canon_left:
            vol = mirror_volume(vol, m)
        swapped = self.train and self.swap_p > 0.0 and random.random() < self.swap_p
        if swapped:                       # canonical knee mirrored: the medial side now sits where lateral belongs
            vol = mirror_volume(vol, m)
        tiles = []            # list of (3,res,res) float32
        slot_ids = []
        win_centers = []
        for c in cs:
            c = max(1, min(c, self._D - 2))
            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0  # (3,H,W)
            H, W = tri.shape[-2], tri.shape[-1]
            if not self.roi:
                tiles.append(self._resize(tri))
                slot_ids.append(slot_of(c))
                win_centers.append(c)
                if self.zoom_px:           # second scale: centred close-up of the same window
                    oy, ox = (H - self.zoom_px) // 2, (W - self.zoom_px) // 2
                    tiles.append(self._resize(tri[:, oy:oy + self.zoom_px, ox:ox + self.zoom_px]))
                    slot_ids.append(slot_of(c))
                    win_centers.append(c)
                continue
            # one joint box from the CENTER slice, applied to all 3 slices (keeps RGB registered)
            sq_mode = "tight" if compart else self.roi_mode
            sq = _roi_square_box(tuple(int(v) for v in self.roi_sbox[row, c]), W=W, H=H,
                                 pad=self.roi_pad, mode=sq_mode,
                                 centroid=tuple(float(v) for v in self.roi_cen[row, c]))
            if compart:
                left, right = _roi_compartments(sq, overlap=self.roi_overlap)
                for box in (left, right):
                    x0, y0, x1, y1 = box
                    tiles.append(self._resize(tri[:, y0:y1, x0:x1]))
                    slot_ids.append(slot_of(c))
                    win_centers.append(c)
            else:
                x0, y0, x1, y1 = sq
                tiles.append(self._resize(tri[:, y0:y1, x0:x1]))
                slot_ids.append(slot_of(c))
                win_centers.append(c)
        if len(tiles) > self.k:
            tiles = tiles[:self.k]
            slot_ids = slot_ids[:self.k]
            win_centers = win_centers[:self.k]
        wins = np.stack(tiles, 0)          # (K,3,res,res)
        x = torch.from_numpy(wins)
        if self.train and self.aug:
            if self.contrast_aug:
                gamma = random.uniform(0.8, 1.25)
                x = x.clamp(0, 1).pow(gamma)
            # light medical-safe aug: NO flips (laterality is signal); mild intensity jitter
            g = 1.0 + (random.random() - 0.5) * 0.20
            x = (x * g).clamp(0, 1)
        if self.norm == "imagenet":        # each backbone at its correct input distribution
            x = (x - self._mean) / self._std
        y = torch.from_numpy(self.labels[uid])
        if swapped:
            y = y[SWAP_ML]
        out = [x, y]
        if self.slot_emb and self.pos_side is not None:
            side = float(self.pos_side.get(uid, 0.0))
            out.append(pack_slot_pos(slot_ids, [window_pos(c, m, side, mirrored) for c in win_centers]))
        elif self.slot_emb:
            out.append(torch.tensor(slot_ids, dtype=torch.long))
        if self.seq_head:
            out.append(torch.tensor(win_centers, dtype=torch.long))
        if self.label_w is not None:
            w = np.ascontiguousarray(self.label_w[uid], dtype=np.float32)
            out.append(torch.from_numpy(w[SWAP_ML] if swapped else w))
        return tuple(out)


def _winit(_): torch.set_num_threads(1)


def collate(batch):
    # Stacks every returned element: (xs, ys[, slots][, centers][, weights]).
    # slots (index 2) exists only when the dataset was built with slot_emb;
    # centers/weights are trailing and read by flag-aware indices in main().
    return tuple(torch.stack([b[k] for b in batch]) for k in range(len(batch[0])))


def weighted_bce(bce_none, logits, y, w):
    """BCE*weights averaged over elements with weight > 0 (keeps pos_weight).

    With all weights 1 this equals plain mean-BCE. Zero-weight elements are
    ignored (label down-weighting for noisy-report supervision).
    """
    l = bce_none(logits, y) * w
    n = int((w > 0).sum())
    return l.sum() / max(n, 1)


# ------------------------------- model ---------------------------------------
def build_backbone(arch="vit_small_patch16_224", pretrained=False):
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if is_vit:
        kw.update(global_pool="token", dynamic_img_size=True)
    else:
        kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)


RADIMAGENET_RESNET = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}


def load_raptor(bb, ckpt_path):
    if ckpt_path in ("timm", "pretrained"):
        return "timm-pretrained"
    if ckpt_path in ("", "none", "None"):
        print("[raptor] RANDOM-INIT control (no SSL weights)", flush=True)
        return "random-init"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    st = ck["student"] if "student" in ck else (ck["model"] if "model" in ck else ck)   # SSL student, or a released RaptorClassifier
    bbst = {k[len("backbone."):]: v for k, v in st.items() if k.startswith("backbone.")}
    if bbst and next(iter(bbst))[0].isdigit():  # RadImageNet ResNet50: torchvision children as nn.Sequential
        bbst = {RADIMAGENET_RESNET[k.split(".", 1)[0]] + "." + k.split(".", 1)[1]: v for k, v in bbst.items()}
        missing, unexpected = bb.load_state_dict(bbst, strict=True)  # --arch resnet50; any mismatch is an error
    missing, unexpected = bb.load_state_dict(bbst, strict=False)
    ep = ck.get("epoch", "?")
    print(f"[raptor] loaded backbone from {os.path.basename(ckpt_path)} (ssl epoch {ep}) | "
          f"loaded {len(bbst)} tensors, missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
    return f"{os.path.basename(ckpt_path)}@ep{ep}"


class RaptorClassifier(nn.Module):
    """Raptor encoder + per-diagnosis attention-MIL head (12 findings)."""
    def __init__(self, backbone, F_dim=384, n=12, drop=0.2, slot_emb=False, seq_head=False, pos_emb=False, n_slots=5):
        super().__init__()
        self.backbone = backbone
        self.F_dim = F_dim
        self.slot_emb = nn.Embedding(n_slots, F_dim) if slot_emb else None
        if self.slot_emb is not None:
            nn.init.zeros_(self.slot_emb.weight)
        self.pos_proj = nn.Linear(3, F_dim) if pos_emb else None   # [p, sin(pi p), cos(pi p)] -> F, starts as a no-op
        if self.pos_proj is not None:
            nn.init.zeros_(self.pos_proj.weight); nn.init.zeros_(self.pos_proj.bias)
        self.seq_head = bool(seq_head)
        self.gru = nn.GRU(F_dim, F_dim // 2, num_layers=2,
                          batch_first=True, bidirectional=True,
                          dropout=0.1) if self.seq_head else None
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.aux_norm = nn.LayerNorm(F_dim) if self.seq_head else None
        self.aux_att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                     nn.Linear(256, n)) if self.seq_head else None
        self.aux_clsW = nn.Parameter(torch.zeros(n, F_dim)) if self.seq_head else None
        self.aux_clsb = nn.Parameter(torch.zeros(n)) if self.seq_head else None
        if self.aux_clsW is not None:
            nn.init.trunc_normal_(self.aux_clsW, std=0.02)
        self.n = n

    win_chunk = 0   # >0: run the backbone over chunks of this many windows under activation checkpointing (8 GB cards)
    stkim_k = 0     # >0 (training only): ACMIL stochastic top-K masking of each finding's top-k windows
    stkim_p = 0.7   # probability that each of those top-k windows is hidden
    seq_tracks = 1  # tiles per window position (2 with zoom: field, close-up); the GRU runs each track on its own

    def _attend(self, h):
        """Per-finding attention weights (B,K,n). In training with stkim_k > 0 each finding's top-k
        windows are hidden with probability stkim_p (ACMIL, ECCV 2024); eval is unchanged."""
        a = self.att(h)
        if self.training and self.stkim_k > 0 and a.shape[1] > self.stkim_k:
            top = a.topk(self.stkim_k, dim=1).indices
            hit = torch.zeros_like(a, dtype=torch.bool).scatter_(1, top, True)
            a = a.masked_fill(hit & (torch.rand_like(a) < self.stkim_p), float("-inf"))
        return torch.softmax(a, dim=1)

    def _seq(self, feats):
        """bi-GRU over the windows in slice order. With seq_tracks = T tiles per position (field, close-up,
        field, ...) each track is its own sequence, so the GRU never alternates between scales."""
        T = self.seq_tracks
        if T == 1:
            return self.gru(feats)[0]
        B, K, F = feats.shape
        S = K // T
        g = feats[:, :S * T].reshape(B, S, T, F).transpose(1, 2).reshape(B * T, S, F)
        g = self.gru(g)[0].reshape(B, T, S, F).transpose(1, 2).reshape(B, S * T, F)
        return torch.cat([g, g.new_zeros(B, K - S * T, F)], 1)  # a cut-off last tile gets no GRU term

    def encode(self, x):
        B, K = x.shape[:2]
        flat = x.flatten(0, 1)
        if self.win_chunk > 0 and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            f = torch.cat([checkpoint(self.backbone, c, use_reentrant=False) for c in flat.split(self.win_chunk, 0)], 0)
        elif self.win_chunk > 0:
            f = torch.cat([self.backbone(c) for c in flat.split(self.win_chunk, 0)], 0)
        else:
            f = self.backbone(flat)
        return f.view(B, K, -1)

    def _add_slots(self, feats, slots):
        if slots is None:
            return feats
        if self.pos_proj is not None and slots.is_floating_point():   # pack_slot_pos: slot id + packed position
            sid = slots.floor()
            p = (slots - sid) / 0.4995 - 1.0
            feats = feats + self.pos_proj(torch.stack([p, torch.sin(math.pi * p), torch.cos(math.pi * p)], -1).to(feats.dtype))
            slots = sid.long()
        if self.slot_emb is not None:
            feats = feats + self.slot_emb(slots)
        return feats

    def head(self, feats, slots=None):
        feats = self._add_slots(feats, slots)
        if self.seq_head:
            feats = feats + self._seq(feats)
        h = self.norm(feats)
        a = self._attend(h)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def aux_head(self, feats, slots=None):
        """Independent attention-MIL copy on the PRE-GRU features (training aux loss)."""
        feats = self._add_slots(feats, slots)
        h = self.aux_norm(feats)
        a = self.aux_att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.aux_clsW).sum(-1) + self.aux_clsb
        return logits

    def forward(self, x, slots=None):
        return self.head(self.encode(x), slots=slots)

    def forward_with_aux(self, x, slots=None):
        """Returns (main_logits, aux_logits). Eval/inference use main only."""
        feats = self.encode(x)
        return self.head(feats, slots=slots), self.aux_head(feats, slots=slots)


# ------------------------------- train ---------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # Defaults to ImageNet-pretrained weights, which is what every released arm was trained from.
    # The previous default pointed at a self-supervised checkpoint that is not distributed here,
    # so following the example run above failed on a missing file.
    ap.add_argument("--ckpt", default="timm",
                    help="'timm' for ImageNet-pretrained weights, or a path to your own "
                         "self-supervised checkpoint")
    ap.add_argument("--arch", default="vit_small_patch16_224")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--k_eval", type=int, default=24,
                    help="eval windows per study; 94 = 96 - 2 is full coverage on the 96 layout "
                         "(190 = 192 - 2 on the 192 layout)")
    ap.add_argument("--slot_sizes", default="26,22,18,12,18",
                    help="comma list of the 5 slot slice counts, e.g. 40,40,36,36,40 for the 192 layout; "
                         "must match the corpus volume depth")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (effective batch = bs * accum)")
    ap.add_argument("--win_chunk", type=int, default=0, help="checkpointed backbone chunks of N windows (memory); 0 = whole study at once")
    ap.add_argument("--oof_every", type=int, default=0, help="score a fixed N-study subset of the held-out fold every epoch (masked AUC vs soft labels)")
    ap.add_argument("--oof_mask", default=None, help="parquet with m_<label> columns (1 = supervised) for the per-epoch OOF subset")
    ap.add_argument("--fp16", action="store_true", help="fp16 autocast + loss scaling (T4 has no bf16)")
    ap.add_argument("--xla", action="store_true", help="train on a TPU core through torch_xla (bf16 autocast, mark_step per micro-batch, no grad scaler)")
    ap.add_argument("--resume", default=None, help="resume state file written by --time_budget_h")
    ap.add_argument("--time_budget_h", type=float, default=0.0, help="stop after the epoch that crosses this wall-clock budget and write a resume state")
    ap.add_argument("--patience", type=int, default=5, help="stop after N consecutive epochs without validation improvement (0 to disable)")
    ap.add_argument("--bb_lr", type=float, default=3e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--freeze_blocks", type=int, default=0)
    # Each kept epoch is a full copy of the weights in system RAM (~87 MB small ViT, ~293 MB
    # CoAtNet). Lower to 1 on a memory-tight machine; at 1 no averaged checkpoint is written.
    ap.add_argument("--topk", type=int, default=3,
                    help="top epochs kept for weight averaging (1 disables averaging)")
    ap.add_argument("--norm", default="none", choices=["none", "imagenet"])
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--labels", default=None,
                    help="parquet of training labels: StudyInstanceUID + the twelve finding "
                         "columns, values 0..1. Not provided with this notebook - see the header.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    # ---- CV fold hook ----
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--fold_file", default=None)
    # ---- anatomical ROI joint-crop (A/B lever). Default OFF -> identical to baseline. ----
    ap.add_argument("--roi", action="store_true")
    ap.add_argument("--roi_mode", default="compartment", choices=["tight", "safe", "compartment"])
    ap.add_argument("--roi_pad", type=float, default=0.06)
    ap.add_argument("--roi_overlap", type=float, default=0.12)
    ap.add_argument("--roi_boxes", default=os.path.join(HERE, "roi_boxes.npz"))
    # ---- data recipe (T05) ----
    ap.add_argument("--clean_windows", action="store_true", help="only windows where c-1, c, c+1 are real & same slot")
    ap.add_argument("--mirror_p", type=float, default=0.0, help="study-level anatomical mirror augmentation probability")
    ap.add_argument("--contrast_aug", action="store_true", help="study-level random gamma contrast augmentation in [0.8, 1.25]")
    ap.add_argument("--slots", default=None, help="specialist: comma list of slot ids whose windows are used, e.g. 2,3 = coronal")
    ap.add_argument("--canon_side", default=None, help="CSV StudyInstanceUID,Laterality: mirror every Left knee to right-knee layout (train and eval)")
    ap.add_argument("--swap_p", type=float, default=0.0, help="with --canon_side: probability of mirror + medial<->lateral label swap (train only)")
    ap.add_argument("--pos_emb", default=None, help="CSV StudyInstanceUID,Laterality (Left/Right): add each window's anatomical position (lateral->medial, front->back, foot->head) to its features")
    ap.add_argument("--slot_emb", action="store_true", help="add learned slot embedding (init zeros) to window features")
    ap.add_argument("--seq_head", action="store_true", help="bi-GRU sequence head over sorted windows + auxiliary attention loss (refuses roi_mode compartment)")
    ap.add_argument("--aux_w", type=float, default=0.3, help="weight of the auxiliary pre-GRU attention loss")
    ap.add_argument("--label_weights", default=None, help="path to per-(study,finding) weight parquet (w_<finding> cols); down-weights noisy labels")
    ap.add_argument("--stkim_k", type=int, default=0, help="ACMIL stochastic top-K window masking (training only): hide each finding's top-k windows")
    ap.add_argument("--stkim_p", type=float, default=0.7, help="probability each top-k window is hidden when --stkim_k > 0")
    ap.add_argument("--zoom_mm", type=float, default=0.0, help="two-scale tiles: each window also yields a centred close-up of this many mm (corpus field is 140 mm); 0 = off")
    ap.add_argument("--extra_labels", default=None, help="parquet of soft labels for studies in RAPTOR_EXTRA_CORPUS (e.g. teacher predictions from tools/predict_corpus.py); they always train, never validate")
    ap.add_argument("--extra_weight", type=float, default=1.0, help="loss weight of every extra-corpus label")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    XLA = None
    if a.xla:
        import torch_xla, torch_xla.core.xla_model as xm
        XLA = xm; dev = torch_xla.device()
        print(f"[xla] device {dev} | torch_xla {torch_xla.__version__}", flush=True)
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    DEVT = "xla" if XLA else dev                      # autocast device_type string
    WIF = _winit if XLA else None                     # loader worker_init_fn (XLA path only)
    AMP_DTYPE = torch.float16 if (a.fp16 or (dev == "cuda" and not torch.cuda.is_bf16_supported())) else torch.bfloat16
    AMP_ON = (dev == "cuda") or (XLA is not None)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda" and AMP_DTYPE == torch.float16))
    print(f"[amp] dtype {AMP_DTYPE} | autocast {AMP_ON} | grad scaler {scaler.is_enabled()}", flush=True)
    if a.smoke:
        a.epochs, a.bs, a.k, a.k_eval, a.limit, a.workers = 2, 4, 4, 6, 40, 0
    print(f"device {dev} | res {a.res} | k {a.k}/{a.k_eval} | bs {a.bs} | tag {a.tag} | "
          f"roi={a.roi}({a.roi_mode})", flush=True)

    # ---- ids / labels ----
    ids = _open_ids()
    id2row = {u: i for i, u in enumerate(ids)}
    idset = set(ids)
    _tcsv = os.path.join(RSNA, "train.csv")
    if not os.path.exists(_tcsv):
        _tcsv = _find("train.csv")
    if not _tcsv:
        raise SystemExit("Could not find train.csv - attach the competition data.")
    tr = pd.read_csv(_tcsv); tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    gold_df = tr[tr[LAB].notna().all(axis=1)].copy().set_index("StudyInstanceUID")
    gold_ids = [u for u in gold_df.index if u in idset]
    _lab = a.labels or _find("labels_llm_soft.parquet")
    if not _lab or not os.path.exists(_lab):
        # Exit cleanly rather than as a failure: running this notebook as published, with no
        # labels attached, is the expected path and should read as an explanation, not a crash.
        print(_NO_LABELS.format(cols=", ".join(LAB)), flush=True)
        raise SystemExit(0)
    soft = pd.read_parquet(_lab)
    soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str); soft = soft.set_index("StudyInstanceUID")
    goldset = set(gold_ids)
    train_ids = [u for u in ids if u in soft.index and u not in goldset]
    # ---- CV fold hook: hold out fold `a.fold`, train on the rest ----
    oof_ids = []
    if a.folds > 0:
        assert 0 <= a.fold < a.folds, f"--fold must be in [0,{a.folds}) when --folds>0"
        fmap = json.load(open(a.fold_file))["folds"]
        held = set(u for u in train_ids if fmap.get(u, -1) == a.fold)
        oof_ids = [u for u in train_ids if u in held]
        train_ids = [u for u in train_ids if u not in held]
        print(f"[cv] fold {a.fold}/{a.folds}: train {len(train_ids)} | OOF held-out {len(oof_ids)} "
              f"| fold_file {os.path.basename(a.fold_file)}", flush=True)
    if a.limit: train_ids = train_ids[:a.limit]
    labels = {u: soft.loc[u, LAB].values.astype(np.float32) for u in train_ids}
    for u in oof_ids: labels[u] = soft.loc[u, LAB].values.astype(np.float32)
    for u in gold_ids: labels[u] = gold_df.loc[u, LAB].values.astype(np.float32)
    print(f"train {len(train_ids)} | gold-val {len(gold_ids)}"
          + (f" | oof {len(oof_ids)}" if oof_ids else ""), flush=True)

    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)
    pw = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)

    # ---- ROI boxes (only when --roi) ----
    roi_sbox = roi_cen = None
    if a.roi:
        rb = np.load(a.roi_boxes, allow_pickle=True)
        rb_ids = rb["ids"].astype(str)
        if not np.array_equal(rb_ids, ids):
            rmap = {u: i for i, u in enumerate(rb_ids)}
            missing = [u for u in ids if u not in rmap]
            if missing:
                raise RuntimeError(f"roi_boxes missing {len(missing)} corpus ids (e.g. {missing[:2]})")
            order = np.array([rmap[u] for u in ids])
            roi_sbox = rb["sbox"][order]; roi_cen = rb["cen"][order]
        else:
            roi_sbox = rb["sbox"]; roi_cen = rb["cen"]
        print(f"[roi] ENABLED mode={a.roi_mode} pad={a.roi_pad} overlap={a.roi_overlap} "
              f"boxes={os.path.basename(a.roi_boxes)} sbox={roi_sbox.shape}", flush=True)
    pos_side = None
    if a.pos_emb:
        if not a.slot_emb:
            raise ValueError("--pos_emb needs --slot_emb (the position rides on the slot tensor)")
        lat = pd.read_csv(a.pos_emb)
        pos_side = dict(zip(lat.StudyInstanceUID.astype(str), lat.Laterality.map({"Left": 1.0, "Right": -1.0}).fillna(0.0)))
        print(f"[pos_emb] window position input; knee side for {sum(v != 0 for v in pos_side.values())} studies", flush=True)
    canon_left = None
    if a.canon_side:
        assert a.mirror_p == 0.0, "--canon_side fixes the knee side; use --swap_p, not --mirror_p"
        _c = pd.read_csv(a.canon_side)
        canon_left = set(_c.StudyInstanceUID[_c.Laterality == "Left"].astype(str))
        print(f"[canon] {len(canon_left)} left knees mirrored to right layout | swap_p {a.swap_p}", flush=True)
    _roi_kw = dict(roi=a.roi, roi_mode=a.roi_mode, roi_pad=a.roi_pad, roi_overlap=a.roi_overlap,
                   roi_sbox=roi_sbox, roi_cen=roi_cen,
                   clean_windows=bool(a.clean_windows), slot_emb=bool(a.slot_emb),
                   slots=tuple(int(v) for v in a.slots.split(',')) if a.slots else None,
                   seq_head=bool(a.seq_head), zoom_px=zoom_px(a.zoom_mm), pos_side=pos_side,
                   canon_left=canon_left, swap_p=a.swap_p)
    if a.zoom_mm:
        print(f"[zoom] two-scale tiles: 140 mm field + centred {a.zoom_mm:g} mm close-up ({zoom_px(a.zoom_mm)} px); --k/--k_eval count tiles", flush=True)

    # ---- slot layout (any 5-slot corpus depth, e.g. 96 or 192) ----
    _sizes = [int(v) for v in a.slot_sizes.split(",")]
    assert len(_sizes) in (5, 6) and all(v > 0 for v in _sizes), f"--slot_sizes must be 5 (or 6 with RAPTOR_AXIAL2) positive ints: {a.slot_sizes}"
    global SLOT_BOUNDS
    SLOT_BOUNDS = bounds_from_sizes(_sizes)
    _corpus_vols, _ = _open_corpus()
    assert _corpus_vols.shape[1] == sum(_sizes), \
        f"corpus depth {_corpus_vols.shape[1]} != sum(slot_sizes) {sum(_sizes)}"
    print(f"[slots] layout={_sizes} depth={sum(_sizes)} bounds={SLOT_BOUNDS}", flush=True)

    # ---- label down-weights (only when --label_weights) ----
    lw_all = None
    if a.label_weights:
        lw_df = pd.read_parquet(a.label_weights).set_index("StudyInstanceUID")
        lw_cols = [c for c in lw_df.columns if c.startswith("w_")]
        assert len(lw_cols) == 12, f"{a.label_weights}: {len(lw_cols)} w_ cols"
        lw_all = {u: lw_df.loc[u, lw_cols].values.astype(np.float32) for u in lw_df.index}
        missing = [u for u in train_ids if u not in lw_all]
        if missing:
            raise RuntimeError(f"label_weights missing {len(missing)} train ids")
        print(f"[label_weights] {os.path.basename(a.label_weights)} rows={len(lw_all)} "
              f"zero_frac={float((lw_df[lw_cols].values == 0).mean()):.3f}", flush=True)
    if a.extra_labels:                                # after pos_weight, which stays on competition labels
        n_comp = len(train_ids)
        train_ids, lw_all = add_extra(train_ids, labels, lw_all, ids, set(tr["StudyInstanceUID"]),
                                      a.extra_labels, a.extra_weight, a.limit)
        print(f"[extra] +{len(train_ids) - n_comp} studies from RAPTOR_EXTRA_CORPUS "
              f"({os.path.basename(a.extra_labels)}), weight {a.extra_weight:g}", flush=True)

    if os.environ.get("RAPTOR_PRELOAD") == "1":
        _open_corpus()                                # parent holds the RAM copy; forked workers inherit it
    tds = StudyWindows(HERE, train_ids, id2row, labels, a.res, a.k, train=True, norm=a.norm,
                       mirror_p=a.mirror_p, contrast_aug=a.contrast_aug, label_w=lw_all, **_roi_kw)
    vds = StudyWindows(HERE, gold_ids, id2row, labels, a.res, a.k_eval, train=False, norm=a.norm, **_roi_kw)
    tl = DataLoader(tds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                    collate_fn=collate, pin_memory=(dev == "cuda"), persistent_workers=a.workers > 0, worker_init_fn=WIF)
    vl = DataLoader(vds, batch_size=max(2, a.bs // 2), shuffle=False, num_workers=a.workers,
                    collate_fn=collate, persistent_workers=a.workers > 0, worker_init_fn=WIF)
    sl = None; sub_mask = None
    if a.oof_every > 0 and a.folds > 0 and oof_ids:
        sub_ids = random.Random(0).sample(list(oof_ids), min(a.oof_every, len(oof_ids)))
        sds = StudyWindows(HERE, sub_ids, id2row, labels, a.res, a.k_eval, train=False, norm=a.norm, **_roi_kw)
        sl = DataLoader(sds, batch_size=max(2, a.bs // 2), shuffle=False, num_workers=a.workers, collate_fn=collate, persistent_workers=False, worker_init_fn=WIF)
        if a.oof_mask:
            _mk = pd.read_parquet(a.oof_mask).set_index("StudyInstanceUID")
            _mc = [c for c in _mk.columns if c.startswith("m_")]
            sub_mask = np.stack([_mk.loc[u, _mc].values.astype(np.float32) if u in _mk.index else np.ones(12, np.float32) for u in sub_ids])
        else:
            sub_mask = np.ones((len(sub_ids), 12), np.float32)
        print(f"[oof-sub] {len(sub_ids)} held-out studies scored every epoch (mask: {'file' if a.oof_mask else 'none'})", flush=True)

    use_timm = a.ckpt in ("timm", "pretrained")
    bb = build_backbone(a.arch, pretrained=use_timm)
    src = load_raptor(bb, a.ckpt)
    if use_timm: print(f"[raptor] timm-pretrained backbone: {a.arch}", flush=True)
    F_dim = bb.num_features
    if a.grad_ckpt:
        try:
            bb.set_grad_checkpointing(True); print("[raptor] gradient checkpointing ON", flush=True)
        except Exception as e:
            print(f"[raptor] grad_ckpt unsupported for {a.arch}: {e}", flush=True)
    model = RaptorClassifier(bb, F_dim=F_dim, slot_emb=bool(a.slot_emb),
                             seq_head=bool(a.seq_head), pos_emb=bool(a.pos_emb), n_slots=len(SLOT_BOUNDS)).to(dev)
    model.win_chunk = a.win_chunk
    model.stkim_k, model.stkim_p = a.stkim_k, a.stkim_p
    model.seq_tracks = 2 if a.zoom_mm else 1   # zoom: field and close-up tiles alternate per window position
    if a.stkim_k > 0:
        print(f"[stkim] training-only top-{a.stkim_k} window masking, p={a.stkim_p}", flush=True)
    if a.freeze_blocks > 0:
        # Backbones name their layer groups differently: ViT-style models use blocks.N., while
        # CoAtNet, MaxViT and ConvNeXt use stages.N. Matching only one of those froze nothing at
        # all on the other family, and did it silently. Try both, and always say what happened.
        pref = tuple([f"blocks.{b}." for b in range(a.freeze_blocks)]
                     + [f"stages.{b}." for b in range(a.freeze_blocks)])
        frozen = 0
        for nm, p in model.backbone.named_parameters():
            if nm.startswith(pref):
                p.requires_grad = False
                frozen += p.numel()
        n_t = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
        if frozen:
            print(f"[raptor] froze the first {a.freeze_blocks} layer group(s): "
                  f"{n_t} tensors, {frozen/1e6:.2f}M parameters", flush=True)
        else:
            print(f"[raptor] WARNING: --freeze_blocks {a.freeze_blocks} matched no parameters on "
                  f"{a.arch}; every parameter is still being trained", flush=True)

    head_params = [p for n_, p in model.named_parameters() if not n_.startswith("backbone.") and p.requires_grad]
    bb_params = [p for n_, p in model.named_parameters() if n_.startswith("backbone.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb_params, "lr": a.bb_lr},
                             {"params": head_params, "lr": a.head_lr}], weight_decay=a.wd)
    steps = max((len(tl) // a.accum) * a.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr],
                                                total_steps=steps, pct_start=0.15)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    bce_none = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")

    @torch.no_grad()
    def evaluate():
        model.eval(); P = []; Y = []
        for batch in vl:
            x = batch[0]
            slots = batch[2].to(dev) if a.slot_emb else None
            with torch.autocast(DEVT, dtype=AMP_DTYPE, enabled=AMP_ON):
                o = torch.sigmoid(model(x.to(dev), slots=slots).float())
            P.append(o.cpu().numpy()); Y.append(batch[1].numpy())
        P = np.concatenate(P); Y = np.concatenate(Y)
        aucs = {}
        for j, name in enumerate(LAB):
            if len(set(Y[:, j].astype(int))) > 1:
                aucs[name] = float(roc_auc_score(Y[:, j], P[:, j]))
        return float(np.mean(list(aucs.values()))), aucs, P, Y

    best = 0.0; best_state = None; best_P = None; t0 = time.time(); hist = []
    TOPK = max(1, a.topk); topk = []; ep_start = 0; patience_counter = 0

    import signal
    _interrupted = False
    def _sig_handler(signum, frame):
        nonlocal _interrupted
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\n[signal] Received {sig_name}! Gracefully saving checkpoint and stopping...", flush=True)
        _interrupted = True

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    if a.resume and os.path.exists(a.resume):
        R_ = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(R_["model"]); opt.load_state_dict(R_["opt"])
        if R_.get("sched") is not None:
            sched_sd = R_["sched"]
            if steps != sched_sd.get("total_steps"):
                print(f"[sched] adapting OneCycleLR total_steps from {sched_sd.get('total_steps')} to {steps} for {a.epochs} epochs", flush=True)
                sched_sd["total_steps"] = steps
            sched.load_state_dict(sched_sd)
        if R_.get("scaler") is not None and scaler.is_enabled(): scaler.load_state_dict(R_["scaler"])
        best, best_state, best_P, hist, topk = R_["best"], R_["best_state"], R_["best_P"], R_["hist"], R_["topk"]
        ep_start = int(R_["epoch"]) + 1; t0 -= float(R_.get("elapsed", 0.0))
        patience_counter = int(R_.get("patience_counter", 0))
        if "rng" in R_ and R_["rng"]:
            rng = R_["rng"]
            if "torch" in rng: torch.set_rng_state(rng["torch"])
            if "cuda" in rng and rng["cuda"] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
            if "numpy" in rng: np.random.set_state(rng["numpy"])
            if "python" in rng: random.setstate(rng["python"])
        print(f"[resume] from {os.path.basename(a.resume)}: continuing at epoch {ep_start}/{a.epochs} (best {best:.4f}, patience {patience_counter}/{a.patience})", flush=True)
        del R_
    for ep in range(ep_start, a.epochs):
        model.train(); tot = 0.0; tot_t = None
        opt.zero_grad(); n_acc = 0
        _te = time.time(); _ts = time.time(); _step = 0
        for batch in tl:
            if _interrupted:
                print(f"[signal] Interrupt triggered at step {_step}/{len(tl)}", flush=True)
                break
            x = batch[0].to(dev, non_blocking=True)
            y = batch[1].to(dev, non_blocking=True)
            slots = batch[2].to(dev, non_blocking=True) if a.slot_emb else None
            w = batch[-1].to(dev, non_blocking=True) if lw_all is not None else None
            with torch.autocast(DEVT, dtype=AMP_DTYPE, enabled=AMP_ON):
                if a.seq_head:
                    logits, aux = model.forward_with_aux(x, slots=slots)
                else:
                    logits = model(x, slots=slots)
                if w is None:
                    loss = lossf(logits.float(), y)
                    if a.seq_head:
                        loss = loss + a.aux_w * lossf(aux.float(), y)
                else:
                    loss = weighted_bce(bce_none, logits.float(), y, w)
                    if a.seq_head:
                        loss = loss + a.aux_w * weighted_bce(bce_none, aux.float(), y, w)
            scaler.scale(loss / a.accum).backward(); n_acc += 1
            if XLA: tot_t = loss.detach() if tot_t is None else tot_t + loss.detach()   # no per-step host sync on XLA
            else: tot += loss.item()
            if XLA and n_acc < a.accum: XLA.mark_step()
            if n_acc == a.accum:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scaler.step(opt); scaler.update()
                if sched.last_epoch + 1 < steps: sched.step()
                opt.zero_grad(); n_acc = 0
                if XLA: XLA.mark_step()
            if XLA and _step < 12:
                XLA.wait_device_ops(); print(f"  [xla] step {_step} {time.time()-_ts:.2f}s", flush=True); _ts = time.time()
            _step += 1
        train_elapsed = time.time() - _te
        sec_per_it = train_elapsed / max(_step, 1)
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            res_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
            print(f"  [timing] {_step} steps in {train_elapsed:.1f}s ({sec_per_it:.3f} s/step) | peak {peak_gb:.2f} GB (reserved {res_gb:.2f} GB)", flush=True)
        if _interrupted:
            print(f"[signal] Interrupted during epoch {ep}. Saving resume state before exiting...", flush=True)
            rp = os.path.join(HERE, f"raptor_resume_{a.tag}.pt")
            rng_state = {"torch": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                         "numpy": np.random.get_state(), "python": random.getstate()}
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict() if scaler.is_enabled() else None, "best": best,
                        "best_state": best_state, "best_P": best_P, "hist": hist, "topk": topk, "epoch": max(0, ep - 1),
                        "elapsed": time.time() - t0, "rng": rng_state, "patience_counter": patience_counter}, rp + ".tmp")
            os.replace(rp + ".tmp", rp)
            break
        if XLA:
            XLA.wait_device_ops(); print(f"  [xla] train loop {time.time()-_te:.0f}s for {_step} micro-steps", flush=True); _te = time.time()
        if XLA and tot_t is not None: tot = float(tot_t.item())
        au, aucs, P, Y = evaluate()
        if XLA: print(f"  [xla] gold eval {time.time()-_te:.0f}s", flush=True); _te = time.time()
        hist.append({"ep": ep, "loss": tot / len(tl), "gold_auc": au})
        if sl is not None:
            model.eval(); Ps = []; Ys = []
            with torch.no_grad():
                for batch in sl:
                    x = batch[0]
                    slots = batch[2].to(dev) if a.slot_emb else None
                    with torch.autocast(DEVT, dtype=AMP_DTYPE, enabled=AMP_ON):
                        Ps.append(torch.sigmoid(model(x.to(dev), slots=slots).float()).cpu().numpy())
                    Ys.append(batch[1].numpy())
            Ps = np.concatenate(Ps); Ys = np.concatenate(Ys); sub = []
            for j in range(len(LAB)):
                kk = sub_mask[:, j] > 0; yb = (Ys[kk, j] >= 0.5).astype(int)
                if len(set(yb)) > 1: sub.append(float(roc_auc_score(yb, Ps[kk, j])))
            so = float(np.mean(sub)) if sub else float("nan"); hist[-1]["oof_sub_auc"] = so
            print(f"ep{ep} OOF{len(Ys)} masked macro-AUC {so:.4f}", flush=True)
            if XLA: print(f"  [xla] oof-sub eval {time.time()-_te:.0f}s", flush=True)
        if XLA and ep == ep_start:
            import torch_xla.debug.metrics as _met; print(_met.short_metrics_report(), flush=True)
        if au > best:
            best = au; best_P = P
            patience_counter = 0
            best_state = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "gold_auc": au, "aucs": aucs, "src": src, "res": a.res, "norm": a.norm,
                          "clean_windows": bool(a.clean_windows), "slot_emb": bool(a.slot_emb), "slots": a.slots, "slot_sizes": a.slot_sizes, "roi": a.roi, "roi_mode": a.roi_mode, "roi_pad": a.roi_pad, "roi_overlap": a.roi_overlap, "seq_head": bool(a.seq_head), "zoom_mm": float(a.zoom_mm), "aux_w": float(a.aux_w), "label_weights": os.path.basename(a.label_weights) if a.label_weights else None,
                          "arch": a.arch, "lab": LAB, "epoch": ep}
        else:
            patience_counter += 1
            if a.patience > 0:
                print(f"  [patience] no improvement for {patience_counter}/{a.patience} epoch(s) (best {best:.4f} at ep{best_state['epoch'] if best_state else 0})", flush=True)

        if best_state is not None:
            _tmp = os.path.join(HERE, f"raptor_ft_{a.tag}.pt.tmp")
            torch.save(best_state, _tmp)
            os.replace(_tmp, os.path.join(HERE, f"raptor_ft_{a.tag}.pt"))
            np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}.npz"),
                     pred=best_P, truth=Y, ids=np.array(gold_ids))
            json.dump({"tag": a.tag, "src": src, "best_gold_auc": best,
                       "aucs": best_state["aucs"], "hist": hist, "res": a.res, "norm": a.norm,
                       "clean_windows": bool(a.clean_windows), "slot_emb": bool(a.slot_emb), "slots": a.slots, "slot_sizes": a.slot_sizes, "label_weights": os.path.basename(a.label_weights) if a.label_weights else None,
                       "epochs": a.epochs, "bb_lr": a.bb_lr, "head_lr": a.head_lr,
                       "n_train": len(train_ids), "n_gold": len(gold_ids),
                       "roi": a.roi, "roi_mode": a.roi_mode, "roi_pad": a.roi_pad, "roi_overlap": a.roi_overlap,
                       "partial": True, "epochs_done": ep + 1},
                      open(os.path.join(HERE, f"raptor_ft_{a.tag}.json"), "w"), indent=1)
            if patience_counter == 0:
                print(f"  [ckpt] best-so-far saved at ep{ep} ({best:.4f})", flush=True)

        sel = hist[-1].get("oof_sub_auc", float("nan"))                 # selection score: masked OOF subset if scored, else gold
        sel = sel if sel == sel else au
        if len(topk) < TOPK or sel > min(t.get("sel", t["gold_auc"]) for t in topk):
            topk.append({"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                         "gold_auc": au, "aucs": aucs, "src": src, "res": a.res, "norm": a.norm,
                         "clean_windows": bool(a.clean_windows), "slot_emb": bool(a.slot_emb), "slots": a.slots, "slot_sizes": a.slot_sizes, "roi": a.roi, "roi_mode": a.roi_mode, "roi_pad": a.roi_pad, "roi_overlap": a.roi_overlap, "seq_head": bool(a.seq_head), "zoom_mm": float(a.zoom_mm), "aux_w": float(a.aux_w), "label_weights": os.path.basename(a.label_weights) if a.label_weights else None,
                         "arch": a.arch, "lab": LAB, "epoch": ep, "P": P, "sel": sel})
            topk.sort(key=lambda t: -t.get("sel", t["gold_auc"]))
            del topk[TOPK:]

            # Save running SWA checkpoint whenever topk updates!
            if len(topk) > 1:
                sds = [t["model"] for t in topk]
                avg = {}
                for k in sds[0]:
                    v0 = sds[0][k]
                    if v0.is_floating_point():
                        avg[k] = sum(sd[k].double() for sd in sds).div(len(sds)).to(v0.dtype)
                    else:
                        avg[k] = v0.clone()
                swa_state = {"model": avg, "gold_auc": None, "aucs": {}, "src": src, "res": a.res, "norm": a.norm,
                             "clean_windows": bool(a.clean_windows), "slot_emb": bool(a.slot_emb), "slots": a.slots, "slot_sizes": a.slot_sizes, "roi": a.roi, "roi_mode": a.roi_mode, "roi_pad": a.roi_pad, "roi_overlap": a.roi_overlap, "seq_head": bool(a.seq_head), "zoom_mm": float(a.zoom_mm), "aux_w": float(a.aux_w), "label_weights": os.path.basename(a.label_weights) if a.label_weights else None,
                             "arch": a.arch, "lab": LAB, "epoch": [int(t["epoch"]) for t in topk],
                             "swa_over": [int(t["epoch"]) for t in topk]}
                torch.save(swa_state, os.path.join(HERE, f"raptor_ft_{a.tag}_swa.pt"))
                print(f"  [swa] running SWA updated over epochs {[int(t['epoch']) for t in topk]}", flush=True)

        print(f"ep{ep} loss {tot/len(tl):.3f} | GOLD macro-AUC {au:.4f} (best {best:.4f}) | topk by {'OOF-subset' if sl is not None else 'gold'}: {[int(t['epoch']) for t in topk]} | {time.time()-t0:.0f}s",
              flush=True)
        # Write resume state after EVERY epoch (atomic replace) so job can be paused/resumed
        rp = os.path.join(HERE, f"raptor_resume_{a.tag}.pt")
        rng_state = {"torch": torch.get_rng_state(),
                     "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                     "numpy": np.random.get_state(), "python": random.getstate()}
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None, "best": best,
                    "best_state": best_state, "best_P": best_P, "hist": hist, "topk": topk, "epoch": ep,
                    "elapsed": time.time() - t0, "rng": rng_state, "patience_counter": patience_counter}, rp + ".tmp")
        os.replace(rp + ".tmp", rp)

        if a.patience > 0 and patience_counter >= a.patience:
            print(f"\n[early-stop] Validation score did not improve for {patience_counter} consecutive epochs! "
                  f"Stopping early at epoch {ep} (best {best:.4f} at epoch {best_state['epoch'] if best_state else 0}).", flush=True)
            break

        if _interrupted:
            print(f"[signal] Interrupted after epoch {ep}. Exiting training loop...", flush=True)
            break

        if a.time_budget_h > 0 and (time.time() - t0) > a.time_budget_h * 3600 and ep < a.epochs - 1:
            print(f"[resume] time budget reached after ep{ep}; state saved to {os.path.basename(rp)}; exiting", flush=True)
            sys.exit(0)
    _, aucs, _, Y = evaluate()
    print(f"\nDONE {src} | BEST GOLD macro-AUC {best:.4f}", flush=True)
    for k, v in (best_state["aucs"] if best_state else aucs).items():
        print(f"   {k:18s} {v:.3f}", flush=True)

    if best_state is not None:
        torch.save(best_state, os.path.join(HERE, f"raptor_ft_{a.tag}.pt"))
        np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}.npz"),
                 pred=best_P, truth=Y, ids=np.array(gold_ids))
        json.dump({"tag": a.tag, "src": src, "best_gold_auc": best, "aucs": best_state["aucs"],
                   "hist": hist, "res": a.res, "norm": a.norm,
                   "clean_windows": bool(a.clean_windows), "slot_emb": bool(a.slot_emb), "slots": a.slots, "slot_sizes": a.slot_sizes, "label_weights": os.path.basename(a.label_weights) if a.label_weights else None,
                   "epochs": a.epochs, "bb_lr": a.bb_lr,
                   "head_lr": a.head_lr, "n_train": len(train_ids), "n_gold": len(gold_ids),
                   "roi": a.roi, "roi_mode": a.roi_mode, "roi_pad": a.roi_pad, "roi_overlap": a.roi_overlap, "partial": False, "epochs_done": a.epochs},
                  open(os.path.join(HERE, f"raptor_ft_{a.tag}.json"), "w"), indent=1)
        print(f"saved raptor_ft_{a.tag}.pt / raptor_gold_{a.tag}.npz / raptor_ft_{a.tag}.json", flush=True)

    # ---- CV OOF: predict the held-out fold at the best (gold-selected) weights ----
    if a.folds > 0 and oof_ids and best_state is not None:
        model.load_state_dict(best_state["model"]); model.eval()
        ods = StudyWindows(HERE, oof_ids, id2row, labels, a.res, a.k_eval, train=False, norm=a.norm, **_roi_kw)
        ol = DataLoader(ods, batch_size=max(2, a.bs // 2), shuffle=False, num_workers=a.workers,
                        collate_fn=collate, persistent_workers=False, worker_init_fn=WIF)
        Po = []
        with torch.no_grad():
            for batch in ol:
                x = batch[0]
                slots = batch[2].to(dev) if a.slot_emb else None
                with torch.autocast(DEVT, dtype=AMP_DTYPE, enabled=AMP_ON):
                    o = torch.sigmoid(model(x.to(dev), slots=slots).float())
                Po.append(o.cpu().numpy())
        Po = np.concatenate(Po)
        Yo = np.stack([labels[u] for u in oof_ids]).astype(np.float32)
        np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_fold{a.fold}.npz"),
                 pred=Po, truth=Yo, ids=np.array(oof_ids), fold=a.fold, nfolds=a.folds)
        print(f"[cv] wrote raptor_oof_{a.tag}_fold{a.fold}.npz ({len(oof_ids)} studies; "
              f"truth = soft labels)", flush=True)

        # --- top-K epoch OOF (new) ---
        # The epoch-ensemble used to be judged on gold only; that gate is too small to
        # resolve the move. Re-run the held-out fold at each retained epoch instead.
        oof_by_ep = {best_state["epoch"]: Po}
        for t in topk:
            e = int(t["epoch"])
            if e in oof_by_ep:
                continue
            model.load_state_dict(t["model"]); model.eval()
            Pe = []
            with torch.no_grad():
                for batch in ol:
                    x = batch[0]
                    slots = batch[2].to(dev) if a.slot_emb else None
                    with torch.autocast(DEVT, dtype=AMP_DTYPE, enabled=AMP_ON):
                        Pe.append(torch.sigmoid(model(x.to(dev), slots=slots).float()).cpu().numpy())
                Pe = np.concatenate(Pe); oof_by_ep[e] = Pe
            np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_ep{e}_fold{a.fold}.npz"),
                     pred=Pe, truth=Yo, ids=np.array(oof_ids), fold=a.fold, nfolds=a.folds)
            print(f"[cv] wrote top-K epoch OOF ep{e}", flush=True)
        if len(oof_by_ep) > 1:
            Pens = np.mean(list(oof_by_ep.values()), axis=0)
            np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_epens_fold{a.fold}.npz"),
                     pred=Pens, truth=Yo, ids=np.array(oof_ids), fold=a.fold, nfolds=a.folds)
            print(f"[cv] wrote epoch-ensemble OOF over epochs {sorted(oof_by_ep)}", flush=True)
        model.load_state_dict(best_state["model"]); model.eval()

    # --- weight-averaged checkpoint (new) ---
    if len(topk) > 1:
        import copy
        sds = [t["model"] for t in topk]
        avg = {}
        for k in sds[0]:
            v0 = sds[0][k]
            if v0.is_floating_point():
                avg[k] = sum(sd[k].double() for sd in sds).div(len(sds)).to(v0.dtype)
            else:
                avg[k] = v0.clone()          # e.g. num_batches_tracked
        swa_state = {"model": avg, "gold_auc": None, "aucs": {}, "src": src, "res": a.res, "norm": a.norm,
                     "clean_windows": bool(a.clean_windows), "slot_emb": bool(a.slot_emb), "slots": a.slots, "slot_sizes": a.slot_sizes, "roi": a.roi, "roi_mode": a.roi_mode, "roi_pad": a.roi_pad, "roi_overlap": a.roi_overlap, "seq_head": bool(a.seq_head), "zoom_mm": float(a.zoom_mm), "aux_w": float(a.aux_w), "label_weights": os.path.basename(a.label_weights) if a.label_weights else None,
                     "arch": a.arch, "lab": LAB, "epoch": [int(t["epoch"]) for t in topk],
                     "swa_over": [int(t["epoch"]) for t in topk]}
        model.load_state_dict(avg); model.eval()
        au_swa, aucs_swa, P_swa, Y_swa = evaluate()
        swa_state["gold_auc"] = au_swa; swa_state["aucs"] = aucs_swa
        torch.save(swa_state, os.path.join(HERE, f"raptor_ft_{a.tag}_swa.pt"))
        np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}_swa.npz"),
                 pred=P_swa, truth=Y_swa, ids=np.array(gold_ids))
        print(f"SWA over epochs {[int(t['epoch']) for t in topk]} | gold {au_swa:.4f} "
              f"(best-epoch {best:.4f}) {'BETTER' if au_swa > best else 'no gain'}", flush=True)
        if a.folds > 0 and oof_ids:
            ods2 = StudyWindows(HERE, oof_ids, id2row, labels, a.res, a.k_eval, train=False,
                                norm=a.norm, **_roi_kw)
            ol2 = DataLoader(ods2, batch_size=max(2, a.bs // 2), shuffle=False,
                             num_workers=a.workers, collate_fn=collate, persistent_workers=False, worker_init_fn=WIF)
            Ps = []
            with torch.no_grad():
                for batch in ol2:
                    x = batch[0]
                    slots = batch[2].to(dev) if a.slot_emb else None
                    with torch.autocast(DEVT, dtype=AMP_DTYPE, enabled=AMP_ON):
                        Ps.append(torch.sigmoid(model(x.to(dev), slots=slots).float()).cpu().numpy())
            Ps = np.concatenate(Ps)
            np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_swa_fold{a.fold}.npz"),
                     pred=Ps, truth=np.stack([labels[u] for u in oof_ids]).astype(np.float32),
                     ids=np.array(oof_ids), fold=a.fold, nfolds=a.folds)
            print(f"[cv] wrote SWA OOF", flush=True)
        model.load_state_dict(best_state["model"]); model.eval()

    if len(topk) > 1:
        eps = [t["epoch"] for t in topk]
        for rank, t in enumerate(topk):
            if rank == 0:
                continue
            P_t = t.pop("P")
            np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}_ep{t['epoch']}.npz"),
                     pred=P_t, truth=Y, ids=np.array(gold_ids))
            t["P"] = P_t
        Pens = np.mean([t["P"] for t in topk], axis=0)
        ens_aucs = {}
        for j, name in enumerate(LAB):
            if len(set(Y[:, j].astype(int))) > 1:
                ens_aucs[name] = float(roc_auc_score(Y[:, j], Pens[:, j]))
        ens = float(np.mean(list(ens_aucs.values())))
        np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}_epens.npz"),
                 pred=Pens, truth=Y, ids=np.array(gold_ids))
        print(f"TOPK epochs {eps} | best {best:.4f} | epoch-ensemble {ens:.4f} "
              f"({'BETTER' if ens > best else 'no gain'})", flush=True)


if __name__ == "__main__":
    main()