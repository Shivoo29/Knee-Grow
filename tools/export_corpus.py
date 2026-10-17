#!/usr/bin/env python3
"""Pack an all_vols.npy corpus into compressed tar shards for upload (Kaggle private dataset).

Each study is stored as <id>.u8d = zlib(level 6) of its row-delta (np.diff along the image width,
uint8 wrap-around), about 1.9x smaller than raw. train_knee.py reads the unpacked folder
(vols/*.u8d + all_ids.npy + all_masks.npy + corpus.json) like any corpus. Shards are tar files
named *.u8dtar so Kaggle stores them as they are; unpack with: tar -xf <shard> -C <corpus>/vols

    python tools/export_corpus.py --src ds/corpus192 --out /path/knee-c192-u8d --shards 16 --workers 4
"""
import argparse
import io
import json
import os
import shutil
import sys
import tarfile
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def encode(vol):
    return zlib.compress(np.diff(vol, axis=-1, prepend=0).astype(np.uint8).tobytes(), 6)


def decode(blob, shape):
    return np.cumsum(np.frombuffer(zlib.decompress(blob), np.uint8).reshape(shape), axis=-1, dtype=np.uint8)


def write_shard(src, out, k, rows, ids):
    vols = np.load(os.path.join(src, "all_vols.npy"), mmap_mode="r")
    path = os.path.join(out, f"vols_{k:02d}.u8dtar")
    with tarfile.open(path + ".tmp", "w") as tf:
        for r in rows:
            blob = encode(np.ascontiguousarray(vols[r]))
            ti = tarfile.TarInfo(f"{ids[r]}.u8d")
            ti.size, ti.mtime = len(blob), 0
            tf.addfile(ti, io.BytesIO(blob))
    os.replace(path + ".tmp", path)
    return k, len(rows), os.path.getsize(path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with all_vols.npy, all_masks.npy, all_ids.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--slot_sizes", default="40,40,36,36,40")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    ids = np.load(os.path.join(a.src, "all_ids.npy"), allow_pickle=True).astype(str)
    vols = np.load(os.path.join(a.src, "all_vols.npy"), mmap_mode="r")
    assert vols.shape[0] == len(ids)
    for f in ("all_ids.npy", "all_masks.npy"):
        shutil.copy(os.path.join(a.src, f), os.path.join(a.out, f))
    json.dump({"format": "u8d", "shape": list(vols.shape[1:]), "slot_sizes": [int(v) for v in a.slot_sizes.split(",")]},
              open(os.path.join(a.out, "corpus.json"), "w"))
    t0 = time.time()
    todo = [(k, rows) for k, rows in enumerate(np.array_split(np.arange(len(ids)), a.shards))
            if not os.path.exists(os.path.join(a.out, f"vols_{k:02d}.u8dtar"))]
    with ProcessPoolExecutor(a.workers) as ex:
        for k, n, size in ex.map(write_shard, *zip(*[(a.src, a.out, k, rows, ids) for k, rows in todo])) if todo else []:
            print(f"shard {k:02d}: {n} studies, {size / 1e9:.2f} GB | {(time.time() - t0) / 60:.1f} min", flush=True)
    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out))
    print(f"EXPORT_DONE {len(ids)} studies, {total / 1e9:.1f} GB (raw {vols.nbytes / 1e9:.1f} GB)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
