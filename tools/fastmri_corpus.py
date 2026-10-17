#!/usr/bin/env python3
"""Stream fastMRI knee DICOM archives (tar.xz) into the 192-slice corpus layout, without unpacking.

Every study goes through build_corpus96_stream._process_single_study unchanged (same series
picking, IPP sort, 2-98 percentile window, 140 mm crop at 384 px), so fastMRI volumes match the
competition corpus. The competition series table only says plane and fluid-sensitive (that flag
equals fat suppression there), so each fastMRI series gets the same two fields: the plane from
ImageOrientationPatient, fat suppression from ScanOptions (FS) or the series description.
Localizers and series under 8 images are dropped.

Output folder (resumable: a rerun skips studies already in index.csv):
  vols/<id>.u8z   zlib-compressed uint8 volume, shape (192, 384, 384)
  index.csv       id, series kept, filled slots, slice mask
  all_ids.npy, all_masks.npy, corpus.json   written at the end (or with --finalize);
                                            train_knee.py reads the folder via RAPTOR_EXTRA_CORPUS

    python tools/fastmri_corpus.py --urls ~/projects/knee/fastmri/.urls --archive knee_DICOMs_batch2 \
        --out ~/projects/knee/fastmri/corpus --workers 48
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import pydicom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import build_corpus96_stream as bc  # noqa: E402

SLOT_SIZES = (40, 40, 36, 36, 40)
FS_TOKENS = {"FS", "SFS", "FSA", "FAT", "FATSAT", "SPAIR", "SPIR", "STIR"}
FS_RE = re.compile(r"(?<![a-z])(fs|fat[ _-]?sat|fatsat|stir|spair|spir|fsat|f/s)(?![a-z])", re.I)
LOC_RE = re.compile(r"loc|scout|survey|3[ _-]?pl|tri[ _-]?pilot|calib", re.I)
PLANES = ("Sagittal", "Coronal", "Axial")
MIN_IMAGES = 8


def study_id(key):
    return "fm_" + hashlib.sha1(key.encode()).hexdigest()[:16]


def is_fat_sat(scan_options, desc):
    opts = scan_options if isinstance(scan_options, (list, tuple, pydicom.multival.MultiValue)) \
        else str(scan_options or "").replace("\\", " ").split()
    return bool({str(t).strip().upper() for t in opts} & FS_TOKENS) or bool(FS_RE.search(str(desc or "")))


def plane_of(iop):
    n = np.abs(np.cross(np.asarray(iop[:3], float), np.asarray(iop[3:], float)))
    return PLANES[int(np.argmax(n))]


def series_rows(files):
    """{SeriesInstanceUID: [file names]} and the series table rows build_corpus96_stream expects."""
    groups, info = {}, {}
    for name, data in files.items():
        try:
            d = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True, force=True)
            sid, iop = str(d.SeriesInstanceUID), d.ImageOrientationPatient
        except Exception:
            continue
        desc = str(getattr(d, "SeriesDescription", "") or "")
        if "LOCALIZER" in [str(t).upper() for t in getattr(d, "ImageType", [])] or LOC_RE.search(desc):
            continue
        groups.setdefault(sid, []).append(name)
        if sid not in info:
            info[sid] = (plane_of(iop), int(is_fat_sat(getattr(d, "ScanOptions", ""), desc)))
    keep = {s: f for s, f in groups.items() if len(f) >= MIN_IMAGES}
    rows = [{"SeriesInstanceUID": s, "Anatomical_Plane": info[s][0], "Fluid_Sensitive": info[s][1]}
            for s in sorted(keep, key=lambda s: -len(keep[s]))]          # fullest series first
    return keep, rows


class _MemZip:
    """The one method build_corpus96_stream needs from a ZipFile, served from memory."""
    def __init__(self, files):
        self.files = files

    def read(self, name):
        return self.files[name]


def _init(sizes):
    bc.SLOTS = [(pl, fl, k) for (pl, fl, _), k in zip(bc.SLOTS, sizes)]
    bc.MAXS = sum(sizes)


def process(key, files, out):
    sid = study_id(key)
    file_map, rows = series_rows(files)
    bc._ZIP_HANDLE, bc._FILE_MAP, bc._SERIES_MAP = _MemZip(files), file_map, {sid: rows}
    _, vol, mask, recs = bc._process_single_study(sid)
    tmp = os.path.join(out, "vols", sid + ".u8z.tmp")
    with open(tmp, "wb") as f:
        f.write(zlib.compress(vol.tobytes(), 3))
    os.replace(tmp, os.path.join(out, "vols", sid + ".u8z"))
    return {"id": sid, "n_series": len(rows), "slots": "".join("1" if r["filled"] else "0" for r in recs),
            "mask": "".join(map(str, mask.tolist()))}


def iter_studies(tf, log=print):
    """(study key, {member name: bytes}) for each study directory, in archive order."""
    key, files = None, {}
    try:
        for m in tf:
            if not m.isfile():
                continue
            parts = m.name.split("/")
            k = "/".join(parts[:2]) if len(parts) > 2 else parts[0]
            if k != key and files:
                yield key, files
                files = {}
            key = k
            files[m.name] = tf.extractfile(m).read()
    except (tarfile.TarError, EOFError, OSError) as e:  # truncated stream: drop the partial study
        log(f"[stream] archive ended early ({type(e).__name__}); last study skipped, rerun resumes")
        return
    if files:
        yield key, files


def open_stream(args):
    if args.tar:
        return None, tarfile.open(args.tar, mode="r|*")
    urls = dict(line.split(None, 1) for line in open(os.path.expanduser(args.urls)) if line.strip())
    proc = subprocess.Popen(["curl", "-sfL", "--retry", "5", urls[args.archive].strip()], stdout=subprocess.PIPE)
    return proc, tarfile.open(fileobj=proc.stdout, mode="r|xz")


def read_index(out):
    p = os.path.join(out, "index.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def finalize(out, sizes=SLOT_SIZES):
    rows = [r for r in read_index(out) if os.path.exists(os.path.join(out, "vols", r["id"] + ".u8z"))]
    np.save(os.path.join(out, "all_ids.npy"), np.array([r["id"] for r in rows], dtype=object))
    np.save(os.path.join(out, "all_masks.npy"),
            np.array([[int(c) for c in r["mask"]] for r in rows], dtype=np.uint8).reshape(len(rows), sum(sizes)))
    json.dump({"format": "u8z", "shape": [sum(sizes), bc.IMG, bc.IMG], "slot_sizes": list(sizes)},
              open(os.path.join(out, "corpus.json"), "w"))
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tar", help="local archive (any tar compression)")
    ap.add_argument("--urls", help="file of '<archive name> <url>' lines")
    ap.add_argument("--archive", help="archive name in --urls, e.g. knee_DICOMs_batch2")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new studies (testing)")
    ap.add_argument("--finalize", action="store_true", help="only rewrite all_ids.npy / all_masks.npy from index.csv")
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(a.out, "vols"), exist_ok=True)
    if a.finalize:
        print(f"[finalize] {finalize(a.out)} studies", flush=True)
        return
    _init(SLOT_SIZES)
    done = {r["id"] for r in read_index(a.out)}
    new_index = not os.path.exists(os.path.join(a.out, "index.csv"))
    fh = open(os.path.join(a.out, "index.csv"), "a", newline="")
    w = csv.DictWriter(fh, fieldnames=["id", "n_series", "slots", "mask"])
    if new_index:
        w.writeheader()
    proc, tf = open_stream(a)
    t0, n_new, n_skip, n_err, pending = time.time(), 0, 0, 0, set()

    def drain():
        nonlocal pending, n_new, n_err
        finished, pending = wait(pending, return_when=FIRST_COMPLETED)
        for f in finished:
            try:
                w.writerow(f.result()); n_new += 1
                if n_new % 100 == 0:
                    print(f"{n_new} new | {n_skip} skipped | {n_err} errors | "
                          f"{(time.time() - t0) / n_new:.2f} s/study", flush=True)
            except Exception as e:
                n_err += 1
                print(f"[error] {type(e).__name__}: {e}", flush=True)
        fh.flush()

    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(SLOT_SIZES,)) as ex:
        for key, files in iter_studies(tf):
            if study_id(key) in done:
                n_skip += 1
                continue
            pending.add(ex.submit(process, key, files, a.out))
            if len(pending) >= 2 * a.workers:
                drain()
            if a.limit and n_new + len(pending) >= a.limit:
                break
        while pending:
            drain()
    fh.close()
    if proc:
        proc.kill()
    n = finalize(a.out)
    print(f"[done] {n_new} new, {n_skip} already there, {n_err} errors in {(time.time() - t0) / 60:.1f} min; "
          f"corpus now {n} studies", flush=True)


if __name__ == "__main__":
    main()
