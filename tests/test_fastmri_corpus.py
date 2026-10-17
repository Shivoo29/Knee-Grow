#!/usr/bin/env python3
"""tests/test_fastmri_corpus.py - tools/fastmri_corpus.py on synthetic DICOM studies packed as tar.xz.
Needs pydicom and opencv (not in .venv):
    <python with pydicom + cv2> -I tests/test_fastmri_corpus.py
"""
import contextlib
import io
import os
import random
import sys
import tarfile
import tempfile
import time
import zlib

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WT, "tools"))
import fastmri_corpus as fc  # noqa: E402

IOP = {"Sagittal": [0, 1, 0, 0, 0, -1], "Coronal": [1, 0, 0, 0, 0, -1], "Axial": [1, 0, 0, 0, 1, 0]}
NORMAL_AXIS = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


def _dcm(study, series, plane, desc, opts, z, n_img, image_type=("ORIGINAL", "PRIMARY")):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
    ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.Modality = study, series, "MR"
    ds.SeriesDescription, ds.ImageType = desc, list(image_type)
    if opts:
        ds.ScanOptions = opts
    ds.ImageOrientationPatient = IOP[plane]
    pos = [0.0, 0.0, 0.0]
    pos[NORMAL_AXIS[plane]] = 4.0 * z
    ds.ImagePositionPatient, ds.PixelSpacing = pos, [0.5, 0.5]
    ds.Rows = ds.Columns = 320
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    a = np.full((320, 320), 100, np.uint16)
    a[100:220, 40 + 8 * z: 60 + 8 * z] = 3000            # bright bar moves with slice position
    ds.PixelData = a.tobytes()
    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds, enforce_file_format=True)
    return buf.getvalue()


SERIES_A = [("Sagittal", "SAG T2 FS", ["SAT1", "FS"], 12), ("Sagittal", "SAG PD", "", 12),
            ("Coronal", "COR PD FS_FBK", "FS", 12), ("Coronal", "COR PD_FBK", "", 10),
            ("Axial", "AX T2 FS", "FS", 12), ("Sagittal", "3-plane localizer", "", 3), ("Coronal", "COR T1", "", 5)]
SERIES_B = [("Sagittal", "pd_tse_fs_SAG_p2_320", ["SAT1", "FS"], 14), ("Coronal", "pd_tse_fs_COR_p2_320", "FS", 14)]


def _archive(path, studies, cut=None):
    members = []
    for sname, series in studies:
        st = generate_uid()
        for j, (plane, desc, opts, n) in enumerate(series):
            se = generate_uid()
            zs = list(range(n))
            random.Random(j).shuffle(zs)                   # file order != slice order
            itype = ("DERIVED", "SECONDARY", "LOCALIZER") if "localizer" in desc else ("ORIGINAL", "PRIMARY")
            for i, z in enumerate(zs):
                members.append((f"knee_batch/{sname}/s{j}/x/IMG{i:04d}.dcm", _dcm(st, se, plane, desc, opts, z, n, itype)))
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:xz") as tf:
        for name, data in members:
            ti = tarfile.TarInfo(name); ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    data = raw.getvalue()
    open(path, "wb").write(data[:cut] if cut else data)


def _run(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fc.main(argv)
    return out.getvalue()


def test_flags():
    assert fc.is_fat_sat(pydicom.multival.MultiValue(str, ["SAT1", "FS"]), "")
    assert fc.is_fat_sat("", "pd_tse_fs_SAG_p2_320") and fc.is_fat_sat("", "SAG T2 FS")
    assert not fc.is_fat_sat("", "pd_tse_cor_p2_512") and not fc.is_fat_sat("", "AX T2 FSE")
    assert not fc.is_fat_sat(["SAT1"], "COR PD_FBK")
    assert [fc.plane_of(IOP[p]) for p in fc.PLANES] == list(fc.PLANES)
    return "fat-sat from ScanOptions or description (not FSE); plane from orientation"


def test_build_resume_finalize():
    with tempfile.TemporaryDirectory() as td:
        tar, out = os.path.join(td, "a.tar.xz"), os.path.join(td, "corpus")
        _archive(tar, [("studyA", SERIES_A), ("studyB", SERIES_B)])
        _run(["--out", out, "--tar", tar, "--workers", "2"])
        idx = fc.read_index(out)
        assert len(idx) == 2 and all(r["id"].startswith("fm_") for r in idx)
        by_slots = sorted(r["slots"] for r in idx)
        assert by_slots == ["10100", "11111"], by_slots     # B: sag FS + cor FS only
        assert sorted(int(r["n_series"]) for r in idx) == [2, 5]   # localizer and the 5-image series dropped
        ids = np.load(os.path.join(out, "all_ids.npy"), allow_pickle=True)
        masks = np.load(os.path.join(out, "all_masks.npy"))
        assert masks.shape == (2, 192)
        for i, u in enumerate(ids):
            vol = np.frombuffer(zlib.decompress(open(os.path.join(out, "vols", u + ".u8z"), "rb").read()), np.uint8)
            vol = vol.reshape(192, 384, 384)
            assert np.array_equal(masks[i], (vol.reshape(192, -1).sum(1) > 0).astype(np.uint8))
            bar = [int(np.argmax(vol[s, 192, :])) for s in range(40) if masks[i, s]]
            assert bar == sorted(bar) or bar == sorted(bar, reverse=True), "slices not in position order"
        log = _run(["--out", out, "--tar", tar, "--workers", "2"])
        assert "0 new, 2 already there" in log and len(fc.read_index(out)) == 2
    return "2 studies, slot fill 11111 / 10100, localizer + short series dropped, slices sorted, rerun skips"


def test_truncated_archive():
    with tempfile.TemporaryDirectory() as td:
        full, cut, out = os.path.join(td, "f.tar.xz"), os.path.join(td, "c.tar.xz"), os.path.join(td, "corpus")
        _archive(full, [("studyA", SERIES_A), ("studyB", SERIES_B)])
        _archive(cut, [("studyA", SERIES_A), ("studyB", SERIES_B)], cut=int(os.path.getsize(full) * 0.97))
        log = _run(["--out", out, "--tar", cut, "--workers", "2"])
        assert "archive ended early" in log and len(fc.read_index(out)) == 1, log
    return "a cut archive keeps the complete study and skips the partial one"


TESTS = [("flags", test_flags), ("build_resume_finalize", test_build_resume_finalize),
         ("truncated_archive", test_truncated_archive)]


def main():
    fails = 0
    t0 = time.time()
    for name, fn in TESTS:
        t = time.time()
        try:
            print(f"PASS {name} {fn()} ({time.time()-t:.1f}s)", flush=True)
        except Exception as e:
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e} ({time.time()-t:.1f}s)", flush=True)
    print(f"done {len(TESTS)-fails}/{len(TESTS)} pass ({time.time()-t0:.1f}s)", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
