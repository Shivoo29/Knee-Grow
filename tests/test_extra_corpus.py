#!/usr/bin/env python3
"""tests/test_extra_corpus.py - RAPTOR_EXTRA_CORPUS in train_knee.py and tools/predict_corpus.py (CPU, synthetic).
    .venv/bin/python -I tests/test_extra_corpus.py
"""
import contextlib
import io
import json
import os
import sys
import tarfile
import tempfile
import time
import zlib

import numpy as np
import pandas as pd
import torch

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
sys.path.insert(0, os.path.join(WT, "tools"))
import train_knee as tk  # noqa: E402
import predict_corpus as pc  # noqa: E402
import export_corpus as ec  # noqa: E402

SHAPE = (192, 384, 384)


def _vol(seed):
    v = np.random.RandomState(seed).randint(0, 256, SHAPE).astype(np.uint8)
    v[180:] = 0                                        # padded tail slices, as in real studies
    return v


def _npy_corpus(d, ids, seeds):
    os.makedirs(d, exist_ok=True)
    v = np.stack([_vol(s) for s in seeds])
    np.save(os.path.join(d, "all_vols.npy"), v)
    np.save(os.path.join(d, "all_masks.npy"), (v.reshape(len(ids), SHAPE[0], -1).sum(-1) > 0).astype(np.uint8))
    np.save(os.path.join(d, "all_ids.npy"), np.array(ids, dtype=object))
    return v


def _z_corpus(d, ids, seeds, fmt="u8z"):
    os.makedirs(os.path.join(d, "vols"), exist_ok=True)
    vs = [_vol(s) for s in seeds]
    for u, v in zip(ids, vs):
        blob = ec.encode(v) if fmt == "u8d" else zlib.compress(v.tobytes(), 3)
        open(os.path.join(d, "vols", f"{u}.{fmt}"), "wb").write(blob)
    np.save(os.path.join(d, "all_ids.npy"), np.array(ids, dtype=object))
    np.save(os.path.join(d, "all_masks.npy"), np.stack([(v.reshape(SHAPE[0], -1).sum(-1) > 0).astype(np.uint8) for v in vs]))
    json.dump({"format": fmt, "shape": list(SHAPE)}, open(os.path.join(d, "corpus.json"), "w"))
    return vs


@contextlib.contextmanager
def _env(here, extra):
    old_here, old_env = tk.HERE, os.environ.get("RAPTOR_EXTRA_CORPUS")
    tk.HERE, tk._CORPUS = here, None
    os.environ["RAPTOR_EXTRA_CORPUS"] = ",".join(extra)
    try:
        yield
    finally:
        tk.HERE, tk._CORPUS = old_here, None
        if old_env is None:
            os.environ.pop("RAPTOR_EXTRA_CORPUS", None)
        else:
            os.environ["RAPTOR_EXTRA_CORPUS"] = old_env


def test_corpus_alignment():
    with tempfile.TemporaryDirectory() as td:
        main = _npy_corpus(os.path.join(td, "main"), ["c0", "c1"], [0, 1])
        ext = _z_corpus(os.path.join(td, "fm"), ["fm_a", "fm_b", "fm_c"], [10, 11, 12])
        with _env(os.path.join(td, "main"), [os.path.join(td, "fm")]):
            ids = tk._open_ids()
            vols, masks = tk._open_corpus()
            assert list(ids) == ["c0", "c1", "fm_a", "fm_b", "fm_c"] and vols.shape == (5,) + SHAPE
            for row, ref in enumerate(list(main) + ext):
                assert np.array_equal(vols[row], ref), row
            assert masks.shape == (5, 192) and int(masks[3].sum()) == 180
            ds = tk.StudyWindows(td, list(ids), {u: i for i, u in enumerate(ids)},
                                 {u: np.zeros(12, np.float32) for u in ids}, 32, 6, train=False)
            assert tuple(ds[3][0].shape) == (6, 3, 32, 32)
    return "ids, volumes and masks line up across an npy part and a u8z part; StudyWindows reads both"


def test_u8d_main_corpus_and_export():
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "c192")
        ref = _npy_corpus(src, ["c0", "c1", "c2", "c3", "c4"], [0, 1, 2, 3, 4])
        out = os.path.join(td, "export")
        with contextlib.redirect_stdout(io.StringIO()):
            ec.main(["--src", src, "--out", out, "--shards", "2", "--workers", "2"])
        shards = sorted(f for f in os.listdir(out) if f.endswith(".u8dtar"))
        assert shards == ["vols_00.u8dtar", "vols_01.u8dtar"]
        run = os.path.join(td, "run")                          # a run dir as the cloud script builds it
        os.makedirs(os.path.join(run, "vols"))
        for s in shards:
            with tarfile.open(os.path.join(out, s)) as tf:
                tf.extractall(os.path.join(run, "vols"), filter="data")
        for f in ("all_ids.npy", "all_masks.npy", "corpus.json"):
            os.symlink(os.path.join(out, f), os.path.join(run, f))
        size = sum(os.path.getsize(os.path.join(out, s)) for s in shards)
        with _env(run, []):
            ids = tk._open_ids()
            vols, masks = tk._open_corpus()
            assert list(ids) == ["c0", "c1", "c2", "c3", "c4"] and vols.fmt == "u8d"
            for row in range(5):
                assert np.array_equal(vols[row], ref[row]), row
    return f"export -> 2 shards ({ref.nbytes / size:.2f}x of raw on random data) -> unpack -> main corpus byte-exact"


def test_add_extra():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "teacher.parquet")
        df = pd.DataFrame({"StudyInstanceUID": ["fm_a", "fm_c", "fm_zz"]})
        for j, c in enumerate(tk.LAB):
            df[c] = 0.1 * (j % 9)
        df.to_parquet(p, index=False)
        labels = {"c0": np.ones(12, np.float32)}
        ids = np.array(["c0", "c1", "fm_a", "fm_b", "fm_c"])
        tr, lw = tk.add_extra(["c0"], labels, None, ids, {"c0", "c1"}, p, 0.5)
        assert tr == ["c0", "fm_a", "fm_c"] and np.allclose(labels["fm_c"][:3], [0.0, 0.1, 0.2])
        assert np.all(lw["c0"] == 1) and np.all(lw["fm_a"] == 0.5)
        tr2, lw2 = tk.add_extra(["c0"], dict(labels), {"c0": np.full(12, 0.7, np.float32)}, ids, {"c0", "c1"}, p, 2.0, limit=1)
        assert tr2 == ["c0", "fm_a"] and np.all(lw2["c0"] == np.float32(0.7)) and np.all(lw2["fm_a"] == 2)
        try:
            tk.add_extra(["c0"], labels, None, np.array(["c0"]), {"c0"}, p, 1.0)
        except SystemExit:
            return "extras appended after competition ids, own weight, --limit honoured, no match refused"
    raise AssertionError("no matching extra ids accepted")


def _ckpt(path, seed):
    torch.manual_seed(seed)
    bb = tk.build_backbone("resnet10t", pretrained=False)
    m = tk.RaptorClassifier(bb, F_dim=bb.num_features, slot_emb=True)
    torch.save({"model": m.state_dict(), "arch": "resnet10t", "res": 32, "norm": "imagenet", "clean_windows": True,
                "slot_emb": True, "slots": None, "slot_sizes": "40,40,36,36,40", "zoom_mm": 100.0, "seq_head": False}, path)
    return m


def test_predict_corpus():
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "fm")
        _z_corpus(d, ["fm_a", "fm_b", "fm_c"], [10, 11, 12])
        m1, m2 = _ckpt(os.path.join(td, "t1.pt"), 1), _ckpt(os.path.join(td, "t2.pt"), 2)
        out = os.path.join(td, "teacher.parquet")
        argv = ["--corpus", d, "--ckpts", f"{td}/t1.pt,{td}/t2.pt", "--out", out, "--k_eval", "8", "--workers", "0"]
        with contextlib.redirect_stdout(io.StringIO()):
            pc.main(argv)
        df = pd.read_parquet(out)
        assert list(df["StudyInstanceUID"]) == ["fm_a", "fm_b", "fm_c"] and (df["n_ckpts"] == 2).all()
        vols, masks, ids = tk._extra_part(d)
        tk._CORPUS = (vols, masks)
        ds = tk.StudyWindows(td, list(ids), {u: i for i, u in enumerate(ids)}, {u: np.zeros(12, np.float32) for u in ids},
                             32, 8, train=False, norm="imagenet", clean_windows=True, slot_emb=True, zoom_px=tk.zoom_px(100))
        with torch.no_grad():
            ref = np.mean([[torch.sigmoid(m.eval()(ds[i][0][None], slots=ds[i][2][None]))[0].numpy() for i in range(3)]
                           for m in (m1, m2)], axis=0)
        tk._CORPUS = None
        assert np.allclose(df[tk.LAB].values, ref, atol=1e-5), np.abs(df[tk.LAB].values - ref).max()
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            pc.main(argv)
        assert log.getvalue().count("[cache]") == 2
    return "mean of 2 checkpoints == direct StudyWindows + model forward; rerun uses the per-checkpoint cache"


def test_calibration():
    z = torch.linspace(-12, 12, 200001)
    for w in (1.0, 2.5, 10.0):
        for q in (0.03, 0.2, 0.5, 0.9):
            loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w), reduction="none")(z, torch.full_like(z, q))
            p = float(torch.sigmoid(z[loss.argmin()]))           # what a model trained on soft label q converges to
            assert abs(p - w * q / (w * q + 1 - q)) < 1e-3, (w, q, p)
            assert abs(pc.uninflate(p, w) - q) < 1e-3, (w, q)
    with tempfile.TemporaryDirectory() as td:
        df = pd.DataFrame({c: [0.6, 0.0, 0.0, 0.0] for c in tk.LAB})
        df["StudyInstanceUID"] = list("abcd")
        df.to_parquet(os.path.join(td, "l.parquet"))
        assert np.allclose(pc.pos_weight(os.path.join(td, "l.parquet")), (1 - 0.15) / 0.15)
    return "BCE pos_weight optimum is wq/(wq+1-q); uninflate recovers q; pos_weight matches train_knee"


TESTS = [("corpus_alignment", test_corpus_alignment), ("u8d_main_corpus_and_export", test_u8d_main_corpus_and_export),
         ("add_extra", test_add_extra), ("predict_corpus", test_predict_corpus), ("calibration", test_calibration)]


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
