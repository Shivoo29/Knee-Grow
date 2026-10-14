#!/usr/bin/env python3
"""tests/test_seq_head.py - CPU-only checks for the bi-GRU sequence head + label weights.

Plain `def test_*` + `__main__` runner (no pytest). Do NOT run training or any
GPU job. Prints only shapes, counts, small scalars and timings (no pixel
arrays, no UIDs, no Report text).

Run from the worktree root with the main checkout's venv:
    /abs/path/.venv/bin/python -I tests/test_seq_head.py
"""
import argparse
import ast
import importlib.util
import inspect
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import types

import numpy as np
import pandas as pd
import torch

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)

# fleet_v6 imports pydicom/cv2 at module top (decode paths only); neither is
# installed in .venv and neither is touched by the head/window code.
for _n in ("pydicom", "pydicom.pixel_data_handlers",
           "pydicom.pixel_data_handlers.util", "cv2"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["pydicom.pixel_data_handlers.util"].apply_modality_lut = lambda *a, **k: None


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tk = _load("wt_train_knee", os.path.join(WT, "train_knee.py"))
fleet = _load("wt_fleet_v6", os.path.join(WT, "kaggle_base954_fleet", "fleet_v6.py"))
blw = _load("wt_build_label_weights", os.path.join(WT, "tools", "build_label_weights.py"))

LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
       'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
NS = {"torch": torch, "nn": torch.nn}


def _legacy_cls():
    src = subprocess.run(["git", "-C", WT, "show", "bd76bdc:train_knee.py"],
                         capture_output=True, text=True, check=True).stdout
    tree = ast.parse(src)
    node = next(n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "RaptorClassifier")
    ns = dict(NS)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<legacy>", "exec"), ns)
    return ns["RaptorClassifier"]


def _tiny_backbone():
    import timm
    return timm.create_model("resnet18", pretrained=False, num_classes=0, in_chans=3)


def test_seq_off_keys():
    """seq_head off: state_dict keys identical to pre-change code (bd76bdc)."""
    Legacy = _legacy_cls()
    old = sorted(Legacy(torch.nn.Identity(), F_dim=16, slot_emb=True).state_dict().keys())
    new = sorted(tk.RaptorClassifier(torch.nn.Identity(), F_dim=16, slot_emb=True,
                                     seq_head=False).state_dict().keys())
    assert old == new, f"{set(old) ^ set(new)}"
    return f"keys={len(old)} identical"


def test_seq_on_order_aware():
    """seq_head on: (B,12) out, permutation changes it, aux returns (B,12)."""
    torch.manual_seed(0)
    B, K, F = 2, 5, 512
    m = tk.RaptorClassifier(_tiny_backbone(), F_dim=F, seq_head=True).eval()
    x = torch.randn(B, K, 3, 64, 64)
    with torch.no_grad():
        o = m(x)
        assert tuple(o.shape) == (B, 12), o.shape
        o_perm = m(x.flip(1))
        d_perm = float((o - o_perm).abs().max())
        assert d_perm > 1e-3, f"order-blind? {d_perm:.2e}"
        main, aux = m.forward_with_aux(x)
        assert tuple(main.shape) == (B, 12) and tuple(aux.shape) == (B, 12)
        assert float((main - o).abs().max()) == 0.0
        m0 = tk.RaptorClassifier(_tiny_backbone(), F_dim=F).eval()
        m0.load_state_dict({k: v for k, v in m.state_dict().items()
                            if not k.startswith(("gru.", "aux_"))}, strict=True)
        # base head on same backbone weights is (up to fp order) permutation-invariant
        with torch.no_grad():
            b0, b0p = m0(x), m0(x.flip(1))
        d_base = float((b0 - b0p).abs().max())
    return f"out=(2,12) perm_diff={d_perm:.2e} base_diff={d_base:.2e} aux_ok"


def test_class_source_identical():
    """train/fleet RaptorClassifier source text identical."""
    a = inspect.getsource(tk.RaptorClassifier)
    b = inspect.getsource(fleet.RaptorClassifier)
    assert a == b, "class drift"
    return f"class chars={len(a)} identical"


def test_ckpt_roundtrip_strict():
    """train-saved seq checkpoint loads in fleet strict=True, same logits."""
    torch.manual_seed(1)
    F = 512
    mt = tk.RaptorClassifier(_tiny_backbone(), F_dim=F, seq_head=True).eval()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "seq.pt")
        torch.save({"model": mt.state_dict(), "seq_head": True, "aux_w": 0.3,
                    "slots": None, "res": 64, "norm": "none"}, p)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        mf = fleet.RaptorClassifier(_tiny_backbone(), F_dim=F,
                                    seq_head=bool(ck.get("seq_head", False))).eval()
        mf.load_state_dict(ck["model"], strict=True)
    x = torch.randn(1, 4, 3, 64, 64)
    with torch.no_grad():
        d = float((mt(x) - mf(x)).abs().max())
    assert d < 1e-6, d
    return f"strict load ok maxdiff={d:.2e}"


def test_seq_compartment_refused():
    """--seq_head + compartment ROI raises ValueError."""
    try:
        tk.StudyWindows(WT, ["u"], {"u": 0}, {"u": np.zeros(12, np.float32)},
                        64, 6, train=False, seq_head=True,
                        roi=True, roi_mode="compartment")
    except ValueError as e:
        assert "ordering" in str(e), str(e)
        return f"ValueError ok ({len(str(e))}ch)"
    raise AssertionError("no ValueError")


def test_weighted_loss():
    """Weighted BCE == plain BCE at all-ones; zero weights ignored."""
    torch.manual_seed(2)
    pw = torch.tensor([1.0, 3.0, 1.5, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5])
    plain = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    none = torch.nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
    logits = torch.randn(2, 12)
    y = (torch.rand(2, 12) > 0.5).float()
    w1 = torch.ones(2, 12)
    d = float(abs(plain(logits, y) - tk.weighted_bce(none, logits, y, w1)))
    assert d < 1e-6, d
    w0 = torch.ones(2, 12)
    w0[0, 0] = 0.0
    got = float(tk.weighted_bce(none, logits, y, w0))
    exp = float((none(logits, y)[1:, :].sum() + none(logits, y)[0, 1:].sum()) / 23)
    assert abs(got - exp) < 1e-6, (got, exp)
    return f"ones_diff={d:.2e} zero_ignored={got:.4f}"


def test_build_label_weights_synthetic():
    """Synthetic teacher/labels pair gives the expected 0/1 mask."""
    uids = ["SYN0", "SYN1", "SYN2"]
    with tempfile.TemporaryDirectory() as td:
        tdir = os.path.join(td, "teacher")
        os.makedirs(tdir)
        for k in range(4):
            # chunk 0 carries the signal; chunks 1-3 are filler (dedupe keeps chunk 0).
            if k == 0:
                vals = {"SYN0": 1.0, "SYN1": 0.0, "SYN2": 0.5}
            else:
                vals = {"SYN0": 0.5, "SYN1": 0.5, "SYN2": 0.5}
            r = pd.DataFrame({"StudyInstanceUID": uids,
                              **{col: [vals[u] for u in uids] for col in LAB}})
            c = pd.DataFrame({"StudyInstanceUID": uids,
                              **{col: [vals[u] for u in uids] for col in LAB}})
            # duplicate SYN0 inside chunk 0 with different values (dedupe keeps first).
            r = pd.concat([r, r.iloc[[0]].assign(**{LAB[0]: 0.0})], ignore_index=True)
            c = pd.concat([c, c.iloc[[0]].assign(**{LAB[0]: 0.0})], ignore_index=True)
            r.to_csv(os.path.join(tdir, f"c{k}_raptor_raw.csv"), index=False)
            c.to_csv(os.path.join(tdir, f"c{k}_compact.csv"), index=False)
        lab = pd.DataFrame({"StudyInstanceUID": uids,
                              **{col: [0.0, 0.1, 0.5] for col in LAB}})
        # SYN0: teacher 1.0 vs label 0.0 -> |d|=1.0 > 0.7 -> 0
        # SYN1: teacher 0.0 vs label 0.1 -> 0.1 -> 1
        # SYN2: teacher 0.5 vs label 0.5 -> 0.0 -> 1
        lab_path = os.path.join(td, "labels.parquet")
        lab.to_parquet(lab_path, index=False)
        w = blw.build_label_weights(tdir, lab_path, 0.7, os.path.join(td, "w.parquet"))
    assert list(w.columns) == [f"w_{c}" for c in LAB], list(w.columns)[:3]
    assert list(w.index) == uids
    assert (w.loc["SYN0"].values == 0.0).all()
    assert (w.loc["SYN1"].values == 1.0).all()
    assert (w.loc["SYN2"].values == 1.0).all()
    return f"rows={len(w)} mask_ok (dedupe kept first)"


TESTS = [("seq_off_keys", test_seq_off_keys),
         ("seq_on_order_aware", test_seq_on_order_aware),
         ("class_source_identical", test_class_source_identical),
         ("ckpt_roundtrip_strict", test_ckpt_roundtrip_strict),
         ("seq_compartment_refused", test_seq_compartment_refused),
         ("weighted_loss", test_weighted_loss),
         ("build_label_weights_synthetic", test_build_label_weights_synthetic)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.parse_args()
    torch.set_num_threads(4)
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
