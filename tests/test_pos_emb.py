#!/usr/bin/env python3
"""tests/test_pos_emb.py - --pos_emb: window positions point the right way and start as a no-op.
    .venv/bin/python -I tests/test_pos_emb.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_knee as tk  # noqa: E402


def test_window_pos():
    tk.SLOT_BOUNDS = tk.bounds_from_sizes([40, 40, 36, 36, 40])
    mask = np.zeros(192, np.uint8)
    for s, (a, b) in enumerate(tk.SLOT_BOUNDS):
        mask[a:a + (b - a) - 4] = 1                 # every slot 4 padded slices at its end
    first, last = 0, 40 - 5                         # first / last real sagittal slice (patient left / right)
    assert tk.window_pos(first, mask, +1, False) == -1.0 and tk.window_pos(last, mask, +1, False) == 1.0  # left: lateral first
    assert tk.window_pos(first, mask, -1, False) == 1.0 and tk.window_pos(last, mask, -1, False) == -1.0  # right: medial first
    assert tk.window_pos(first, mask, +1, True) == 1.0                  # mirrored left knee reads like a right knee
    assert tk.window_pos(first, mask, 0, False) == 0.0                  # unknown side: no sagittal position
    c0 = tk.SLOT_BOUNDS[2][0]
    assert tk.window_pos(c0, mask, +1, True) == -1.0 == tk.window_pos(c0, mask, -1, False)  # coronal: front, side-free
    assert abs(tk.window_pos(tk.SLOT_BOUNDS[4][0] + 17, mask, 1, False)) < 0.06            # axial middle ~ 0


def test_pack_and_noop():
    slots = tk.pack_slot_pos([0, 1, 2, 4], [-1.0, 0.0, 0.5, 1.0])
    sid = slots.floor()
    assert sid.long().tolist() == [0, 1, 2, 4]
    assert torch.allclose((slots - sid) / 0.4995 - 1.0, torch.tensor([-1.0, 0.0, 0.5, 1.0]), atol=1e-4)
    a = tk.RaptorClassifier(torch.nn.Identity(), F_dim=8, slot_emb=True, pos_emb=True)
    b = tk.RaptorClassifier(torch.nn.Identity(), F_dim=8, slot_emb=True)
    b.load_state_dict({k: v for k, v in a.state_dict().items() if not k.startswith("pos_proj")})
    a.eval(); b.eval(); feats = torch.randn(1, 4, 8)
    assert torch.allclose(a.head(feats, slots[None]), b.head(feats, slots[None].floor().long()))  # zero-init = old model


if __name__ == "__main__":
    test_window_pos(); test_pack_and_noop()
    print("PASS pos_emb: lateral/medial by knee side and mirror, coronal/axial side-free, packing, zero-init no-op")


def test_fleet_copy_identical():
    import re
    src = lambda p: open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), p)).read()
    tk_src, fl_src = src("train_knee.py"), src("kaggle_base954_fleet/fleet_v6.py")
    for name in ("window_pos", "pack_slot_pos"):
        f = lambda s: re.search(r"def %s\(.*?(?=\n\n\n)" % name, s, re.S).group(0)
        assert f(tk_src) == f(fl_src), name + " drifted between training and Kaggle inference"
    assert "import os, sys, glob, time, math," in fl_src  # the model head uses math.pi


if __name__ == "__main__" and len(sys.argv) == 1:
    test_fleet_copy_identical()
    print("PASS pos_emb fleet copy identical")
