#!/usr/bin/env python3
"""tests/test_radimagenet.py - a RadImageNet ResNet50 checkpoint (torchvision children as nn.Sequential) loads into timm resnet50.
    .venv/bin/python -I tests/test_radimagenet.py
"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_knee as tk  # noqa: E402

src = tk.build_backbone("resnet50")
back = {v: k for k, v in tk.RADIMAGENET_RESNET.items()}
sd = {"backbone." + back[k.split(".", 1)[0]] + "." + k.split(".", 1)[1]: v + 1.0 if v.is_floating_point() else v
      for k, v in src.state_dict().items()}  # RadImageNet layout, values shifted so a silent no-load would show
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "ResNet50.pt")
    torch.save(sd, p)
    bb = tk.build_backbone("resnet50")
    tk.load_raptor(bb, p)
assert all(torch.equal(bb.state_dict()[k], v + 1.0 if v.is_floating_point() else v) for k, v in src.state_dict().items())
print("PASS radimagenet: all 318 tensors land on timm resnet50")
