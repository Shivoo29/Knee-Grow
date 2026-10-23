import torch
import timm
import sys
import os

KNEE = os.path.expanduser("~/projects/knee")
sys.path.insert(0, KNEE)
import train_knee

bb = train_knee.build_backbone("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", pretrained=False)
types = set(type(m) for m in bb.modules())
print("Unique module classes in CoAtNet:")
for t in sorted(types, key=lambda x: x.__name__):
    if any(k in t.__name__.lower() for k in ["attn", "rel", "self", "trans", "pool"]):
        print(" ", t.__name__, t.__module__)
