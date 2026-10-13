import torch
import timm
import sys
import os
import inspect

KNEE = os.path.expanduser("~/projects/knee")
sys.path.insert(0, KNEE)
import train_knee

bb = train_knee.build_backbone("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", pretrained=False)
for name, m in bb.named_modules():
    if "TransformerBlock2d" in type(m).__name__:
        print(f"Module: {name} ({type(m).__name__})")
        for sub_name, sub_m in m.named_children():
            print(f"   child: {sub_name} ({type(sub_m).__name__}, {type(sub_m).__module__})")
        print("\nTransformerBlock2d.forward source:")
        print(inspect.getsource(type(m).forward))
        attn_child = getattr(m, 'attn', None)
        if attn_child is not None:
            print("\nattn.forward source:")
            print(inspect.getsource(type(attn_child).forward))
        break
