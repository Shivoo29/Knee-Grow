import torch
import timm
import sys
import os

KNEE = os.path.expanduser("~/projects/knee")
sys.path.insert(0, KNEE)
import train_knee

bb = train_knee.build_backbone("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", pretrained=False).cuda().eval()
ck = torch.load(os.path.join(KNEE, "raptor_ft_v6.pt"), map_location="cpu", weights_only=False)
bb_state = {k.replace("backbone.", ""): v for k, v in ck["model"].items() if k.startswith("backbone.")}
bb.load_state_dict(bb_state, strict=True)

x = torch.randn(1, 3, 384, 384, device="cuda")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    x_s0 = bb.stages[0](bb.stem(x))
    blk = bb.stages[1].blocks[0]
    
    t = blk.conv2_kxk(blk.norm1(blk.conv1_1x1(blk.down(blk.pre_norm(x_s0)))))
    print("conv2_kxk output NaNs:", torch.isnan(t).sum().item(), "dtype:", t.dtype)
    
    # 1. Native autocast norm2:
    t_fp16 = blk.norm2(t)
    print("Native norm2 NaNs:", torch.isnan(t_fp16).sum().item())
    
    # 2. FP32 norm2:
    with torch.autocast("cuda", enabled=False):
        t_fp32 = blk.norm2(t.float()).half()
    print("FP32 norm2 NaNs:", torch.isnan(t_fp32).sum().item(), "Max:", t_fp32.abs().max().item())
