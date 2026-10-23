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
    x_stem = bb.stem(x)
    print("Stem output NaNs:", torch.isnan(x_stem).sum().item(), "Max:", x_stem.abs().max().item())

    x_s0 = bb.stages[0](x_stem)
    print("Stage 0 output NaNs:", torch.isnan(x_s0).sum().item(), "Max:", x_s0.abs().max().item())

    x_s1 = bb.stages[1](x_s0)
    print("Stage 1 output NaNs:", torch.isnan(x_s1).sum().item(), "Max:", x_s1.abs().max().item())

    # Stage 2: let's test block by block!
    cur = x_s1
    for i, blk in enumerate(bb.stages[2].blocks):
        cur = blk(cur)
        has_nan = torch.isnan(cur).sum().item()
        print(f"Stage 2 Block {i} NaNs: {has_nan} | Max: {cur.abs().max().item()}")
        if has_nan > 0:
            # Inspect inside this block
            print(f"--> Found first NaN in Stage 2 Block {i}! Breaking.")
            break
