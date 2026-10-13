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
    print("Stage 0 output Max:", x_s0.abs().max().item())

    cur = x_s0
    for i, blk in enumerate(bb.stages[1].blocks):
        print(f"\n--- Stage 1 Block {i} ({type(blk).__name__}) ---")
        for child_name, child in blk.named_children():
            out = child(cur)
            print(f"  {child_name} ({type(child).__name__}): NaNs={torch.isnan(out).sum().item()}, Max={out.abs().max().item()}")
        cur = blk(cur)
        print(f"Block {i} final output NaNs={torch.isnan(cur).sum().item()}, Max={cur.abs().max().item()}")
        if torch.isnan(cur).any():
            break
