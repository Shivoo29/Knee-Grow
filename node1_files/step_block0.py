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
    
    # Step through forward:
    shortcut = blk.shortcut(x_s0)
    print("shortcut NaNs:", torch.isnan(shortcut).sum().item(), "Max:", shortcut.abs().max().item())
    
    t = blk.pre_norm(x_s0)
    print("pre_norm NaNs:", torch.isnan(t).sum().item(), "Max:", t.abs().max().item())
    
    t = blk.down(t)
    print("down NaNs:", torch.isnan(t).sum().item(), "Max:", t.abs().max().item())
    
    t = blk.conv1_1x1(t)
    print("conv1_1x1 NaNs:", torch.isnan(t).sum().item(), "Max:", t.abs().max().item())
    
    t = blk.norm1(t)
    print("norm1 NaNs:", torch.isnan(t).sum().item(), "Max:", t.abs().max().item())
    
    t = blk.conv2_kxk(t)
    print("conv2_kxk NaNs:", torch.isnan(t).sum().item(), "Max:", t.abs().max().item())
    
    if blk.se_early is not None:
        t = blk.se_early(t)
        print("se_early NaNs:", torch.isnan(t).sum().item())
        
    t = blk.norm2(t)
    print("norm2 NaNs:", torch.isnan(t).sum().item(), "Max:", t.abs().max().item())
    
    if blk.se is not None:
        t = blk.se(t)
        print("se NaNs:", torch.isnan(t).sum().item(), "Max:", t.abs().max().item())
        
    t = blk.conv3_1x1(t)
    print("conv3_1x1 NaNs:", torch.isnan(t).sum().item(), "Max:", t.abs().max().item())
    
    out = blk.drop_path(t) + shortcut
    print("final block output NaNs:", torch.isnan(out).sum().item(), "Max:", out.abs().max().item())
