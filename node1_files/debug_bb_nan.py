import torch
import timm
import sys
import os
import numpy as np
from timm.models.maxxvit import Attention2d

KNEE = os.path.expanduser("~/projects/knee")
sys.path.insert(0, KNEE)
import train_knee

# Check if Attention2d is even being called!
call_count = 0
orig_forward = Attention2d.forward

def patched_attention_forward(self, x: torch.Tensor, shared_rel_pos=None) -> torch.Tensor:
    global call_count
    call_count += 1
    B, C, H, W = x.shape

    if self.head_first:
        q, k, v = self.qkv(x).view(B, self.num_heads, self.dim_head * 3, -1).chunk(3, dim=2)
    else:
        q, k, v = self.qkv(x).reshape(B, 3, self.num_heads, self.dim_head, -1).unbind(1)

    # Compute attention dot-product and softmax in float32 to prevent FP16 overflow!
    q_f32 = q.float() * float(self.scale)
    k_f32 = k.float()
    attn = q_f32.transpose(-2, -1) @ k_f32
    if self.rel_pos is not None:
        attn = self.rel_pos(attn)
    elif shared_rel_pos is not None:
        attn = attn + shared_rel_pos.float()
    attn = attn.softmax(dim=-1).to(v.dtype)
    x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)
    x = self.proj(x)
    return x

Attention2d.forward = patched_attention_forward

bb = train_knee.build_backbone("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", pretrained=False).cuda().eval()
ck = torch.load(os.path.join(KNEE, "raptor_ft_v6.pt"), map_location="cpu", weights_only=False)
bb_state = {k.replace("backbone.", ""): v for k, v in ck["model"].items() if k.startswith("backbone.")}
bb.load_state_dict(bb_state, strict=True)

# Test single chunk of 12 slices
dummy = torch.randn(12, 3, 384, 384, device="cuda")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    out = bb(dummy)

print("Attention2d called count:", call_count)
print("Backbone out NaNs with dummy:", torch.isnan(out).sum().item())

# Test with real slice chunk
npz = np.load(os.path.join(KNEE, "preproc_gold_20.npz"))
vol = npz["vols"][0]
tri = np.stack([vol[0], vol[1], vol[2]], 0).astype(np.float32) / 255.0
real_input = torch.from_numpy(tri).cuda().unsqueeze(0) # (1, 3, 384, 384)
mean_t = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
std_t  = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
real_input = (real_input - mean_t) / std_t

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    out_real = bb(real_input)

print("Backbone out NaNs with real input:", torch.isnan(out_real).sum().item())
print("Backbone max value:", out_real.abs().max().item())
