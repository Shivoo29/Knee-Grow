import torch
import timm
import sys
import os
import numpy as np

KNEE = os.path.expanduser("~/projects/knee")
sys.path.insert(0, KNEE)
import train_knee

print("=== Testing CoAtNet-96 Attention Monkey-Patch ===")
bb = train_knee.build_backbone("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", pretrained=False).cuda().eval()
head = train_knee.RaptorClassifier(bb, F_dim=bb.num_features).cuda().eval()
head.win_chunk = 4
ck = torch.load(os.path.join(KNEE, "raptor_ft_v6.pt"), map_location="cpu", weights_only=False)
head.load_state_dict(ck["model"])

# Load study 0 tensor
npz = np.load(os.path.join(KNEE, "preproc_gold_20.npz"))
vols = npz["vols"]
masks = npz["masks"]

mean_t = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
std_t  = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

def get_x(vol, mask):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3: valid = np.arange(3)
    cs = np.linspace(valid.min(), valid.max(), 94).round().astype(int)
    tiles = []
    for c in cs:
        c = max(1, min(c, 94))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        tiles.append(tri)
    wins = np.stack(tiles, 0)
    x = torch.from_numpy(wins).cuda()
    x = (x - mean_t) / std_t
    return x.unsqueeze(0)

x = get_x(vols[0], masks[0])

# Reference BF16
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    out_bf16 = torch.sigmoid(head(x)).cpu().numpy()[0]
print("BF16 reference:", out_bf16[:4], "NaNs:", np.isnan(out_bf16).sum())

# Inspect the backbone attention modules:
from timm.models.maxxvit import RelAttention2d, RelAttention

attn_modules = []
for name, m in head.named_modules():
    if isinstance(m, (RelAttention2d, RelAttention)):
        attn_modules.append((name, m))
print(f"Found {len(attn_modules)} relative attention modules in CoAtNet.")

# Let's inspect the forward method of RelAttention2d
# We patch its forward so the dot product & softmax run in float32:
orig_forward_2d = RelAttention2d.forward

def patched_forward_2d(self, x, shared_rel_attn=None):
    # Run Q, K, V projections in whatever dtype (FP16 under autocast)
    B, C, H, W = x.shape
    qkv = self.qkv(x)
    q, k, v = qkv.chunk(3, dim=1)
    # Reshape for multi-head attention
    # Cast q and k to float32 for dot product & softmax to prevent overflow
    q = q.view(B, self.num_heads, self.head_dim, -1).transpose(-1, -2).float()
    k = k.view(B, self.num_heads, self.head_dim, -1).float()
    v = v.view(B, self.num_heads, self.head_dim, -1).transpose(-1, -2)

    attn = torch.matmul(q, k) * float(self.scale)
    if self.has_rel_pos:
        attn = self._apply_rel_pos(attn, shared_rel_attn)
    attn = torch.softmax(attn, dim=-1)
    if self.attn_drop.p > 0:
        attn = self.attn_drop(attn)

    # Cast attention back to v's dtype for final matmul
    attn = attn.to(v.dtype)
    out = torch.matmul(attn, v).transpose(-1, -2).contiguous().view(B, C, H, W)
    out = self.proj(out)
    out = self.proj_drop(out)
    return out

# Check if RelAttention2d is the exact class
for cls in [RelAttention2d, RelAttention]:
    print(f"Inspecting {cls.__name__}...")
    import inspect
    print("Methods:", [m for m in dir(cls) if "forward" in m])

# Let's see what attention class is actually in coatnet:
if attn_modules:
    first_attn = attn_modules[0][1]
    print("Actual attention class:", type(first_attn))
    import inspect
    print("Source lines:\n", inspect.getsource(type(first_attn).forward))
