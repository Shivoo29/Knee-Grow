import torch
import timm
import sys
import os
import time
import numpy as np
from timm.layers.norm_act import BatchNormAct2d
from timm.models.maxxvit import Attention2d

KNEE = os.path.expanduser("~/projects/knee")
sys.path.insert(0, KNEE)
import train_knee

# 1. Patch BatchNormAct2d to run batch norm in FP32:
orig_bn_forward = BatchNormAct2d.forward
def patched_bn_forward(self, x):
    # Run batch norm in float32 to prevent FP16 variance/mean overflow
    x_dtype = x.dtype
    with torch.autocast("cuda", enabled=False):
        out = orig_bn_forward(self, x.float())
    return out.to(x_dtype)
BatchNormAct2d.forward = patched_bn_forward

# 2. Patch Attention2d to run dot-product & softmax in FP32:
orig_attn_forward = Attention2d.forward
def patched_attn_forward(self, x: torch.Tensor, shared_rel_pos=None) -> torch.Tensor:
    B, C, H, W = x.shape
    if self.head_first:
        q, k, v = self.qkv(x).view(B, self.num_heads, self.dim_head * 3, -1).chunk(3, dim=2)
    else:
        q, k, v = self.qkv(x).reshape(B, 3, self.num_heads, self.dim_head, -1).unbind(1)

    q_f32 = q.float() * float(self.scale)
    k_f32 = k.float()
    attn = q_f32.transpose(-2, -1) @ k_f32
    if self.rel_pos is not None:
        attn = self.rel_pos(attn)
    elif shared_rel_pos is not None:
        attn = attn + shared_rel_pos.float()
    attn = attn.softmax(dim=-1).to(v.dtype)
    if self.attn_drop.p > 0:
        attn = self.attn_drop(attn)
    x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)
    x = self.proj(x)
    return x
Attention2d.forward = patched_attn_forward

print("✓ Patched BatchNormAct2d and Attention2d for safe FP16 inference.")

# Load model
bb = train_knee.build_backbone("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", pretrained=False).cuda().eval()
head = train_knee.RaptorClassifier(bb, F_dim=bb.num_features).cuda().eval()
head.win_chunk = 4
ck = torch.load(os.path.join(KNEE, "raptor_ft_v6.pt"), map_location="cpu", weights_only=False)
head.load_state_dict(ck["model"])

# Load 20 gold studies
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

# Evaluate all 20 studies in BF16 vs Patched FP16
preds_bf16 = []
preds_fp16 = []
nan_bf16 = 0
nan_fp16 = 0

t0 = time.perf_counter()
for i in range(20):
    x = get_x(vols[i], masks[i])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = torch.sigmoid(head(x)).cpu().numpy()[0]
    preds_bf16.append(out)
    nan_bf16 += int(np.isnan(out).sum())
t_bf16 = time.perf_counter() - t0

t0 = time.perf_counter()
for i in range(20):
    x = get_x(vols[i], masks[i])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        out = torch.sigmoid(head(x)).cpu().numpy()[0]
    preds_fp16.append(out)
    nan_fp16 += int(np.isnan(out).sum())
t_fp16 = time.perf_counter() - t0

preds_bf16 = np.array(preds_bf16)
preds_fp16 = np.array(preds_fp16)
diff = np.abs(preds_fp16 - preds_bf16)

from scipy.stats import pearsonr
corrs = [pearsonr(preds_fp16[:, j], preds_bf16[:, j])[0] for j in range(12)]

print("\n" + "="*70)
print("HYBRID FP16 BENCHMARK RESULTS (ALL 20 GOLD STUDIES):")
print("="*70)
print(f"BF16 Time (20 studies):     {t_bf16:.2f}s ({t_bf16/20*1000:.1f} ms/study) | NaNs: {nan_bf16}")
print(f"Patched FP16 (20 studies):  {t_fp16:.2f}s ({t_fp16/20*1000:.1f} ms/study) | NaNs: {nan_fp16}")
print(f"Max Absolute Diff vs BF16:  {diff.max():.6f}")
print(f"Mean Absolute Diff vs BF16: {diff.mean():.6f}")
print(f"Mean Pearson Correlation:   {np.mean(corrs):.6f}")
print("="*70)
