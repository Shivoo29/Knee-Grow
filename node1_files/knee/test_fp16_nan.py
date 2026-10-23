import torch
import timm
import sys
import os

KNEE = os.path.expanduser("~/projects/knee")
sys.path.insert(0, KNEE)
import train_knee

print("=== Testing CoAtNet-96 FP16 / FP32 Behavior ===")
bb = train_knee.build_backbone("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", pretrained=False).cuda().eval()
head = train_knee.RaptorClassifier(bb, F_dim=bb.num_features).cuda().eval()
head.win_chunk = 4
ck = torch.load(os.path.join(KNEE, "raptor_ft_v6.pt"), map_location="cpu", weights_only=False)
head.load_state_dict(ck["model"])

# Load real preprocessed volumes from preproc_gold_20.npz
cached = torch.load if False else None
import numpy as np
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

# Test on study 0
x = get_x(vols[0], masks[0])
print("Study 0 tensor shape:", x.shape)

# 1. Pure BF16
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    out_bf16 = torch.sigmoid(head(x)).cpu().numpy()
print("BF16 Output:", out_bf16[0][:4], "NaN count:", np.isnan(out_bf16).sum())

# 2. Pure FP16
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    out_fp16 = torch.sigmoid(head(x)).cpu().numpy()
print("FP16 Output:", out_fp16[0][:4], "NaN count:", np.isnan(out_fp16).sum())

# 3. Where does the NaN come from?
with torch.inference_mode():
    flat = x.flatten(0, 1)
    chunk = flat[:14] # 14 windows
    with torch.autocast("cuda", dtype=torch.float16):
        f_fp16 = head.backbone(chunk)
    print("Backbone FP16 NaN count:", torch.isnan(f_fp16).sum().item(), "Max val:", f_fp16.abs().max().item())

    # Full encoding:
    head.win_chunk = 6
    with torch.autocast("cuda", dtype=torch.float16):
        encoded_fp16 = head.encode(x)
    print("Full encode FP16 NaN count:", torch.isnan(encoded_fp16).sum().item(), "Max val:", encoded_fp16.abs().max().item())

    # Now run head in FP16 vs FP32:
    with torch.autocast("cuda", dtype=torch.float16):
        logits_fp16_head = head.head(encoded_fp16)
    print("Head in FP16 NaN count:", torch.isnan(logits_fp16_head).sum().item())

    logits_fp32_head = head.head(encoded_fp16.float())
    print("Head in FP32 (with FP16 backbone feats) NaN count:", torch.isnan(logits_fp32_head).sum().item())
    prob_hybrid = torch.sigmoid(logits_fp32_head.float()).cpu().numpy()
    print("Hybrid (FP16 BB + FP32 Head) Output:", prob_hybrid[0][:4])
    print("Diff vs BF16:", np.abs(prob_hybrid - out_bf16).max())
