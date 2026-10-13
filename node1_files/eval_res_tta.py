"""Gold-58 AUC of 96-slice checkpoints at several eval resolutions, with/without mirror TTA.
Uses preproc_gold_58.npz (raw-DICOM preprocessing, 96x384). Same window logic as the Kaggle v6 script."""
import sys, numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from train_knee import build_backbone, RaptorClassifier
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker\x27s","Contusion","Fracture"]
SLOTS = [26, 22, 18, 12, 18]; S_MAP = np.repeat(np.arange(5), SLOTS); K = 94
z = np.load("preproc_gold_58.npz"); vols, masks = z["vols"], z["masks"]
ref = np.load("raptor_gold_v6.npz", allow_pickle=True); ids = ref["ids"]; T = ref["truth"]
dev = "cuda"

def wins(vol, mask, res, mirror):
    v = np.where(mask > 0)[0]; v = v if len(v) >= 3 else np.arange(3)
    lo, hi = int(v.min()), int(v.max()); cs = list(range(lo + 1, hi)) or [max(1, (lo + hi) // 2)]
    cs = [cs[i] for i in np.linspace(0, len(cs) - 1, min(K, len(cs))).round().astype(int)]
    x = torch.from_numpy(np.stack([vol[c - 1:c + 2] for c in cs])).to(dev).float() / 255.0
    if mirror:
        sag = torch.from_numpy(np.isin(S_MAP[cs], (0, 1))).to(dev).view(-1, 1, 1, 1)
        x = torch.where(sag, x.flip(1), x.flip(-1))
    if res != 384:
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    return x

@torch.no_grad()
def run(ck_path):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    bb = build_backbone(ck.get("arch", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"), pretrained=False)
    m = RaptorClassifier(bb, F_dim=bb.num_features).to(dev).eval(); m.load_state_dict(ck["model"])
    out = {}
    for res in (384, 352, 320):
        for mir in (False, True):
            P = []
            for i in range(len(vols)):
                with torch.autocast("cuda", dtype=torch.float16):
                    P.append(torch.sigmoid(m(wins(vols[i], masks[i], res, mir)[None]).float())[0].cpu().numpy())
            out[(res, mir)] = np.array(P)
    for res in (384, 352, 320):
        for name, p in (("plain", out[(res, False)]), ("tta", (out[(res, False)] + out[(res, True)]) / 2)):
            a = [roc_auc_score(T[:, j], p[:, j]) for j in range(12)]
            print(f"{ck_path:28s} res{res} {name:5s} macro {np.mean(a):.4f} | LatOA {a[5]:.3f} PFOA {a[6]:.3f}", flush=True)
    return out

assert (np.load("raptor_gold_v6_s3_swa.npz", allow_pickle=True)["ids"] == ids).all()
for ck in sys.argv[1:]:
    run(ck)
