import torch
import numpy as np
import os
import sys

KNEE = os.path.expanduser("~/projects/knee")
sys.path.insert(0, KNEE)
import train_knee
from sklearn.metrics import roc_auc_score

v6_npz = np.load(os.path.join(KNEE, "raptor_gold_v6.npz"), allow_pickle=True)
gold_ids = v6_npz["ids"].astype(str)
gold_truth = v6_npz["truth"].astype(float)
gold_offline_preds = v6_npz["pred"].astype(float)

# Load model
bb = train_knee.build_backbone("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", pretrained=False)
model = train_knee.RaptorClassifier(bb, F_dim=bb.num_features).cuda().eval()
model.win_chunk = 6
ck = torch.load(os.path.join(KNEE, "raptor_ft_v6.pt"), map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"], strict=True)

# Build vds using exact StudyWindows class from train_knee.py!
all_ids = np.load(os.path.join(KNEE, "all_ids.npy"), allow_pickle=True).astype(str)
id2row = {uid: i for i, uid in enumerate(all_ids)}
dummy_labels = {uid: np.zeros(12, dtype=np.float32) for uid in gold_ids}

vds = train_knee.StudyWindows(
    KNEE, gold_ids, id2row, dummy_labels, res=384, k=94, train=False, norm="imagenet"
)

preds_sw = []
for i in range(len(gold_ids)):
    x, _ = vds[i] # (94, 3, 384, 384)
    x = x.unsqueeze(0).cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = torch.sigmoid(model(x).float()).cpu().numpy()[0]
    preds_sw.append(out)

preds_sw = np.array(preds_sw)
diff = np.abs(preds_sw - gold_offline_preds)
print("Max diff between StudyWindows forward pass and saved raptor_gold_v6.npz:", diff.max())
print("Mean diff:", diff.mean())

aucs = [roc_auc_score(gold_truth[:, j], preds_sw[:, j]) for j in range(12)]
print("Macro AUC using StudyWindows:", np.mean(aucs))
print("Offline saved Macro AUC:", np.mean([roc_auc_score(gold_truth[:, j], gold_offline_preds[:, j]) for j in range(12)]))
