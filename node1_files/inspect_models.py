import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

FINDINGS = [
    "acl", "pcl", "mcl", "lcl",
    "medial_meniscus", "lateral_meniscus",
    "medial_oa", "lateral_oa", "pf_oa",
    "effusion", "synovitis", "bakers_cyst",
    "contusion", "fracture"
]

# Let's inspect the files
v6_data = np.load('raptor_gold_v6.npz')
b44_data = np.load('raptor_gold_bilstm_v1_ens.npz')
b96_data = np.load('raptor_gold_bilstm_c96_ens.npz')

print("v6 keys:", list(v6_data.keys()))
print("b44 keys:", list(b44_data.keys()))
print("b96 keys:", list(b96_data.keys()))

y_true = v6_data.get('y_true', v6_data.get('targets', None))
if y_true is None:
    # try other keys
    for k in v6_data.keys():
        if 'true' in k or 'label' in k or 'target' in k:
            y_true = v6_data[k]
            print(f"y_true found as {k}, shape {y_true.shape}")

# Let's find predictions
for name, d in [("v6", v6_data), ("b44", b44_data), ("b96", b96_data)]:
    for k in d.keys():
        arr = d[k]
        if hasattr(arr, 'shape'):
            print(f"{name}[{k}]: shape={arr.shape}, min={np.nanmin(arr):.3f}, max={np.nanmax(arr):.3f}")

# Check cnext if exists
try:
    cnext_data = np.load('raptor_gold_cnext_base_96.npz')
    print("cnext keys:", list(cnext_data.keys()))
    for k in cnext_data.keys():
        arr = cnext_data[k]
        if hasattr(arr, 'shape'):
            print(f"cnext[{k}]: shape={arr.shape}")
except Exception as e:
    print(f"cnext error: {e}")
