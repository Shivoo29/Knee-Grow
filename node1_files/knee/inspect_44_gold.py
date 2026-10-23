import numpy as np
import pandas as pd
import os

KNEE = os.path.expanduser("~/projects/knee")
v6 = np.load(os.path.join(KNEE, "raptor_gold_v6.npz"), allow_pickle=True)
gold_id0 = str(v6["ids"][0])

ids_a = np.load(os.path.join(KNEE, "ds/corpus1/all_ids.npy"), allow_pickle=True).astype(str)
ids_b = np.load(os.path.join(KNEE, "ds/corpus2/extra_ids.npy"), allow_pickle=True).astype(str)
all_ids_44 = np.concatenate([ids_a, ids_b])

print("Gold study 0 in 44-slice corpus:", gold_id0 in all_ids_44)
idx = np.where(all_ids_44 == gold_id0)[0][0]
print("Index in 44-slice corpus:", idx)

# Check series_df for this study
s_df = pd.read_csv(os.path.join(KNEE, "ds/comp/train_series.csv"))
s_rows = s_df[s_df["StudyInstanceUID"].astype(str) == gold_id0]
print("\nSeries for this study in train_series.csv:")
for _, r in s_rows.iterrows():
    print(f"  Plane: {r.get('Anatomical_Plane')} | Fluid: {r.get('Fluid_Sensitive')} | Series: {r.get('SeriesInstanceUID')}")

# Load masks
m_a = np.load(os.path.join(KNEE, "ds/corpus1/all_masks.npy"))
m_b = np.load(os.path.join(KNEE, "ds/corpus2/extra_masks.npy"))
all_masks_44 = np.concatenate([m_a, m_b])
print("\nMask for this study (valid slices count):", all_masks_44[idx].sum(), "/ 44")
print("Non-zero mask indices:", np.where(all_masks_44[idx] > 0)[0])
