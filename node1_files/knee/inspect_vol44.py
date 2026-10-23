import numpy as np
import pandas as pd
import pydicom
import zipfile
import os
import cv2

KNEE = os.path.expanduser("~/projects/knee")
ZIP_PATH = os.path.join(KNEE, "raw", "rsna-knee-abnormality-detection.zip")

v6 = np.load(os.path.join(KNEE, "raptor_gold_v6.npz"), allow_pickle=True)
gold_id0 = str(v6["ids"][0])

ids_a = np.load(os.path.join(KNEE, "ds/corpus1/all_ids.npy"), allow_pickle=True).astype(str)
idx = np.where(ids_a == gold_id0)[0][0]

vol44 = np.load(os.path.join(KNEE, "ds/corpus1/all_vols.npy"), mmap_mode="r")[idx] # (44, 336, 336)
print("vol44 shape:", vol44.shape)

# Let's inspect slice 20:
print("Slice 20 min, max, mean:", vol44[20].min(), vol44[20].max(), vol44[20].mean())

# Check series for this study
s_df = pd.read_csv(os.path.join(KNEE, "ds/comp/train_series.csv"))
s_rows = s_df[s_df["StudyInstanceUID"].astype(str) == gold_id0]
print("Series available:")
for _, r in s_rows.iterrows():
    print(f"  {r.get('Anatomical_Plane')} | Fluid={r.get('Fluid_Sensitive')} | UID={r.get('SeriesInstanceUID')}")
