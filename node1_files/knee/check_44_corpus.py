import numpy as np
import os

KNEE = os.path.expanduser("~/projects/knee")
a = np.load(os.path.join(KNEE, "ds/corpus1/all_vols.npy"), mmap_mode="r")
b = np.load(os.path.join(KNEE, "ds/corpus2/extra_vols.npy"), mmap_mode="r")
print("corpus1 shape:", a.shape, a.dtype)
print("corpus2 shape:", b.shape, b.dtype)
m_a = np.load(os.path.join(KNEE, "ds/corpus1/all_masks.npy"))
m_b = np.load(os.path.join(KNEE, "ds/corpus2/extra_masks.npy"))
print("masks shape:", m_a.shape, m_b.shape)

ids_a = np.load(os.path.join(KNEE, "ds/corpus1/all_ids.npy"), allow_pickle=True)
ids_b = np.load(os.path.join(KNEE, "ds/corpus2/extra_ids.npy"), allow_pickle=True)
print("ids total:", len(ids_a) + len(ids_b))
