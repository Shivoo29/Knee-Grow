import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = "/home/s05814802723/projects/knee"
roi_path = f"{ROOT}/roi_boxes.npz"

print("=== INSPECTING ROI_BOXES.NPZ ===")
if os.path.exists(roi_path):
    data = np.load(roi_path, allow_pickle=True)
    print("Keys in roi_boxes.npz:", list(data.files))
    for k in data.files:
        arr = data[k]
        print(f"  {k:20s}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")
        if len(arr) > 0:
            print(f"    sample [0, 0]: {arr[0, 0] if len(arr.shape) > 1 else arr[0]}")

print("\n=== INSPECTING ROI_LOCALIZE.PY ===")
loc_path = f"{ROOT}/roi_localize.py"
if os.path.exists(loc_path):
    with open(loc_path, "r") as f:
        print(f.read())
