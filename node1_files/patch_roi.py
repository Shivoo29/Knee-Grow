target = "/home/s05814802723/projects/knee/train_knee.py"
with open(target, "r", encoding="utf-8") as f:
    code = f.read()

old_str = "rb = np.load(a.roi_boxes)"
new_str = "rb = np.load(a.roi_boxes, allow_pickle=True)"

if old_str in code:
    code = code.replace(old_str, new_str)
    with open(target, "w", encoding="utf-8") as f:
        f.write(code)
    print("SUCCESS: Patched train_knee.py with allow_pickle=True")
else:
    print("ALREADY PATCHED or string not found")
