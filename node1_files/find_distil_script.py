import os
import glob

ROOT = "/home/s05814802723/projects/knee"
py_files = glob.glob(f"{ROOT}/*.py")
print("=== PYTHON SCRIPTS MENTIONING DISTIL ===")
for p in py_files:
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            if "distil" in content.lower():
                print(f"\n--- {os.path.basename(p)} ---")
                for line in content.splitlines():
                    if "distil" in line.lower() or "teacher" in line.lower() or "labels_avg" in line.lower():
                        print(f"  {line}")
    except Exception as e:
        pass
