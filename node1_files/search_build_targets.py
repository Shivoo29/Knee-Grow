import os

ROOT = "/home/s05814802723/projects/knee"
print("=== DEEP SEARCH FOR build_targets AND base954 ===")
for root, dirs, files in os.walk(ROOT):
    if "venv" in root or ".git" in root or "__pycache__" in root:
        continue
    for f in files:
        if f.endswith(('.py', '.sh', '.md', '.json', '.csv', '.txt', '.ipynb')):
            path = os.path.join(root, f)
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as fp:
                    content = fp.read()
                    if 'build_targets' in content:
                        print(f"FOUND 'build_targets' in: {path}")
                    if 'base954' in content:
                        print(f"FOUND 'base954' in: {path}")
                    elif '954' in content:
                        for line in content.splitlines():
                            if '954' in line and any(w in line for w in ['auc', 'score', 'base', 'lb', 'model', 'blend']):
                                print(f"MATCH 954 in {os.path.basename(path)}: {line.strip()[:100]}")
            except Exception:
                pass
