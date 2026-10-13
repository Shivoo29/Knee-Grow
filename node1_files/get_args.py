with open("/home/s05814802723/projects/knee/train_knee.py") as f:
    lines = f.readlines()

in_parser = False
for i, line in enumerate(lines):
    if "ArgumentParser" in line:
        in_parser = True
    if in_parser:
        if "add_argument" in line or "ArgumentParser" in line:
            print(f"L{i}: {line.strip()}")
        if "parse_args" in line:
            print(f"L{i}: {line.strip()}")
            break
