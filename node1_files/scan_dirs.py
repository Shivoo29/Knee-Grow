import os, glob

KNEE = "/home/s05814802723/projects/knee"
print("Scanning directories under projects/knee:")
for item in os.listdir(KNEE):
    full = os.path.join(KNEE, item)
    if os.path.isdir(full):
        sub_count = len(os.listdir(full))
        print(f"  Dir: {item} (items: {sub_count})")

if os.path.exists("/home/s05814802723/rsna"):
    print("\nScanning /home/s05814802723/rsna:")
    for item in os.listdir("/home/s05814802723/rsna"):
        full = os.path.join("/home/s05814802723/rsna", item)
        print(f"  Item: {item}")
