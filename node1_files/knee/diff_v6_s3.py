import json
import os

with open('raptor_ft_v6.json') as f:
    v6_meta = json.load(f)

print("=== raptor_ft_v6.json stored parameters ===")
for k in ['tag', 'src', 'res', 'epochs', 'bb_lr', 'head_lr', 'n_train', 'n_gold', 'roi', 'roi_mode', 'partial', 'epochs_done']:
    print(f"  {k}: {v6_meta.get(k)}")

print("\n=== v6.log header ===")
if os.path.exists('v6.log'):
    with open('v6.log') as f:
        for i in range(5):
            print(" ", f.readline().strip())

print("\n=== train_coat_v6_s3.log header ===")
if os.path.exists('train_coat_v6_s3.log'):
    with open('train_coat_v6_s3.log') as f:
        for i in range(5):
            print(" ", f.readline().strip())
