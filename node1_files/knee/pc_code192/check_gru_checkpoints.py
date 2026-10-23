import torch
import glob
import os

pts = glob.glob('/home/s05814802723/projects/knee/*.pt') + glob.glob('/home/s05814802723/projects/knee/pc_code192/*.pt')
print(f"Checking {len(pts)} checkpoints for GRU keys...")
found = []
for p in pts:
    try:
        sd = torch.load(p, map_location='cpu')
        state = sd.get('model', sd.get('state_dict', sd)) if isinstance(sd, dict) else sd
        if isinstance(state, dict):
            if any('gru' in k.lower() for k in state.keys()):
                found.append(p)
    except Exception as e:
        pass

print("Checkpoints with GRU:", found)
