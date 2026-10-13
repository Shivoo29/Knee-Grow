import json

with open('/home/s05814802723/projects/knee/kaggle_downloads/5_family/rsna-knee-5-family-grand-ensemble.ipynb') as f:
    nb = json.load(f)

for i, l in enumerate(nb['cells'][31]['source'][:15]):
    print(f'{i}: {l[:80]}')
