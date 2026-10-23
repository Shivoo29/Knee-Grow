import json

with open('/home/s05814802723/projects/knee/kaggle_downloads/5_family/rsna-knee-5-family-grand-ensemble.ipynb') as f:
    nb = json.load(f)

for ci, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if '_B64' in src:
        print(f'Found _B64 in cell {ci}')
        for line in src.splitlines():
            if '_B64' in line:
                print(' ', line[:120])
