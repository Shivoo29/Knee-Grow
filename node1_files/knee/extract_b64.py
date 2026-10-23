import json
import base64
import re

with open('/home/s05814802723/projects/knee/kaggle_downloads/5_family/rsna-knee-5-family-grand-ensemble.ipynb') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][31]['source'])
m = re.search(r'_B64\s*=\s*"""(.*?)"""', src, re.DOTALL)
if not m:
    m = re.search(r'_B64\s*=\s*"([^"]+)"', src)

if m:
    raw_b64 = m.group(1).strip()
    decoded = base64.b64decode(raw_b64).decode('utf-8')
    with open('/tmp/runner_5family.py', 'w') as out:
        out.write(decoded)
    print(f'SUCCESS! Decoded {len(decoded.splitlines())} lines into /tmp/runner_5family.py')
else:
    print('Failed to extract _B64 regex!')
