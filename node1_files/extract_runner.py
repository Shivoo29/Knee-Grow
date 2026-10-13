import json
import base64
import re

with open('/home/s05814802723/projects/knee/kaggle_downloads/5_family/rsna-knee-5-family-grand-ensemble.ipynb') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][31]['source'])
for line in src.splitlines():
    if 'b64decode' in line:
        print('Decode line:', line[:100])
        # Find string inside quotes
        m = re.search(r"b64decode\(['\"]([^'\"]+)['\"]\)", line)
        if m:
            raw = base64.b64decode(m.group(1)).decode('utf-8')
            with open('/tmp/runner_5family.py', 'w') as out:
                out.write(raw)
            print('Extracted runner script! Total lines:', len(raw.splitlines()))
            break
        else:
            # Maybe the variable was defined above
            print('b64decode found without inline string. Searching for variable...')
