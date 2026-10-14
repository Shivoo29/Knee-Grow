import subprocess
import os

TOKEN = os.environ['KAGGLE_API_TOKEN']   # never hard-code; a leaked token was revoked
BASE_DIR = os.path.expanduser('~/projects/knee/kaggle_downloads')
os.makedirs(BASE_DIR, exist_ok=True)

KERNELS = [
    'shivoo29/rsna-knee-clean-rank-blend',
    'shivoo29/rsna-knee-grand-ensemble-v2',
    'shivoo29/rsna-knee-grand-ensemble-v1',
    'shivoo29/rsna-knee-blend-plus-d4',
    'shivoo29/rsna-knee-v6-fast-96-slice-swa',
    'shivoo29/rsna-knee-student-fleet-96-slice',
    'shivoo29/rsna-knee-teacher-probe',
    'shivoo29/rsna-knee-5-family-grand-ensemble',
    'shivoo29/rsna-knee-v5-blend-3-seeds',
    'shivoo29/rsna-knee-v5-blend-2-seeds',
]

env = os.environ.copy()
env['KAGGLE_API_TOKEN'] = TOKEN
KAGGLE_BIN = os.path.expanduser('~/miniconda3/envs/knee/bin/kaggle')

for k in KERNELS:
    slug = k.split('/')[1]
    out_dir = os.path.join(BASE_DIR, slug)
    os.makedirs(out_dir, exist_ok=True)
    print(f'=== Pulling source for {k} ===')
    subprocess.run([KAGGLE_BIN, 'kernels', 'pull', k, '-p', out_dir], env=env)
    print(f'=== Pulling output for {k} ===')
    subprocess.run([KAGGLE_BIN, 'kernels', 'output', k, '-p', out_dir], env=env)

print('=== ALL DOWNLOADS COMPLETE ===')
