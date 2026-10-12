"""Build teacher954.ipynb: run the unchanged base954 pipeline on a chunk of TRAIN studies (distillation teacher).

A fake competition root is built in /kaggle/temp: test.csv = one chunk of train study ids (gold-58 first, in chunk 0),
test_series.csv = their rows of train_series.csv, test_series -> symlink to the real train_series DICOM folder.
Both base954 stages are pointed at it through RSNA_COMP_ROOT (two small patches, marked [teacher]).
Outputs kept in /kaggle/working: raptor_raw.csv (Raptor probabilities), ours_sub.csv (compact probabilities),
submission.csv (base954 rank blend, ranks within the chunk).

    python kaggle_teacher954/build_nb.py 0 4    # chunk 0 of 4 -> teacher954.ipynb
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
CHUNK, N_CHUNKS = int(sys.argv[1]), int(sys.argv[2])
nb = json.load(open(HERE.parent / 'kaggle_base954' / 'base954.ipynb'))

FAKE_ROOT = r'''
# [teacher] fake competition root whose "test" set is one chunk of the TRAIN studies
import os
from pathlib import Path
import pandas as pd
CHUNK, N_CHUNKS = %d, %d
REAL = next(p for p in (Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
                        Path('/kaggle/input/rsna-knee-abnormality-detection')) if (p / 'train.csv').is_file())
FAKE = (Path('/kaggle/temp') if Path('/kaggle/temp').is_dir() else Path('/tmp')) / 'fakecomp'
FAKE.mkdir(parents=True, exist_ok=True)
tr = pd.read_csv(REAL / 'train.csv', usecols=['StudyInstanceUID', 'ACL'])   # never load the Report column
tr['StudyInstanceUID'] = tr['StudyInstanceUID'].astype(str)
ids = tr.loc[tr.ACL.notna(), 'StudyInstanceUID'].tolist() + sorted(tr.loc[tr.ACL.isna(), 'StudyInstanceUID'])  # gold first
ids = ids[CHUNK::N_CHUNKS] if CHUNK else [u for i, u in enumerate(ids) if i %% N_CHUNKS == 0 or i < 58]
ids = list(dict.fromkeys(ids))
pd.DataFrame({'StudyInstanceUID': ids}).to_csv(FAKE / 'test.csv', index=False)
ts = pd.read_csv(REAL / 'train_series.csv')
ts[ts.StudyInstanceUID.astype(str).isin(ids)].to_csv(FAKE / 'test_series.csv', index=False)
ss = pd.read_csv(REAL / 'sample_submission.csv', nrows=1)
pd.DataFrame({c: (ids if c == 'StudyInstanceUID' else 0.5) for c in ss.columns}).to_csv(FAKE / 'sample_submission.csv', index=False)
if not (FAKE / 'test_series').exists():
    (FAKE / 'test_series').symlink_to(REAL / 'train_series')
assert all((FAKE / 'test_series' / u).is_dir() for u in ids[:20]), 'train_series layout differs'
os.environ['RSNA_COMP_ROOT'] = str(FAKE)
print('TEACHER_CHUNK', CHUNK, 'of', N_CHUNKS, '| studies', len(ids), '| gold in chunk', int(tr.ACL.notna().sum()) if CHUNK == 0 else 0, flush=True)
''' % (CHUNK, N_CHUNKS)

def patch(cell_idx, old, new):
    src = ''.join(nb['cells'][cell_idx]['source'])
    assert src.count(old) == 1, (cell_idx, old[:50])
    nb['cells'][cell_idx]['source'] = src.replace(old, new)

patch(2, 'def find_test_root():\n',
      'def find_test_root():\n    if os.environ.get("RSNA_COMP_ROOT"):  # [teacher]\n        return os.environ["RSNA_COMP_ROOT"]\n')
patch(4, "DATA = find_mount(COMPETITION, 'test.csv')",
      "DATA = Path(os.environ['RSNA_COMP_ROOT']) if os.environ.get('RSNA_COMP_ROOT') else find_mount(COMPETITION, 'test.csv')  # [teacher]")
code = {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': FAKE_ROOT}
first_code = next(i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code')
nb['cells'].insert(first_code, code)
json.dump(nb, open(HERE / 'teacher954.ipynb', 'w'), indent=1)
print('chunk', CHUNK, 'of', N_CHUNKS, '| cells', len(nb['cells']))
