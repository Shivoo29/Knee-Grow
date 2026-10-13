"""Build two notebooks that run the distilled students (CoAtNet version 6, 96-slice) on the hidden test set.

  students_only.ipynb       rank-average of the students, nothing else      (tells us how good the students are alone)
  base954_students.ipynb    unchanged base954 + students as a rank branch   (weight W, default 0.15)

Uses the same proven inference script as the fleet (fleet_v6.py, K_EVAL=94, RAPTOR_CKPT picks the checkpoint).
Weights come from the Kaggle dataset shivoo29/knee-students (raptor_ft_v6d_s7_swa.pt, raptor_ft_v6d_s8_swa.pt).

    python kaggle_students/build_nb.py [W]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
W = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
CKPTS = {'s7': 'raptor_ft_v6d_s7_swa.pt', 's8': 'raptor_ft_v6d_s8_swa.pt'}
FLEET_V6 = (ROOT / 'kaggle_base954_fleet' / 'fleet_v6.py').read_text()
code = lambda s: {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s}

RUN = r'''
def find_one(name):
    hits = glob.glob('/kaggle/input/*/' + name) + glob.glob('/kaggle/input/datasets/*/*/' + name)  # no recursive walk over test DICOMs
    assert len(hits) >= 1, name + ' not attached'
    return hits[0]

def pct(v):
    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values

STOP_AFTER_H = 7.3   # no new student starts after this
SCRIPTS = {%s}
outs = {}
for tag, ckpt in SCRIPTS.items():
    elapsed_h = (time.time() - NB_T0) / 3600
    if elapsed_h > STOP_AFTER_H:
        print('STUDENT_SKIPPED %%s: %%.2f h elapsed' %% (tag, elapsed_h), flush=True)
        continue
    path = Path('/tmp/student_%%s.py' %% tag)
    path.write_text(FLEET_V6)
    out = WORK / ('student_%%s.csv' %% tag)
    env = dict(os.environ, RSNA_COMP_ROOT=str(DATA), RAPTOR_OUTPUT_CSV=str(out), RAPTOR_CKPT=ckpt)
    t = time.time()
    rc = subprocess.run([sys.executable, '-u', str(path)], env=env, cwd='/tmp').returncode
    if rc != 0:
        raise RuntimeError('student %%s failed with exit code %%d' %% (tag, rc))
    print('STUDENT_DONE %%s %%.0fs' %% (tag, time.time() - t), flush=True)
    outs[tag] = pd.read_csv(out, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[test_ids, labels]
''' % ', '.join("'%s': find_one('%s')" % kv for kv in CKPTS.items())

# ---- students only -------------------------------------------------------------------------------------------
ONLY_HEAD = r'''
import glob, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
NB_T0 = time.time()
WORK = Path('/kaggle/working')
DATA = next(Path(p) for p in ('/kaggle/input/competitions/rsna-knee-abnormality-detection',
                              '/kaggle/input/rsna-knee-abnormality-detection') if (Path(p) / 'test.csv').is_file())
test_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
cols = list(pd.read_csv(DATA / 'sample_submission.csv', nrows=1).columns)
labels = cols[1:]
n = len(test_ids)
'''
ONLY_TAIL = r'''
assert len(outs) >= 1, 'no student finished'
final = np.column_stack([np.mean([pct(o[c].values) for o in outs.values()], axis=0) for c in labels])
sub = pd.DataFrame(final.astype(np.float32), columns=labels)
sub.insert(0, 'StudyInstanceUID', test_ids)
sub = sub[cols]
assert np.isfinite(sub[labels].values).all() and ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()
sub.to_csv(WORK / 'submission.csv', index=False)
print('STUDENTS_ONLY_COMPLETE', json.dumps({'students': list(outs), 'rows': n, 'seconds': round(time.time() - NB_T0, 1)}), flush=True)
'''
nb_only = {'cells': [code('FLEET_V6 = %r\n' % FLEET_V6), code(ONLY_HEAD + RUN + ONLY_TAIL)],
           'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}},
           'nbformat': 4, 'nbformat_minor': 5}
(HERE / 'students_only').mkdir(exist_ok=True)
json.dump(nb_only, open(HERE / 'students_only' / 'students_only.ipynb', 'w'), indent=1)

# ---- base954 + students --------------------------------------------------------------------------------------
BLEND_HEAD = r'''
import glob, shutil
shutil.copyfile(WORK / 'submission.csv', WORK / 'base954_sub.csv')
W_FLEET = %s
''' % W
BLEND_TAIL = r'''
if not outs:
    print('STUDENTS_NONE: keeping base954 submission', flush=True)
else:
    base = pd.read_csv(WORK / 'base954_sub.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[test_ids, labels]
    final = np.column_stack([(1 - W_FLEET) * pct(base[c].values) + W_FLEET * np.mean([pct(o[c].values) for o in outs.values()], axis=0)
                             for c in labels])
    sub = pd.DataFrame(final.astype(np.float32), columns=labels)
    sub.insert(0, 'StudyInstanceUID', test_ids)
    sub = sub[cols]
    assert sub['StudyInstanceUID'].tolist() == test_ids and np.isfinite(sub[labels].values).all()
    assert ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()
    sub.to_csv(WORK / 'submission.csv', index=False)
    print('BASE954_STUDENTS_COMPLETE', json.dumps({'w': W_FLEET, 'students': list(outs), 'seconds': round(time.time() - NB_T0, 1)}), flush=True)
    display(sub.head())
'''
nb = json.load(open(ROOT / 'kaggle_base954' / 'base954.ipynb'))
nb['cells'] += [code('FLEET_V6 = %r\n' % FLEET_V6), code(BLEND_HEAD + RUN + BLEND_TAIL)]
(HERE / 'base954_students').mkdir(exist_ok=True)
json.dump(nb, open(HERE / 'base954_students' / 'base954_students.ipynb', 'w'), indent=1)
print('built: students_only, base954_students (W=%s)' % W)
