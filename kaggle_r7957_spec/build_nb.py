"""R7 + Raptor (LB 0.956) + our specialists as PER-FINDING rank branches.

Each specialist is run with the proven fleet_v6.py (it rebuilds slots / clean windows from the checkpoint) and is
blended only into the findings it is good at, with a fixed weight; every other finding stays pure R7 + Raptor.
Default: Specialist 2 (sagittal-fluid + axial windows) at 0.5 on PF OA and Lateral OA (gold-58: 0.869 -> 0.884,
0.793 -> 0.828 against the base954 teacher).

    python kaggle_r7957_spec/build_nb.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
SPECS = {'spec_pf': ('raptor_ft_spec_pf_s23_swa.pt', {'PF OA': 0.5, 'Lateral OA': 0.5})}
FLEET_V6 = (ROOT / 'kaggle_base954_fleet' / 'fleet_v6.py').read_text()
code = lambda s: {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s}

CELL = r'''
# [spec] per-finding specialist branches on top of the R7 + Raptor submission written above
import glob, json, os, shutil, subprocess, sys, time
import numpy as np, pandas as pd
SPECS = %r
STOP_AFTER_H = 7.0
shutil.copyfile(WORK / 'submission.csv', WORK / 'r7957_sub.csv')
test_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
cols = list(pd.read_csv(DATA / 'sample_submission.csv', nrows=1).columns)
labels = cols[1:]
pct = lambda v: pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values
def find_one(name):
    hits = glob.glob('/kaggle/input/*/' + name) + glob.glob('/kaggle/input/datasets/*/*/' + name)
    assert hits, name + ' not attached'
    return hits[0]
base = pd.read_csv(WORK / 'r7957_sub.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[test_ids, labels]
final = {c: pct(base[c].values) for c in labels}
for tag, (ckpt, weights) in SPECS.items():
    if (time.time() - NB_T0) / 3600 > STOP_AFTER_H:
        print('SPEC_SKIPPED', tag, flush=True); continue
    Path('/tmp/spec.py').write_text(FLEET_V6)
    out = WORK / ('spec_%%s.csv' %% tag)
    env = dict(os.environ, RSNA_COMP_ROOT=str(DATA), RAPTOR_OUTPUT_CSV=str(out), RAPTOR_CKPT=find_one(ckpt))
    t = time.time()
    rc = subprocess.run([sys.executable, '-u', '/tmp/spec.py'], env=env, cwd='/tmp').returncode
    if rc != 0:
        raise RuntimeError('specialist %%s failed with exit code %%d' %% (tag, rc))
    s = pd.read_csv(out, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[test_ids, labels]
    for c, w in weights.items():
        final[c] = (1 - w) * final[c] + w * pct(s[c].values)
    print('SPEC_DONE', tag, '%%.0fs' %% (time.time() - t), json.dumps(weights), flush=True)
sub = pd.DataFrame({c: final[c].astype(np.float32) for c in labels})
sub.insert(0, 'StudyInstanceUID', test_ids)
sub = sub[cols]
assert np.isfinite(sub[labels].values).all() and ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()
sub.to_csv(WORK / 'submission.csv', index=False)
print('R7957_SPEC_COMPLETE', round(time.time() - NB_T0), 's', flush=True)
''' % SPECS

nb = json.load(open(ROOT / 'kaggle_r7_957' / 'r7_957.ipynb'))
nb['cells'] += [code('FLEET_V6 = %r\nfrom pathlib import Path\n' % FLEET_V6), code(CELL)]
json.dump(nb, open(HERE / 'r7957_spec.ipynb', 'w'), indent=1)
m = json.load(open(ROOT / 'kaggle_r7_957' / 'kernel-metadata.json'))
m.update(id='shivoo29/rsna-knee-r7957-spec', title='rsna-knee-r7957-spec', code_file='r7957_spec.ipynb')
m['dataset_sources'] = m['dataset_sources'] + ['shivoo29/knee-students']
json.dump(m, open(HERE / 'kernel-metadata.json', 'w'), indent=2)
print('built r7957_spec.ipynb', SPECS)
