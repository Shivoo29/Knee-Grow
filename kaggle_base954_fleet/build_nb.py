"""Build base954_fleet.ipynb: the unchanged public 0.954 notebook + our seed fleet as a 15% rank branch.

Stage 4 (new cell) runs our two proven scripts as subprocesses:
  fleet_v5x3.py  3 seeds of the 44-slice CoAtNet (LB 0.941 on its own)
  fleet_v6.py    96-slice CoAtNet, run once per seed (v6, v6_s3) via RAPTOR_CKPT
then rewrites submission.csv = 0.85 * pct(base954) + 0.15 * mean(pct(v5x3), pct(v6), pct(v6_s3)).

    python kaggle_base954_fleet/build_nb.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
nb = json.load(open(HERE.parent / 'kaggle_base954' / 'base954.ipynb'))

STAGE4 = r'''
# Stage 4 (ours): seed fleet as a small rank branch on top of the base954 submission written above.
import glob, os, shutil, subprocess, sys, time
import numpy as np, pandas as pd
W_FLEET = 0.15
SKIP_AFTER_H = 6.0   # our branch needs ~2 h on the hidden test; past this the base954 submission is kept as is
shutil.copyfile(WORK / 'submission.csv', WORK / 'base954_sub.csv')

def find_one(name):
    hits = glob.glob('/kaggle/input/*/' + name) + glob.glob('/kaggle/input/datasets/*/*/' + name)  # no recursive walk over test DICOMs
    assert len(hits) >= 1, name + ' not attached'
    return hits[0]

SCRIPTS = {'v5x3': (FLEET_V5X3, None),
           'v6': (FLEET_V6, find_one('raptor_ft_v6_swa.pt')),
           'v6_s3': (FLEET_V6, find_one('raptor_ft_v6_s3_swa.pt'))}
elapsed_h = (time.time() - NB_T0) / 3600
if elapsed_h > SKIP_AFTER_H:
    print('FLEET_SKIPPED: %.2f h elapsed > %.1f h, keeping base954 submission' % (elapsed_h, SKIP_AFTER_H), flush=True)
else:
    outs = {}
    for tag, (src, ckpt) in SCRIPTS.items():
        path = Path('/tmp/fleet_%s.py' % tag)
        path.write_text(src)
        out = WORK / ('fleet_%s.csv' % tag)
        env = dict(os.environ, RSNA_COMP_ROOT=str(DATA), RAPTOR_OUTPUT_CSV=str(out))
        if ckpt:
            env['RAPTOR_CKPT'] = ckpt
        t = time.time()
        rc = subprocess.run([sys.executable, '-u', str(path)], env=env, cwd='/tmp').returncode
        if rc != 0:
            raise RuntimeError('fleet %s failed with exit code %d' % (tag, rc))
        print('FLEET_DONE %s %.0fs' % (tag, time.time() - t), flush=True)
        outs[tag] = pd.read_csv(out, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[test_ids, labels]

    base = pd.read_csv(WORK / 'base954_sub.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[test_ids, labels]
    final = np.zeros((n, len(labels)))
    rho = {}
    for j, c in enumerate(labels):
        fleet = np.mean([pct(o[c].values) for o in outs.values()], axis=0)
        final[:, j] = (1 - W_FLEET) * pct(base[c].values) + W_FLEET * fleet
        if n >= 3:
            rho[c] = round(float(pd.Series(base[c].values).corr(pd.Series(fleet), method='spearman')), 4)
    sub = pd.DataFrame(final.astype(np.float32), columns=labels)
    sub.insert(0, 'StudyInstanceUID', test_ids)
    sub = sub[cols]
    assert sub['StudyInstanceUID'].tolist() == test_ids and np.isfinite(sub[labels].values).all()
    assert ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()
    sub.to_csv(WORK / 'submission.csv', index=False)
    print('FLEET_BLEND_COMPLETE', json.dumps({'w_fleet': W_FLEET, 'total_seconds': round(time.time() - NB_T0, 1),
                                               'spearman_base_vs_fleet': rho}), flush=True)
    display(sub.head())
'''

src_cell = 'FLEET_V5X3 = %r\nFLEET_V6 = %r\n' % ((HERE / 'fleet_v5x3.py').read_text(), (HERE / 'fleet_v6.py').read_text())
code = lambda s: {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s}
nb['cells'] += [code(src_cell), code(STAGE4)]
json.dump(nb, open(HERE / 'base954_fleet.ipynb', 'w'), indent=1)
print('cells', len(nb['cells']))
