"""Blend our own model families into a base notebook as one rank branch.

    python kaggle_blend_own/build_nb.py --base blendmine --slug rsna-knee-blend-own --out kaggle_blend_own/w30 \
        --datasets shivoo29/knee-cnxt-folds --group cnxt=raptor_ft_cnxt_f0_swa.pt,raptor_ft_cnxt_f1_swa.pt,... --w 0.3

Each --group is one fleet_v6 run (one DICOM decode pass) averaging same-layout checkpoints (RAPTOR_CKPTS);
groups are rank-averaged with equal weight into the own branch. --base blendmine keeps heliosli's 0.958 blend
and adds the branch at weight W (a failed or late group keeps the base); --base none submits the own branch
alone (the OAI/fastMRI-free hedge).
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent

SETUP_NONE = r'''
# [own] names the stage expects when there is no base notebook
import json, time
from pathlib import Path
import numpy as np, pandas as pd
NB_T0 = time.time()
WORK = Path('/kaggle/working')
DATA = next(p for p in [Path('/kaggle/input/rsna-knee-abnormality-detection'),
                        Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')] if (p / 'test.csv').is_file())
test_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
cols = list(pd.read_csv(DATA / 'sample_submission.csv', nrows=1).columns)
labels = cols[1:]
n = len(test_ids)
def pct(v):
    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values
'''

STAGE = r'''
# [own] our model families as one rank branch; each group = one fleet_v6 decode pass over the test set
import glob, os, subprocess, sys, time
import numpy as np, pandas as pd
W_OWN = __W__
GROUPS = __GROUPS__
STOP_AFTER_H = __STOP__   # no group starts after this; skipped groups leave the rest unchanged
HAS_BASE = __HAS_BASE__

def find_one(name):
    hits = glob.glob('/kaggle/input/*/' + name) + glob.glob('/kaggle/input/datasets/*/*/' + name)
    assert hits, name + ' not attached'
    return hits[0]

Path('/tmp/fleet_v6.py').write_text(FLEET_V6)
outs = {}
for tag, names in GROUPS.items():
    h = (time.time() - NB_T0) / 3600
    if h > STOP_AFTER_H:
        print('OWN_SKIPPED %s at %.2f h' % (tag, h), flush=True)
        continue
    out = WORK / ('own_%s.csv' % tag)
    env = dict(os.environ, RSNA_COMP_ROOT=str(DATA), RAPTOR_OUTPUT_CSV=str(out),
               RAPTOR_CKPTS=','.join(find_one(x) for x in names))
    t = time.time()
    rc = subprocess.run([sys.executable, '-u', '/tmp/fleet_v6.py'], env=env, cwd='/tmp').returncode
    if rc != 0:
        if HAS_BASE:
            print('OWN_FAILED %s exit %d: group skipped, base kept' % (tag, rc), flush=True)
            continue
        raise RuntimeError('own group %s failed with exit code %d' % (tag, rc))
    print('OWN_DONE %s %.0fs' % (tag, time.time() - t), flush=True)
    outs[tag] = pd.read_csv(out, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[test_ids, labels]

if not outs:
    assert HAS_BASE, 'no own group ran and there is no base submission'
    print('OWN_NONE: keeping the base submission', flush=True)
else:
    base = (pd.read_csv(WORK / 'submission.csv', dtype={'StudyInstanceUID': str})
              .set_index('StudyInstanceUID').loc[test_ids, labels]) if HAS_BASE else None
    final = np.zeros((n, len(labels)))
    rho = {}
    for j, c in enumerate(labels):
        own = np.mean([pct(o[c].values) for o in outs.values()], axis=0)
        if HAS_BASE:
            final[:, j] = (1 - W_OWN) * pct(base[c].values) + W_OWN * own
            if n >= 3:
                rho[c] = round(float(pd.Series(base[c].values).corr(pd.Series(own), method='spearman')), 4)
        else:
            final[:, j] = own
    sub = pd.DataFrame(final.astype(np.float32), columns=labels)
    sub.insert(0, 'StudyInstanceUID', test_ids)
    sub = sub[cols]
    assert sub['StudyInstanceUID'].tolist() == test_ids and np.isfinite(sub[labels].values).all()
    assert ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()
    sub.to_csv(WORK / 'submission.csv', index=False)
    print('OWN_BLEND_COMPLETE', json.dumps({'w_own': W_OWN if HAS_BASE else 1.0, 'groups': sorted(outs),
                                             'total_seconds': round(time.time() - NB_T0, 1),
                                             'spearman_base_vs_own': rho}), flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', choices=['blendmine', 'none'], required=True)
    ap.add_argument('--slug', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--group', action='append', required=True, help='TAG=file1.pt,file2.pt (names inside attached datasets)')
    ap.add_argument('--datasets', default='')
    ap.add_argument('--w', type=float, default=0.3)
    ap.add_argument('--stop_h', type=float, default=7.0)
    a = ap.parse_args()
    groups = {}
    for g in a.group:
        tag, names = g.split('=', 1)
        groups[tag] = [x for x in names.split(',') if x]
    code = lambda s: {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s}
    fleet = (ROOT / 'kaggle_base954_fleet' / 'fleet_v6.py').read_text()
    stage = (STAGE.replace('__W__', repr(a.w)).replace('__GROUPS__', repr(groups))
             .replace('__STOP__', repr(a.stop_h)).replace('__HAS_BASE__', repr(a.base != 'none')))
    nb = json.load(open(ROOT / 'kaggle_blendmine' / 'blendmine.ipynb'))
    meta = json.load(open(ROOT / 'kaggle_blendmine' / 'kernel-metadata.json'))
    extra = [d for d in a.datasets.split(',') if d]
    if a.base == 'none':
        nb['cells'] = [code(SETUP_NONE)]
        meta['dataset_sources'] = extra
    else:
        meta['dataset_sources'] = meta['dataset_sources'] + [d for d in extra if d not in meta['dataset_sources']]
    nb['cells'] += [code('FLEET_V6 = %r\n' % fleet), code(stage)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta.update(id='shivoo29/' + a.slug, title=a.slug, code_file=a.slug + '.ipynb')
    json.dump(nb, open(out / (a.slug + '.ipynb'), 'w'), indent=1)
    json.dump(meta, open(out / 'kernel-metadata.json', 'w'), indent=2)
    print('built', out / (a.slug + '.ipynb'), 'cells', len(nb['cells']), 'groups', groups, 'w', a.w)


if __name__ == '__main__':
    main()
