"""R7 ConvNeXt-small alone (no Raptor, so no OAI-trained weights): the R7 + Raptor 0.957 notebook with stage 1 and the
blend removed. Tells us R7's own LB and is the base of an OAI-free final pick.

    python kaggle_r7_only/build_nb.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
nb = json.load(open(HERE.parent / 'kaggle_r7_957' / 'r7_957.ipynb'))
code = lambda s: {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s}
c = nb['cells']
assert c[3]['source'] and 'raptor_run.py' in ''.join(c[3]['source']) and 'Stage 3' in ''.join(c[6]['source'])
stage2 = ''.join(c[5]['source'])
old = "'--max-fallback-fraction', '1.0'"
assert stage2.count(old) == 1
stage2 = stage2.replace(old, "'--max-fallback-fraction', '0.03'")   # [r7-only] no Raptor rank to fall back on
START = "import os, sys, time, subprocess\nNB_T0 = time.time()   # [r7-only] Raptor stage removed\n"
FINISH = r'''
import pandas as pd, numpy as np
sub = pd.read_csv(WORK / 'ours_sub.csv')
cols = list(pd.read_csv(DATA / 'sample_submission.csv', nrows=1).columns)
assert list(sub.columns) == cols and len(sub) == len(pd.read_csv(DATA / 'test.csv'))
assert np.isfinite(sub[cols[1:]].values).all()
sub.to_csv(WORK / 'submission.csv', index=False)
print('R7_ONLY_COMPLETE rows', len(sub), flush=True)
'''
nb['cells'] = [c[0], c[1], code(START), c[4], code(stage2), code(FINISH)]
json.dump(nb, open(HERE / 'r7_only.ipynb', 'w'), indent=1)
print('built r7_only.ipynb, cells', len(nb['cells']))
