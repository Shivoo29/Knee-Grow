"""OAI-free safe pick: R7 ConvNeXt-small alone (LB 0.949) + our label-trained CoAtNet fleet as a rank branch.

The fleet (CoAtNet version 5 seeds 0+1+2, version 6 seeds 0/3/4) is trained on report labels only, never distilled
from Raptor, so no OAI-trained weights are involved anywhere. Reuses the fleet stage from kaggle_base954_fleet.

    python kaggle_r7_fleet/build_nb.py [W_FLEET] [r7_only|r7_957]
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
W = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
BASE = sys.argv[2] if len(sys.argv) > 2 else 'r7_only'          # r7_only (OAI-free) or r7_957 (R7 + Raptor)
OUT = HERE if BASE == 'r7_only' else ROOT / 'kaggle_r7957_fleet'
nb = json.load(open(ROOT / ('kaggle_' + BASE) / (BASE + '.ipynb')))
fleet_src = (ROOT / 'kaggle_base954_fleet' / 'build_nb.py').read_text()
STAGE4 = fleet_src[fleet_src.index("STAGE4 = r'''") + len("STAGE4 = r'''"):fleet_src.index("'''\n\nsrc_cell")]
old = 'W_FLEET = 0.15'
assert STAGE4.count(old) == 1
STAGE4 = STAGE4.replace(old, 'W_FLEET = %s' % W).replace('STOP_AFTER_H = 7.3', 'STOP_AFTER_H = 6.5')
PRE = r'''
# [r7-fleet] names the fleet stage expects (base954 defined them; the R7-only notebook does not)
import numpy as np, pandas as pd
test_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
cols = list(pd.read_csv(DATA / 'sample_submission.csv', nrows=1).columns)
labels = cols[1:]
n = len(test_ids)
def pct(v):
    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values
'''
code = lambda s: {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s}
src_cell = 'FLEET_V5X3 = %r\nFLEET_V6 = %r\n' % ((ROOT / 'kaggle_base954_fleet' / 'fleet_v5x3.py').read_text(),
                                              (ROOT / 'kaggle_base954_fleet' / 'fleet_v6.py').read_text())
nb['cells'] += [code(src_cell), code(PRE + STAGE4.replace("'base954_sub.csv'", "'r7_sub.csv'"))]
OUT.mkdir(exist_ok=True)
name = 'r7_fleet.ipynb' if BASE == 'r7_only' else 'r7957_fleet.ipynb'
json.dump(nb, open(OUT / name, 'w'), indent=1)
print('built', name, 'W_FLEET=%s cells %d' % (W, len(nb['cells'])))
