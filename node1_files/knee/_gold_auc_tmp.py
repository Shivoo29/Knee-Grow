import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

L = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
     'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

tr = pd.read_csv('train.csv')
g = tr.dropna(subset=['ACL']).set_index('StudyInstanceUID')[L]
print('gold rows:', len(g))

def load(f):
    d = np.load(f, allow_pickle=True)
    if 'pred' in d and 'ids' in d:
        return np.array([str(x) for x in d['ids']]), d['pred'].astype(float), list(d.keys())
    ks = list(d.keys())
    probs = None
    for k in ks:
        a = d[k]
        if a.ndim == 2 and a.shape[1] == 12 and 'truth' not in k.lower():
            probs = a.astype(float)
    uids = None
    for k in ks:
        a = d[k]
        if a.ndim == 1 and probs is not None and a.size == probs.shape[0] and a.dtype.kind in 'USO':
            uids = np.array([str(x) for x in a])
    return uids, probs, ks

models = {
    'v6_swa':          'raptor_gold_v6_swa.npz',
    'v6_s3_swa':       'raptor_gold_v6_s3_swa.npz',
    'v5s2_swa':        'raptor_gold_v5s2_swa.npz',
    'cnext_base_96_swa':'raptor_gold_cnext_base_96_swa.npz',
    'eva02_base_96_swa':'raptor_gold_eva02_base_96_swa.npz',
    'v5_swa':          'raptor_gold_v5_swa.npz',
    'v5s1_swa':        'raptor_gold_v5s1_swa.npz',
}

def auc_rows(p):
    return [roc_auc_score(g[L[i]].values, p[:, i]) if g[L[i]].nunique() > 1 else np.nan
            for i in range(12)]

def show(name, a):
    print(f"{name:26s} macro={np.nanmean(a):.4f} | " +
          ' '.join(f"{l[:5]}={x:.3f}" for l, x in zip(L, a)))

store = {}
for name, f in models.items():
    try:
        uids, probs, ks = load(f)
        if probs is None:
            print(name, 'NO PROBS', ks); continue
        if uids is not None and len(set(uids) & set(g.index)) == 58:
            p = pd.DataFrame(probs, index=uids, columns=L).loc[g.index].values
        else:
            p = probs  # assume saved in gold order
        store[name] = p
        show(name, auc_rows(p))
    except Exception as e:
        print(name, 'ERR', repr(e))

def rankz(p):
    return np.column_stack([rankdata(p[:, i]) / len(p) for i in range(12)])

blends = {
    'FLEET top10 (40/35/25)': [('v6_swa', .40), ('cnext_base_96_swa', .35), ('v5s2_swa', .25)],
    'FLEET + eva02 (30/25/20/25)': [('v6_swa', .30), ('cnext_base_96_swa', .25), ('v5s2_swa', .20), ('eva02_base_96_swa', .25)],
    '6-model grand (coat+v6s3+cnx+eva+v5s2+v5s1)': [('v6_swa', .22), ('v6_s3_swa', .18), ('cnext_base_96_swa', .18), ('eva02_base_96_swa', .18), ('v5s2_swa', .14), ('v5s1_swa', .10)],
}
print()
for bname, members in blends.items():
    if not all(m in store for m in (m for m, _ in members)):
        print(bname, 'SKIP (missing npz)'); continue
    b = np.zeros_like(store[members[0][0]])
    for m, w in members:
        b += w * rankz(store[m])
    show(bname, auc_rows(b))
