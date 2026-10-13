import pandas as pd, numpy as np, glob, re
from sklearn.metrics import roc_auc_score
L=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
norm=lambda s: re.sub(r'[^a-z]','',s.lower())
tr=pd.read_csv('train.csv'); g=tr.dropna(subset=['ACL']).set_index('StudyInstanceUID')[L]
for f in sorted(glob.glob('ds/**/*.csv',recursive=True)+glob.glob('ds/**/*.parquet',recursive=True)):
    d=pd.read_csv(f) if f.endswith('csv') else pd.read_parquet(f)
    uid=next((c for c in d.columns if 'study' in c.lower()),None)
    if uid is None: print(f,'NO UID cols=',list(d.columns)[:15]); continue
    m={norm(c):c for c in d.columns}
    cols={l:m.get(norm(l)) for l in L}
    miss=[l for l,c in cols.items() if c is None]
    d=d.drop_duplicates(uid).set_index(uid)
    gg=g.loc[g.index.intersection(d.index)]
    if len(gg)<10 or miss: print(f,'n_gold=',len(gg),'missing=',miss,'cols=',list(d.columns)[:20]); continue
    p=d.loc[gg.index,[cols[l] for l in L]].astype(float).values
    aucs=[roc_auc_score(gg[l],p[:,i]) if gg[l].nunique()>1 else np.nan for i,l in enumerate(L)]
    copy=np.mean(np.isclose(p,gg.values))
    print(f"{f}\n  n={len(d)} gold={len(gg)} macroAUC={np.nanmean(aucs):.4f} exact_match={copy:.2f}\n  "+' '.join(f'{l[:6]}={a:.2f}' for l,a in zip(L,aucs)))
