import os,glob,numpy as np,pandas as pd,SimpleITK as s
from scipy import ndimage as nd
from sklearn.metrics import roc_auc_score
from kneeseg.bone_rf import detect_medial_is_low_z
K=os.path.expanduser("~/projects/knee"); D=f"{K}/spike_kneeseg"
g=np.load(f"{K}/raptor_gold_v6.npz",allow_pickle=True); ids=g["ids"].astype(str); T=g["truth"]
rows=[]
for f in sorted(glob.glob(f"{D}/pred/*.mhd")):
    sid=os.path.basename(f)[:-4]; im=s.ReadImage(f); a=s.GetArrayFromImage(im); sp=im.GetSpacing()[::-1]; vv=np.prod(sp)/1000
    r=dict(sid=sid,nvox_fc=(a==2).sum(),nvox_tc=(a==4).sum())
    try: med_low=bool(detect_medial_is_low_z((a==1)|(a==2),sp))
    except Exception as e: med_low=True; r["medfail"]=1
    for lab,nm,bone in [(2,"fem",1),(4,"tib",3)]:
        m=a==lab; r[f"{nm}_vol"]=m.sum()*vv
        # thickness: 2*max inscribed distance in-plane (slices thick) -> use per-slice 2D EDT
        th=np.zeros(m.shape,np.float32)
        for k in range(m.shape[0]):
            if m[k].any(): th[k]=nd.distance_transform_edt(m[k],sampling=sp[1:])*2
        zs=np.where(m.any((1,2)))[0]
        if len(zs)==0: continue
        zc=(zs.min()+zs.max())/2
        for side,sel in [("lo",np.arange(m.shape[0])<zc),("hi",np.arange(m.shape[0])>=zc)]:
            mm=m&sel[:,None,None]; key=("med" if (side=="lo")==med_low else "lat")
            r[f"{nm}_{key}_vol"]=mm.sum()*vv
            r[f"{nm}_{key}_thk"]=th[mm].mean() if mm.any() else 0
            r[f"{nm}_{key}_thk_p10"]=np.percentile(th[mm],10) if mm.any() else 0
    for c in ["vol","thk"]:
        r[f"medlat_ratio_{c}"]=(r.get(f"tib_med_{c}",0)+r.get(f"fem_med_{c}",0))/(r.get(f"tib_lat_{c}",0)+r.get(f"fem_lat_{c}",0)+1e-6)
    rows.append(r)
F=pd.DataFrame(rows).set_index("sid"); F.to_csv(f"{D}/features.csv")
print("n seg:",len(F),"fem cart vox median/min:",F.nvox_fc.median(),F.nvox_fc.min(),"tib:",F.nvox_tc.median(),F.nvox_tc.min(),"near-empty(<500 vox):",((F.nvox_fc<500)|(F.nvox_tc<500)).sum())
idx={s:i for i,s in enumerate(ids)}; y=T[[idx[s] for s in F.index]]
feats=[c for c in F.columns if c not in("nvox_fc","nvox_tc","medfail")]
for j,nm in [(4,"Medial OA"),(5,"Lateral OA"),(6,"PF OA")]:
    yy=y[:,j]; ok=~np.isnan(yy); res=[]
    for c in feats:
        x=F[c].values.astype(float); o=ok&~np.isnan(x)
        if len(set(yy[o]))<2: continue
        a=roc_auc_score(yy[o],x[o]); res.append((max(a,1-a),c,"+" if a>=.5 else "-",o.sum(),int(yy[o].sum())))
    res.sort(reverse=True); print(nm, "top3:",[(round(a,3),c,sg) for a,c,sg,_,_ in res[:3]],"n",res[0][3],"pos",res[0][4])
