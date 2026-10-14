import zipfile, io, os, numpy as np, pandas as pd, pydicom, SimpleITK as sitk, collections
K=os.path.expanduser("~/projects/knee"); OUT=f"{K}/spike_kneeseg/vols"; os.makedirs(OUT,exist_ok=True)
ids=np.load(f"{K}/raptor_gold_v6.npz",allow_pickle=True)["ids"].astype(str)
s=pd.read_csv(f"{K}/ds/comp/train_series.csv"); s=s[s.StudyInstanceUID.isin(ids)]
print("series per plane/fluid/fs:\n",s.groupby(["Anatomical_Plane","Fluid_Sensitive","Fat_Suppression"]).size())
pick={}
for sid,g in s.groupby("StudyInstanceUID"):
    sg=g[g.Anatomical_Plane.str.capitalize()=="Sagittal"]
    if not len(sg): continue
    # prefer non-fat-sat (closer to SKI10 T1), then non-fluid
    sg=sg.sort_values(["Fat_Suppression","Fluid_Sensitive"])
    pick[sg.iloc[0].SeriesInstanceUID]=(sid,int(sg.iloc[0].Fat_Suppression),int(sg.iloc[0].Fluid_Sensitive))
print("studies w/ sagittal:",len(pick),"of",len(ids)); print("picked (fs,fluid):",collections.Counter((v[1],v[2]) for v in pick.values()))
z=zipfile.ZipFile(f"{K}/raw/rsna-knee-abnormality-detection.zip")
fm=collections.defaultdict(list)
for n in z.namelist():
    p=n.split("/")
    if n.endswith(".dcm") and len(p)>=3 and p[-2] in pick: fm[p[-2]].append(n)
print("example path depth:",len(next(iter(fm.values()))[0].split("/")))
meta=[]
for ser,(sid,fs,fl) in pick.items():
    sl=[]
    for f in fm[ser]:
        d=pydicom.dcmread(io.BytesIO(z.read(f)))
        iop=np.array(d.ImageOrientationPatient,float); n=np.cross(iop[:3],iop[3:])
        sl.append((float(np.dot(np.array(d.ImagePositionPatient,float),n)),d.pixel_array.astype(np.float32),[float(x) for x in d.PixelSpacing]))
    sl.sort(key=lambda t:t[0]); shp=collections.Counter(t[1].shape for t in sl).most_common(1)[0][0]
    sl=[t for t in sl if t[1].shape==shp]
    dz=float(np.median(np.diff([t[0] for t in sl]))) if len(sl)>1 else 3.0
    v=np.stack([t[1] for t in sl]); v=(v/np.percentile(v,99.5)*1000).clip(0,2000).astype(np.int16)
    img=sitk.GetImageFromArray(v); img.SetSpacing((sl[0][2][1],sl[0][2][0],abs(dz)))
    sitk.WriteImage(img,f"{OUT}/{sid}.mhd")
    meta.append(dict(sid=sid,fs=fs,fluid=fl,shape=v.shape,dz=abs(dz),ps=sl[0][2][0]))
m=pd.DataFrame(meta); m.to_csv(f"{K}/spike_kneeseg/meta.csv",index=False)
print(m[["dz","ps"]].describe().T); print("nslices",m["shape"].map(lambda x:x[0]).describe().to_dict())
