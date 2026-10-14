import SimpleITK as s,glob,os
os.makedirs("vols1",exist_ok=True); SRC="vols_thick"
for f in glob.glob(SRC+"/*.mhd"):
    im=s.ReadImage(f); sp=im.GetSpacing(); sz=im.GetSize()
    nz=max(40,int(round(sz[2]*sp[2]/1.0))); f2=max(1.0,0.6/sp[0]); nx,ny=int(sz[0]/f2),int(sz[1]/f2)
    r=s.Resample(im,(nx,ny,nz),s.Transform(),s.sitkLinear,im.GetOrigin(),(sp[0]*sz[0]/nx,sp[1]*sz[1]/ny,sp[2]*sz[2]/nz),im.GetDirection(),0,im.GetPixelID())
    s.WriteImage(r,"vols1/"+os.path.basename(f))
print("resampled")
