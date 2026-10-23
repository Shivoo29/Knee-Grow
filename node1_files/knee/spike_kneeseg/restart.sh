cd ~/projects/knee/spike_kneeseg
for p in $(pgrep -f "^venv/bin/python seg.py"); do kill $p; done; sleep 3
python3 - <<EOF
s=open("resamp.py").read()
s=s.replace('os.makedirs("vols1",exist_ok=True)','os.makedirs("vols1",exist_ok=True); SRC="vols_thick"').replace('glob.glob("vols/*.mhd")','glob.glob(SRC+"/*.mhd")')
s=s.replace("nz=max(40,int(round(sz[2]*sp[2]/1.0)))","nz=max(40,int(round(sz[2]*sp[2]/1.0))); f2=max(1.0,0.6/sp[0]); nx,ny=int(sz[0]/f2),int(sz[1]/f2)")
s=s.replace("(sz[0],sz[1],nz)","(nx,ny,nz)").replace("(sp[0],sp[1],sp[2]*sz[2]/nz)","(sp[0]*sz[0]/nx,sp[1]*sz[1]/ny,sp[2]*sz[2]/nz)")
open("resamp.py","w").write(s)
s=open("seg.py").read().replace("import sys,os,time,json","import sys,os,time,json\nos.cpu_count=lambda:12")
open("seg.py","w").write(s)
EOF
cat > go.sh <<EOF
cd ~/projects/knee/spike_kneeseg
rm -rf vols1 pred; venv/bin/python resamp.py || exit 1; rm -rf vols; mv vols1 vols
export OMP_NUM_THREADS=4 LOKY_MAX_CPU_COUNT=32 CUDA_VISIBLE_DEVICES=
taskset -c 0-31 venv/bin/python seg.py > seg.log 2>&1
EOF
nohup bash go.sh > go.log 2>&1 &
echo restarted
