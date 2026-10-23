import sys,os,time,json
os.cpu_count=lambda:12
import kneeseg.pipeline.inference as I
from kneeseg.pipeline.inference import inference_improved
_o=I._process_one_file
def _w(f):
    t=time.time()
    try: r=_o(f); print("OK",f[-12:],round(time.time()-t),flush=True); return r
    except Exception as e: print("FAIL",f[-12:],type(e).__name__,str(e)[:80],flush=True); return f,{}
I._process_one_file=_w
D=os.path.expanduser("~/projects/knee/spike_kneeseg")
files=sorted(f for f in os.listdir(f"{D}/vols") if f.endswith(".mhd"))
n=int(sys.argv[1]) if len(sys.argv)>1 else len(files)
cfg={"data_config":{"image_directory":f"{D}/vols","select_files":files[:n]},
 "inference_config":{"inference_proximity_mm":10.0},
 "model_config":{"target_bones":["femur","tibia"],"model_directory":f"{D}/model_ski10"},
 "output_config":{"prediction_directory":f"{D}/pred"}}
t=time.time(); inference_improved(cfg); print("TOTAL s",time.time()-t,"n",n)
