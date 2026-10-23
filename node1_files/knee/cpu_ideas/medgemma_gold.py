"""Zero-shot MedGemma 1.5 on the gold-58 studies, straight from the raw competition DICOMs (no corpus).
Runs locally on node1 (rule 4.b: nothing leaves the machine). For each study and question it shows ~10 slices of the
matching plane and reads P(Yes) from the first answer token; AUC vs expert labels, side by side with our ConvNeXt.
    mg_env/bin/python -I medgemma_gold.py ZIP GOLD_NPZ_GLOB LATERALITY_CSV"""
import glob, io, sys, zipfile
import numpy as np, pandas as pd, pydicom, torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForImageTextToText, AutoProcessor

ZIP, GOLD_GLOB, LAT = sys.argv[1:4]
MODEL = "google/medgemma-1.5-4b-it"
N_SLICES = 10
F = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
OA_DEF = ("high-grade cartilage loss (more than 50% of cartilage thickness, over at least 1 cm, or full-thickness loss)")
QUESTIONS = {   # finding -> (plane, prefer fat-suppressed?, question)
    "Lateral OA": ("Coronal", False, f"Is there osteoarthritis of the LATERAL tibiofemoral compartment, meaning {OA_DEF} on the lateral femoral condyle or lateral tibial plateau?"),
    "Medial OA": ("Coronal", False, f"Is there osteoarthritis of the MEDIAL tibiofemoral compartment, meaning {OA_DEF} on the medial femoral condyle or medial tibial plateau?"),
    "PF OA": ("Axial", True, f"Is there patellofemoral osteoarthritis, meaning {OA_DEF} on the patella or trochlea?"),
    "Synovitis": ("Sagittal", True, "Is there synovitis (synovial thickening, or effusion-synovitis with thickened synovium or Hoffa fat pad signal change)?"),
    "Effusion": ("Sagittal", True, "Is there a joint effusion (abnormal fluid distending the suprapatellar recess or joint)?"),
}

z = zipfile.ZipFile(ZIP)
names = [n for n in z.namelist() if n.startswith("train_series/") and n.endswith(".dcm")]
by_series = {}
for n in names:
    _, st, se, _ = n.split("/")
    by_series.setdefault((st, se), []).append(n)
ser = pd.read_csv(io.BytesIO(z.read("train_series.csv")))
lat = pd.read_csv(LAT).set_index("StudyInstanceUID").Laterality

gz = [np.load(p) for p in sorted(glob.glob(GOLD_GLOB))]
gold_ids, truth = gz[0]["ids"], gz[0]["truth"]
ours = np.mean([g["pred"] for g in gz], 0)


def slices(study, plane, fs):
    s = ser[(ser.StudyInstanceUID == study) & (ser.Anatomical_Plane == plane)]
    if s.empty:
        return None
    pick = s[s.Fat_Suppression == int(fs)]
    row = (pick if len(pick) else s).iloc[0]
    ds = [pydicom.dcmread(io.BytesIO(z.read(n))) for n in by_series[(study, row.SeriesInstanceUID)]]
    try:   # true anatomical order: position along the slice normal
        nrm = np.cross(*np.array(ds[0].ImageOrientationPatient, float).reshape(2, 3))
        ds.sort(key=lambda d: float(np.dot(nrm, np.array(d.ImagePositionPatient, float))))
    except Exception:
        ds.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0)))
    lo, hi = int(len(ds) * 0.2), max(int(len(ds) * 0.8), int(len(ds) * 0.2) + 1)   # central 60% holds the joint
    idx = np.linspace(lo, hi - 1, min(N_SLICES, hi - lo)).round().astype(int)
    out = []
    for i in idx:
        a = ds[i].pixel_array.astype(np.float32)
        p1, p99 = np.percentile(a, [1, 99])
        a = np.clip((a - p1) / max(p99 - p1, 1e-6), 0, 1)
        out.append(Image.fromarray((a * 255).astype(np.uint8)).convert("RGB").resize((448, 448)))
    return out


proc = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
tok = proc.tokenizer
YES, NO = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]

res = {f: np.full(len(gold_ids), np.nan) for f in QUESTIONS}
for k, u in enumerate(gold_ids):
    side = lat.get(u, "Right")
    lat_img = "left" if side == "Right" else "right"   # radiological display: patient right on image left
    for f, (plane, fs, q) in QUESTIONS.items():
        ims = slices(u, plane, fs)
        if not ims:
            continue
        side_txt = f" The patient's lateral side is on the image {lat_img}." if plane in ("Coronal", "Axial") else ""
        msgs = [{"role": "system", "content": [{"type": "text", "text": "You are an expert musculoskeletal radiologist."}]},
                {"role": "user", "content": [{"type": "text", "text": f"These are {len(ims)} {plane.lower()} MRI slices of a {side.lower()} knee.{side_txt}"}]
                 + [{"type": "image", "image": im} for im in ims]
                 + [{"type": "text", "text": q + " Answer with only Yes or No."}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                       return_tensors="pt").to(model.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            lg = model(**inp).logits[0, -1].float()
        res[f][k] = torch.softmax(lg[[YES, NO]], 0)[0].item()
    print(f"{k + 1}/{len(gold_ids)} done", flush=True)

print("\nGOLD-58 AUC   MedGemma zero-shot | our ConvNeXt (round-2, 4 folds)")
for f in QUESTIONS:
    j = F.index(f); m = ~np.isnan(res[f])
    print(f"  {f:11s}  {roc_auc_score(truth[m, j], res[f][m]):.3f}  |  {roc_auc_score(truth[m, j], ours[m, j]):.3f}   (n={m.sum()}, pos={int(truth[m, j].sum())})")
pd.DataFrame({"StudyInstanceUID": gold_ids, **{f"mg_{f}": v for f, v in res.items()}}).to_csv("medgemma_gold58.csv", index=False)
