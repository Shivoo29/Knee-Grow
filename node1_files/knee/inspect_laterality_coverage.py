import os
import glob
import pydicom
import zipfile
import pandas as pd
import numpy as np

ROOT = "/home/s05814802723/projects/knee"

# 1. Check existing metadata CSVs/parquets
print("=== CHECKING EXISTING METADATA FILES ===")
meta_candidates = glob.glob(f"{ROOT}/**/*meta*.csv", recursive=True) + glob.glob(f"{ROOT}/**/*series*.csv", recursive=True)
for p in meta_candidates:
    rel = os.path.relpath(p, ROOT)
    if "venv" in rel or ".git" in rel:
        continue
    try:
        df = pd.read_csv(p, nrows=5)
        print(f"\nFile: {rel} | Shape (head): {df.shape} | Columns: {list(df.columns)}")
        print(df.head(2).to_string())
    except Exception as e:
        print(f"Error reading {rel}: {e}")

# 2. Inspect raw DICOM metadata across 100 studies from rsna-knee-abnormality-detection.zip
zip_path = f"{ROOT}/raw/rsna-knee-abnormality-detection.zip"
print(f"\n=== INSPECTING RAW DICOM TAGS IN {os.path.basename(zip_path)} ===")

# Load gold study IDs
train = pd.read_csv(f"{ROOT}/train.csv")
LAB = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
       'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
gold = train.dropna(subset=LAB)
gold_uids = set(gold['StudyInstanceUID'].astype(str))

# Scan zip entries
has_lat_tag = 0
has_img_lat = 0
has_desc_lr = 0
has_ipp = 0
has_iop = 0
studies_checked = set()
total_studies_target = 100

with zipfile.ZipFile(zip_path, 'r') as z:
    infolist = z.infolist()
    # Filter for train_series or test_series .dcm
    dcm_entries = [item for item in infolist if item.filename.endswith('.dcm')]
    
    # Group by study
    study_map = {}
    for item in dcm_entries:
        parts = item.filename.split('/')
        # train_series/<StudyUID>/<SeriesUID>/<SOPUID>.dcm
        if len(parts) >= 3:
            s_uid = parts[1]
            if s_uid not in study_map:
                study_map[s_uid] = item
                if len(study_map) >= total_studies_target:
                    break
                    
    print(f"Found {len(study_map)} studies to sample from zip.")
    
    for s_uid, item in study_map.items():
        with z.open(item) as fp:
            d = pydicom.dcmread(fp, stop_before_pixels=True)
            lat = getattr(d, 'Laterality', None)
            img_lat = getattr(d, 'ImageLaterality', None)
            desc = str(getattr(d, 'SeriesDescription', '')).upper()
            ipp = getattr(d, 'ImagePositionPatient', None)
            iop = getattr(d, 'ImageOrientationPatient', None)
            
            if lat in ('L', 'R', 'LEFT', 'RIGHT'):
                has_lat_tag += 1
            if img_lat in ('L', 'R', 'LEFT', 'RIGHT'):
                has_img_lat += 1
            if 'LEFT' in desc or 'RIGHT' in desc or ' LT' in desc or ' RT' in desc or desc.startswith('L ') or desc.startswith('R '):
                has_desc_lr += 1
            if ipp is not None and len(ipp) == 3:
                has_ipp += 1
            if iop is not None and len(iop) == 6:
                has_iop += 1
        studies_checked.add(s_uid)

N = len(studies_checked)
print(f"\n--- LATERALITY & SPATIAL TAG COVERAGE ACROSS {N} STUDIES ---")
print(f"Laterality tag present (L/R):               {has_lat_tag:3d}/{N} ({has_lat_tag/N:6.1%})")
print(f"ImageLaterality tag present (L/R):          {has_img_lat:3d}/{N} ({has_img_lat/N:6.1%})")
print(f"SeriesDescription contains L/R/Left/Right:  {has_desc_lr:3d}/{N} ({has_desc_lr/N:6.1%})")
print(f"ImagePositionPatient present (3D position): {has_ipp:3d}/{N} ({has_ipp/N:6.1%})")
print(f"ImageOrientationPatient present (6D basis): {has_iop:3d}/{N} ({has_iop/N:6.1%})")
