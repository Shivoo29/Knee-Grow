import pydicom
import glob, os, zipfile
import pandas as pd

KNEE = "/home/s05814802723/projects/knee"
comp_zip = os.path.join(KNEE, "raw", "rsna-knee-abnormality-detection.zip")

print("Checking Laterality fields in raw DICOM metadata...")
with zipfile.ZipFile(comp_zip, 'r') as z:
    dcms = [f for f in z.namelist() if f.endswith('.dcm')][:20]
    for dcm_path in dcms:
        with z.open(dcm_path) as fp:
            d = pydicom.dcmread(fp, stop_before_pixels=True)
            lat = getattr(d, 'ImageLaterality', getattr(d, 'Laterality', None))
            body_part = getattr(d, 'BodyPartExamined', None)
            orient = getattr(d, 'ImageOrientationPatient', None)
            pos = getattr(d, 'ImagePositionPatient', None)
            series_desc = getattr(d, 'SeriesDescription', '')
            print(f"File: {os.path.basename(dcm_path)} | Laterality: {lat} | BodyPart: {body_part} | Series: {series_desc}")
