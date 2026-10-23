"""One DICOM header per series (no pixels, no reports) -> series_headers.csv. Usage: python -I header_scan.py ZIP OUT"""
import sys, io, zipfile, collections, multiprocessing as mp
import pandas as pd, pydicom
ZIP, OUT = sys.argv[1], sys.argv[2]
TAGS = ["InstitutionName", "StationName", "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "PatientSex",
        "SliceThickness", "SpacingBetweenSlices", "RepetitionTime", "EchoTime", "InversionTime", "FlipAngle",
        "ContrastBolusAgent", "SeriesDescription", "ProtocolName", "ScanningSequence", "SequenceVariant", "Laterality",
        "BodyPartExamined", "Rows", "Columns"]
def one(args):
    member, n = args
    try:
        d = pydicom.dcmread(io.BytesIO(zipfile.ZipFile(ZIP).read(member)), stop_before_pixels=True)
    except Exception as e:
        return {"err": str(e)[:80]}
    r = {t: (str(getattr(d, t, "")) if getattr(d, t, None) is not None else "") for t in TAGS}
    ps = getattr(d, "PixelSpacing", None); r["PixelSpacing"] = float(ps[0]) if ps else None
    iop = getattr(d, "ImageOrientationPatient", None); r["IOP"] = ",".join(f"{float(v):.3f}" for v in iop) if iop else ""
    r["n_images"] = n
    return r
if __name__ == "__main__":
    names = [x for x in zipfile.ZipFile(ZIP).namelist() if x.endswith(".dcm")]
    first, count = {}, collections.Counter()
    for x in names:
        k = tuple(x.split("/")[:3]); count[k] += 1; first.setdefault(k, x)
    keys = sorted(first)
    with mp.Pool(48) as p:
        rows = p.map(one, [(first[k], count[k]) for k in keys], chunksize=16)
    df = pd.DataFrame(rows); df.insert(0, "split", [k[0] for k in keys]); df.insert(1, "StudyInstanceUID", [k[1] for k in keys])
    df.insert(2, "SeriesInstanceUID", [k[2] for k in keys]); df.to_csv(OUT, index=False)
    print(df.shape, "errors:", df.get("err", pd.Series(dtype=str)).notna().sum())
