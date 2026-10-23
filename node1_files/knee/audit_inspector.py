"""
RSNA Knee Abnormality Detection 2026 - Comprehensive Artifact Auditor
Inspects CSV, Parquet, JSON, DICOM/Images, Audio, and Archives (NPZ, NPY, PT, ZIP).
"""

import os
import sys
import json
import glob
import zipfile
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

REPORT_DATA = {
    "system_info": {},
    "csv_files": {},
    "parquet_files": {},
    "json_files": {},
    "dicom_and_images": {},
    "audio_files": {},
    "archives_and_tensors": {},
}

ROOT = Path("/home/s05814802723/projects/knee")

def audit_system():
    print("[1/6] Auditing System & GPU Status...")
    import subprocess
    try:
        gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    except Exception as e:
        gpu_info = str(e)
    REPORT_DATA["system_info"] = {
        "root": str(ROOT),
        "python": sys.version,
        "gpu_smi": gpu_info
    }

def audit_csv():
    print("[2/6] Auditing CSV files...")
    csv_paths = list(ROOT.rglob("*.csv"))
    # filter out git or cache if any
    csv_paths = [p for p in csv_paths if ".git" not in str(p) and "venv" not in str(p)]
    
    for p in sorted(csv_paths):
        rel = str(p.relative_to(ROOT))
        if p.is_symlink() and not p.exists():
            print(f"  [BROKEN SYMLINK] {rel} -> {os.readlink(p)}")
            REPORT_DATA["csv_files"][rel] = {
                "path": str(p),
                "is_symlink": True,
                "broken": True,
                "target": str(os.readlink(p))
            }
            continue
        try:
            sz = p.stat().st_size
        except Exception as e:
            print(f"  [ERROR STAT] {rel}: {e}")
            REPORT_DATA["csv_files"][rel] = {"path": str(p), "error": str(e)}
            continue
        print(f"  Inspecting CSV: {rel} ({sz / 1024:.1f} KB)")
        try:
            df = pd.read_csv(p, nrows=1000)
            full_len = sum(1 for _ in open(p, "rb")) - 1 # approximate row count
            info = {
                "path": str(p),
                "size_bytes": sz,
                "size_human": f"{sz / 1024 / 1024:.2f} MB" if sz > 1024*1024 else f"{sz / 1024:.1f} KB",
                "estimated_rows": full_len,
                "columns": list(df.columns),
                "num_columns": len(df.columns),
                "dtypes": {c: str(df[c].dtype) for c in df.columns},
                "null_counts_first_1k": {c: int(df[c].isnull().sum()) for c in df.columns},
                "head_sample": df.head(3).to_dict(orient="records")
            }
            if "StudyInstanceUID" in df.columns:
                info["unique_sample_uids"] = int(df["StudyInstanceUID"].nunique())
            REPORT_DATA["csv_files"][rel] = info
        except Exception as e:
            REPORT_DATA["csv_files"][rel] = {"path": str(p), "size_bytes": sz, "error": str(e)}

def audit_parquet():
    print("[3/6] Auditing Parquet files...")
    pq_paths = list(ROOT.rglob("*.parquet"))
    pq_paths = [p for p in pq_paths if ".git" not in str(p) and "venv" not in str(p)]
    
    for p in sorted(pq_paths):
        rel = str(p.relative_to(ROOT))
        if p.is_symlink() and not p.exists():
            print(f"  [BROKEN SYMLINK] {rel} -> {os.readlink(p)}")
            REPORT_DATA["parquet_files"][rel] = {"path": str(p), "is_symlink": True, "broken": True, "target": str(os.readlink(p))}
            continue
        try:
            sz = p.stat().st_size
        except Exception as e:
            REPORT_DATA["parquet_files"][rel] = {"path": str(p), "error": str(e)}
            continue
        print(f"  Inspecting Parquet: {rel} ({sz / 1024:.1f} KB)")
        try:
            df = pd.read_parquet(p)
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            stats = {}
            for col in num_cols[:15]: # first 15 numeric cols
                stats[col] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std())
                }
            REPORT_DATA["parquet_files"][rel] = {
                "path": str(p),
                "size_bytes": sz,
                "size_human": f"{sz / 1024 / 1024:.2f} MB" if sz > 1024*1024 else f"{sz / 1024:.1f} KB",
                "shape": list(df.shape),
                "columns": list(df.columns),
                "null_counts": {c: int(df[c].isnull().sum()) for c in df.columns},
                "summary_stats": stats,
                "head_sample": df.head(2).to_dict(orient="records")
            }
        except Exception as e:
            REPORT_DATA["parquet_files"][rel] = {"path": str(p), "size_bytes": sz, "error": str(e)}

def audit_json():
    print("[4/6] Auditing JSON files...")
    json_paths = list(ROOT.rglob("*.json"))
    json_paths = [p for p in json_paths if ".git" not in str(p) and "venv" not in str(p)]
    
    for p in sorted(json_paths):
        rel = str(p.relative_to(ROOT))
        if p.is_symlink() and not p.exists():
            print(f"  [BROKEN SYMLINK] {rel} -> {os.readlink(p)}")
            REPORT_DATA["json_files"][rel] = {"path": str(p), "is_symlink": True, "broken": True, "target": str(os.readlink(p))}
            continue
        try:
            sz = p.stat().st_size
        except Exception as e:
            REPORT_DATA["json_files"][rel] = {"path": str(p), "error": str(e)}
            continue
        print(f"  Inspecting JSON: {rel} ({sz} bytes)")
        try:
            with open(p, "r") as f:
                data = json.load(f)
            
            summary = {
                "path": str(p),
                "size_bytes": sz,
                "type": type(data).__name__,
            }
            if isinstance(data, dict):
                summary["top_keys"] = list(data.keys())[:20]
                # extract useful model metrics if present
                for key in ["best_gold_macro_auc", "gold_macro_auc", "best_epoch", "best_loss", "macro_auc", "arch", "seed", "epochs"]:
                    if key in data:
                        summary[key] = data[key]
            elif isinstance(data, list):
                summary["length"] = len(data)
                if len(data) > 0 and isinstance(data[0], dict):
                    summary["first_element_keys"] = list(data[0].keys())
            REPORT_DATA["json_files"][rel] = summary
        except Exception as e:
            REPORT_DATA["json_files"][rel] = {"path": str(p), "size_bytes": sz, "error": str(e)}

def audit_dicom_and_images():
    print("[5/6] Auditing Images and DICOM files...")
    image_exts = {".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    img_counts = Counter()
    sample_files = []
    
    for root, dirs, files in os.walk(ROOT):
        if ".git" in root or "venv" in root:
            continue
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in image_exts:
                img_counts[ext] += 1
                if len(sample_files) < 5:
                    sample_files.append(os.path.join(root, f))
    
    # Also inspect rsna-knee-abnormality-detection.zip for DICOM info
    zip_path = ROOT / "raw" / "rsna-knee-abnormality-detection.zip"
    zip_dicom_count = 0
    zip_sample_dicoms = []
    zip_exists = zip_path.exists()
    
    if zip_exists:
        try:
            print(f"  Inspecting zip archive headers: {zip_path} ({zip_path.stat().st_size / (1024**3):.1f} GB)...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                # Sample up to first 200 items in zip
                infolist = z.infolist()
                total_in_zip = len(infolist)
                dicom_in_zip = sum(1 for item in infolist if item.filename.endswith('.dcm'))
                sample_members = [item.filename for item in infolist[:10]]
                
                # Read 1 DICOM header if possible
                dcm_sample_info = {}
                dcm_candidates = [item for item in infolist if item.filename.endswith('.dcm')]
                if dcm_candidates:
                    try:
                        import pydicom
                        import io
                        with z.open(dcm_candidates[0]) as dcm_file:
                            ds = pydicom.dcmread(io.BytesIO(dcm_file.read()), stop_before_pixels=True)
                            dcm_sample_info = {
                                "file": dcm_candidates[0].filename,
                                "Modality": getattr(ds, "Modality", "Unknown"),
                                "SeriesDescription": getattr(ds, "SeriesDescription", "Unknown"),
                                "BodyPartExamined": getattr(ds, "BodyPartExamined", "Unknown"),
                                "PatientPosition": getattr(ds, "PatientPosition", "Unknown"),
                                "Rows": getattr(ds, "Rows", None),
                                "Columns": getattr(ds, "Columns", None),
                                "PixelSpacing": [float(x) for x in getattr(ds, "PixelSpacing", [])] if getattr(ds, "PixelSpacing", None) else None,
                                "SliceThickness": float(getattr(ds, "SliceThickness", 0.0)) if getattr(ds, "SliceThickness", None) else None,
                                "PhotometricInterpretation": getattr(ds, "PhotometricInterpretation", "Unknown"),
                            }
                    except Exception as ex_dcm:
                        dcm_sample_info = {"sample_file": dcm_candidates[0].filename, "parse_error": str(ex_dcm)}

                REPORT_DATA["dicom_and_images"]["zip_archive"] = {
                    "path": str(zip_path),
                    "size_gb": zip_path.stat().st_size / (1024**3),
                    "total_entries": total_in_zip,
                    "dicom_count": dicom_in_zip,
                    "sample_paths": sample_members,
                    "parsed_dicom_sample": dcm_sample_info
                }
        except Exception as e:
            REPORT_DATA["dicom_and_images"]["zip_archive"] = {"error": str(e)}

    REPORT_DATA["dicom_and_images"]["filesystem_images"] = {
        "extension_counts": dict(img_counts),
        "total_on_filesystem": sum(img_counts.values()),
        "sample_files": sample_files
    }

def audit_audio():
    print("[6/6] Auditing Audio files...")
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
    audio_found = []
    
    for root, dirs, files in os.walk(ROOT):
        if ".git" in root or "venv" in root:
            continue
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in audio_exts:
                audio_found.append(os.path.join(root, f))
                
    REPORT_DATA["audio_files"] = {
        "count": len(audio_found),
        "files": audio_found,
        "note": "RSNA Knee Abnormality Detection is a Musculoskeletal Magnetic Resonance Imaging (MRI) challenge. Zero audio files expected; 0 found on filesystem."
    }

def audit_archives_and_tensors():
    print("[7/6] Auditing Archives, Checkpoints, and Tensors (.npz, .npy, .pt, .zip)...")
    tensors = {}
    
    # 1. NPZ files
    npz_files = list(ROOT.rglob("*.npz"))
    for p in sorted(npz_files):
        rel = str(p.relative_to(ROOT))
        if p.is_symlink() and not p.exists():
            tensors[rel] = {"type": "NPZ", "is_symlink": True, "broken": True, "target": str(os.readlink(p))}
            continue
        try:
            sz = p.stat().st_size
        except Exception as e:
            tensors[rel] = {"type": "NPZ", "error": str(e)}
            continue
        try:
            with np.load(p) as data:
                files = list(data.files)
                shapes = {k: list(data[k].shape) for k in files[:10]}
                dtypes = {k: str(data[k].dtype) for k in files[:10]}
                tensors[rel] = {
                    "type": "NPZ",
                    "size_human": f"{sz / 1024:.1f} KB" if sz < 1024*1024 else f"{sz / 1024 / 1024:.2f} MB",
                    "keys": files,
                    "shapes": shapes,
                    "dtypes": dtypes
                }
        except Exception as e:
            tensors[rel] = {"type": "NPZ", "error": str(e)}

    # 2. Key NPY files
    npy_files = list(ROOT.rglob("*.npy"))
    for p in sorted(npy_files):
        rel = str(p.relative_to(ROOT))
        if p.is_symlink() and not p.exists():
            tensors[rel] = {"type": "NPY", "is_symlink": True, "broken": True, "target": str(os.readlink(p))}
            continue
        try:
            sz = p.stat().st_size
        except Exception as e:
            tensors[rel] = {"type": "NPY", "error": str(e)}
            continue
        try:
            # Memory map to avoid loading 62GB into memory
            arr = np.load(p, mmap_mode="r")
            tensors[rel] = {
                "type": "NPY",
                "size_human": f"{sz / (1024**3):.2f} GB" if sz > 1024**3 else f"{sz / 1024 / 1024:.2f} MB",
                "shape": list(arr.shape),
                "dtype": str(arr.dtype)
            }
        except Exception as e:
            tensors[rel] = {"type": "NPY", "error": str(e)}

    # 3. PT Checkpoints
    pt_files = list(ROOT.rglob("*.pt"))
    for p in sorted(pt_files):
        rel = str(p.relative_to(ROOT))
        if p.is_symlink() and not p.exists():
            tensors[rel] = {"type": "PyTorch Checkpoint", "is_symlink": True, "broken": True, "target": str(os.readlink(p))}
            continue
        try:
            sz = p.stat().st_size
        except Exception as e:
            tensors[rel] = {"type": "PyTorch Checkpoint", "error": str(e)}
            continue
        tensors[rel] = {
            "type": "PyTorch Checkpoint",
            "size_human": f"{sz / (1024**3):.2f} GB" if sz > 1024**3 else f"{sz / 1024 / 1024:.2f} MB",
            "size_bytes": sz
        }
        
    REPORT_DATA["archives_and_tensors"] = tensors

if __name__ == "__main__":
    audit_system()
    audit_csv()
    audit_parquet()
    audit_json()
    audit_dicom_and_images()
    audit_audio()
    audit_archives_and_tensors()
    
    out_path = ROOT / "audit_results.json"
    with open(out_path, "w") as f:
        json.dump(REPORT_DATA, f, indent=2)
    print(f"\n[DONE] Audit saved to: {out_path}")
