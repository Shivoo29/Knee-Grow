#!/usr/bin/env python3
"""Extract DICOM site/scanner metadata from the raw competition zip for GroupKFold.

Reads ONE DICOM file per study (first file found), extracts:
  - InstitutionName
  - Manufacturer
  - ManufacturerModelName
  - StationName
  - MagneticFieldStrength

Outputs:
  - site_metadata.csv: raw metadata per study
  - site_groups.json: {"groups": {StudyInstanceUID: cluster_id}} for --fold_file

Runs on CPU only. Does NOT interfere with GPU training.
Takes ~10-15 min on the 247 GB zip (reads only 1 DICOM header per study).

Usage:
    python extract_site_metadata.py --zip ~/projects/knee/raw/rsna-knee-abnormality-detection.zip \\
                                     --output ~/projects/knee/ds/
"""
import os, sys, json, time, argparse, zipfile, io, hashlib
from collections import defaultdict
import numpy as np

def extract_metadata(zip_path, output_dir):
    """Extract site metadata from raw DICOM zip."""
    try:
        import pydicom
    except ImportError:
        print("ERROR: pydicom not found. Install with: pip install pydicom", flush=True)
        sys.exit(1)

    t0 = time.time()
    print(f"[site] Opening zip: {zip_path}", flush=True)

    # Index zip: StudyInstanceUID -> first .dcm path
    study_to_dcm = {}
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if not name.endswith('.dcm'):
                continue
            parts = name.split('/')
            # Expected: rsna-knee-abnormality-detection/train_series/STUDY_UID/SERIES_UID/file.dcm
            # Or:       train_series/STUDY_UID/SERIES_UID/file.dcm
            study_uid = None
            for i, p in enumerate(parts):
                if p == 'train_series' and i + 1 < len(parts):
                    study_uid = parts[i + 1]
                    break
            if study_uid and study_uid not in study_to_dcm:
                study_to_dcm[study_uid] = name

    n_studies = len(study_to_dcm)
    print(f"[site] Found {n_studies} studies in zip. Extracting metadata...", flush=True)

    FIELDS = ['InstitutionName', 'Manufacturer', 'ManufacturerModelName',
              'StationName', 'MagneticFieldStrength']

    rows = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for i, (study_uid, dcm_path) in enumerate(study_to_dcm.items()):
            try:
                dcm_bytes = zf.read(dcm_path)
                ds = pydicom.dcmread(io.BytesIO(dcm_bytes), stop_before_pixels=True)
                row = {'StudyInstanceUID': study_uid}
                for f in FIELDS:
                    val = getattr(ds, f, None)
                    if val is not None:
                        row[f] = str(val).strip()
                    else:
                        row[f] = ''
                rows.append(row)
            except Exception as e:
                rows.append({'StudyInstanceUID': study_uid,
                             'InstitutionName': '', 'Manufacturer': '',
                             'ManufacturerModelName': '', 'StationName': '',
                             'MagneticFieldStrength': ''})
                if i < 5:
                    print(f"  [warn] study {study_uid}: {e}", flush=True)

            if (i + 1) % 500 == 0 or i == n_studies - 1:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n_studies - i - 1)
                print(f"  [{i+1}/{n_studies}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)

    # Write raw CSV
    csv_path = os.path.join(output_dir, 'site_metadata.csv')
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['StudyInstanceUID'] + FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[site] Wrote {csv_path} ({len(rows)} rows)", flush=True)

    # Build site clusters
    # Cluster by (InstitutionName, Manufacturer, ManufacturerModelName)
    # This captures the true "site" — same hospital + same scanner model
    site_key_to_id = {}
    groups = {}
    site_counts = defaultdict(int)

    for row in rows:
        # Create a composite site key
        key_parts = [row.get('InstitutionName', ''),
                     row.get('Manufacturer', ''),
                     row.get('ManufacturerModelName', '')]
        key = '|'.join(key_parts)

        if key not in site_key_to_id:
            site_key_to_id[key] = len(site_key_to_id)
        gid = site_key_to_id[key]
        groups[row['StudyInstanceUID']] = gid
        site_counts[gid] += 1

    n_sites = len(site_key_to_id)
    print(f"\n[site] Discovered {n_sites} unique site clusters:", flush=True)
    # Print top clusters
    sorted_sites = sorted(site_key_to_id.items(), key=lambda x: site_counts[x[1]], reverse=True)
    for key, gid in sorted_sites[:25]:
        parts = key.split('|')
        print(f"  Site {gid:3d}: {site_counts[gid]:5d} studies | "
              f"Inst='{parts[0][:40]}' Mfr='{parts[1][:25]}' Model='{parts[2][:25]}'", flush=True)
    if n_sites > 25:
        print(f"  ... and {n_sites - 25} more sites", flush=True)

    # Build balanced N-fold assignment using site groups
    # We want 5 folds where no site straddles two folds
    # Sort sites by size (descending), greedily assign to smallest fold
    N_FOLDS = 5
    fold_sizes = [0] * N_FOLDS
    site_to_fold = {}

    for key, gid in sorted_sites:
        # Assign this site to the fold with fewest studies
        min_fold = min(range(N_FOLDS), key=lambda f: fold_sizes[f])
        site_to_fold[gid] = min_fold
        fold_sizes[min_fold] += site_counts[gid]

    folds = {}
    for uid, gid in groups.items():
        folds[uid] = site_to_fold[gid]

    fold_dist = defaultdict(int)
    for f in folds.values():
        fold_dist[f] += 1

    print(f"\n[site] Fold distribution (5-fold GroupKFold by site):", flush=True)
    for f in range(N_FOLDS):
        print(f"  Fold {f}: {fold_dist[f]} studies", flush=True)

    # Save
    json_path = os.path.join(output_dir, 'site_groups.json')
    with open(json_path, 'w') as f:
        json.dump({
            'n_sites': n_sites,
            'n_folds': N_FOLDS,
            'fold_sizes': fold_sizes,
            'site_key_to_id': site_key_to_id,
            'site_counts': dict(site_counts),
            'folds': folds,
            'groups': groups
        }, f, indent=1)
    print(f"[site] Wrote {json_path}", flush=True)

    # Also write the simpler fold-file format that train_knee.py expects
    fold_file_path = os.path.join(output_dir, 'folds_site.json')
    with open(fold_file_path, 'w') as f:
        json.dump({'folds': folds}, f)
    print(f"[site] Wrote {fold_file_path} (train_knee.py --fold_file format)", flush=True)

    print(f"\n[site] Done in {time.time()-t0:.1f}s", flush=True)
    return n_sites, fold_dist


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True, help='Path to rsna-knee-abnormality-detection.zip')
    ap.add_argument('--output', required=True, help='Output directory for site_metadata.csv and site_groups.json')
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    extract_metadata(args.zip, args.output)
