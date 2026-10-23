import os
import glob
import json
import re
import pandas as pd
import numpy as np

ROOT = "/home/s05814802723/projects/knee"

print("=" * 90)
print("COMPREHENSIVE AUDIT: WHEN WAS MIRROR USED & HOW DID IT AFFECT PERFORMANCE?")
print("=" * 90)

# 1. Search all JSON receipts in ROOT and subdirectories
jsons = glob.glob(f"{ROOT}/**/*.json", recursive=True)
records = []

for jp in sorted(jsons):
    if "node_modules" in jp or ".git" in jp:
        continue
    try:
        with open(jp, "r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            continue
        tag = d.get("tag", os.path.basename(jp).replace(".json", ""))
        aucs = d.get("aucs", {})
        if not isinstance(aucs, dict) or len(aucs) < 5:
            continue
        
        # Check mirror_p setting
        mirror_p = d.get("mirror_p", None)
        
        # Also check corresponding log or launch command if possible
        log_name = f"train_coat_{tag}.log"
        log_paths = [
            os.path.join(os.path.dirname(jp), log_name),
            os.path.join(ROOT, log_name),
            os.path.join(ROOT, "pc_code", log_name),
            os.path.join(ROOT, "pc_code192", log_name)
        ]
        cmd_mirror = None
        for lp in log_paths:
            if os.path.exists(lp):
                with open(lp, "r", encoding="utf-8", errors="ignore") as lf:
                    head_lines = [lf.readline() for _ in range(15)]
                for hl in head_lines:
                    if "--mirror_p" in hl:
                        m = re.search(r"--mirror_p\s+([\d\.]+)", hl)
                        if m:
                            cmd_mirror = float(m.group(1))
                            break
                        cmd_mirror = "flag_present"
                        break
        
        effective_mirror = mirror_p if mirror_p is not None else cmd_mirror
        
        best_auc = d.get("best_gold_auc", None)
        if best_auc is None and "MACRO_AUC" in aucs:
            best_auc = aucs["MACRO_AUC"]
        
        mtime = os.path.getmtime(jp)
        records.append({
            "File": os.path.relpath(jp, ROOT),
            "Tag": tag,
            "Mirror_P": effective_mirror if effective_mirror is not None else 0.0,
            "Macro_AUC": round(best_auc, 4) if best_auc is not None else None,
            "MCL": round(aucs.get("MCL", 0.0), 4),
            "Lateral OA": round(aucs.get("Lateral OA", 0.0), 4),
            "Medial OA": round(aucs.get("Medial OA", 0.0), 4),
            "Medial Men": round(aucs.get("Medial Meniscus", 0.0), 4),
            "Lat Men": round(aucs.get("Lateral Meniscus", 0.0), 4),
            "Timestamp": pd.to_datetime(mtime, unit="s").strftime("%Y-%m-%d %H:%M")
        })
    except Exception as e:
        pass

df = pd.DataFrame(records)
if not df.empty:
    df = df.sort_values("Timestamp")
    print(df.to_string(index=False))

# 2. Check git / code history for when mirror_volume was added to train_knee.py
print("\n" + "=" * 90)
print("2. CODE TIMELINE: WHEN WAS MIRROR_VOLUME INTRODUCED?")
print("=" * 90)
knee_scripts = [
    f"{ROOT}/train_knee.py",
    f"{ROOT}/pc_code/train_knee.py",
    f"{ROOT}/pc_code192/train_knee.py"
]
for ks in knee_scripts:
    if os.path.exists(ks):
        with open(ks, "r", encoding="utf-8", errors="ignore") as f:
            src = f.read()
        has_mirror = "def mirror_volume" in src
        mtime = os.path.getmtime(ks)
        dt = pd.to_datetime(mtime, unit="s").strftime("%Y-%m-%d %H:%M")
        print(f"{os.path.relpath(ks, ROOT):30s} | Has mirror_volume: {has_mirror} | Last Modified: {dt}")

# 3. Direct Head-to-Head Comparison: Runs with Mirror vs Runs without Mirror
print("\n" + "=" * 90)
print("3. STATISTICAL COMPARISON: RUNS WITH MIRROR vs RUNS WITHOUT MIRROR")
print("=" * 90)
if not df.empty:
    mirror_runs = df[df["Mirror_P"].apply(lambda x: str(x) not in ("0.0", "0", "None"))]
    no_mirror_runs = df[df["Mirror_P"].apply(lambda x: str(x) in ("0.0", "0", "None"))]
    print(f"\n--- RUNS WITH MIRROR_P > 0 (N={len(mirror_runs)}) ---")
    print(mirror_runs[["Tag", "Mirror_P", "Macro_AUC", "MCL", "Lateral OA", "Timestamp"]].to_string(index=False))
    
    print(f"\n--- TOP RUNS WITHOUT MIRROR (MIRROR_P = 0) (N={len(no_mirror_runs)}) ---")
    top_clean = no_mirror_runs.sort_values("Macro_AUC", ascending=False).head(10)
    print(top_clean[["Tag", "Mirror_P", "Macro_AUC", "MCL", "Lateral OA", "Timestamp"]].to_string(index=False))

    print("\n--- MEAN METRIC COMPARISON ---")
    print(f"Mean MCL with Mirror:     {mirror_runs['MCL'].mean():.4f}")
    print(f"Mean MCL without Mirror:  {no_mirror_runs['MCL'].mean():.4f}")
    print(f"MCL Delta (Harm):         {mirror_runs['MCL'].mean() - no_mirror_runs['MCL'].mean():+.4f}")
    print(f"Mean Lat OA with Mirror:   {mirror_runs['Lateral OA'].mean():.4f}")
    print(f"Mean Lat OA without Mirror:{no_mirror_runs['Lateral OA'].mean():.4f}")
