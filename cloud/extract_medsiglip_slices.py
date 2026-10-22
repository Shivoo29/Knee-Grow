import os
import io
import time
import json
import tarfile
import numpy as np
import pandas as pd
from PIL import Image

def select_indices(vol, slot_fs, slot_plain, target_k):
    # Find non-zero slices in primary fat-sat slot
    nz = [i for i in range(slot_fs[0], slot_fs[1]) if vol[i].max() > 0]
    if len(nz) == 0 and slot_plain is not None:
        nz = [i for i in range(slot_plain[0], slot_plain[1]) if vol[i].max() > 0]
    if len(nz) == 0:
        idx = np.linspace(slot_fs[0], slot_fs[1] - 1, target_k).round().astype(int)
        return list(idx)
    if len(nz) < target_k:
        idx = np.linspace(0, len(nz) - 1, target_k).round().astype(int)
        return [nz[j] for j in idx]
    # Middle 60%
    s = int(round(0.20 * len(nz)))
    e = int(round(0.80 * len(nz)))
    mid = nz[s:e]
    if len(mid) < target_k:
        mid = nz
    idx = np.linspace(0, len(mid) - 1, target_k).round().astype(int)
    return [mid[j] for j in idx]

def main():
    out_dir = "pcr2/knee-medsiglip-slices"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading memmap all_vols.npy and all_ids.npy...")
    vols = np.load("pcr2/all_vols.npy", mmap_mode="r")
    ids = np.load("pcr2/all_ids.npy", allow_pickle=True)
    n_studies = len(ids)
    print(f"Total studies: {n_studies}")

    # Metadata for Kaggle dataset
    meta = {
        "title": "knee-medsiglip-slices",
        "id": "shivoo29/knee-medsiglip-slices",
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True
    }
    with open(os.path.join(out_dir, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 4 shards
    num_shards = 4
    shard_size = int(np.ceil(n_studies / num_shards))

    csv_rows = []
    t_start = time.time()

    current_shard_idx = -1
    current_tar = None

    for i in range(n_studies):
        shard_idx = min(i // shard_size, num_shards - 1)
        shard_name = f"shard_{shard_idx}.tar"
        shard_path = os.path.join(out_dir, shard_name)

        if shard_idx != current_shard_idx:
            if current_tar is not None:
                current_tar.close()
                print(f"Closed {os.path.basename(current_tar.name)}, size: {os.path.getsize(current_tar.name) / 1e6:.1f} MB")
            current_shard_idx = shard_idx
            current_tar = tarfile.open(shard_path, "w")
            print(f"Opened new shard: {shard_name}")

        study_id = str(ids[i])
        vol = vols[i]

        sag_idx = select_indices(vol, (0, 40), (40, 80), 6)
        cor_idx = select_indices(vol, (80, 116), (116, 152), 6)
        ax_idx = select_indices(vol, (152, 192), None, 4)

        planes = [
            ("sag", sag_idx),
            ("cor", cor_idx),
            ("ax", ax_idx)
        ]

        for plane_name, slice_list in planes:
            for k, g_idx in enumerate(slice_list):
                slice_arr = vol[g_idx]
                fname = f"{study_id}_{plane_name}_{k}.jpg"

                # Encode to JPEG q90
                pil_img = Image.fromarray(slice_arr, mode="L")
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=90)
                jpeg_bytes = buf.getvalue()

                ti = tarfile.TarInfo(name=fname)
                ti.size = len(jpeg_bytes)
                ti.mtime = int(time.time())
                current_tar.addfile(ti, io.BytesIO(jpeg_bytes))

                csv_rows.append({
                    "study_id": study_id,
                    "plane": plane_name,
                    "slice_idx": k,
                    "global_vol_idx": int(g_idx),
                    "shard": shard_name,
                    "filename": fname
                })

        if (i + 1) % 200 == 0 or (i + 1) == n_studies:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            rem = (n_studies - (i + 1)) / rate if rate > 0 else 0
            print(f"Progress: {i+1}/{n_studies} studies ({rate:.1f} std/s, ~{rem:.0f}s left)")

    if current_tar is not None:
        current_tar.close()
        print(f"Closed {os.path.basename(current_tar.name)}, size: {os.path.getsize(current_tar.name) / 1e6:.1f} MB")

    df_slices = pd.DataFrame(csv_rows)
    df_slices.to_csv(os.path.join(out_dir, "slices.csv"), index=False)
    print(f"Saved slices.csv with {len(df_slices)} rows.")
    total_mb = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)) / 1e6
    print(f"Done! Total dataset size: {total_mb:.1f} MB in {time.time() - t_start:.1f}s")

if __name__ == "__main__":
    main()
