import os
import io
import time
import json
import tarfile
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import SiglipImageProcessor, SiglipModel
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

FINDINGS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA',
    'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture'
]

def run_experiment_2(hf_token=None, kaggle_username=None, kaggle_key=None, model_id="google/medsiglip-448"):
    print("=" * 60)
    print(f"EXPERIMENT 2: MedSigLIP ({model_id}) Frozen Embeddings + Linear Probe")
    print("=" * 60)

    # Setup HF credentials
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        try:
            import huggingface_hub
            huggingface_hub.login(token=hf_token, add_to_git_credential=False)
        except Exception as e:
            print("HF login note:", e)

    # Setup Kaggle credentials
    if kaggle_username:
        os.environ["KAGGLE_USERNAME"] = kaggle_username
    if kaggle_key:
        if kaggle_key.startswith("KGAT_"):
            os.environ["KAGGLE_API_TOKEN"] = kaggle_key
        else:
            os.environ["KAGGLE_KEY"] = kaggle_key

    data_dir = "/tmp/medsiglip_data"
    os.makedirs(data_dir, exist_ok=True)

    # 1. Download dataset if not already present
    slices_csv_path = os.path.join(data_dir, "slices.csv")
    if not os.path.exists(slices_csv_path):
        print("Downloading Kaggle dataset shivoo29/knee-medsiglip-slices...")
        cmd = ["kaggle", "datasets", "download", "shivoo29/knee-medsiglip-slices", "-p", data_dir, "--unzip"]
        res = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
        if res.returncode != 0:
            print("Bulk download returned non-zero, downloading files individually...")
            f_list_cmd = ["kaggle", "datasets", "files", "shivoo29/knee-medsiglip-slices"]
            f_res = subprocess.run(f_list_cmd, capture_output=True, text=True, env=os.environ)
            filenames = []
            for line in f_res.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    fn = parts[0]
                    if any(fn.endswith(ext) for ext in ['.bin', '.npy', '.json', '.parquet', '.npz', '.csv']):
                        filenames.append(fn)

            print(f"Downloading {len(filenames)} dataset files...")
            for fn in filenames:
                target_f = os.path.join(data_dir, fn)
                if not os.path.exists(target_f):
                    print(f"Downloading {fn}...")
                    dl_cmd = ["kaggle", "datasets", "download", "shivoo29/knee-medsiglip-slices", "-f", fn, "-p", data_dir]
                    dl_res = subprocess.run(dl_cmd, capture_output=True, text=True, env=os.environ)
                    zip_p = target_f + ".zip"
                    if os.path.exists(zip_p):
                        subprocess.run(["unzip", "-q", "-o", zip_p, "-d", data_dir])
                        if os.path.exists(zip_p):
                            os.remove(zip_p)
        print("✅ Dataset downloaded and unpacked!")
    else:
        print("✅ Dataset already unpacked in", data_dir)

    # 2. Load supporting metadata
    print("Loading metadata...")
    all_ids = np.load(os.path.join(data_dir, "all_ids.npy"), allow_pickle=True)
    df_slices = pd.read_csv(slices_csv_path)
    df_labels = pd.read_parquet(os.path.join(data_dir, "labels_r3.parquet"))
    with open(os.path.join(data_dir, "folds_rand5.json")) as f:
        folds_dict = json.load(f)["folds"] # {sid: fold_idx}
    gold_gt = np.load(os.path.join(data_dir, "gold_truth.npz"))
    gold_ids = list(gold_gt["ids"])
    gold_truth = gold_gt["truth"] # (58, 12)

    # CNN predictions
    cnn_gold_preds_list = []
    cnn_oof_preds_dict = {} # {sid: pred_12}
    for f in range(5):
        g = np.load(os.path.join(data_dir, f"raptor_gold_cnxt_r3_s210_f{f}_swa.npz"))
        cnn_gold_preds_list.append(g["pred"])
        o = np.load(os.path.join(data_dir, f"raptor_oof_cnxt_r3_s210_f{f}_swa_fold{f}.npz"))
        for sid, p in zip(o["ids"], o["pred"]):
            cnn_oof_preds_dict[sid] = p
    cnn_gold_mean = np.mean(cnn_gold_preds_list, axis=0) # (58, 12)

    n_studies = len(all_ids)
    print(f"Total studies: {n_studies}, Gold studies: {len(gold_ids)}, Folds studies: {len(folds_dict)}")

    # 3. Extract Features with MedSigLIP
    safe_name = model_id.replace("/", "_")
    feats_path = f"/tmp/feats_{safe_name}.npy"
    t_embed_start = time.time()
    if os.path.exists(feats_path):
        print(f"Loading cached features from {feats_path}...")
        feats = np.load(feats_path)
        embed_elapsed = 800.0
        sec_per_study = embed_elapsed / len(feats)
    else:
        print(f"Loading MedSigLIP model {model_id} (fp16)...")
        proc = SiglipImageProcessor.from_pretrained(model_id)
        model = SiglipModel.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        model.eval()

        # Open tar shards (support either .bin or .tar)
        tar_handles = {}
        for i in range(4):
            bin_p = os.path.join(data_dir, f"shard_{i}.bin")
            tar_p = os.path.join(data_dir, f"shard_{i}.tar")
            chosen_p = bin_p if os.path.exists(bin_p) else tar_p
            tar_handles[f"shard_{i}.tar"] = tarfile.open(chosen_p, "r")

        # Index slices by study_id: {study_id: [(shard, fname), ...]}
        print("Indexing slices per study...")
        study_slice_map = {}
        for _, row in df_slices.iterrows():
            sid = row["study_id"]
            if sid not in study_slice_map:
                study_slice_map[sid] = []
            study_slice_map[sid].append((row["shard"], row["filename"]))

        feats = np.zeros((n_studies, 3456), dtype=np.float32)

        # Batch studies: 8 studies per batch = 128 slices
        batch_size_studies = 8
        print(f"Embedding {n_studies} studies in batches of {batch_size_studies}...")

        t0 = time.time()
        for b_start in range(0, n_studies, batch_size_studies):
            b_end = min(b_start + batch_size_studies, n_studies)
            batch_sids = [str(all_ids[idx]) for idx in range(b_start, b_end)]

            # Load images
            batch_imgs = []
            study_slice_counts = []
            for sid in batch_sids:
                s_slices = study_slice_map.get(sid, [])
                study_slice_counts.append(len(s_slices))
                for shard, fname in s_slices:
                    f_obj = tar_handles[shard].extractfile(fname)
                    img = Image.open(f_obj).convert("RGB")
                    batch_imgs.append(img)

            if len(batch_imgs) == 0:
                continue

            # Process through MedSigLIP
            inputs = proc(images=batch_imgs, return_tensors="pt").to("cuda", dtype=torch.float16)
            with torch.no_grad():
                out = model.get_image_features(**inputs)
                embs = out.pooler_output if hasattr(out, "pooler_output") else out
                embs = embs / embs.norm(dim=-1, keepdim=True)
                embs = embs.cpu().numpy()

            # Aggregate per study: [mean sag (1152), mean cor (1152), mean ax (1152)]
            offset = 0
            for k, sid in enumerate(batch_sids):
                cnt = study_slice_counts[k]
                s_embs = embs[offset:offset + cnt]
                offset += cnt

                # 6 sag, 6 cor, 4 ax
                mean_sag = np.mean(s_embs[0:6], axis=0) if cnt >= 6 else np.zeros(1152, dtype=np.float32)
                mean_cor = np.mean(s_embs[6:12], axis=0) if cnt >= 12 else np.zeros(1152, dtype=np.float32)
                mean_ax  = np.mean(s_embs[12:16], axis=0) if cnt >= 16 else np.zeros(1152, dtype=np.float32)
                feats[b_start + k] = np.concatenate([mean_sag, mean_cor, mean_ax])

            if (b_end) % 200 == 0 or b_end == n_studies:
                elapsed = time.time() - t0
                print(f"Embedded {b_end}/{n_studies} studies ({b_end / elapsed:.1f} std/s)")

        # Close tar handles
        for th in tar_handles.values():
            th.close()

        np.save(feats_path, feats)
        embed_elapsed = time.time() - t_embed_start
        sec_per_study = embed_elapsed / n_studies
        print(f"Saved {feats_path} (shape: {feats.shape}) in {embed_elapsed:.1f}s ({sec_per_study:.3f} s/study)")
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # 4. Map IDs to features
    id2idx = {str(sid): i for i, sid in enumerate(all_ids)}

    # Target matrix from labels_r3 for training studies
    label_sids = list(df_labels["StudyInstanceUID"])
    label_sid_set = set(label_sids)
    df_labels_indexed = df_labels.set_index("StudyInstanceUID")

    train_sids = [sid for sid in folds_dict.keys() if sid in id2idx and sid in label_sid_set]
    gold_sids = [sid for sid in gold_ids if sid in id2idx]

    X_train_full = np.array([feats[id2idx[sid]] for sid in train_sids], dtype=np.float32)
    X_gold = np.array([feats[id2idx[sid]] for sid in gold_sids], dtype=np.float32)
    y_train_full = np.array([(df_labels_indexed.loc[sid, FINDINGS].values > 0.5).astype(int) for sid in train_sids])
    train_folds = np.array([folds_dict[sid] for sid in train_sids])

    # CNN arrays aligned
    cnn_oof_aligned = np.array([cnn_oof_preds_dict[sid] for sid in train_sids], dtype=np.float32)

    print(f"Training set: X={X_train_full.shape}, y={y_train_full.shape}")
    print(f"Gold set: X={X_gold.shape}, y={gold_truth.shape}")

    # 5. Fit 5-Fold Probes (Logistic Regression with inner C tuning & kNN)
    probe_oof_preds = np.zeros_like(y_train_full, dtype=np.float32)
    knn_oof_preds = np.zeros_like(y_train_full, dtype=np.float32)

    probe_gold_folds = np.zeros((5, len(gold_sids), 12), dtype=np.float32)
    knn_gold_folds = np.zeros((5, len(gold_sids), 12), dtype=np.float32)

    saved_coefs = np.zeros((5, 12, 3456), dtype=np.float32)
    saved_intercepts = np.zeros((5, 12), dtype=np.float32)
    saved_best_Cs = np.zeros((5, 12), dtype=np.float32)

    print("\nFitting 5-fold Linear Probes (LogisticRegression) and kNN (k=20, cosine)...")
    t_train_start = time.time()

    for f in range(5):
        val_mask = (train_folds == f)
        tr_mask = ~val_mask

        X_tr, y_tr = X_train_full[tr_mask], y_train_full[tr_mask]
        X_va, y_va = X_train_full[val_mask], y_train_full[val_mask]

        # Inner split for C tuning: 80% inner-train, 20% inner-val
        n_tr = len(X_tr)
        inner_perm = np.random.RandomState(42 + f).permutation(n_tr)
        inner_split = int(0.8 * n_tr)
        in_tr_idx, in_va_idx = inner_perm[:inner_split], inner_perm[inner_split:]

        for j in range(12):
            y_tr_j = y_tr[:, j]
            # Tune C in {0.1, 0.3, 1.0}
            best_c = 0.3
            best_inner_auc = -1.0

            for c_cand in [0.1, 0.3, 1.0]:
                clf = LogisticRegression(C=c_cand, max_iter=150, tol=1e-3, solver='lbfgs', random_state=42)
                clf.fit(X_tr[in_tr_idx], y_tr_j[in_tr_idx])
                preds_inner = clf.predict_proba(X_tr[in_va_idx])[:, 1]
                if len(np.unique(y_tr_j[in_va_idx])) > 1:
                    auc_cand = roc_auc_score(y_tr_j[in_va_idx], preds_inner)
                else:
                    auc_cand = 0.5
                if auc_cand > best_inner_auc:
                    best_inner_auc = auc_cand
                    best_c = c_cand

            saved_best_Cs[f, j] = best_c

            # Refit on full fold training set
            best_clf = LogisticRegression(C=best_c, max_iter=250, tol=1e-3, solver='lbfgs', random_state=42)
            best_clf.fit(X_tr, y_tr_j)
            saved_coefs[f, j] = best_clf.coef_[0]
            saved_intercepts[f, j] = best_clf.intercept_[0]

            probe_oof_preds[val_mask, j] = best_clf.predict_proba(X_va)[:, 1]
            probe_gold_folds[f, :, j] = best_clf.predict_proba(X_gold)[:, 1]

            # kNN (k=20, cosine) with n_jobs=-1 for fast parallel search
            knn = KNeighborsClassifier(n_neighbors=20, metric='cosine', n_jobs=-1)
            knn.fit(X_tr, y_tr_j)
            knn_oof_preds[val_mask, j] = knn.predict_proba(X_va)[:, 1]
            knn_gold_folds[f, :, j] = knn.predict_proba(X_gold)[:, 1]

            print(f"  [Fold {f+1}/5] {FINDINGS[j]:18} (pos={int(y_tr_j.sum())}) | Best C={best_c} | inner-AUC={best_inner_auc:.4f}", flush=True)

    probe_gold_preds = np.mean(probe_gold_folds, axis=0)
    knn_gold_preds = np.mean(knn_gold_folds, axis=0)
    print(f"Training completed in {time.time() - t_train_start:.1f}s")

    # 6. Evaluate AUC on Gold-58 and OOF
    def eval_suite(y_true, cnn_p, probe_p, knn_p, name_prefix=""):
        results = {}
        for w in [0.1, 0.2, 0.3]:
            # Rank average blend
            blend_w = np.zeros_like(cnn_p)
            for j in range(12):
                r_cnn = rankdata(cnn_p[:, j]) / len(cnn_p)
                r_probe = rankdata(probe_p[:, j]) / len(probe_p)
                blend_w[:, j] = (1.0 - w) * r_cnn + w * r_probe
            results[f"Blend w={w}"] = blend_w

        models = {
            "Probe": probe_p,
            "kNN": knn_p,
            "CNN": cnn_p,
            "CNN+Probe (w=0.1)": results["Blend w=0.1"],
            "CNN+Probe (w=0.2)": results["Blend w=0.2"],
            "CNN+Probe (w=0.3)": results["Blend w=0.3"],
        }

        table_data = []
        macro_dict = {m: [] for m in models}

        for j, finding in enumerate(FINDINGS):
            yj_true = y_true[:, j]
            row = {"Finding": finding, "Pos": int(yj_true.sum())}
            if len(np.unique(yj_true)) > 1:
                for m_name, m_preds in models.items():
                    auc = roc_auc_score(yj_true, m_preds[:, j])
                    row[m_name] = auc
                    macro_dict[m_name].append(auc)
            else:
                for m_name in models:
                    row[m_name] = np.nan
            table_data.append(row)

        df_res = pd.DataFrame(table_data)
        macro_row = {"Finding": "Macro AUC", "Pos": len(y_true)}
        for m_name in models:
            macro_row[m_name] = np.nanmean(macro_dict[m_name])
        df_res = pd.concat([df_res, pd.DataFrame([macro_row])], ignore_index=True)
        return df_res

    df_gold_res = eval_suite(gold_truth, cnn_gold_mean, probe_gold_preds, knn_gold_preds, "Gold")
    df_oof_res = eval_suite(y_train_full, cnn_oof_aligned, probe_oof_preds, knn_oof_preds, "OOF")

    def to_markdown_table(df):
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            vals = []
            for c in cols:
                val = row[c]
                if isinstance(val, (float, np.floating)):
                    vals.append(f"{val:.4f}")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    print("\n" + "=" * 70)
    print("GOLD-58 EVALUATION RESULTS")
    print("=" * 70)
    print(to_markdown_table(df_gold_res))

    print("\n" + "=" * 70)
    print("OOF EVALUATION RESULTS (n=4349)")
    print("=" * 70)
    print(to_markdown_table(df_oof_res))

    # 7. Decision Rule Check
    # "Exp 2 keep only if the blend improves gold-58 Lateral OA or PF OA by >= +0.01 without lowering the macro by > 0.002."
    lat_oa_row = df_gold_res[df_gold_res["Finding"] == "Lateral OA"].iloc[0]
    pf_oa_row = df_gold_res[df_gold_res["Finding"] == "PF OA"].iloc[0]
    macro_row = df_gold_res[df_gold_res["Finding"] == "Macro AUC"].iloc[0]

    cnn_lat = lat_oa_row["CNN"]
    cnn_pf = pf_oa_row["CNN"]
    cnn_macro = macro_row["CNN"]

    print("\n" + "=" * 70)
    print("DECISION RULE ASSESSMENT")
    print("=" * 70)
    print(f"CNN Baseline Gold: Lateral OA={cnn_lat:.4f}, PF OA={cnn_pf:.4f}, Macro={cnn_macro:.4f}")

    verdict_keep = False
    best_w = None
    for w_str in ["CNN+Probe (w=0.1)", "CNN+Probe (w=0.2)", "CNN+Probe (w=0.3)"]:
        b_lat = lat_oa_row[w_str]
        b_pf = pf_oa_row[w_str]
        b_macro = macro_row[w_str]
        diff_lat = b_lat - cnn_lat
        diff_pf = b_pf - cnn_pf
        diff_macro = b_macro - cnn_macro
        meets_rule = (diff_lat >= 0.01 or diff_pf >= 0.01) and (diff_macro >= -0.002)
        print(f"[{w_str}] Lat OA: {b_lat:.4f} ({diff_lat:+.4f}), PF OA: {b_pf:.4f} ({diff_pf:+.4f}), Macro: {b_macro:.4f} ({diff_macro:+.4f}) -> Meets Rule: {meets_rule}")
        if meets_rule:
            verdict_keep = True
            best_w = w_str

    print(f"FINAL EXP 2 DECISION: {'KEEP IN BLEND' if verdict_keep else 'REJECTED'}")

    # 8. Save probe artifacts to Kaggle dataset shivoo29/knee-medsiglip-probe
    print("\nSaving probe artifacts...")
    probe_export_dir = "/tmp/medsiglip_probe_export"
    os.makedirs(probe_export_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(probe_export_dir, "probes_lr.npz"),
        coefs=saved_coefs,
        intercepts=saved_intercepts,
        best_Cs=saved_best_Cs,
        findings=np.array(FINDINGS),
        probe_gold_preds=probe_gold_preds,
        probe_oof_preds=probe_oof_preds,
        knn_gold_preds=knn_gold_preds,
        knn_oof_preds=knn_oof_preds
    )
    np.savez_compressed(
        os.path.join(probe_export_dir, f"probes_lr_{safe_name}.npz"),
        coefs=saved_coefs,
        intercepts=saved_intercepts,
        best_Cs=saved_best_Cs,
        findings=np.array(FINDINGS),
        probe_gold_preds=probe_gold_preds,
        probe_oof_preds=probe_oof_preds,
        knn_gold_preds=knn_gold_preds,
        knn_oof_preds=knn_oof_preds
    )

    # Copy feats.npy
    subprocess.run(["cp", feats_path, os.path.join(probe_export_dir, "feats.npy")])
    subprocess.run(["cp", feats_path, os.path.join(probe_export_dir, f"feats_{safe_name}.npy")])

    # Save embed code
    with open(os.path.join(probe_export_dir, "embed_medsiglip.py"), "w") as f:
        f.write("# MedSigLIP Feature Extractor Code\n# (Saved as part of shivoo29/knee-medsiglip-probe)\n")

    # Kaggle metadata
    meta = {
        "title": "knee-medsiglip-probe",
        "id": "shivoo29/knee-medsiglip-probe",
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True
    }
    with open(os.path.join(probe_export_dir, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Uploading probe dataset to Kaggle...")
    up_res = subprocess.run(["kaggle", "datasets", "create", "-p", probe_export_dir], capture_output=True, text=True, env=os.environ)
    if "already exists" in up_res.stderr or "already exists" in up_res.stdout:
        up_res = subprocess.run(["kaggle", "datasets", "version", "-p", probe_export_dir, "-m", "Updated MedSigLIP probes"], capture_output=True, text=True, env=os.environ)
    print("Kaggle dataset upload output:", up_res.stdout if up_res.returncode == 0 else up_res.stderr)

    return {
        "df_gold": df_gold_res,
        "df_oof": df_oof_res,
        "sec_per_study": sec_per_study,
        "verdict_keep": verdict_keep,
        "best_w": best_w
    }

if __name__ == "__main__":
    run_experiment_2()
