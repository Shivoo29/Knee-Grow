# Pre-GPU Diagnostic & Architectural Audit (`pre_gpu_checks.md`)

**Date:** 2026-10-19  
**Execution Mode:** STRICT READ-ONLY & CPU-ONLY  
**Target Environment:** Remote node1 (`172.27.92.5`), local workspace `c:\Users\ASHISH\kaggle_rsna`  
**Ground Rules Observed:** No GPU utilization, zero modification of existing code/weights/labels, no disruption of running tmux sessions (`r2f2`). Facts and numbers only, no solutions.

---

## 1. Soft Label Bin Distributions & The [0.31, 0.34] Cluster

### 1.1 Bin Counts Across All 12 Classes (4,349 Non-Gold Studies)

Bins evaluated: `< 0.05`, `[0.05, 0.15)`, `[0.15, 0.45)`, `[0.45, 0.85)`, `> 0.85`.

#### A. `labels_hybrid_v1.parquet` (n = 4,349)
| Finding | < 0.05 | 0.05 - 0.15 | 0.15 - 0.45 | 0.45 - 0.85 | > 0.85 | [0.31, 0.34] Count (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACL** | 2,752 | 487 | 370 | 258 | 482 | 22 (0.5%) |
| **MCL** | 3,361 | 283 | 291 | 185 | 229 | 17 (0.4%) |
| **Medial Meniscus** | 1,061 | 682 | 871 | 674 | 1,061 | 100 (2.3%) |
| **Lateral Meniscus** | 2,668 | 499 | 439 | 286 | 457 | 45 (1.0%) |
| **Medial OA** | 1,514 | 778 | 838 | 622 | 597 | 93 (2.1%) |
| **Lateral OA** | 2,217 | 647 | 688 | 398 | 399 | 67 (1.5%) |
| **PF OA** | 1,481 | 632 | 754 | 569 | 913 | 77 (1.8%) |
| **Effusion** | 805 | 586 | 1,170 | 954 | 834 | 129 (3.0%) |
| **Synovitis** | 50 | 99 | 2,938 | 934 | 328 | 11 (0.3%) |
| **Baker's** | 2,746 | 557 | 425 | 308 | 313 | 47 (1.1%) |
| **Contusion** | 2,987 | 471 | 374 | 222 | 295 | 30 (0.7%) |
| **Fracture** | 3,745 | 240 | 174 | 88 | 102 | 16 (0.4%) |

#### B. `labels_sev_v1.parquet` (n = 4,349)
| Finding | < 0.05 | 0.05 - 0.15 | 0.15 - 0.45 | 0.45 - 0.85 | > 0.85 | [0.31, 0.34] Count (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACL** | 2,709 | 490 | 441 | 245 | 464 | 24 (0.6%) |
| **MCL** | 3,365 | 290 | 297 | 175 | 222 | 18 (0.4%) |
| **Medial Meniscus** | 1,065 | 684 | 889 | 670 | 1,041 | 102 (2.3%) |
| **Lateral Meniscus** | 2,674 | 497 | 450 | 285 | 443 | 46 (1.1%) |
| **Medial OA** | 1,514 | 778 | 838 | 622 | 597 | 93 (2.1%) |
| **Lateral OA** | 2,217 | 647 | 688 | 398 | 399 | 67 (1.5%) |
| **PF OA** | 1,019 | 1,046 | 1,135 | 1,149 | 0 | 0 (0.0%) |
| **Effusion** | 805 | 586 | 1,170 | 954 | 834 | **738 (17.0%)** |
| **Synovitis** | **0** | 51 | **3,748 (86.2%)** | 415 | 135 | 0 (0.0%) |
| **Baker's** | 2,746 | 557 | 425 | 308 | 313 | **317 (7.3%)** |
| **Contusion** | 2,987 | 471 | 374 | 222 | 295 | 30 (0.7%) |
| **Fracture** | 3,745 | 240 | 174 | 88 | 102 | 16 (0.4%) |

### 1.2 Identification of the [0.31, 0.34] Cluster
1. **In `labels_sev_v1.parquet`:**
   - **Effusion:** Has **738 studies (17.0%)** in $[0.31, 0.34]$.
   - **Baker's Cyst:** Has **317 studies (7.3%)** in $[0.31, 0.34]$.
   - **Synovitis:** Has **0** studies in $[0.31, 0.34]$, but **3,748 studies (86.2%)** collapsed into $[0.15, 0.45)$ with median `0.1677` and zero cases below `0.05`.
   - **PF OA:** Has **0** cases $> 0.85$ (max label is capped at `0.7850`).
2. **In `result_labels.csv` / `result_labels.parquet` (n = 4,407):**
   - **Synovitis** exhibits an extreme artificial peak: **3,487 studies (79.1%)** fall strictly in $[0.31, 0.34]$ (10th percentile is `0.3100`, 85th percentile is `0.3340`).

---

## 2. Gold Reports vs. Gold Labels (58 Gold Scans in `train.csv`)

### 2.1 Synovitis Stem Audit
- **Stems evaluated (case-insensitive regex):** `\b(synov\w*|sinov\w*|hoffa\w*|plic\w*|capsul\w*|hypertroph\w*|hyperplas\w*|inflam\w*)`
- **Overall Synovitis Contingency Table (n = 58):**

| Report Text Mention | True Gold Synovitis = 1 | True Gold Synovitis = 0 | Total |
| :--- | :---: | :---: | :---: |
| **Stem Mentioned (Yes)** | 16 | 12 | 28 |
| **No Stem Mentioned (No)** | **11** | 19 | 30 |
| **Total** | 27 | 31 | 58 |

*Diagnostic Fact:* **11 out of 27 (40.7%)** true Gold Synovitis positive scans contain **zero** mentions of any synovitis, Hoffa's fat pad, plica, capsule, or inflammation stems in their narrative report.

- **Split by Gold Effusion (0 vs. 1):**
  - **Effusion == 0 (n = 23):**
    - Stem Mentioned: 5 Pos / 6 Neg
    - No Stem Mentioned: 0 Pos / 12 Neg (100% negative predictive value when Effusion=0 and no stem).
  - **Effusion == 1 (n = 35):**
    - Stem Mentioned: 11 Pos / 6 Neg
    - No Stem Mentioned: **11 Pos / 7 Neg**  
    *Diagnostic Fact:* When Effusion is present, **11 of 22 (50.0%)** Gold Synovitis positives have **zero** text mention in the radiology report.

### 2.2 PF OA and Lateral OA Stem Audit
- **Stems evaluated:** `osteoarth*`, `arthros*`, `arthrit*`, `chondro*`, `cartilage*`, `knorpel*`, `osteophyt*`
- **Regex:** `\b(osteoarth\w*|arthros\w*|arthrit\w*|chondro\w*|cartilage\w*|knorpel\w*|osteophyt\w*)`

- **Patellofemoral Osteoarthritis (PF OA) (n = 58):**

| Report Text Mention | True Gold PF OA = 1 | True Gold PF OA = 0 | Total |
| :--- | :---: | :---: | :---: |
| **Stem Mentioned (Yes)** | 6 | 17 | 23 |
| **No Stem Mentioned (No)** | **15** | 20 | 35 |
| **Total** | 21 | 37 | 58 |

*Diagnostic Fact:* **15 out of 21 (71.4%)** true Gold PF OA positive scans have no generalized degenerative/cartilage stems mentioned in the text report.

- **Lateral Osteoarthritis (Lateral OA) (n = 58):**

| Report Text Mention | True Gold Lateral OA = 1 | True Gold Lateral OA = 0 | Total |
| :--- | :---: | :---: | :---: |
| **Stem Mentioned (Yes)** | 5 | 18 | 23 |
| **No Stem Mentioned (No)** | **6** | 29 | 35 |
| **Total** | 11 | 47 | 58 |

*Diagnostic Fact:* **6 out of 11 (54.5%)** true Gold Lateral OA cases have no osteoarthritis/cartilage stems in the text report.

---

## 3. The 300-Study Out-of-Fold (OOF) Set

### 3.1 Partitioning & Inclusion Audit
- **IDs Source File:** `/home/s05814802723/projects/knee/pc_code192/folds_rand4.json`
- **Fold Partitions:** 4,349 studies split into Fold 0 (1,088), Fold 1 (1,087), Fold 2 (1,087), Fold 3 (1,087).
- **Sampling Code (`pc_code192/train_knee.py:734`):**
  `sub_ids = random.Random(0).sample(list(oof_ids), min(300, len(oof_ids)))`
- **Model Training Inclusion Audit for the Fold 0 300-Study Evaluation Subset:**
  - `CoAtNet-384 192` (`d192_s50`): **IN TRAINING DATA** (Trained on all 4,349 studies without fold partitioning, `--folds 0`).
  - `ConvNeXt-Small` (`cnxt_s_breakthrough`): **NOT IN TRAINING DATA** (Fold 0 held-out model; 300 studies strictly unexposed).
  - `ConvNeXt-Tiny-F1` (`cnxt_f1`): **IN TRAINING DATA** (Fold 1 model; held out Fold 1, Fold 0 was part of its 3,262 training set).
  - `ConvNeXt-Tiny Sev2 Fold 0` (`cnxt_sev2_f0`): **NOT IN TRAINING DATA** (Fold 0 held out; 300 studies strictly unexposed).
  - `ConvNeXt-Tiny Sev2 Fold 1` (`cnxt_sev2_f1`): **IN TRAINING DATA** (Fold 1 held out; Fold 0 in training set).
  - `ConvNeXt-Tiny Sev2 Fold 2` (`cnxt_sev2_f2`): **IN TRAINING DATA** (Fold 2 held out; Fold 0 in training set).
  - `ConvNeXt-Tiny Sev2 Fold 3` (`cnxt_sev2_f3`): **IN TRAINING DATA** (Fold 3 held out; Fold 0 in training set).

### 3.2 Confident vs. Ambiguous Soft Label Counts in Fold 0 OOF-300
Confident defined as $\le 0.05$ (Negative) or $\ge 0.85$ (Positive); Ambiguous defined as $(0.05, 0.85)$.

| Bottleneck Class | Set | Positives ($\ge 0.85$) | Negatives ($\le 0.05$) | Ambiguous ($0.05 - 0.85$) | % Ambiguous |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Synovitis** | `labels_sev_v1` | 17 | **0** | 283 | **94.3%** |
| | `labels_hybrid_v1` | 34 | 4 | 262 | 87.3% |
| **PF OA** | `labels_sev_v1` | **0** | 71 | 229 | **76.3%** |
| | `labels_hybrid_v1` | 79 | 104 | 117 | 39.0% |
| **Lateral OA** | `labels_sev_v1` | 36 | 134 | 130 | 43.3% |
| | `labels_hybrid_v1` | 36 | 134 | 130 | 43.3% |
| **Lateral Meniscus** | `labels_sev_v1` | 31 | 50 | 219 | 73.0% |
| | `labels_hybrid_v1` | 32 | 189 | 79 | 26.3% |

---

## 4. Window Center Selection, Series Pooling & Physical Slice Spacing

### 4.1 Window Centers Selection Logic
- **Code Locations:**
  - `/home/s05814802723/projects/knee/pc_code192/train_knee.py:222-254` (`pick_centers`)
  - `/home/s05814802723/projects/knee/pc_code192/train_knee.py:357-358` (`StudyWindows.__getitem__`)
  - `/home/s05814802723/projects/knee/train_knee.py:217-230` (`StudyWindows._centers`)
- **Mechanism:**
  - **In Evaluation / Validation (`train=False`):**
    Window candidate centers `cs` are filtered such that if `clean_windows=True`, all 3 slices of the triplet ($c-1, c, c+1$) belong to the exact same slot (`slot_of(c-1) == slot_of(c) == slot_of(c+1)`) and have valid pixel masks ($mask > 0$).
    Deterministically samples `count` indices across the candidates via:
    `idx = np.linspace(0, len(cs) - 1, count).round().astype(int)`
    `return [cs[i] for i in idx]`
  - **In Training (`train=True`):**
    Candidate centers `cs` are replicated and shuffled: `pool = cs * reps; random.shuffle(pool); return pool[:count]`.
  - **Two-Scale Zoom Mode (`--zoom_mm > 0`):**
    Emits 2 crops per center (full 140 mm field + centered 100 mm close-up). Therefore, `n_centers = (k + 1) // 2` is requested (for `--k 48`, exactly 24 centers are chosen, generating $24 \times 2 = 48$ tiles).

### 4.2 Series Pooling Within Planes
- **Is series pooling performed within a plane before sampling?**
  - **NO.** Series are strictly **NOT** pooled or concatenated.
  - **Code References:**
    - `rsna_preproc.py:111-125` (`pick_series_for_slot`)
    - `verify_inference_preproc.py:89-94`
    - `submission_kernel.py:114-120` (`pick_series`)
  - **Mechanism:**
    For each slot, `pick_series_for_slot` searches for the first available series matching `(plane, fluid_sensitivity)`. If multiple series share the same plane and fluid tag, only the first candidate is taken, and all others are ignored. Slices are sampled solely from that single isolated series.

### 4.3 Physical Inter-Slice Distance on the 58 Gold Studies (from DICOM `ImagePositionPatient`)
Measured across all 58 Gold studies (336 total series parsed directly from `raw/rsna-knee-abnormality-detection.zip` via projection along slice normal vector $\vec{n} = \vec{r} \times \vec{c}$):

| Model Family | Plane | Sampled Gaps Count | Studies Eval | Median Gap | P90 Gap | Max Gap | % Studies with Gap > 4 mm | % Gaps > 4 mm |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CoAtNet-384 (96 slices)** | **Axial** | 986 | 58 / 58 | **6.60 mm** | 9.90 mm | 12.60 mm | **93.1%** | 89.0% |
| | **Sagittal** | 2,626 | 58 / 58 | **4.00 mm** | 6.60 mm | 9.04 mm | **100.0%** | 51.4% |
| | **Coronal** | 1,470 | 58 / 58 | **6.60 mm** | 10.40 mm | 14.40 mm | **100.0%** | 84.7% |
| **192-Slice Depth Family**<br>*(CoAtNet-192, ConvNeXt-Tiny zoom,<br>ConvNeXt-Small, ConvNeXt-Tiny Sev2)* | **Axial** | 2,262 | 58 / 58 | **3.30 mm** | 5.00 mm | 7.00 mm | **55.2%** | 34.2% |
| | **Sagittal** | 4,446 | 58 / 58 | **3.30 mm** | 4.50 mm | 5.50 mm | **41.4%** | 18.6% |
| | **Coronal** | 3,570 | 58 / 58 | **3.30 mm** | 4.50 mm | 7.20 mm | **46.6%** | 19.0% |
| **CoAtNet-384 (44 slices, v5s2)** | **Sagittal** | 2,398 | 58 / 58 | **4.40 mm** | 6.60 mm | 9.12 mm | **100.0%** | 60.6% |

---

## 5. Positional Encoding vs. Attention-Weighted Pooling & Slice Reversal

### 5.1 Architecture Inspection
- **Code Reference:** `train_knee.py:350-356` and `submission_kernel.py:260-265` (`RaptorClassifier.head`):
  ```python
  def head(self, feats):
      h = self.norm(feats)
      a = self.att(h)
      a = torch.softmax(a, dim=1)
      pooled = torch.einsum("bkn,bkf->bnf", a, h)
      logits = (pooled * self.clsW).sum(-1) + self.clsb
      return logits
  ```
- **Findings:**
  - Pooling is a **pure attention-weighted sum** over the window/bag dimension ($K$).
  - **Zero positional embeddings** and **zero sequence layers** are present in any released model family (`CoAtNet-384 96`, `CoAtNet-192`, `ConvNeXt-Tiny zoom`, `ConvNeXt-Small`, `ConvNeXt-Tiny Sev2`).
  - While an experimental `--seq_head` GRU option exists in `pc_code192/train_knee.py:359-360` and `train_bilstm.py`, none of the submitted or ensembled checkpoints used it.

### 5.2 CPU Test: Permutation & Slice Reversal Invariance
- **Test Setup:** Executed on CPU using `torch.float32` with randomly initialized `RaptorClassifier` head on input tensor `(B=1, K=48, F=384)`.
- **Results:**
  - Original vs. Slice-Reversed (`feats.flip(1)`):
    $$\max |\text{Logits}_{\text{orig}} - \text{Logits}_{\text{rev}}| = 2.42 \times 10^{-8}$$
  - Original vs. Arbitrary Slice-Permuted (`feats[:, torch.randperm(48), :]`):
    $$\max |\text{Logits}_{\text{orig}} - \text{Logits}_{\text{perm}}| = 2.89 \times 10^{-8}$$
- **Conclusion:** Reversing slice order changes output by machine epsilon ($< 3 \times 10^{-8}$), empirically verifying **mathematical permutation invariance**.

---

## 6. Training Profiles Across All Model Families

Parsed directly from training run logs on the training server:

| Model Family / Checkpoint | Run Log | Epochs | Seconds / Epoch (Mean) | Seconds / Step (Mean) | Total Wall-Clock (s / h) | Hardware |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CoAtNet-384 192** (`d192_s50`) | `pc_code192/train_coat_d192_s50.log` | 20 | 2,419.8s | 1.049s | 48,395s (13.44h) | NVIDIA L4 (24GB) |
| **ConvNeXt-Small** (`cnxt_s_breakthrough`) | `pc_code192/train_breakthrough.log` | 15 | 1,439.7s | 0.683s | 21,596s (6.00h) | NVIDIA L4 (24GB) |
| **ConvNeXt-Tiny zoom Fold 0** (`cnxt_f0`) | `pc_code192/train_cnxt_f0.log` | 15 | 924.9s | 0.432s | 13,873s (3.85h) | NVIDIA L4 (24GB) |
| **ConvNeXt-Tiny zoom Fold 1** (`cnxt_f1`) | `pc_code192/train_cnxt_f1.log` | 15 | 921.5s | 0.432s | 13,823s (3.84h) | NVIDIA L4 (24GB) |
| **ConvNeXt-Tiny Sev2 Fold 0** (`cnxt_sev2_f0`) | `pc_code192/train_cnxt_sev2_f0.log` | 15 | 931.5s | 0.434s | 13,972s (3.88h) | NVIDIA L4 (24GB) |
| **ConvNeXt-Tiny Sev2 Fold 1** (`cnxt_sev2_f1`) | `pc_code192/train_cnxt_sev2_f1.log` | 14 | 923.9s | 0.433s | 12,935s (3.59h) | NVIDIA L4 (24GB) |
| **ConvNeXt-Tiny Sev2 Fold 2** (`cnxt_sev2_f2`) | `pc_code192/train_cnxt_sev2_f2.log` | 14 | 923.3s | 0.432s | 12,926s (3.59h) | NVIDIA L4 (24GB) |
| **ConvNeXt-Tiny Sev2 Fold 3** (`cnxt_sev2_f3`) | `pc_code192/train_cnxt_sev2_f3.log` | In Prog | ~925s | ~0.432s | Running (tmux `r2f2`) | NVIDIA L4 (24GB) |
| **CoAtNet-384 96** (`v6d24_s9`) | `train_coat_v6d24_s9.log` | 21 | 2,363.7s | 1.050s | 49,637s (13.79h) | NVIDIA L4 (24GB) |
| **CoAtNet-384 96** (`v6d_s7`) | `train_coat_v6d_s7.log` | 16 | 2,364.7s | 1.050s | 37,835s (10.51h) | NVIDIA L4 (24GB) |
| **CoAtNet-384 96** (`v6`) | `v6.log` | 16 | 2,363.6s | 1.050s | 37,817s (10.50h) | NVIDIA L4 (24GB) |
| **CoAtNet-384 44** (`v5s2`) | `v5s2.log` | 17 | 1,818.9s | 0.812s | 30,921s (8.59h) | NVIDIA L4 (24GB) |

---

## 7. Inference Timing & Hardware Benchmark Evidence

### 7.1 Provenance of the 1.1 Hour Number
- **Source Citation:** `gold_and_augmentation_check.md:237`:
  `- Total projected runtime for 1,300 studies: $\approx 1.1\text{ hours}$, safely within the $2.5\text{h}$ operational target and $9.0\text{h}$ hard cutoff.`
- **Underlying Measurement:**
  - Measured in `verify_inference_preproc.py` on the remote server's **NVIDIA L4 (24GB)** GPU across 20 full raw DICOM studies:
    - Raw DICOM read + decompression + preprocessing: $\approx 1.65\text{s} - 1.80\text{s}$ / study.
    - Model forward pass: $\approx 0.45\text{s} - 0.70\text{s}$ (CoAtNet-96) to $1.20\text{s}$ (ConvNeXt).
    - End-to-end total per study: $\approx 2.12\text{s} - 3.05\text{s}$ / study.
    - Extrapolation across 1,300 test studies:
      $$1,300 \times 3.05\text{ s} = 3,965\text{ seconds} = 1.10\text{ hours}$$

### 7.2 Kaggle Production Target Hardware
- **Target GPU on Kaggle:** **2x Tesla T4 (16GB)**.
- **Evidence:**
  - `submission_kernel.py:95-98`: Explicit multi-GPU dispatch across `cuda:0` and `cuda:1` round-robin.
  - `t4_timing_notebook.py:48-53`: Explicit Kaggle T4 benchmark establishing FP16 vs. FP32 throughput and memory limits without BF16 support.

---

## 8. Saved Forward-Only vs. Mirror-Only Predictions on Gold Set

- **Status:** **NOT AVAILABLE / NOT CHECKED**.
- **Audit Verification:**
  - Searched all 221 `.npz` prediction files across `/home/s05814802723/projects/knee/` and `/home/s05814802723/projects/knee/pc_code192/`.
  - Every `.npz` (e.g. `raptor_gold_v6.npz`, `raptor_gold_d192_s50.npz`, `raptor_oof_cnxt_*.npz`) strictly stores 3 keys: `pred`, `truth`, `ids`.
  - No saved `.npz` file logged separate `pred_fwd` and `pred_mirror` prediction arrays on disk.
  - As per the strict READ-ONLY and NO-GPU rules, new inference was not executed.

---
*Report generated strictly using static code audits, parsed training logs, and CPU-only numerical verification.*
