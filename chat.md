# Handoff: RSNA Knee Abnormality Detection (Kaggle 2026)

Read this first if you are a new model/session continuing this work. **Solo (teammates declined, Oct 8).** Last updated: 2026-10-09 ~01:05 IST.

## 1. Goal and working agreement
- **Goal: top 10 on the leaderboard, ideally 1st.** Gold (~rank 20) is the floor. Leaderboard-first decisions.
- **AI writes all code**; the user reads and understands every line before pushing to their GitHub → keep code small, plain, commented where non-obvious.
- **Always explain** what you are doing and what each piece is for. Flag things the user should research.
- User talks Hinglish; prefers short, direct answers.
- **Deadlines (23:59 UTC):** team merge **Oct 15**, final submission **Oct 22**. 5 submissions/day.

## 2. Competition facts
- Predict 12 findings per knee MRI study: `ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`. Metric: **macro ROC-AUC**.
- Submission: `submission.csv` with `StudyInstanceUID` + 12 columns, produced by a **Kaggle notebook** (≤9 h, T4×2, internet off).
- Train: 4,407 studies. `train.csv` = `StudyInstanceUID, Report` + 12 labels; **only 58 rows have labels (gold)**; the rest have only the free-text report. `train_series.csv` = `Fluid_Sensitive, Fat_Suppression, Anatomical_Plane` per series.
- **Test has images only (no report).** Reports are useful only as a training-label source.
- No site column (can only get it from DICOM headers).
- Rules: external data and models allowed if free and public. **Do not paste report text or images into any chat/hosted API** (Rule 4.b data security). Prize winners must open-source (CC-BY-NC 4.0, plus an OSI licence that does not limit commercial use).

## 3. Leaderboard (Oct 8, 5,426 teams)
| Rank | 1 | 10 | 20 (gold) | 50 | 100 | 265 (silver) | 530 (bronze) |
|---|---|---|---|---|---|---|---|
| Score | 0.964 | 0.961 | 0.959 | 0.956 | 0.952 | 0.946 | 0.944 |

**Us: 0.943, rank ~1300** (the public fork score, shared by hundreds of teams).

## 4. Machines (see also memory `machines.md`)
| Machine | Access | Use | Notes |
|---|---|---|---|
| **node1** | `ssh node1` (alias in `~/.ssh/config`, key `~/.ssh/id_ed25519`) | Main training. L4 24 GB, 64 cores, 251 GB RAM | **Shared box: someone kills jobs after ~7 AM IST → plan runs to finish before 7 AM** (or ask the user about daytime availability). /home is shared NFS: keep our footprint ≲60 GB. Conda env `knee` (torch 2.11 cu128; the driver only supports CUDA 12.8). **In tmux call `~/miniconda3/envs/knee/bin/python` directly** (`conda run` fails). Set env vars *inside* the tmux command. |
| **PC (local)** | this machine | Control, analysis, second trainer | RTX 4070S 12 GB, 30 GB RAM (often 20+ GB used). Project venv `.venv` (torch cu128, timm, pandas, sklearn, pyarrow). **The user runs their own `finetune.py` on this GPU (~6.4 GB). Never kill it.** |
| Local LLM | `qwen start / stop / status / logs / chat` | Qwen3.5-9B Q6_K via llama.cpp, `127.0.0.1:8080/v1` | systemd user service, disabled by default. Source: `~/.local/src/llama.cpp`, model `~/models/`. |
| Kaggle | `kaggle` CLI (user `shivoo29`) | Submissions, inference | `kaggle kernels push` is allowed in `.claude/settings.local.json`. 30 GPU-h/week. |
| **Azure "MinisAi"** | — | **NEVER USE** | Company production backend. |

## 5. Files
**PC: `~/Documents/still_learning/rsna-knee-abnormality-detection/`**
- `status.sh`: one command showing submissions, notebook status, node1 epochs and download progress.
- `data/train.csv` (+ test/sample csvs), `data/llm_labels_v4_blend.csv`, `data/llm_qwen.*` (Qwen labels for the 58 gold studies only)
- `ds/corpus1`, `ds/corpus2`: preprocessed image corpus (21 GB), `ds/comp/train.csv`
- `train_knee.py`: the public training script (from `nartaa/rsna-knee-hpo-assets`), used as-is
- `labels_v4.parquet`: training labels (v4_blend, gold rows removed)
- `label_llm.py`: Qwen report labeller (result: 0.850 vs gold, **worse than v4**, abandoned)
- `docs/superpowers/specs/2026-10-07-rsna-knee-top10-design.md`: original design spec (partly outdated by the research)
- `reports/Knee MRI prior work and tricks.md`: **research report, read it**; notes in `research_notes/`

**node1: `~/projects/knee/`**
- `ds/corpus1`, `ds/corpus2`, `ds/comp/train.csv`, `ds/rsna-knee-*` (label tables, folds `folds_v0.json`)
- `train_knee.py`, `labels_v4.parquet`, `eval_labels.py`
- `raptor_ft_v4base.pt` (+ `_swa.pt`, `raptor_gold_v4base.npz`, `v4base.log`): **our first trained model**

## 6. What has been done
1. Accepted rules, verified identity. Forked `pjmathematician/rsna-knee-d4-blend` as `shivoo29/rsna-knee-d4-blend`; removed its last 3 "OUR LEG" cells (they need the author's private datasets). **Version 2 submitted → 0.943.**
2. Label tables vs gold-58 (macro AUC): `llm_labels_v4_blend` **0.893** (best; weak: Synovitis .79, Fracture .79, Lat OA .83, Contusion .86), v2 0.887, full 0.878. Our Qwen-9B 0.850, and blending it in hurts. Tables in `nartaa/...` and `dreaddevelopment/...` exclude the gold rows (cannot be scored). `train_labels_v1.parquet` likely contains gold copies (`y_`/`m_` columns): never use it for validation.
3. **First own model `v4base`** on node1: CoAtNet-rmlp-2 384, 10 epochs, bs 4, accum 2, grad_ckpt, ~40 min/epoch. **Gold-58 macro AUC 0.8925**. Per finding: ACL .933, MCL .859, MedMen .975, LatMen .837, MedOA .957, **LatOA .766, PFOA .798**, Effusion .952, **Synovitis .827**, Baker's .995, Contusion .954, Fracture .858.
4. Teacher probe notebook `shivoo29/rsna-knee-teacher-probe` (public ensemble predicting 58 gold + 300 train studies via a fake `RSNA_COMP_ROOT`): **COMPLETE, output not analysed yet** (download failed). Concern: the public models trained on these studies → predictions may be memorised.
