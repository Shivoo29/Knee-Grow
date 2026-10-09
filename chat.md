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
