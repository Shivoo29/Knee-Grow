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
