# RSNA Knee Abnormality Detection: Top-10 System Design

Date: 2026-10-07 · Deadlines: merge **Oct 15**, final **Oct 22** (23:59 UTC)
Goal: **top 10, ideally 1st** (public LB ≈ 0.956–0.958 today). Gold is the floor.
Approach: **B, labels + distillation + external data**, with **C, team merge** as the backup.

## 1. Facts this design rests on

| Fact | Consequence |
|---|---|
| Test set has **images only** (`test.csv` = StudyInstanceUID) | Reports are useful only as training labels |
| 58 gold-labelled studies, 4,349 report-only | Gold-58 is the only trustworthy validation (SE ≈ ±0.02 macro-AUC) |
| Best public label table (`llm_labels_v4_blend`) = **0.893** AUC vs gold | Weak findings: Synovitis 0.79, Fracture 0.79, Lateral OA 0.83, Contusion 0.86 |
| Public ensemble (d4-blend) ≈ **0.943** public LB | Our baseline and our teacher |
| `train_knee.py` (public): 2.5D attention-MIL CoAtNet on the preprocessed corpus | Reused as-is; no new model code |
| Kaggle inference: ≤ 9 h, internet off; weights must be a public Kaggle dataset | Students must be fast enough to add to the ensemble |
| Metric = macro AUC → only **ranking within each column** matters | Blends are done in rank space |

## 2. Architecture

```
PC (4070S + local Qwen)
  [1] label_llm.py   reports → 12 probabilities per study          → llm_qwen.parquet
  [2] calibrate.py   combine labellers, calibrate on gold-58       → labels_cal.parquet
Kaggle (T4×2)
  [3] teacher.ipynb  public ensemble predicts the 4,407 TRAIN studies → teacher.parquet
PC
  [4] targets.py     per-finding mix: α·teacher + (1−α)·labels_cal (+ external) → targets.parquet
node1 (L4 24 GB, 251 GB RAM)
  [5] train_knee.py  students: CoAtNet + 1–2 other backbones, 5-fold → weights + OOF preds
PC
  [6] blend.py       fit rank-blend weights on OOF + gold-58        → blend.json
Kaggle
  [7] submit.ipynb   public ensemble + our students + blend.json    → submission.csv
```

### What each component is for
- **[1] label_llm.py.** A second, independent reading of every report by Qwen3.5-9B (local, `qwen start`). Its errors differ from v4's, so combining the two cancels noise. Output: a probability per finding, not a yes/no answer, because hedged reports ("suspected tear") carry information.
- **[2] calibrate.py.** Maps each labeller's raw score to a true probability, per finding, fitted on gold-58 (logistic / isotonic). The labellers are then combined with weights estimated from how well each agrees with gold. A labeller that is bad at synovitis gets little say on synovitis.
- **[3] teacher.ipynb.** The core of distillation. The 0.943 ensemble looked at the *image*, and its opinion is often a better label than the report text. It runs on Kaggle because the public weights and notebook already live there; one CPU+GPU run gives predictions for all 4,407 training studies.
- **[4] targets.py.** The student's training target per finding: `α·teacher + (1−α)·labels_cal`. α is chosen per finding by AUC on gold-58. Effusion may trust the teacher; ACL may trust the report.
- **[5] train_knee.py (reused).** The proven public pipeline. We only patch what's needed: reading external studies, more backbones. 5 folds produce out-of-fold (OOF) predictions for the blender.
- **[6] blend.py.** Rank-space weighted blend of the public ensemble and our students, with weights fitted on OOF + gold, **never on the public LB**. This protects the private-LB rank.
- **[7] submit.ipynb.** Fork of the working d4-blend (Version 2) with our weights attached as a Kaggle dataset.

### External data (lever 3)
Candidates: **fastMRI+** (knee pathology boxes), **OAI / MOAKS** (OA, effusion, BML, meniscus, ACL scoring), **MRNet** (ACL, meniscus, abnormal). They enter at [4] and [5] as extra training studies, with labels mapped to the subset of our 12 findings they cover (masked loss on the rest). **Blocked on your research** (§6). Everything else proceeds without it.

## 3. Machines

| Machine | Role | Notes |
|---|---|---|
| node1 (`ssh node1`) | All student training | /home is shared: keep under ~60 GB. Run from tmux, with the env python called directly |
| PC | Qwen labelling, calibration, targets, blend, control | `qwen start/stop`; free RAM before long runs |
| Kaggle | Teacher inference, final submissions | 30 GPU-h/week; 5 submissions/day |
| Azure MinisAi | **Never used** | Company production |

## 4. Schedule and decision gates

| Day | Work | Gate / output |
|---|---|---|
| Oct 7 (tonight) | `v4base` reference student trains on node1 | Gold-58 AUC of a single student on v4 labels |
| Oct 8 | [1] Qwen labelling (PC, ~3–4 h), [2] calibration · [3] teacher notebook on Kaggle · baseline LB score in | **G1:** `labels_cal` > 0.893 vs gold → use it; otherwise keep v4 |
| Oct 9 | [4] targets → student #1 (CoAtNet, distilled) on node1 · your external-data research back | **G2:** go/no-go on external data |
| Oct 10–11 | Student #2 (different backbone) · external data ingest if G2 = go | |
| Oct 12 | [6] blend, [7] submit our first own blend | **G3:** LB ≥ 0.948 → keep going solo; < 0.946 → push merge (C) hard |
| Oct 13–14 | Merge outreach with OOF numbers (C) · more students | **Oct 15: merge deadline** |
| Oct 15–20 | Noisy-student round 2 (retrain on the new ensemble's predictions), efficiency variant | |
| Oct 21–22 | Final picks: 1 robust OOF-fitted blend + 1 efficiency-optimised blend | **Oct 22 final** |

**GPU budget (node1):** about 7–8 h per CoAtNet student run, so ~2 runs/day and ~25 runs total. 5-fold for every model is too expensive. **Rule:** full-data runs, validated on gold-58, for exploration; 5-fold only for models that make the final blend.

## 5. Risks and handling

| Risk | Handling |
|---|---|
| Overfitting gold-58 (only 58 studies) | Use it only for go/no-go and α; prefer OOF-on-teacher for blend weights; never pick on public LB |
| Teacher errors copied by the student | Mix with labels (α < 1) + strong augmentation (noisy student) |
| Kaggle hidden-run failure (decode errors, stale dataset versions) | Keep Version 2 as a known-good fallback; every new submit notebook is diffed against it |
| node1 shared/killed | Checkpoints every epoch (script already does); resumable via `--time_budget_h` |
| External data licence incompatible with prize open-sourcing | No use until licence verified (§6) |
| Inference time over 9 h | Each student timed on Kaggle before adding; d4-blend uses ~2.5–3 h |

## 6. Research tasks for you

1. **fastMRI+**: licence, findings covered (meniscus, ACL, effusion, contusion?), download size.
2. **OAI / MOAKS**: registration and approval time; is MOAKS scoring downloadable?
3. **MRNet**: licence; does it allow open-sourcing and commercial use (prize rule 3.6c)?
4. **Competition forum**: any host ruling on external data, or posted results using it.

## 7. Testing / verification

- Every label table: AUC vs gold-58 + **gold-copy check** (`exact_match` must be ≈ 0).
- Every student: gold-58 macro-AUC logged per epoch (built into `train_knee.py`).
- Blend: OOF AUC, plus gold-58 AUC, **before** spending a submission.
- Submit notebooks: run on the 3-study dummy test first; confirm `submission.csv` has 13 columns and no NaN.

## 8. Code rules

- Claude writes all code; you review every line before it goes to GitHub.
- One small script per component, plain Python, no framework or config system.
- Competition data and reports never leave your machines; no hosted LLM APIs.
