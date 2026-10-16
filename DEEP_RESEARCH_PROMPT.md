# Deep research brief: RSNA Knee Abnormality Detection (Kaggle, 2026) — from public LB 0.956 to top 5 (≥ 0.966)

## 0. Your role and what you must return

You are the research lead for a solo Kaggle team. A separate engineer (an AI coding agent that has our code, data,
two GPUs and full context) will implement your recommendations during the last 6 days of the competition.
You cannot see our data and must not ask for it. Your job is to research **as thoroughly as possible, across every
relevant field**, and return **one ranked, implementation-ready plan** whose every claim is sourced.

Fields to cover: this competition's own forum, code and rulings; musculoskeletal (MSK) radiology and the official
label criteria; knee-MRI deep learning; learning with noisy labels and a small trusted set; weak supervision and
learning from radiology reports; multiple-instance learning (MIL) and 2.5D/multi-view architectures; AUC/ranking
optimisation; domain shift across scanners; ensembling and model-selection maths for macro ROC-AUC; Kaggle
inference engineering; and transferable recipes from past Kaggle medical competitions. Where existing methods do
not fit, derive new ones (equations welcome) and say why they should work here.

Depth and correctness matter more than breadth. A wrong number costs us days. Today is **2026-10-16**; the final
submission deadline is **2026-10-22 23:59 UTC**.

---

## 1. The competition (facts)

- Task: per-study probability for 12 findings on knee MRI: ACL injury, MCL injury, Medial meniscus tear, Lateral
  meniscus tear, Medial (tibiofemoral) OA, Lateral (tibiofemoral) OA, Patellofemoral (PF) OA, Effusion, Synovitis,
  Baker's cyst, Bone contusion, Fracture. Metric: **macro ROC-AUC over the 12 findings** (rank-based per column).
- Data: each study = several DICOM series (sagittal / coronal / axial; flags `Fluid_Sensitive`, `Fat_Suppression`).
  Series typically 20–45 slices (median 30), long tail to a few hundred. DICOMs are stripped to 86 allow-listed tags
  (patient age is removed; sex is present). Mixed transfer syntaxes (JPEG Lossless, JPEG 2000, uncompressed).
  Diverse international sites, scanners and protocols.
- Training set: **4,349 studies with only the free-text radiology report** (reports are **multilingual**), plus
  **58 "gold" studies with expert labels**. The forum (hengck23) states the gold labels were made by radiologists
  **reading the images, independently of the report** (example: a gold study labelled Effusion = 1 whose report
  says "No knee effusion is present"). Test labels are clinically adjudicated, image-based.
- Test: ~1,300 studies, no reports. Public LB is reportedly ~30% (~390 studies; noise SD of one submission
  ≈ 0.003 at macro AUC 0.955, per a participant's simulation). Host warns prevalence may differ between train,
  public and private test.
- Kaggle limits: code competition, inference on 2× T4 (16 GB each), **9 h**, no internet; 5 submissions per day.
- Official label definitions exist in the host's competition overview / discussion post (topic 733343, host
  "javaduke95"); a participant quotes one: *"Medial meniscus tear: Abnormal signal that definitely contacts the
  meniscal surface on at least two images, or a morphologic abnormality such as a truncated, diminutive, or displaced
  fragment … Intrasubstance degeneration that does not reach the surface …"*. **We do not have the full text.**

## 2. Leaderboard (2026-10-15 snapshot, 6,013 teams)

#1 0.967 (2 teams); ≥0.964: 14 teams; ≥0.963: 24; ≥0.960: 66; ≥0.957: 266; ≥0.956: 1,193 (a big tie of public-notebook
forks). **Us: rank 942 at 0.956.** Top 5 needs ≈ 0.966. The best public notebooks are 0.956–0.957.

## 3. What we have measured (do not re-propose tested ideas unless you explain why our test was flawed)

### 3.1 Leaderboard results of our submissions
| Submission | Public LB |
|---|---|
| Fork of heliosli "R7 + Raptor" (Raptor CoAtNet 384 + heliosli R7 ConvNeXt-small, 50/50 per-finding rank blend) | **0.956** |
| Older public base (Raptor 70% + heliosli compact ConvNeXt 30%) | 0.954 |
| R7 ConvNeXt alone (no OAI-trained Raptor) | 0.949 |
| R7 + our label-trained CoAtNets (rank weight 0.35) | 0.952 |
| R7 + Raptor + our label-trained CoAtNets (0.25) | 0.954 |
| R7 + Raptor + our "Specialist 2" only on PF OA and Lateral OA (0.5) | 0.954 |
| base + our distilled students (0.15) | 0.954 |
| Our distilled students alone (2 CoAtNets) | 0.948 |
| Our label-trained CoAtNets alone (3 seeds) | 0.941 |
Pending: fork of heliosli's newest "Blend mine in" notebook (Raptor 384 + Raptor 224 + heliosli R6 ConvNeXt with
flip TTA, per-finding weights).

### 3.2 Gold-58 measurements (noise: ±0.02 macro, ±0.06 per finding)
- Report-derived labels: mean of 17 public LLM label tables **0.899**; best single tables ≈ 0.89–0.90 (JEV hybrid
  0.904, Claude Opus 4.5 0.892, Qwen-32B 0.869, a GPT model 0.849). Bigger LLMs did not help.
- Public image "teacher" (Raptor + compact) **0.932** (Raptor may have seen gold in training).
  Per finding (teacher / LLM-mean): ACL .975/.993, MCL .959/.993, Med meniscus .978/.931, Lat meniscus .917/.892,
  Med OA .989/.922, **Lat OA .793/.845**, **PF OA .869/.934**, Effusion .978/.887, **Synovitis .842/.802**,
  Baker's .980/.913, Contusion .946/.866, Fracture .943/.812.
- 0.6 × image prediction + 0.4 × report labels scores 0.939–0.941 on gold (both with the teacher and with our own
  out-of-sample models) — but reports are absent at test time.
- Positive rate gold vs train report labels (%): ACL 41/23, MCL 16/16, Med men 45/46, **Lat men 40/23**,
  Med OA 26/37, Lat OA 19/29, PF OA 36/44, Effusion 60/50, Synovitis 47/39, Baker's 21/31, Contusion 33/27,
  **Fracture 31/18**. (Gold may be an enriched sample.)
- Our models (CoAtNet `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` unless stated):
  label-trained (report labels) seeds 0.915–0.927; **distilled from the public teacher (0.6 teacher + 0.4 report)
  0.927–0.931** (best 0.9312); 5-seed average 0.926, 4-distilled average 0.931; DINOv2-B (distilled) 0.922;
  ConvNeXt-base 0.895 and EVA02-base 0.905 (older recipe); coronal compartment-ROI specialist 0.906;
  sagittal-fluid + axial specialist 0.909 (best PF OA 0.888 and Lat OA 0.847 of our models).
- Seed choice per finding (split-half test): picking the best model per finding gains +0.0064, plain averaging +0.0075.
- Cross-finding mixing on gold (e.g. synovitis ← effusion): +0.006–0.017, i.e. noise level.
- Student-vs-teacher Spearman 0.958, DINOv2-vs-teacher 0.942; adding either to the teacher at weight 0.15–0.3 moves
  gold by ≤ 0.001.

### 3.3 Experiments that failed, and our explanation (challenge it if you disagree)
1. **Distillation from the public teacher**: students reached the teacher's level alone (LB 0.948) but added 0 in
   the blend. Explanation: the teacher was trained on our training studies, so its predictions there are in-sample
   and mostly echo the labels it was trained on; students become near-copies.
2. **OOF relabelling**: we trained 4 folds on report labels only (each 0.904–0.914 on gold), built targets
   0.6 × out-of-fold image prediction + 0.4 × report label (0.938 "quality" on gold), and trained final models on
   them: **gold 0.918**, no better than plain report labels. Explanation: the targets look good only because they
   contain report information unavailable at test time, and the image part came from models no stronger than the
   student. Tell us whether any variant (EM, privileged information, stronger/different teachers, masking) can work.
3. Mirror TTA (no gain), lower resolution (CoAtNet-rmlp fixed at 384), a 3-slice-window bug fix + augmentation +
   slot embedding (T05 flags; ablation in progress), label down-weighting where teacher and report strongly
   disagree (only 0.5% of cells affected).

### 3.4 Our current pipeline
2.5D: 3 adjacent slices → RGB window → CoAtNet → **per-finding attention-MIL** head (12 attention queries), no
positional information. Corpus: 5 series slots per study (Sag fluid 26, Sag non-fluid 22, Cor fluid 18, Cor
non-fluid 12, Axial 18 = 96 slices), **140 mm crop around the image centre** → 384 px (~0.36 mm/px), per-series
2–98% percentile scaling to uint8 (bright fluid clips). Training: 12 random windows per step, BCE with soft
labels, OneCycle, 16 epochs, SWA of top-3 epochs, gold-58 for model selection. Inference: 94 windows per study.
**Found flaw:** the 96-slot layout keeps only ~65% of the slices of the 5 chosen series (148 → 96 per study) and
"adjacent" windows are really 1.5–2 slices apart; 15% of series (incl. 1,491 extra axial series) are never used.
A 192-slice corpus (40/40/36/36/40, keeps ~90%) is built; runs with 12 and 24 windows per step finish Oct 17.

### 3.5 What the best public model does (heliosli R6/R7, from its manifest and code)
ConvNeXt-small (ImageNet-22k) with the first 16 blocks **frozen**; **176 training windows per study**; the knee is
located first (an SKM-TEA-trained anchor model, fallback: an intensity-based `knee_anchor()`), then **100 mm
close-ups in millimetres** around it (kneecap: 40 mm anterior + 25 mm superior; posterior: 35 mm; medial/lateral
halves: ±30 mm on coronal) plus the full view; thin slices averaged into 3.5 mm slabs; random 352 crop; 4-fold CV;
a "state" head predicting graded categories from reports (e.g. Baker's none/small/moderate/large/ruptured);
matched-pair ranking loss (supported in code); anatomical mirror with medial/lateral label swap; flip TTA; a
fastMRI+ auxiliary task (licence risk); per-finding blend weights (R6 gets 80% of Lateral OA, 65% of Baker's;
the CoAtNets get 80% of Synovitis). We are building our own version of the knee-anchored close-ups (task T11).

## 4. Constraints you must plan within

- Hardware: PC with RTX 4070 Super 12 GB (≈ 28 min per CoAtNet epoch at 12 windows); "node1" with an L4 24 GB,
  64 CPU cores, 251 GB RAM (≈ 38–40 min per epoch). One training job per GPU at full speed; two at once halve both.
- Time: ~6 days, so roughly 8–10 full training runs in total, plus Kaggle commit runs (weekly GPU quota ~30 h).
- Kaggle inference: 2× T4, 9 h, offline; the two public components already use a large share of the budget.
- Rules: never share competition report text or images with any external service (rule 4.b). External data:
  OAI is disputed (topic 749145, rule 2.6.b, no host ruling yet); KneeCoT is banned; fastMRI likely excluded
  ("institutional research agreement" wording); public weights trained on competition data were ruled OK (Oct 14).
  Verify each of these against the primary source and quote it. One final pick must avoid disputed data.

## 5. Research questions (answer all; priority order A > B > G > E > D > C > J > F > I > H)

### A. This competition (do first)
A1. Find and quote **verbatim the host's definitions/criteria for all 12 findings** and any annotation protocol
    (who labelled, adjudication, thresholds, bilateral studies, multilingual reports). Sources: competition
    Overview/Data tabs, discussion 733343 (first post), host replies.
A2. Every host ruling on external data, public weights, commercial LLMs on reports (topic 733965) and rule 2.6.b,
    with quotes and dates; classify what is safe for a final submission.
A3. Every technical hint from top-LB participants (hengck23, Cody_Null, tascj/"吃白饭的大肥鱼", datnt114,
    CentralyDS, azamatt, "RB & Matheus & Tucker & Tony", Sergey Bryansky, joinT fusion, Ilya Kolb, MURAD, Evan zhu,
    Prateek Grover, Ali Ozan Memetoglu, Kirderf, varshitha kolupuri, etc.): quote and link. Include topic 748111
    ("Single sub 0.960!") and any Rednote/Xiaohongshu/Zhihu/X/LinkedIn post it refers to; topics 745861 ("Tricks"),
    747658 (LLM label benchmark), 748011 (CV vs LB), 743416 (external data), 746792 ("Reaching 0.949").
A4. All public notebooks/datasets scoring ≥ 0.956, their components and which ones we do not use yet; any strong
    public model trained without OAI/fastMRI.
A5. Confirm public/private split size; estimate shake-up; recommend how to choose the final 2 submissions.

### B. Noisy labels with a small trusted set (our 58 gold studies)
B1. Methods that use a small clean set to correct or reweight noisy labels: Gold Loss Correction (Hendrycks et al.
    2018), forward/backward correction (Patrini et al. 2017), Learning to Reweight (Ren et al. 2018), Meta-Weight-Net,
    Meta Label Correction, and newer ones. For each: does it work with **58 samples × 12 binary labels**, multi-label
    and MIL settings? Sample complexity, reported gains, failure modes.
B2. Systematic, finding-specific concept mismatch between report labels and image-based expert labels (see the
    prevalence table). Methods: per-finding noise-transition estimation, PU learning ("not mentioned" ≠ negative),
    CheXpert's U-Ignore / U-Ones / U-Zeros findings (Irvin et al. 2019), asymmetric smoothing, uncertainty masking.
B3. Label models over many weak labellers (Dawid–Skene, Snorkel/MeTaL, FlyingSquid): we have 31 public LLM label
    tables (+ image models). Do they beat averaging? How to use the 58 gold for calibration? Evidence.
B4. **Report and image as two views of one latent truth**: co-training (Blum & Mitchell), multi-view learning,
    learning using privileged information (LUPI, Vapnik), report+image teacher → image-only student. Derive a
    latent-variable/EM formulation (latent z per finding; report evidence r ~ p(r|z) with per-finding confusion
    estimated on gold; image model p(z|x); E-step posterior; M-step image-model training), its identifiability with
    58 gold, and whether it avoids the failure in §3.3-2. Give update equations and a training recipe.
B5. Noise-robust losses and selection (GCE, SCE, ELR, bootstrapping, co-teaching, DivideMix, JoCoR) with evidence on
    report-derived medical labels (CheXpert, MIMIC-CXR, ChestX-ray14, knee).

### C. Report → label extraction (multilingual, MSK)
C1. CheXpert labeler, CheXbert, **VisualCheXbert** (maps reports to labels that agree with *image-based* radiologist
    labels — directly our problem), RadGraph; anything MSK- or knee-specific.
C2. LLM extraction best practice: atomic rubric questions written from the official criteria, an explicit
    "not mentioned/uncertain" option, self-consistency, few-shot with gold, multilingual handling. Papers comparing
    LLM labelers with expert image labels. Best open models runnable locally on 24 GB (Qwen 2.5/3 14–32B quantised,
    Llama 3.x, MedGemma, Gemma 3, Mistral) and throughput for 4,349 reports.
C3. Mapping report language to image criteria with radiology grading systems (MOAKS, WORMS, ICRS, Outerbridge,
    meniscal tear grades, effusion grading, effusion-synovitis vs Hoffa-synovitis): e.g. "tricompartmental
    degenerative change", "chondromalacia patellae grade 2", "small effusion", "synovial thickening".

### D. Knee-MRI deep learning
D1. Per-finding state of the art and what made it work: MRNet (Bien et al. 2018) and follow-ups (ELNet, MRPyrNet,
    transformers), fastMRI+ pathology, SKM-TEA, OAI/MOAKS automation (e.g. Astuto et al. 2021), KneePreM
    (arXiv 2609.31461), KJD-25K (MICCAI 2026), SGMRI-VQA. Inputs (planes, slices, resolution, crops), per-finding AUC.
D2. Our weakest findings: **Synovitis on non-contrast MRI**, **Lateral tibiofemoral OA**, **PF OA**, **Lateral
    meniscus tear**. Imaging features, best sequences/planes/regions, any DL evidence.
D3. Preprocessing: intensity standardisation across scanners (Nyúl–Udupa histogram matching, per-series z-score,
    N4), multi-window inputs (e.g. a fluid window), slab averaging, slice count, and **label-free knee/compartment
    localisation** (intensity heuristics, atlas registration, self-supervised keypoints).

### E. Architecture and training for 2.5D multi-series MIL
E1. MIL and fusion heads (gated attention, TransMIL, DSMIL, CLAM, bi-GRU/LSTM, transformers with slice-position and
    plane embeddings, per-finding query tokens à la Query2Label/ML-Decoder, cross-plane attention); small-data evidence.
E2. Training with many windows per study cheaply (frozen early blocks, cached features + head training, gradient
    checkpointing, mixed precision); optimal windows per study; partial fine-tuning schedules.
E3. **AUC-oriented losses**: Deep AUC Maximization / LibAUC AUC-margin loss (Yuan et al. 2021; won the CheXpert
    competition), pairwise/listwise ranking, matched-pair hard negatives, two-stage BCE → AUC loss; multi-label macro.
E4. Augmentation for MRI MIL (mirror with medial/lateral swap, slice jitter, gamma, bag-level mixup/cutmix).
E5. Backbones under our constraints (CoAtNet, ConvNeXt-S/T, EfficientNetV2, MaxViT, DINOv2/v3, RadImageNet).

### F. Domain shift and test time
F1. Multi-site generalisation and test-time adaptation (BN-stat adaptation, TENT) feasible within 9 h; whether
    per-site normalisation helps a rank-based metric; site/scanner tags that survive the 86-tag allow-list.
F2. TTA that actually helps MRI MIL (flip with label swap, multi-crop), with evidence.

### G. Ensembling and selection maths for macro ROC-AUC
G1. Theory: when does rank-averaging a weaker model with correlation ρ improve a stronger model's AUC (e.g. binormal
    model)? Plug in our numbers (base 0.956; candidates 0.948 alone, Spearman 0.94–0.96 with the base). What
    correlation/strength does a new model need to add +0.005 macro?
G2. Per-finding weights from a tiny validation set (58 gold) plus a noisy public LB (SD ~0.003): shrinkage,
    empirical Bayes, hierarchical priors across findings; overfitting risk; a concrete protocol.
G3. Choosing the final 2 submissions under a 30/70 public/private split and the OAI risk.
G4. Single-class LB probing ((11 × 0.5 + AUC_c)/12; ~0.012 AUC resolution at 3 decimals): worth it? rules/ethics?

### H. Kaggle inference engineering
H1. Fast DICOM decoding (JPEG 2000 / JPEG Lossless: GDCM, pylibjpeg, nvJPEG), TensorRT / torch.compile / fp16 /
    channels_last for CoAtNet and ConvNeXt, caching decoded volumes across models, fitting 4–6 models for ~1,300
    studies × ~150 slices into 9 h on 2× T4.

### I. Past competitions (recipes with numbers)
I1. RSNA 2024 lumbar spine (top-10 write-ups), RSNA 2023 abdominal trauma, RSNA 2022 cervical spine, RSNA 2025
    intracranial aneurysm, CheXpert competition (LibAUC), SIIM-ISIC, RANZCR, UBC-OCEAN: pseudo-label rounds, label
    denoising, use of small clean sets, what failed. Quote write-ups.

### J. New, first-principles methods
J1. Be creative but rigorous. Candidates to evaluate (and add your own): the EM / privileged-information model from
    B4; a few-parameter "annotator concept" transform fitted on gold (hengck23: "use 58 samples to train a model
    with only ~10 parameters … unlikely to overfit"); a structured multi-label head modelling finding co-occurrence
    (OA ↔ effusion ↔ synovitis; ACL ↔ contusion pivot-shift pattern; meniscus ↔ OA) as a CRF or graph layer;
    any new maths you find necessary. For each: derivation sketch, why it should beat what we tried, expected gain,
    cost, risks.

---

## 6. Deliverable (one Markdown report)

1. **Executive summary (≤ 1 page):** the top 8 interventions ranked by expected LB gain × probability it works ÷
   GPU-hours, each with an evidence grade: A = published and replicated; B = one paper or one competition write-up;
   C = forum anecdote; D = theory or your estimate.
2. **One card per intervention:** name; mechanism; which measured bottleneck in §3 it addresses; evidence with links
   and numbers; exact recipe (hyper-parameters, losses, schedules); pseudocode or equations with tensor shapes;
   cost in hours on a 4070 Super / L4 and Kaggle inference minutes; how to validate cheaply (remember gold noise
   ±0.02 macro, ±0.06 per finding; LB SD ~0.003); risks; interactions with other cards.
3. **Official label criteria for all 12 findings (verbatim) and a table of host rulings (verbatim, dated).**
4. **Competition intelligence digest** (A3–A5) with links.
5. **Do-not-do list** with evidence, so we do not waste runs.
6. **Day-by-day plan Oct 17–22** for the two GPUs and the Kaggle quota, including which 2 final submissions to pick.
7. Bibliography with links.

## 7. Research rules

- Every factual claim needs a link and, where possible, a short quote. Papers: venue and year.
- Mark anything not directly sourced as **ESTIMATE** or **UNVERIFIED**. Never invent numbers or merge results from
  different sources. Distinguish "reported in the paper" from "your projection for us".
- Be sceptical of Kaggle notebook titles (scores in titles are often wrong) and of other AI-generated summaries
  (a previous AI report invented team details and gains).
- Do not recommend data or tools that break the rules; for grey areas quote the exact rule and ruling.
- Prefer primary sources. If something cannot be found, say "not found" rather than guessing.
- If time runs short, finish sections A, B, G and E fully before the others.
