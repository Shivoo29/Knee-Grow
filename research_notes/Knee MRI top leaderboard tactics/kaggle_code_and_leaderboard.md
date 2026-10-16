# RSNA Knee Abnormality Detection: public Kaggle code, datasets and leaderboard (snapshot 16 Oct 2026)

Method. Everything here was collected on 2026-10-16 (about 15:30–16:10 UTC):
- **Notebook lists.** Kaggle CLI 2.2.4 (`kaggle kernels list --competition … --sort-by {scoreDescending,dateRun,voteCount,hotness,dateCreated}`, two pages each).
- **Real best public scores.** Kaggle's public, no-login web endpoints (`kernels.KernelsService/ListKernels`, `discussions.DiscussionsService/GetForumTopicById`). varshithakolupuri's "real scores" notebook uses the same endpoints. They returned 921 competition notebooks; 513 had a score and 140 scored ≥ 0.950.
- **Notebook sources.** 33 notebooks pulled with `kaggle kernels pull -m` and read as text only. None was executed.
- **Datasets.** Small `manifest.json` / `README.md` / `run_config.json` files from the model datasets.
- **Leaderboard.** `kaggle competitions leaderboard --download`, file timestamp 2026-10-16T15:59:42.

No competition data, reports or DICOMs were downloaded. "Real score" means Kaggle's `bestPublicScore` for the notebook: the best of its submitted versions, at three decimals. Scores in titles are often overstated, so this note uses real scores.

Local copies (paths under `research_notes/Knee MRI top leaderboard tactics/`):
- `kernels/<owner>__<slug>/` — pulled notebook sources.
- `listings/kernels_api_2026-10-16.json` — 921 notebooks with real scores and data sources.
- `listings/notebooks_ge_0950_2026-10-16.csv` — the 140 notebooks at ≥ 0.950.
- `leaderboard/…publicleaderboard-2026-10-16T15:59:42.csv`
- `efficiency_lb/full_leaderboard.csv`
- `datasets/` — model manifests.

---

## Q1. Which public notebooks score ≥ 0.950, and what is inside each pipeline?

### Takeaway
There are 140 public notebooks with a real public score ≥ 0.950. They break down as 2 at 0.958, 8 at 0.957, 47 at 0.956, 4 at 0.955, 29 at 0.954, 1 at 0.953, 4 at 0.952, 6 at 0.951 and 39 at 0.950.

All of them are rank blends of the same small set of public checkpoints:
- **nartaa's Raptor CoAtNet** (384 px and 224 px). This one is OAI-trained.
- **heliosli's compact ConvNeXt-small models** (S6, R6, R7). These use knee-anchored 100 mm views; R7 has a fastMRI auxiliary task.
- **nartaa's S6 retrain on all data.**
- **goodpjw2008's 2.5D ConvNeXt-Tiny reader.**

Single public models top out at 0.950. The best public blend is **0.958**. That blend is heliosli's "Blend mine in": the 384 + 224 Raptor half against R6 with flip TTA, using per-finding weights. goodpjw2008's v25–v27, which add the 224-px mirror view and their reader at 1/10 weight, also score 0.958. Nothing public scores above 0.958.

### Cited Findings

**A. Standalone public scores of each public component (real scores)**

| Component | Real public score alone | Source |
|---|---|---|
| nartaa Raptor CoAtNet-RMLP-2 384 SWA, anatomical mirror | 0.949 | [nartaa 0949](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror) |
| nartaa Raptor CoAtNet 224 (0.8 centre crop), one view | 0.945 | [nartaa 0945](https://www.kaggle.com/code/nartaa/rsna-knee-0945-efficient-224crop) |
| heliosli R6 ConvNeXt-small, 4 folds, no flip TTA (11 Oct notebook, private dataset; the current version pins bundle `fe055406`; the scored version's bundle was not verified) | 0.950 | [heliosli rsna-knee-r6-cnx-fs-ss](https://www.kaggle.com/code/heliosli/rsna-knee-r6-cnx-fs-ss) |
| nartaa S6 ConvNeXt-small on all data ("s6fall") | 0.950; the folds 1–4 version scored 0.948 | [nartaa s6fall-solo](https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo); [discussion 749359](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749359) |
| goodpjw2008 2.5D ConvNeXt-Tiny reader, 3 folds | 0.929 (author) | [goodpjw2008 0.958 notebook](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958) |
| hengck23 MaViT-288 baseline (1.5 s/study) | 0.935 | [hengck23](https://www.kaggle.com/code/hengck23/mavit-288-baseline-01-1-5sec-study-1gpu) |
| heliosli R7 alone, heliosli S6 alone | no public solo score found | — |

**B. Component technical cards** (from the code, manifests and READMEs)

- **Raptor CoAtNet 384** (nartaa; Dread Development lineage):
  - Backbone `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` (timm).
  - Input: 140 mm physical crop at native 384 px, then centre crop to 320 px for inference (`INPUT_RES = 320`, `CROP_MM = 140`).
  - Series and slices: 96 slices in 5 slots: sagittal fat-sat 26, sagittal non-fat-sat 22, coronal fat-sat 18, coronal non-fat-sat 12, axial 18.
  - Windows: 94 three-slice windows (`K_EVAL = 94`, full coverage).
  - Head: per-finding attention pooling over windows.
  - TTA: anatomical mirror (sagittal slice order reversed; coronal/axial width flipped), plain and mirror probabilities averaged.
  - Source: [Blend mine in, stage 1 code](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters).
  - Training data and selection:
    - "2,399 baseline OAI knees for masked external supervision".
    - SWA of epochs 14/11/12.
    - Gold-58 was reused for selection.
    - Labels: stevenleehans + pilkwang + riadmohamed42 JEV tables plus nartaa's own report-derived targets ("JEV-blend").
    - Source: [nartaa/rsna-knee-publication-swa-weights-20261007 README](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007).
  - OAI supervised only PF OA, Lateral OA and Synovitis, with missing cells masked — [discussion 746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792); [discussion 749858](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749858)
- **Raptor CoAtNet 224** (nartaa):
  - Same 96-slice corpus and K = 94.
  - `CROP_FRAC = 0.8` centre crop of the 140 mm field, resized to 224 px.
  - One view in nartaa's notebook; varshithakolupuri added the mirror view.
  - SWA of epochs 15/13/12. Same OAI recipe.
  - Sources: [README](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007); [one free flip](https://www.kaggle.com/code/varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up)
- **heliosli R7** (bundle `8c7b666e27dc0e8f`, dataset `heliosli/rsna-knee-compact-r7cnxfsss`). From [manifest.json](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss):
  - Backbone and init: `convnext_small`, initialised from `convnext_small_in22k_ft_in1k`, first 16 blocks frozen.
  - Inputs: views `v4` (full field of view plus 100 mm knee-anchored crops), size 384 with crop 352, 3 slices per window, 3.5 mm physical slab, `eval_stride 1`.
  - Folds and schedule: 4 folds, 16 epochs, lr 1e-4, head lr 1e-3, wd 0.05, batch 2, EMA 0.999.
  - Training windows: 176 per study, "cover" sampling.
  - Augmentation: mirror augmentation 0.5, slice jitter 0.25.
  - Ranking loss: pair-rank 0.5 (pair_frac 0.33).
  - Graded-state head: weight 0.5, targets `state_targets_opusv11b_rr7`.
  - Extra losses and options: "extent" auxiliary for the three OA classes (weight 0.2), `syn_decomp`, `rare_boost`, `spatial_attn`.
  - **`fm_aux` (fastMRI auxiliary) at weight 0.3, 96 windows.**
  - Targets: `probe_targets_opusv11b_rr7_plr7_cdr.csv`, with cell weights `cell_weights_plr7_v7.csv`.
  - Studies: pool 4,344.
  - Data policy and gold: `data_policy: "No KneeCoT inputs or derivatives; gold never trained"`, `gold_in_training 0`.
  - Extras: an Effusion fluid-volume stacker, a Baker's graded-state stacker and a left/right side classifier ship with the bundle.
  - Compute: heliosli says the models were trained on 8× RTX PRO 6000 GPUs ([blend-gold comments](https://www.kaggle.com/code/heliosli/rsna-knee-blend-gold-old-dataset/comments)).
- **heliosli R6** (bundle `fe055406fad6a12f`, tag `cnx_r6fm_c352`). From [the -w1 manifest](https://www.kaggle.com/datasets/heliosli/rsna-knee-r6-frozen-fe055406-w1):
  - Same core recipe as R7 (convnext_small, 16 frozen blocks, crop 352, views v4, 3.5 mm, 176 windows, 16 epochs, 4 folds).
  - Targets: `probe_targets_opusv11b_rr6_plr6_cdr.csv`.
  - Effusion fluid stacker and Baker's state stacker (6 state probabilities) adopted.
  - The manifest records fewer run_config keys and has no `fm_aux` entry. The run name `cnx_r6fm` suggests fastMRI, which is unverified ([749858](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749858)).
  - In the 0.957 and 0.958 notebooks R6 is run with `FLIP_TTA = True`: every fold also scores mirrored windows, medial and lateral outputs are swapped back, the two passes are averaged in logit space, and MCL comes from the plain pass ([Blend mine in](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters)).
- **heliosli S6** (bundle `209a32c8b50a6b1c`, dataset `heliosli/rsna-knee-compact-v1`; built 2026-10-09):
  - Same family as R6/R7. Targets `probe_targets_opusv11b_rr3_plv21_cdr.csv`.
  - The dataset was withdrawn and then restored ("someone told me it wasn't allowed, but it seems to be allowed after all").
  - Source: [compact-v1 README/manifest](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-v1).
- **nartaa S6 all-data ("s6fall")**:
  - heliosli's S6 architecture retrained from `convnext_small.fb_in22k_ft_in1k`, with per-finding spatial attention plus window-attention MIL.
  - Views v4, placed by the SKM-TEA anatomy V-Net anchors. Cache 320 px, crop 288; 3 slices, 3.5 mm.
  - Labels: JEV-blended soft labels with a prior fill for unaddressed cells and a weight-normalised BCE.
  - Training: 64 windows/study, 16 epochs, AdamW lr 1e-4 / 1e-3, wd 0.05, batch 2, cosine schedule with 5 % warm-up, bf16, seed 42. All 4,263 studies; gold-58 excluded. Trained on an RTX 5060 laptop GPU.
  - "No OAI data was used for this model." Licence CC BY-NC 4.0.
  - Fold-0 OOF 0.907 vs 0.890 for the CoAtNet; gold-58 0.931 vs 0.919; yet public 0.950 vs 0.949.
  - Sources: [README + run_config](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights); [749359](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749359)
- **goodpjw2008 2.5D ConvNeXt reader**:
  - Preprocessing: canonical volumes at 0.4 mm/px, orientation from `ImageOrientationPatient`, slices sorted by position.
  - Model: ConvNeXt-Tiny over 3-slice windows, a 2-layer study transformer, one attention pool per finding.
  - Training: 5 folds grouped by report text (3 folds public). Soft labels are the mean of the stevenleehans, pilkwang, lixin73 and laymond tables. Gold-58 never trained on.
  - The current version fills a missing fat-sat or non-fat-sat series with its counterpart and uses 16 windows per series.
  - Runs in about 20 min alone. It ships offline JPEG decoders because "the hidden test has compressed DICOMs".
  - Source: [goodpjw2008 0.958 notebook](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958); [dataset](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader)

**C. Distinct public recipes at ≥ 0.954 (real scores; version id = Kaggle scriptVersionId of the current version; date = current version)**

| Real | Notebook (author) | Version / date | Recipe | Notes |
|---|---|---|---|---|
| **0.958** | [Blend mine in](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters) (heliosli) | 358751289 / 15 Oct 17:03 UTC | Raptor half = 0.65·rank(384, mirror TTA) + 0.35·rank(224, one view). Compact half = R6 with flip TTA. **Per-finding Raptor weight: Synovitis 0.80, Lateral OA 0.20, PF OA 0.60, Baker's 0.35, else 0.50.** | Weights are a "user decision 10-16, fixed before this run". Attaches `heliosli/rsna-knee-r6-frozen-fe055406-w1`. Commit run 496 s on 3 visible studies. Comments are now switched off ([748181](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748181)). |
| **0.958** | [CoAtNet + R6 + 2.5D MIL](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958) (goodpjw2008) | v27 = 359009299 / 16 Oct | v25 and v27: Blend mine in recipe + **224 model with mirror view** + goodpjw reader at 1/10 of v6 per-finding weights. v26: also R7 at 0.4 of the compact slot. | v25 and v26 scored 0.958; v23 (flip TTA for R7 too) 0.957; v21/22/24 0.957 (R7 at 0.4–0.5 of the compact slot "made no difference at three decimals"). "Rank 195 of 6,077 on 16 October". Uses rostigor's byte-identical R6 copy. |
| 0.957 | [This is best public dataset… 0.9576](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9576) (heliosli) | 358332966 / 14 Oct | Raptor half (0.65/0.35) vs R6 with flip TTA, flat 0.5/0.5 | Title claims 0.9576 (4th decimal not verifiable). Renamed from "this-is-new-version-0-9576-in-one-click". |
| 0.957 | [This one is old version 0.9573](https://www.kaggle.com/code/heliosli/this-one-is-old-version-0-9573-newone-you-can-use) (heliosli) | 358425369 / 14 Oct | Raptor half (0.65/0.35) vs **R7 without flip**, 0.5/0.5 | heliosli: submission takes "5 hours maybe" ([comments](https://www.kaggle.com/code/heliosli/this-one-is-old-version-0-9573-newone-you-can-use/comments)) |
| 0.957 | [R7 + Raptor + knee-centred ConvNeXt](https://www.kaggle.com/code/bliverrigpdoge/r7-raptor-knee-centred-convnext) (bliverrigpdoge) | 358269940 / 14 Oct | heliosli R7 + Raptor (0.956) at 0.8 + **own ConvNeXt-small and EfficientNetV2-S ensemble on knee-centred medial/lateral compartment crops (sagittal, coronal) plus axial** at 0.2 | The own weights are private, so a fork scores 0.956 |
| 0.957 | [The bee's knees – final RSNA push](https://www.kaggle.com/code/prvsiyan/the-bee-s-knees-final-rsna-push) (prvsiyan) | 358804172 / 15 Oct | heliosli's R6-flip recipe with flat family weights: R6 55 %, Raptor-384 29.25 %, Raptor-224 15.75 % | The current version is mostly a CPU "ownership control" demo; the 0.957 badge comes from the R6 reproduction |
| 0.957 | [RSNA Knee Template](https://www.kaggle.com/code/kiyoshiohno/rsna-knee-template) (kiyoshiohno) | 358788844 / 15–16 Oct | Thin launcher that runs a **private** "logic dataset" (`rsna_knee_pipeline.py`) with R6 -w1 and nartaa attached | Not reproducible from public inputs |
| 0.957 | [four model ensemble](https://www.kaggle.com/code/laimerbotrolenk/rsna-four-model-ensemble) (laimerbotrolenk) / [0.957 Explained](https://www.kaggle.com/code/leoprovorov/rsna-knee-0-957-explained-why-the-open-ensembl) (AlekseiProvorov) | 358703526 / 15 Oct; 359002180 / 16 Oct | Raptor half = 0.65·384 (mirror) + 0.35·224 (mirror), compact half = **0.60·R7 + 0.40·nartaa S6-all-data**, 0.5/0.5. Contributions: R7 30 %, S6 20 %, 384 32.5 %, 224 17.5 % | leoprovorov: "base recipe copied unchanged prints 0.956, this notebook prints 0.957, place 206" (15 Oct). laimerbotrolenk's V33 "mirror-mix half" variant is unscored. |
| 0.957 | [RSNA Knee 9576 R6 V1 Fixed (Readable)](https://www.kaggle.com/code/picond/rsna-knee-9576-r6-v1-fixed-readable) (picond) | 358614659 / 15 Oct | Copy of the 0.9576 recipe, with a private copy of R6 v1 | — |
| 0.956 | [≈0.957 Latest dataset, one click](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click) (heliosli) | 357456820 / 11 Oct | Raptor-384 + R7, 0.5/0.5 | The base of the 0.956 block |
| 0.956 | [One unused checkpoint](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956) / [One free flip](https://www.kaggle.com/code/varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up) (varshithakolupuri) | 357929858 / 13 Oct; 358309699 / 14 Oct | + Raptor-224 at 35 % of the Raptor slot; then + mirror view for 224 | Display unchanged at 0.956, but rank moved 233 → 156 (224 added) and 357 → 289 (mirror) |
| 0.956 | [RSNA_Versia_7](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7) (evgendvorkin, 432 votes) | 358019432 / 13 Oct | Same executable cells as heliosli's R7 notebook ([ensemble map](https://www.kaggle.com/code/karttikjangid05/rsna-ensemble-map-models-scores-runtime)) | — |
| 0.956 | [blend-gold (old dataset)](https://www.kaggle.com/code/heliosli/rsna-knee-blend-gold-old-dataset) (heliosli, 311 votes) | 357360972 / 11 Oct | Raptor + S6, 0.5/0.5 | v1 used private components; "V2 only scores 0.954" ([comments](https://www.kaggle.com/code/heliosli/rsna-knee-blend-gold-old-dataset/comments)) |
| 0.956 | [Versia 8 multiscale r224mask](https://www.kaggle.com/code/rjings/rsna-versia-8-multiscale-r224mask) (rjings), [top-1-gold](https://www.kaggle.com/code/lavinwins/top-1-gold) (lavinwins) | 15 Oct | The old 0.943 stack datasets (DINOv2, RadImageNet heads, Mattia CoAt readers, Dread corpora) attached alongside the new models | No gain over 0.956 |
| 0.956 | [0.957 SOTA multi-scale foundation](https://www.kaggle.com/code/beraterolelk/0-957-sota-rsna-knee-multi-scale-foundation) (beraterolelk), [kiza123123 sub3](https://www.kaggle.com/code/kiza123123/rsna-knee-sub3-multi-scale-quad-arm-foundation) | 15 Oct | Add `bliverrigpdoge/rsna-knee-leg-r9all` (`loop_R9_all.pt`, a CoAtNet "student of the mirror", 224 px) | Title overstates (real 0.956) |
| 0.955 | [velci-kcd25-20](https://www.kaggle.com/code/bliverrigpdoge/velci-kcd25-20) (bliverrigpdoge) | 357204606 / 11 Oct | 0.954 base (Raptor mirror 0.7 + S6 0.3) + private "KCD25" (knee-centred compartment view, five ImageNet members on d25 soft report labels) at 0.2 | +0.001 from a private knee-centred model, as in the 0.957 notebook |
| 0.954 | [DINOsaur the last](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-the-last) (romantamrazov), [Velciraptor](https://www.kaggle.com/code/matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed) (matterhorn3838), [blend-gold restored](https://www.kaggle.com/code/heliosli/rsna-knee-blend-gold-dataset-restored) (heliosli) | 10–11 Oct | Raptor + S6 (0.5/0.5 or 0.7/0.3) ± goodpjw reader with per-label residuals | — |
| 0.954 | [CoAtNet + S6 + 2.5D MIL](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-s6-2-5d-mil-lb-0-954) (goodpjw2008) | 359023807 / 16 Oct | The new version tests R6-flip + **S6 as second compact model at 0.3** + Raptor 384 + 224 (mirror) + heliosli per-finding weights + reader at 1/10 | Badge is from older versions; the new version is unscored at snapshot time |

**D. Score claims in titles vs real scores**
- [rabari9999 "0.956+ Micro-Boost"](https://www.kaggle.com/code/rabari9999/rsna-knee-0-956-micro-boost-clinical-tri-reader) scored a real 0.950. Its "temperature damping" and "clinical cross-prior" post-processing did not deliver.
- yuvrajlamba01 "LUPI … 0.957+" scored 0.956, and beraterolelk "0.957 SOTA" scored 0.956.
- hitarthjain0 "S6 25D MIL 0957" scored 0.954, and ltcsmokeboi "gp958d/e" scored 0.956.
- Source for all real scores: Kaggle ListKernels `bestPublicScore`; local `listings/notebooks_ge_0950_2026-10-16.csv`.

**E. Runtime evidence**
- The commit runs of all 0.956–0.958 blends took 400–540 s, but that is only 3 visible test studies (Kaggle `lastRunExecutionTimeSeconds`, [real-scores listing](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-real-scores-of-every-public-notebook)).
- Hidden-test runtime of the R7 + Raptor 384 + 224 notebook: "5 hours maybe" (heliosli, [comments](https://www.kaggle.com/code/heliosli/this-one-is-old-version-0-9573-newone-you-can-use/comments)).
- The R6/R7 stage has an internal deadline of `DEADLINE_HOURS = 8.3`. goodpjw's Part B is skipped after 8.2 h ([goodpjw2008 0.958](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958)).
- Raptor-384 alone took 26.1 min submit-to-score; the 224 model about 7.2 min ([746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792)).

### Inferences
- In these blends the compact R6/R7 stage dominates runtime: roughly 4 h of the reported ~5 h (inference from 26 + 7 min for the two Raptor stages and the 5 h total). That leaves about 3–4 h of headroom in the 9 h limit for extra members. The 0.958 notebooks fit in that headroom.
- Every public ≥ 0.954 notebook is built from the same 5–6 checkpoints. The model families differ (CoAtNet vs ConvNeXt), but the label families overlap: the JEV blend for nartaa's models and heliosli's own "opusv11b" report targets for R6/R7/S6. The ceiling of re-weighting these is close. AlekseiProvorov's equal-correlation estimate puts the shared-error correlation at ρ ≈ 0.86–0.89, so eight such models would reach only about 0.960 ([0.957 Explained](https://www.kaggle.com/code/leoprovorov/rsna-knee-0-957-explained-why-the-open-ensembl); an estimate, not a measurement).
- The heliosli target names (`opusv11b_rr7_plr7_cdr`) suggest Claude-Opus-extracted report labels ("opus v11b") with re-read and pseudo-label rounds ("rr7", "plr7"). This is an inference from file names only.

### Gaps
- Exact hidden-test runtimes of the 0.958 notebooks are not published; goodpjw's comment thread asked and got no answer by the snapshot.
- R7-alone and S6-alone public scores were not found.
- Whether R6 was trained with the fastMRI auxiliary task is not recorded in its manifest.
- heliosli's per-finding weights have no stated basis. The notebook code contains a `PROBE_CLASS` single-class LB probe switch (all other columns set to 0.5, so public = (11·0.5 + AUC_c)/12), set to `None` in the public version. Whether the weights came from such probes is unknown.

---

## Q2. What is newest (8–16 Oct), and does anything score above 0.956?

### Takeaway
Yes. Two public notebooks score **0.958** and eight score **0.957**, all published or updated on 14–16 Oct. Each step since 7 Oct came from a newly published trained model, not from re-weighting:
- nartaa's CoAtNet (0.949–0.950), 7 Oct;
- heliosli's S6 (0.954), 10 Oct;
- R7 (0.956), 11 Oct;
- R6 with flip TTA plus the 224 checkpoint (0.957), 14 Oct;
- per-finding weights (0.958), 15–16 Oct.

139 of the 140 ≥ 0.950 notebooks have a current version dated 8–16 Oct.

### Cited Findings
- Timeline of public frontier moves (real scores):
  - **7 Oct:** nartaa 0.949 / Apex 0.950.
  - **8 Oct:** heliosli blend-gold 0.951 (Raptor + a heliosli compact model; that version's contents not checked).
  - **10 Oct:** blend-gold restored 0.954 and Velciraptor 0.954.
  - **11 Oct:** heliosli R7 + Raptor 0.956; velci-kcd25-20 0.955.
  - **13 Oct:** + 224 checkpoint, still 0.956 (rank only).
  - **14 Oct:** R6 flip 0.957 and R7 + 224 0.957; bliverrigpdoge 0.957 (private); nartaa s6fall 0.950 solo.
  - **15 Oct:** Blend mine in 0.958; four-model 0.957; prvsiyan 0.957.
  - **16 Oct:** goodpjw v27 0.958; leoprovorov 0.957.
  - Sources: [Kaggle notebook listing, real scores](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-real-scores-of-every-public-notebook) and the notebooks linked in Q1.
- "Every visible step up came from a newly published trained model, not from re-weighting." The steps were 0.943 (new CoAt readers), 0.944 (2.5D reader), 0.950 (nartaa), 0.956 (R7). An extra reader on top of 0.956 displayed 0.956 but with a lower unrounded score — [hoshinoyumemi](https://www.kaggle.com/code/hoshinoyumemi/knee-0-939-to-0-956-what-moved-the-score)
- Versioned-date counts of the 140 notebooks at ≥ 0.950: 8 Oct 7, 9 Oct 6, 10 Oct 22, 11 Oct 21, 12 Oct 10, 13 Oct 22, 14 Oct 26, 15 Oct 21, 16 Oct 4 — [ListKernels data](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-real-scores-of-every-public-notebook) (local CSV)
- goodpjw2008's own probes on 11–13 Oct:
  - On top of the S6 recipe, the reader at v6 weights gave 0.954 (= base) and at 1/4 weight 0.954.
  - Their "S6-distilled reader" (gold-58 0.931 alone) gave 0.953.
  - On the current bases, "our reader at full weight no longer adds (0.956 base + reader → 0.954)".
  - Source: [goodpjw2008 0.958](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958)
- Earlier goodpjw probe: retraining on labels mixed with the reader's own predictions helped a single fold (0.922 → 0.927) but not the 5-fold ensemble — [goodpjw comments](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958/comments)
- A single submission reached 0.960 on 10 Oct ("basically a single model", a second-hand claim relayed from Rednote) — [discussion 748111](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748111)
- Kaggle's final-week lock blocks *publishing* new public notebooks. A participant says *updating* existing public notebooks is not locked (unverified) — [discussion 748181](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748181)
- Newest datasets (13–16 Oct) relevant to blends:
  - `nartaa/rsna-knee-s6fall-weights` (14 Oct);
  - `heliosli/rsna-knee-r6-frozen-fe055406-w1` (15 Oct);
  - `rostigor/rsna-knee-r6-fe055406` (15 Oct);
  - `riadmohamed42/rsna-knee-dino-5fold-weights` (16 Oct, five DINO fold checkpoints, no public score seen);
  - `dkim54/rsna-knee-models-v10*/v11*` (14–15 Oct, 5-fold `.pt` sets with code; not used by any ≥ 0.950 public notebook).
  - Source: [Kaggle datasets search](https://www.kaggle.com/datasets?search=rsna+knee)

### Inferences
- The leaderboard tie block at exactly 0.958 holds 411 teams, which matches forks of the two 0.958 notebooks on 15–16 Oct. A plain fork of Blend mine in should therefore reproduce 0.958. Our team's fork scored 0.956. Likely causes, to check in order:
  1. the attached R6 dataset version (the non-`-w1` dataset now holds R7);
  2. `BUNDLE_ID` / `FLIP_TTA`;
  3. whether the 224 stage was skipped by the clock;
  4. whether our own model was blended in;
  5. which version of Blend mine in was forked. The per-finding weights were set on 10-16; earlier versions had flat weights.
- More 0.958–0.959 public notebooks may still appear through updates of existing notebooks, despite the publish lock.

### Gaps
- No public notebook ≥ 0.959 exists. The 0.960 single-model claim has no public code.
- Exact version numbers (as opposed to scriptVersionIds) are known only where authors state them (goodpjw v21–v27).

---

## Q3. Which public datasets hold model weights or label tables, and what do they contain?

### Takeaway
The weights behind every top public blend sit in six datasets: two from nartaa, three from heliosli and one from goodpjw2008. heliosli's code/anatomy dataset is attached to nearly all of them. External-data exposure:
- nartaa's Raptor checkpoints: **OAI**.
- heliosli R7: **fastMRI/fastMRI+** auxiliary.
- heliosli's anatomy V-Net: **SKM-TEA**.
- nartaa's S6-all-data and goodpjw's reader: no external data.
- heliosli's orthofoundation dataset also contains an OAI-pretrained ViT-L, but R6/R7/S6 do not load it.

### Cited Findings

**Model-weight datasets (files from `kaggle datasets files`)**

| Dataset | Contents | External data / licence | Source |
|---|---|---|---|
| `nartaa/rsna-knee-publication-swa-weights-20261007` (4,834 downloads) | `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` (0.949), `…_c08_r224_swa.pt` (0.945), checkpoint manifest | **OAI** 2,399 knees, masked. Use terms: "research and educational use, including the RSNA Knee competition". | [dataset](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007) |
| `nartaa/rsna-knee-s6fall-weights` (1,630) | `E_prior_s42_fall.pt` (0.950 solo), run_config, code manifest, `s6_extras.bin` (heliosli's compact scripts + 16 MONAI files) | "No OAI data was used". CC BY-NC 4.0. Needs heliosli's anatomy dataset. | [dataset](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights) |
| `nartaa/rsna-knee-hpo-assets` (2,061) | **Label tables**: `labels_v0raw`, `labels_v1_blend`, `labels_v1_gemlow`, `labels_v1_perlabel`, `train_labels_v0/v1` (parquet), `folds_v0.json`, `train_knee.py` | report-derived | [dataset](https://www.kaggle.com/datasets/nartaa/rsna-knee-hpo-assets) |
| `heliosli/rsna-knee-r6-frozen-fe055406-w1` (686) | R6: 4 × `E_lrprior_s0_f*.pt` (200 MB each), `side_cls.pt`, fluid/state stackers, `state_spec.json`, manifest | R6 manifest has no `fm_aux` entry; run name `cnx_r6fm` | [dataset](https://www.kaggle.com/datasets/heliosli/rsna-knee-r6-frozen-fe055406-w1) |
| `heliosli/rsna-knee-r6-frozen-fe055406` (3,105) | **Latest version holds R7, not R6** (version 1 = R6) | — | [749858](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749858); [goodpjw2008](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958) |
| `rostigor/rsna-knee-r6-fe055406` (374) | Versioned copy of R6 v1, byte-identical | — | [dataset](https://www.kaggle.com/datasets/rostigor/rsna-knee-r6-fe055406) |
| `heliosli/rsna-knee-compact-r7cnxfsss` (4,029) | R7: 4 folds, side classifier, stackers, manifest | **fastMRI + fastMRI+ boxes auxiliary (`fm_aux` 0.3)**; "No KneeCoT" | [dataset](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss) |
| `heliosli/rsna-knee-compact-v1` (6,056) / `heliosli/rsna-knee-compact-r6cnxfsss` (1,790) | S6 bundle `209a32c8b50a6b1c`; compact-r6cnxfsss **version 2** holds the same S6 files despite the "r6" name ([blend-gold old dataset](https://www.kaggle.com/code/heliosli/rsna-knee-blend-gold-old-dataset)) | nothing external found ([749858](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749858)) | [compact-v1](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-v1) |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` (39,904) | `anatomy.pt` (SKM-TEA V-Net for joint/patella anchors), `cloud_code/` (`px_cache.py`, `compact_train.py`, anatomy modules), wheels, OrthoFoundation ViT-L weights | anatomy = **SKM-TEA**; the OFL ViT-L is OAI-pretrained but not loaded by R6/R7/S6 | [dataset](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004); [749858](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749858) |
| `goodpjw2008/rsna-knee-2-5d-convnext-reader` (7,945) | `cnxt_v0_fold0–2.pt`, `train.py`, `make_labels.py`, `preprocess.py`, `knee.py`, `infer.py`, `oof.py`, pylibjpeg / openjpeg wheels | none (public label tables only) | [dataset](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader) |
| `bliverrigpdoge/rsna-knee-leg-r9all` (935) | `loop_R9_all.pt` (CoAtNet student, 224 px), `leg_main.py`, `pack_submit.py` | Distilled from the mirror CoAtNet ("student of the mirror"), so indirectly OAI-exposed (inference) | [dataset](https://www.kaggle.com/datasets/bliverrigpdoge/rsna-knee-leg-r9all); [velci-kcd25-20](https://www.kaggle.com/code/bliverrigpdoge/velci-kcd25-20) |
| `isotriage/rsna-jev-oai-arm` | `raptor_ft_jev_oai384_swa.pt` | OAI by name | [dataset](https://www.kaggle.com/datasets/isotriage/rsna-jev-oai-arm) |
| `dreaddevelopment/raptor-knee-*` (widedense, maxspan, native384, native384dense, finespacing, arms) | Raptor weights and pre-built corpora (Aug) | nothing external found | [dreaddevelopment datasets](https://www.kaggle.com/dreaddevelopment/datasets) |
| Mirrors | `varianceofx/rsna-knee-compact-r6cnxfsss` and `-r7cnxfsss`, `dimaspashaakrilian/…`, `goodjane/…`, `mizeroluckygall/rsna-knee-r6-frozen-fe055406` | copies | [datasets search](https://www.kaggle.com/datasets?search=rsna+knee) |
| `leoprovorov/rsna-knee-ensemble-analytical-handbook` | 56-page handbook (per-model gold-58 AUCs, runtimes, recipe comparison) | **private: HTTP 403** | [0.957 Explained](https://www.kaggle.com/code/leoprovorov/rsna-knee-0-957-explained-why-the-open-ensembl) |

**Label tables updated 13–16 Oct**
- `lingyu07/rsna-knee-v53-master-labels`
- `siamrahman29/rsna-knee-w4-labels` and `w5-labels`
- `rajesh1joc/rsna-knee-llm-labels-gemini-flash`
- `genesis96839/rsna-knee-train-ab-labels`
- `hukumjailer/rsna-knee-consensus-labels-clean` ("5-Source Consensus")
- `haungjiawei/rsna-knee-train-v2-code-and-labels`
- `zzzzzz11132/rsna-m1m2-bundle` (`llm_labels_v2.csv`, `llm_labels_v4_blend.csv`, `report_labels_v2.csv`, a DINOv2 ViT-S checkpoint)
- Source: [Kaggle datasets search "rsna knee"](https://www.kaggle.com/datasets?search=rsna+knee)
- The tables behind the top models:
  - nartaa: stevenleehans + pilkwang + riadmohamed42 JEV + own ([README](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007));
  - goodpjw: stevenleehans, pilkwang, lixin73, laymond ([goodpjw2008](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958));
  - heliosli: own `opusv11b` targets (not public; [R7 manifest](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss)).

**Host rule status (external data)**
- RSNA (Chris Carr, 14 Oct) will not designate datasets. "Datasets that require only registration and acknowledgement of licensing terms would clearly meet these criteria. Datasets that require an institutional research agreement or limit access to users with institutional affiliations clearly do not." Teams are asked to disclose training data — [743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)
- Still open: OAI under NIH geo-restrictions. Login.gov returns error LG22 from China; NOT-OD-25-083 is cited — [749145](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749145); [743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)
- No host answer on OAI-trained public weights — [747455](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747455)
- KneeCoT is ruled out ([749145](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749145), citing 734109). R7's manifest declares no KneeCoT ([R7 manifest](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss)).
- hengck23 pointed to `rajpurkarlab/3DReasonKnee` on Hugging Face: OAI-derived boxes and segmentations for most of the 12 classes, licence not checked. It is not used in any public notebook seen — [745861](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745861)

### Inferences
- External-data exposure of each component, under our team's rules:
  - **OAI-conditional:** Raptor 384/224 (and its distilled students, such as `loop_R9_all`).
  - **fastMRI+-unresolved:** R7, and probably R6.
  - **SKM-TEA (registration plus research-use terms):** the anatomy anchors in R6/R7/S6/S6-all-data.
  - **No external data:** nartaa S6-all-data (apart from SKM-TEA anchors), heliosli S6 (as far as can be seen), goodpjw's reader.
- Our current 0.956 fork (Raptor + R6) therefore carries both the OAI and fastMRI flags. The cleanest public components are nartaa's S6-all-data and goodpjw's reader.

### Gaps
- SKM-TEA's exact access terms were not checked here.
- The OFL ViT-L's OAI provenance comes from a participant post, not from heliosli.
- dkim54's v10/v11 model sets and riadmohamed42's DINO weights have no public score or README seen.

---

## Q4. How do the top public blends differ, and which components are missing from "Blend mine in" + Raptor?

### Takeaway
All ≥ 0.957 public blends share the Raptor half: rank(384) and rank(224) at 0.65 / 0.35. They differ only in:
1. the compact model: R6 with flip TTA, R7, or R7 + S6-all-data;
2. whether the 224 model gets the mirror view;
3. flat vs per-finding weights;
4. small extras: goodpjw's reader at about 1–3.5 %, or private knee-centred CNNs at 20 %.

Public components that are *not* in Blend mine in:
- the 224 mirror view;
- R7 as a second compact model;
- nartaa's S6-all-data;
- heliosli's S6;
- goodpjw's reader;
- `loop_R9_all`.

None of these has moved the displayed score above 0.958.

### Cited Findings

| Blend | Raptor half | Compact half | Weights | Extras | Real |
|---|---|---|---|---|---|
| heliosli R7 | 384 (mirror) only | R7, no flip | 0.5 / 0.5 | — | 0.956 |
| + varshithakolupuri | 0.65·384 + 0.35·224 (one view, then mirror) | R7 | 0.5 / 0.5 | — | 0.956 (higher rank) |
| heliosli "0.9573" | 0.65·384 + 0.35·224 (one view) | R7, no flip | 0.5 / 0.5 | — | 0.957 |
| heliosli "0.9576" | same | R6 + flip TTA | 0.5 / 0.5 | — | 0.957 |
| prvsiyan | same | R6 + flip | R6 0.55 / Raptor 0.45, flat | — | 0.957 |
| AlekseiProvorov / laimerbotrolenk | 0.65·384 + 0.35·224, both mirror | 0.6·R7 + 0.4·S6-all-data | 0.5 / 0.5 | — | 0.957 |
| bliverrigpdoge | R7 + Raptor base (0.956) | — | base 0.8 | private knee-centred ConvNeXt-S + EffNetV2-S, 0.2 | 0.957 |
| **heliosli Blend mine in** | 0.65·384 + 0.35·224 (one view) | R6 + flip | **Raptor: Synovitis 0.8, Lateral OA 0.2, PF OA 0.6, Baker's 0.35, else 0.5** | — | **0.958** |
| **goodpjw2008 v25/v27** | 0.65·384 + 0.35·224 **(mirror)** | R6 + flip (v26: + R7 at 0.4 of the slot) | heliosli per-finding | reader at 1/10 of v6 weights: Baker's 0.035, Contusion 0.03, Medial OA 0.025, ACL 0.025, Medial Meniscus 0.02, MCL 0.02, Lateral Meniscus / Fracture / Lateral OA / PF OA 0.008, Effusion 0.004, Synovitis 0.003 | **0.958** |

Sources: notebooks linked in Q1; [goodpjw2008 0.958 cell "Stage 3" and Part B](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958); [Blend mine in Stage 3](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters); [laimerbotrolenk](https://www.kaggle.com/code/laimerbotrolenk/rsna-four-model-ensemble); [prvsiyan](https://www.kaggle.com/code/prvsiyan/the-bee-s-knees-final-rsna-push).

Further findings:
- How the blends combine:
  - All blends use **per-finding percentile ranks** over each model's non-failed studies.
  - A study one model failed on takes the other model's rank.
  - The notebook stops if Raptor fails on more than 3 % of studies, or our stage on more than 50 %.
  - Source: [Blend mine in](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters)
- Known blend bug: the extra-224 recipe uses the R384 success mask for the R224 ranks, so a 224-only failure is not detected — [ensemble map](https://www.kaggle.com/code/karttikjangid05/rsna-ensemble-map-models-scores-runtime)
- Diversity claims:
  - "The top is a monoculture… Diversity is the lever" — [varshithakolupuri](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-what-the-top-notebooks-are-made-of)
  - goodpjw: "a third reader only pays when it is independent of the two it joins, and S6/R7 already cover what a ConvNeXt reader sees" — [goodpjw2008 0.958](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958)
- Private knee-centred CNNs added about +0.001 at 0.2 weight twice: on the 0.954 base (KCD25, compartment view, five ImageNet members) and on the 0.956 base (ConvNeXt-small + EfficientNetV2-S on medial/lateral compartment crops plus axial) — [velci-kcd25-20](https://www.kaggle.com/code/bliverrigpdoge/velci-kcd25-20); [R7 + Raptor + knee-centred](https://www.kaggle.com/code/bliverrigpdoge/r7-raptor-knee-centred-convnext)
- Equal-correlation estimate: published blends sit at ρ ≈ 0.86–0.89, and eight equally good members would reach about 0.960 (estimate) — [0.957 Explained](https://www.kaggle.com/code/leoprovorov/rsna-knee-0-957-explained-why-the-open-ensembl)
- Noise:
  - The public split's standard deviation is about 0.003, "larger than the whole spread of the top 400 teams (0.0022)" — [0.957 Explained](https://www.kaggle.com/code/leoprovorov/rsna-knee-0-957-explained-why-the-open-ensembl)
  - nartaa gives roughly 0.003 — [749359](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749359)
  - The public split is "30 % of the test set" per [varshithakolupuri](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-real-scores-of-every-public-notebook) and [hoshinoyumemi](https://www.kaggle.com/code/hoshinoyumemi/knee-0-939-to-0-956-what-moved-the-score). AlekseiProvorov's simulation assumes "about 640 studies". These conflict.

### Inferences
- **Components still addable to our fork, in expected value order:**
  1. The 224 **mirror view**: free, same checkpoint, already in goodpjw v25/27.
  2. **nartaa S6-all-data** as a second compact model next to R6 (e.g. 0.6·R6 + 0.4·S6fall). It is the only ≥ 0.950 public model trained on different labels (JEV blend) with heliosli's view geometry, and it is OAI-free. Its R7 pairing scored 0.957; the R6-flip + per-finding pairing has no public test.
  3. **R7 next to R6**: no displayed gain at 0.4–0.5 of the compact slot.
  4. goodpjw's reader at ≤ 3.5 %: no displayed gain.
  5. heliosli S6 at 0.3: goodpjw is testing it on 16 Oct, result unknown.
- None of these is likely to reach 0.969. Our own model only helps if it is genuinely decorrelated from all of them. The public evidence that helps is that knee-centred compartment-crop CNNs added +0.001 twice.
- heliosli's per-finding weights (Lateral OA 0.8 to R6, Synovitis 0.8 to Raptor) carry public-LB overfitting risk if they came from single-class probes. Keep a flat-weight variant as the second final pick. This mirrors the 8 Oct advice in our earlier notes.

### Gaps
- No public per-finding AUCs (public LB or gold-58) exist for R6, R7 or S6-all-data. AlekseiProvorov's gold-58 figure exists only as an image and a private handbook.
- No public test of R6-flip + S6-all-data + per-finding weights.

---

## Q5. Leaderboard shape, top 30, and how fast the top moved

### Takeaway
On 16 Oct 2026 at 15:59 UTC there were 6,095 teams. **No team is at 0.969.** #1 CentralyDS has 0.968; three teams have 0.967. **7 teams are at ≥ 0.966 and 18 at ≥ 0.964.** The approximate medal cut-offs are gold (top 22) 0.963, silver (top 304) 0.958 and bronze (top 609) 0.957. The public 0.956–0.958 blends fill huge tie blocks: 411 teams at 0.958, 322 at 0.957 and 636 at 0.956. The top moved from 0.964 (8 Oct) to 0.967 (14 Oct) to 0.968 (16 Oct). The count of teams at ≥ 0.960 went from about 18 to 50 to 77.

### Cited Findings
- **Counts at thresholds** (16 Oct, 6,095 teams):

| Threshold | Teams |
|---|---|
| ≥ 0.969 | 0 |
| ≥ 0.968 | 1 |
| ≥ 0.967 | 4 |
| ≥ 0.966 | 7 |
| ≥ 0.965 | 9 |
| ≥ 0.964 | 18 |
| ≥ 0.963 | 25 |
| ≥ 0.962 | 38 |
| ≥ 0.961 | 57 |
| ≥ 0.960 | 77 |
| ≥ 0.958 | 530 |
| ≥ 0.956 | 1,488 |
| ≥ 0.955 | 1,525 |
| ≥ 0.950 | 1,971 |

  Exact-score blocks: 0.958 = 411, 0.957 = 322, 0.956 = 636, 0.954 = 205, 0.950 = 179. Source: [public leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard), CLI download 2026-10-16T15:59:42.
- **Top 30** (team, score, last submission UTC, submission count, members):

| # | Team | Score | Last submission | Subs | Members |
|---|---|---|---|---|---|
| 1 | CentralyDS | 0.968 | 10-16 07:39 | 205 | centralyds |
| 2 | RB & Matheus & Tucker & Tony | 0.967 | 10-16 11:08 | 325 | rashmibanthia, tomirol, tonylica, tuckerarrants |
| 3 | azamatt | 0.967 | 10-16 00:03 | 74 | azamat1ch, matthewliew |
| 4 | Sergey Bryansky | 0.967 | 10-16 05:41 | 67 | sggpls |
| 5 | MEJAL | 0.966 | 10-16 08:59 | 179 | jakkma, lihu6666, max6296, minminandy, zhuyifanss |
| 6 | joinT fusion | 0.966 | 10-15 23:41 | 261 | nantohock, tomokihirose, wadakoki |
| 7 | Nischay and Sho, Aillis CEO | 0.966 | 10-16 10:11 | 226 | nischaydnk, shookiyama |
| 8 | Ilya Kolb | 0.965 | 10-16 15:09 | 141 | ilyakolb |
| 9 | GM-Enhanced Kaggle Agent | 0.965 | 10-16 00:05 | 135 | arc144, dott1718, jiweiliu, maxjeblick, theoviel |
| 10 | Team Kangaroo | 0.964 | 10-16 10:14 | 83 | liushuzhi, tascj0 |
| 11 | 「愛と情熱の戦士！…」 | 0.964 | 10-16 10:00 | 257 | kosuke123 |
| 12 | YOKKOISHO | 0.964 | 10-16 10:19 | 312 | shin9915, syamoji141chan, tennogh |
| 13 | monnu | 0.964 | 10-16 10:53 | 85 | fuumin621 |
| 14 | Claudio Verdú Ruiz | 0.964 | 10-16 12:18 | 161 | claverru |
| 15 | TomKroumov | 0.964 | 10-16 04:12 | 173 | tom1115 |
| 16 | Zhengru Li | 0.964 | 10-16 14:52 | 140 | guoqingu, hongori, leechengju, leeczzz, yeeelzh |
| 17 | knee-deep | 0.964 | 10-16 12:34 | 358 | shunyaiida, t88take, takashisomeya, takeharuyoshikawa, yosukeyama |
| 18 | Pa3฿aJluHа | 0.964 | 10-16 07:35 | 88 | hettpugymajlhuk |
| 19 | ~~Zo3ama~~ | 0.963 | 10-16 13:25 | 265 | bunnywasnothere, dude497, wassimdobbi |
| 20 | tine.sh agent | 0.963 | 10-16 09:58 | 39 | sidcodegg |
| 21 | Ex-Aggr | 0.963 | 10-16 12:07 | 240 | mathormad, minhtu123, pntan17, trantrikien239 |
| 22 | 吃白饭的大肥鱼 | 0.963 | 10-15 11:17 | **8** | deepios |
| 23 | rhoskeri | 0.963 | 10-15 00:13 | 116 | rhoskeri |
| 24 | matsurih | 0.963 | 10-16 07:57 | 62 | matsurihime |
| 25 | aleksandr3312 | 0.963 | 10-16 10:58 | 82 | gapchenko |
| 26 | Pavel | 0.962 | 10-15 17:39 | 49 | pavelsavchenkov |
| 27 | tomoon33 | 0.962 | 10-16 13:36 | 195 | tomoon33 |
| 28 | dalab | 0.962 | 10-16 11:50 | 236 | junpeimorioka, shuyaakagawa, sotakurihara1, sugiuraakie, tomoyamiyazawa26 |
| 29 | NMarkS | 0.962 | 10-15 14:55 | 33 | nmarks |
| 30 | Attention Is All You Knee | 0.962 | 10-16 10:42 | 167 | aleksanderhan, aurax7, coolinlai, davidsarrat, zongzishuang |

  Source: [leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard). The median submission count of the top 30 is 161. All 77 teams at ≥ 0.960 last submitted on 15–16 Oct.
- **Rank anchors (16 Oct):**

| Rank | Score |
|---|---|
| 5 | 0.966 |
| 10 | 0.964 |
| 25 | 0.963 |
| 50 | 0.961 |
| 100 | 0.959 |
| 195 / 300 / 530 | 0.958 |
| 600 | 0.957 |
| 1,000 / 1,488 | 0.956 |
| 2,000 | 0.949 |

  Gold/silver/bronze rank cut-offs follow Kaggle's ≥ 1,000-team rule (10 + 0.2 %·N, 5 %, 10 %), giving 22 / 304 / 609 → 0.963 / 0.958 / 0.957. Source: [leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard); medal rule as stated in [varshithakolupuri](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-real-scores-of-every-public-notebook).
- **Earlier snapshots:**
  - **8 Oct** (our notes, CLI): #1 Ilya Kolb 0.964; Pa3฿aJluHа, CentralyDS and rhoskeri 0.963; teams #14–#18 at 0.960 — [this_competition_public_writeups.md / leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)
  - **14 Oct 21:01 UTC** (Kaggle efficiency-LB output: the latest submission in the file; 5,955 rows):
    - CentralyDS 0.967 (submitted 14 Oct), RB & Matheus & Tucker & Tony 0.966, joinT fusion 0.966, azamatt 0.965, Ilya Kolb 0.965.
    - Counts: ≥ 0.964: 12, ≥ 0.962: 23, ≥ 0.960: 50, ≥ 0.958: 92, ≥ 0.956: 945.
    - Caveat: this column is the score of the submission evaluated for efficiency, which may not be each team's best.
    - Source: [efficiency LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb)
  - The bronze line moved from 0.943 to 0.956 in 13 days — [hoshinoyumemi](https://www.kaggle.com/code/hoshinoyumemi/knee-0-939-to-0-956-what-moved-the-score). It is 0.957 on 16 Oct (computed above).
- **Fast notebooks at the top.** Efficiency #1 is Claudio Verdú Ruiz with public 0.964 (14 Oct). Then Pa3฿aJluHа 0.961, Scott Willis 0.959, Optimo 0.960, Wassim Dobbi 0.960, tomoon33 0.960, aleksandr3312 0.960 and Sergey Bryansky 0.961 (efficiency rank 19) — [efficiency LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb)
- **Tie mechanics:**
  - Rank inside a displayed-score block follows the unrounded score, and exact ties go to the earlier submission.
  - Kaggle's auto-selection markers reveal which of your own equal-looking submissions is higher.
  - Sources: [hoshinoyumemi](https://www.kaggle.com/code/hoshinoyumemi/knee-0-939-to-0-956-what-moved-the-score); [varshithakolupuri](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-real-scores-of-every-public-notebook)

### Inferences
- The top moved about +0.004 in 8 days (0.964 → 0.968), about +0.001 in the last two days. Teams at ≥ 0.964 went from 1 to about 12 to 18. If that pace holds for the final 6 days, a top-5 line of about 0.967 and a #1 of about 0.969–0.970 by 22 Oct is plausible. This is an extrapolation, not data.
- The 0.964 public score of the efficiency leader, and the 8-submission team at 0.963 (consistent with the "single sub 0.960" thread, unverified), show that 0.96+ comes from fast own models, not from 5-hour public blends. The public 0.958 blend sits about 0.010 below the top-5 line.
- For our goal of 0.969, about +0.011 over the best public blend, public components alone are very unlikely to be enough. The estimated ceiling of the public family is about 0.960.

### Gaps
- Kaggle does not expose historical leaderboard snapshots. The 8 Oct and 14 Oct points come from our earlier notes and the efficiency-LB output, which has the caveat above.
- No top-30 team has published code or a write-up. The "single sub 0.960" team and method are unverified.
- Private-LB shake-up cannot be measured. The public/private split share is reported inconsistently (30 % vs about 640 studies).
