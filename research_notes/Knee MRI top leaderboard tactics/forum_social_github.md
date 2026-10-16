# RSNA Knee Abnormality Detection: forum, social and GitHub evidence on what moves the score (as of 16 Oct 2026)

Method and conventions. Collected on 2026-10-16.
- **Forum.** All 198 competition topics were listed with the Kaggle CLI (`kaggle competitions topics list -s new`, 10 pages). I fetched the opening posts and comments of 126 topics with `kaggle competitions topics show` and the kagglesdk `GetTopic` call. The page-scraping route used on 8 Oct is now behind a reCAPTCHA, so I did not use it.
- **Leaderboard.** Taken with `kaggle competitions leaderboard --show/--download`; snapshot 2026-10-16 16:23 UTC, 6,096 teams. Efficiency board taken from the output of Kaggle staff's notebook.
- **Public notebooks.** Pulled with `kaggle kernels pull`. I read the markdown only and ran nothing.
- **GitHub.** The `gh` CLI is not installed here, so I used the unauthenticated GitHub REST search: 7 queries, 213 unique repos. I read READMEs and one SOLUTION.md through raw.githubusercontent.com. No repo was cloned or run.
- **Scores.** Unless marked otherwise, every score is a **public-LB number claimed by its author**.
- **Competition data.** No report text, study IDs or images are reproduced here. Some forum posts quote gold reports; I only describe them.
- "Gold-58" means the 58 expert-labelled training studies.

## Q1. Kaggle discussion forum (focus 1–16 Oct 2026): score gains, labels, host rulings, geometry, models, TTA/ensembling, runtime, split

### Takeaway
After 8 Oct the forum moved from "how to beat the 0.943 stack" to three things:
- **Image-informed labels.** A #3-team member's whole public answer was "Short answer is labels... AI, not by hand. And don't rely on reports alone."
- **The external-data question.** The host ruled on 14 Oct that registration-plus-licence datasets are fine and that public weights are prize-eligible if disclosed. OAI geo-blocking is still unanswered.
- **A public blend frontier at 0.956–0.958.** It is built from nartaa's OAI-trained CoAtNet (384 + 224) and heliosli's knee-anchored ConvNeXt R6/R7 (flip TTA, per-finding weights). About 1,370 teams sit at 0.956–0.958.

Top competitors repeatedly say that simple 2.5D CNNs at 224–288 px with 5 folds reach 0.955–0.959 as single architectures, and that the extra points come from labels and pseudo-labels, not from capacity.

### Cited Findings

**Leaderboard state and how much it can resolve**
- **Top of the board, 16 Oct 16:23 UTC:**
  - 1 CentralyDS 0.968
  - 2 RB & Matheus & Tucker & Tony 0.967 (rashmibanthia, tomirol, tonylica, tuckerarrants)
  - 3 azamatt 0.967 (azamat1ch, matthewliew; merged since 15 Oct, when azamat1ch was team "MURAD")
  - 4 Sergey Bryansky 0.967
  - 5 MEJAL 0.966 (the former "Evan zhu" team: jakkma, lihu6666, max6296, minminandy, zhuyifanss)
  - 6 joinT fusion 0.966 (nantohock, tomokihirose, wadakoki)
  - 7 Nischay and Sho, Aillis CEO 0.966 (nischaydnk, shookiyama)
  - 8 Ilya Kolb 0.965
  - 9 GM-Enhanced Kaggle Agent 0.965 (arc144, dott1718, jiweiliu, maxjeblick, theoviel)
  - 10–18 at 0.964: Team Kangaroo (liushuzhi, tascj0), kosuke123's team, YOKKOISHO (shin9915, syamoji141chan, tennogh), monnu, Claudio Verdú Ruiz, TomKroumov, Zhengru Li (5 members), knee-deep (yosukeyama + 4), Pa3฿aJluHа
  - Source: [Kaggle leaderboard via CLI](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)
- **Score histogram, same snapshot:**
  - 18 teams ≥ 0.964 and 77 teams ≥ 0.960.
  - 418 teams at 0.958, 322 at 0.957 and 631 at 0.956: the public-fork mass.
  - Source: [leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)
- **Public-split resolution (Shenzai25, 2026-10-15):** the board header says it is computed on ~30% of the test set. About 2,030 teams sit between 0.941 and 0.956, roughly 135 teams per 0.001. Per-label weight changes worth < ~0.002 are "inside the noise band" — [749570](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749570)
- **Tie blocks (varshitha kolupuri, 2026-10-13):** simulating macro AUC ≈ 0.955 on ~390 studies gives SD ≈ 0.003 per submission. Ranks inside a block of equal 3-decimal scores still follow the unrounded score, so rank movement is a finer paired signal (437→320, 233→156 for one added checkpoint) — [748011](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748011); [real-scores notebook](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-real-scores-of-every-public-notebook)
  - **Conflict:** leoprovorov's analysis assumes a public half of ~640 studies (SD ≈ 0.003) — [0.957 Explained notebook](https://www.kaggle.com/code/leoprovorov/rsna-knee-0-957-explained-why-the-open-ensembl). The host never stated the fraction in a thread I could find.
- **Final-days board movement (Optimo, 2026-10-10):** posted a screenshot of a new team, "吃白饭的大肥鱼" (deepios), at **0.960 with a single submission**. A commenter (william.wu, 2026-10-12) says the author "posted on Rednote and said this sub is basically a single model" (second-hand, unverified). The team was at 0.963 with 8 submissions on 16 Oct — [748111](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748111); [leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)
- **Hidden strong teams (colum2131, 2026-10-10):** because AUC is rank-based, submitting 1−p shows 1−AUC. Strong teams could hide progress this way, and colum2131 flags past-RSNA high finishers yu4u & tattaka sitting at 0.949–0.951 as "suspicious" (speculation) — [748011](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748011)

**Host and Kaggle-staff rulings and clarifications**
- **External data (RSNA, Christopher Carr, Director of Informatics, 2026-10-14):** RSNA will not designate datasets.
  - "Datasets that require only registration and acknowledgement of licensing terms would clearly meet these criteria. Datasets that require an institutional research agreement or limit access to users with institutional affiliations clearly do not."
  - Teams should apply "reasonableness" and disclose training methods, including datasets used.
  - Source: [743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)
- **Public weights (Carr, 2026-10-14):** "submissions incorporating publicly shared weights are eligible for awards". Winners must list the shared weights and their source datasets — [744056](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744056)
- **Open external-data questions, unanswered as of 16 Oct:**
  - Geo-blocking: NIH/NDA bars "countries of concern" including China, and login.gov rejects Chinese registrations with error LG22. Rule 2.6.b names "geo restrictions". Raised by dragoshh1984 (11 Oct), Mobassir (14 Oct) and Richard_61 (16 Oct, also asking about OAI-trained public weights) — [748276](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748276); [749145](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749145); [743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)
  - An NDA help-desk reply posted by Tucker Arrants (8 Oct) says OAI is meant to be open to all, with no country exclusions — [743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)
  - A labeller whose weights are proprietary, used only to make soft labels — unanswered — [748925](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748925)
  - fastMRI/fastMRI+ application plus redistribution clause — unanswered — [748480](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748480)
- **Earlier OAI ruling (Po-Hao "Howard" Chen, host, 2026-09-21):** OAI "does not violate the... rules if it is generally accessible to all participants without an escalated (i.e. institutional) review and sign-off... and is free of charge" — [741819](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819)
- **KneeCoT ban (host, 2026-08-25):** banned because it requires a formal agreement with the institution — [734109](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109)
- **Non-commercial licences (host, 2026-08-27):** NC licences are not excluded "solely because" of the NC clause — [733965](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965)
- **Commercial LLM APIs (host, 2026-08-09 and 08-27):** sending report text to commercial LLM APIs for label extraction is allowed and is not "private sharing" — [733965](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965)
- **How labels were made (host, 2026-08-07/08):**
  - Gold and test labels come from image review with the challenge rubric: two independent readers, a third adjudicating. "The same process was used for the testing set."
  - Where image and report disagree, the image label is authoritative. Discrepancies are expected because "image-based labels uses multiple readers with stricter image-based thresholds".
  - Bilateral studies: report text or DICOM metadata were adjusted so the scored knee can be told apart.
  - Sources: [733491](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733491); [733826](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826)
- **Official label rubric (Overview thread, 2026-08-06).** Borderline findings are graded negative "to favor specificity".
  - ACL: high-grade partial (>50% of fibres) or full tear.
  - MCL: high-grade partial or complete *acute* tear with fibre disruption and edema. Low-grade or chronic changes are negative.
  - Meniscus: signal that definitely contacts the surface on ≥2 images, or a morphologic abnormality. Intrasubstance degeneration is negative.
  - OA (medial, lateral, PF): a "moderate or large area (roughly 1 cm or greater) of high-grade cartilage loss (>50% thickness)".
  - Effusion: moderate or large.
  - Synovitis: "inflammation and thickening of the synovial lining".
  - Baker's cyst: moderate or large.
  - Contusion: edema-like marrow signal without a fracture line.
  - Fracture: acute cortical break or fracture line.
  - Source: [733343](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)
- **Other host notes (2026-08-28):** Fluid_Sensitive and Fat_Suppression are identical in train but "should not be interpreted as... interchangeable... in another dataset" — [737312](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737312). Reports are not available at test time, either before or after submission — [733592](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733592)
- **Kaggle staff (Ryan Holbrook, 2026-10-12):** there is a one-week notebook publishing lock before the deadline in this competition, and "low-effort ensembles like this usually suffer from optimism bias from overfitting and tend to drop on the private leaderboard" — [748181](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748181)
- **Timeline:** entry and team-merger deadline 15 Oct (passed); final submission 22 Oct; winners' requirements (training code, video, method description) due 5 Nov — [Timeline page via CLI](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline)
- **Data page:** about 1,300 test studies. Prevalence "is not guaranteed to be the same across the training, public leaderboard, and final evaluation datasets" — [Data page via CLI](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)
- **Never answered by the host:**
  - Whether the public/private split is site-stratified — [734681](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734681)
  - Whether every hidden study has all planes and a fluid-sensitive series — [744519](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744519)
  - Whether hand-labelling *training* images is allowed. Participants (PC Jimmmy, ParsaAshrafi) read it as allowed — [740718](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740718)

**Labels: tables, consensus, soft labels, image-informed labels**
- **azamat1ch (team azamatt, #3 at 0.967 on 16 Oct; mapping display name → username assumed):**
  - "haven't used any external datasets so far, only the competition data" (2026-10-12).
  - "Short answer is labels" (2026-10-12).
  - "AI, not by hand. And don't rely on reports alone." (2026-10-13).
  - Source: [748295](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748295)
- **colum2131 (0.960):**
  - Public scores: ConvNeXt-T 2.5D 0.953, DINOv2 ViT-S 2.5D 0.951, the two ensembled 0.955. Both are trained with pseudo-labels and no external data (2026-10-09).
  - Pseudo-labels "are generated from the images, with the image model taking the same input as at submission time". Tried separate heads for LLM labels vs pseudo-labels; now "just blend the two at a fixed ratio and use the blended labels as training targets" (2026-10-16).
  - Source: [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)
- **colum2131, gold-58 numbers (2026-10-05).** Values are AUC/accuracy.

  | Label source | Effusion | Synovitis |
  |---|---|---|
  | LLM, reports only | 0.835/0.707 | 0.799/0.741 |
  | LLM, reports + MRI | ~0.91–0.92/0.845 | ~0.84–0.86 |
  | Image model on own pseudo-labels | 0.986/0.931 | 0.859/0.776 |

  - About 73% of reports say nothing about synovitis.
  - Synovitis "not mentioned" in the report: 11/36 (31%) are gold-positive.
  - Effusion reported absent or normal: 10/26 (38%) gold-positive.
  - Effusion reported small or equivocal: 14/18 (78%) gold-positive.
  - "The expert panel calls fluid positive more often than the reports do."
  - Source: [745949](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745949)
- **Eight public LLM label sets vs gold-58 (karttikjangid05, 2026-10-08):**
  - 11 of 12 findings show no clearly different best set (bootstrap 95% CIs). MCL (9 positives) cannot be ranked.
  - Macro AUC: plain mean of six CC0 sets 0.8887; steven_v4 0.8927; steven_v2 0.8873.
  - JEV showed that per-finding blending tuned on gold overfits.
  - Sources: [747540](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747540); [consensus dataset](https://www.kaggle.com/datasets/karttikjangid05/rsna-knee-public-llm-label-sets-compared)
- **hengck23 (0.956 on 15 Oct, 16 submissions), 4–15 Oct.** Statements, mostly without ablations:
  - Show a multimodal LLM the report *and* images and ask it to label the ROI. Use "active labelling" and spend the costliest annotator on the most uncertain samples (4–6 Oct) — [745861](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745861)
  - Gold labels were likely made without looking at the report, so report-to-label agreement tops out near ~0.90 (9 Oct) — [747658](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747658)
  - Claude 4.5 labels from the NTejas-1 repo score AUC 0.8919 on gold (10 Oct). Local unsloth Qwen3.8-27B report labels score 0.87 (14 Oct) — [747658](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747658)
  - Training MRI and text models on the non-gold 4.3k with Claude Opus 4.5 labels and blending them gives gold-58 CV 0.938 (10 Oct). An MRI model trained to mimic JEV, ensembled 50/50, scored LB 0.926 (11 Oct) — [745861](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745861)
  - Give the labelling prompt "the right to say I don't know" and exclude those cells from training (12 Oct) — [745861](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745861)
  - Rule of thumb: an LLM label set at 0.89 on gold gives ~0.95 train AUC, ~0.92 gold validation and ~0.930 public LB (13 Oct) — [745861](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745861)
  - "model is not the key. it is your labels." (9 Oct) — [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)
- **Agreement filter (Prateek Grover, 0.950, 2026-10-12):** train on a cell only if the LLM labellers agree strongly (> 0.9 or < 0.1). This dropped ~25% of samples "but improved my results on CV and public LB" — [745861](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745861)
- **Prompt examples (Ali Ozan Memetoglu, 2026-10-05/06):** Claude labels gave at best a 0.942 single vision model. Passing labelled examples in the prompt instead of a summary moved local CV −0.004 but public LB +0.002 — [745861](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745861)
- **Tucker Arrants, team #2 (2026-10-02):** "I use a simple extraction process with Qwen, no frontier models at all, and most of my models are in the low 0.95 range on the LB. I am doubtful that 'improving labels' matters much". Gold-58 and the LB "are created by the same radiologists". This view conflicts with azamat1ch's above — [745214](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745214)
- **Label swap alone (SpeedSci, 2026-10-02):** changing only the labels moved a DINOv2 from LB 0.931 to 0.942 — [745214](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745214)
- **Unverified offers:**
  - OmerZalman offers a pseudo-label set covering ~60% of labels at "97.2% accuracy" on expert labels. That is accuracy, not AUC (2026-10-10) — [748151](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748151)
  - Kirderf's vision-LLM benchmark (images rendered as a mosaic and scored by VLMs on gold) had incomplete runs across models, and the author asks readers to disregard early numbers (13–16 Oct) — [747658](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747658)
- **nartaa / Danial Zakaria label recipe for the 0.949 CoAtNet (2026-10-13):**
  - Mean of his own Gemini 3 Flash (low thinking) labels and stevenleehans v4_blend; where v4 = 0.25 ("not addressed", ~19% of cells) he keeps his own.
  - Then averaged with riadmohamed42's JEV HYBRID where available.
  - pos_weight = (1−p)/p, clipped to 1–10.
  - Masked cells still count in the 12-cell denominator.
  - Gold-58 excluded from gradients but used for epoch selection; JEV itself was tuned on gold, "so gold-58 is not an independent estimate".
  - Source: [746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792)

**Geometry, series, crops, resolution, backbones, heads**
- **azamat1ch, 5-fold single architectures (2026-10-09):** ConvNeXt-T **0.959**, EfficientNetV2-S 0.956, CoAtNet 0.952 — [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)
- **Other single-model claims since 1 Oct:**
  - CoolinLai (team Attention Is All You Knee, 0.962): 5-fold ResNet-50 at 224 px, 0.954 (28 Sep and 2 Oct).
  - Tanuki_boosting: 0.955 single model, 5-fold (6 Oct).
  - Musa Tur Farazi: 0.956 single model, single fold, 224 px (6 Oct).
  - k256.dev: 0.945 no-fold and 0.947 fold-0 with LLM labels only (10–11 Oct).
  - Adel Zebiche: DINOv2 ViT-S fold-0 at 224 px 0.952 (gold 0.934); best blend 0.957; own labels; no external data (11 and 14 Oct).
  - Sources: [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304); [749426](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749426)
- **nartaa's S6 ConvNeXt retrained on his own labels: 0.950 (2026-10-14).**
  - Backbone convnext_small.fb_in22k_ft_in1k with per-finding spatial attention and window-attention MIL.
  - Input: full field of view **plus 100 mm knee-centred crops placed by the SKM-TEA anatomy V-Net anchors** from heliosli's checkpoint. Cached at 320 px, cropped to 288, 3 slices per window at 3.5 mm physical spacing.
  - Training: 64 windows per study, 16 epochs, AdamW (encoder 1e-4, head 1e-3, wd 0.05), batch 2, cosine with 5% warm-up, bf16. Trains on 4,263 report-labelled studies; gold excluded.
  - Inference in fp16 on T4.
  - Fold-0 beat the CoAtNet locally (OOF 0.907 vs 0.890; gold 0.931 vs 0.919) but lost by 0.001 on the board. Author: public SD ≈ 0.003.
  - Sources: [749359](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749359); [s6fall-solo notebook](https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo); [weights](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights)
- **Knee-centred compartment crops (bliverrigpdoge, 2026-10-14):** ConvNeXt-small + EfficientNetV2-S trained on medial and lateral compartments from sagittal and coronal series plus the axial. Rank-blended at 0.2 into the 0.956 R7 + Raptor base, it gave **0.957** (weights private) — [R7 + Raptor + knee-centred ConvNeXt](https://www.kaggle.com/code/bliverrigpdoge/r7-raptor-knee-centred-convnext)
- **Earlier geometry evidence still cited:**
  - Ziad Ahmed (2026-09-23): window-coverage curve flattens after ~31 windows (k=16 0.8867 → k=31 0.8932 → k=62 0.8948, weak-label macro). RadImageNet R50 does better with the centre slice repeated ×3 than with 3 adjacent slices (0.8542 vs 0.8339 fold-0 OOF): "the channel contract was worth another +0.020". coatnet_rmlp_2 at 256 px instead of 384 loses 0.029.
  - Dread Development (2026-09-23): swapping pooling at inference on weights trained with attention costs ~0.02.
  - Source: [737696](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737696)
- **Encoder size:**
  - DINOv2-S → B: +0.0011, inside the noise floor of 0.0020 (stevenleehans, 2026-08-14) — [735154](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154)
  - DINOv2 B vs S at 224 px converge to the same endpoint (Komil Parmar, 0.961; 2026-08-29) — [738096](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738096)
  - 384 → 288 px "barely had any effect" (Raymond Yuen/laymond, team TRRXT, 0.961; 2026-09-25) — [743148](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743148)
- **Slice coverage (Sanjib Biswas, 2026-09-25):** going from 24 to 32 cached slices (69% → 84% of slices kept) was worth more than any label work. Swapping the LLM (qwen2.5-14b / qwen3-14b / qwen3-30b-a3b) was worth nothing (0.8619 / 0.8572 / 0.8677 on gold). Qwen3's thinking block silently breaks JSON decoding — [743148](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743148)
- **Slice order and FOV (stevenleehans, 2026-08-16):**
  - Sort by ImagePositionPatient projected on the slice normal; filename order matches true order only ~5% of the time.
  - Take 3 anchors × 3 physically adjacent slices.
  - Crop a fixed physical size, since FOV takes 71 distinct values (median 160 mm).
  - Source: [735154](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154)
- **Data traps (gchauhan, 2026-09-14):**
  - InstanceNumber runs against slice position in over a third of series.
  - Laterality is missing in about half of studies.
  - The "non-FS" bucket mixes T1, PD and T2.
  - Studies have 3–14 series; the most common combination covers only 40%.
  - Source: [741310](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741310)
- **Bilateral studies:** both knees are in the DICOM for 7 training studies (Tim Krige, 2026-08-17; Zoltan, 2026-10-11). Whether the test set has such studies is unanswered — [735639](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735639); [733343](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)
- **Synthetic negatives (hengck23, 2026-10-12, from X-ray work, not a knee ablation):** "random images (definitely negative) helps to suppress FP… mri from other body parts are also definite negative samples" — [745861](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745861)

**TTA, ensembling and the public-notebook frontier (0.956–0.958)**
- **Lineage in goodpjw2008's notebook (updated 2026-10-16):**
  - nartaa CoAtNet alone 0.949; goodpjw's ConvNeXt reader alone (3 folds) 0.929.
  - nartaa + heliosli S6: 0.954, measured five times.
  - nartaa + heliosli R7, 50/50: 0.956. Adding nartaa's 224 px / 0.8-crop checkpoint at 35% of the Raptor slot keeps 0.956 but ranks higher.
  - nartaa (384 + 224 with mirror) + heliosli **R6 with flip TTA**: 0.957.
  - The same plus **heliosli's per-finding Raptor weights** (Synovitis 0.8, Lateral OA 0.2, PF OA 0.6, Baker's 0.35, else 0.5): **0.958**, rank 195/6,077.
  - The reader at full weight now *hurts* (0.956 → 0.954), and an S6-distilled reader gives 0.953. "A third reader only pays when it is independent of the two it joins".
  - Source: [goodpjw2008 0.958 notebook](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958)
- **heliosli R6 (2026-10-14):**
  - ConvNeXt-small, bundle fe055406, 4 fold models, mean logit.
  - Its own Effusion "fluid" and Baker's "state" stackers.
  - Flip TTA: every fold also scores mirrored windows, medial/lateral outputs are swapped back and the two passes are averaged in logit space; **MCL comes from the plain pass**.
  - Final blend: per-finding percentile rank, 0.5/0.5 with the Raptor half = 0.65 rank(384) + 0.35 rank(224).
  - Sources: [heliosli 0.9576 notebook](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9576); [Blend mine in](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters)
- **External data inside the public blends (edteoh, 2026-10-16; read from manifests):**
  - nartaa 384 and 224 CoAtNets: OAI, 2,399 knees.
  - heliosli R7: fastMRI images with **fastMRI+ boxes as an auxiliary task** ("fm_aux", weight 0.3).
  - R6 and S6: no external data found.
  - anatomy.pt, which places the crops: **SKM-TEA**.
  - Raptor (dreaddevelopment): none found.
  - The dataset `heliosli/rsna-knee-r6-frozen-fe055406` now holds R7, not R6. The "orthofoundation" dataset is attached for code and anatomy.pt, and its OAI-pretrained ViT is not loaded.
  - Source: [749858](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749858)
- **Ensemble ceiling (leoprovorov, 2026-10-16; estimates):** shared-error correlation among the published models is ρ ≈ 0.86–0.89, where "eight models would reach about 0.960". Report labels reproduce gold at 0.89 (95% CI 0.86–0.92), yet the models score ~0.957 — [0.957 Explained](https://www.kaggle.com/code/leoprovorov/rsna-knee-0-957-explained-why-the-open-ensembl)
- **Mirror and second-checkpoint gains:**
  - nartaa's anatomical mirror (sagittal channel reversal plus coronal/axial width flip, sigmoids averaged): +0.001 earlier — [746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792)
  - Mirroring the 224 checkpoint: "One Free Flip, 68 Ranks Up" inside the 0.956 block (varshitha, 2026-10-14) — [notebook](https://www.kaggle.com/code/varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up)
  - The 224 / 0.8-crop checkpoint helps because "Different resolution and field of view seem to make it wrong on different studies" (varshitha, 2026-10-13) — [746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792)
- **Adding a weaker reader can lose (prvsiyan):** mirror arm alone 0.949; adding goodpjw's ConvNeXt reader at a pre-declared 0.70/0.30 gave 0.948, so it was not adopted. The author avoids per-label weights and calls the six public 0.950 notebooks per-label hand-set or gold-tuned — [The bee's knees](https://www.kaggle.com/code/prvsiyan/the-bee-s-knees-final-rsna-push)
- **Per-label routing risk (Shenzai25, 2026-10-15):** in the most-forked 0.941 notebook, Lateral Meniscus = 1.00 to CoAtNet zeroes three stages for that label, and "this instrument has no reading for it". Ablation to uniform 0.60 pending — [749570](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749570)
- **Will public notebooks hold?** k256.dev (2026-10-14): the public 0.954 notebook "might surprisingly not be overfitting"; BirdCLEF+ 2026 is a precedent. Cody_Null: "public notebooks nearly always overfit… But forks win medals more often than you would think" — [748181](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748181)

**Runtime, T4, hidden-rerun failures**
- **Chris Deotte (GM, 2026-10-15):** to debug "Notebook Threw Exception", make a commit notebook that infers 1,300 random train studies. Look for NaN predictions and malformed-DICOM errors — [749623](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749623)
- Others report repeated hidden-rerun failures after the 3 visible studies and 1,300 train studies pass (16 Oct, 4 Oct) — [749798](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749798); [745826](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745826)
- **bf16 on T4/P100** is emulated, about 50× slower; a time guard then wrote 0.5 for most studies (0.851 → 0.710). Check `torch.cuda.is_bf16_supported(including_emulation=False)` (2026-09-29 / 10-01) — [744230](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744230)
- **Decoding:** the data page lists JPEG Lossless and JPEG 2000 transfer syntaxes. Sampled files were uncompressed, but bring pylibjpeg or gdcm (Eesh saxena, 2026-10-09) — [747865](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747865)
- **Mount path:** the input can mount at `/kaggle/input/competitions/rsna-knee-abnormality-detection` or `/kaggle/input/rsna-knee-abnormality-detection` (Mingquan Zhuang, 2026-10-06) — [746124](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746124)
- **Efficiency board (Kaggle staff notebook output, submissions up to 14 Oct):**
  - #1 Claudio Verdú Ruiz 0.964; #2 Pa3฿aJluHа 0.961; #3 Scott Willis 0.959; #4 Optimo 0.960; #5 Wassim Dobbi 0.960; #8 tomoon33 0.960; #11 TomKroumov 0.958; #19 Sergey Bryansky 0.961.
  - Several top-10 public teams have their best efficiency ranks far down: CentralyDS #3909, RB & Tucker #1285, joinT fusion #3390, Nischay & Sho #5115.
  - Source: [efficiency LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb)

**Validation, CV vs LB**
- **colum2131 (2026-10-10), 15 own submissions:** gold-58 AUC vs public LB r = 0.66; OOF on ~4,300 LLM-labelled studies vs LB r = 0.87 — [748011](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748011)
- **datnt114 (0.961–0.962):** "i trust cv local… best pb 0.961 with oof (5 fold) 0.937 and gold58: 0.938" (2026-10-10) — [748011](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748011)
- **hengck23 (2026-10-10):** "submit classwise. you should see something" and "Assume: (local cv + local train)/2 = (public lb + private lb)/2" — [748011](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748011)
- **hengck23 (2026-10-06):** split by label *type* for validation instead of by data — [745823](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745823)
- **Cody_Null (0.958; 2026-10-14):** "Local validation is close to useless here. Only 58 labels are scored the same way as the leaderboard" — [748181](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748181)

### Inferences
- The evidence leans toward the gap from 0.956–0.958 (public blends) to 0.964–0.968 (top) coming from **own models trained on image-informed labels**, plus multi-fold, multi-architecture ensembling. Blending public checkpoints does not seem to supply it. The image-informed labels are LLM labels blended with image-model pseudo-labels or VLM reads (azamat1ch, colum2131, hengck23). No top team has confirmed this recipe, so it is an inference.
- External data is now low-risk if it needs only registration plus a licence click (host, 14 Oct), and disclosure is expected. OAI geo-blocking remains a live risk for prizes, and less so for medals. Two people near the top say they used none (azamat1ch; YYama of knee-deep), so external data is not necessary to reach 0.964–0.967.
- Within the public frontier, the moves that work are: adding a decorrelated view (different FOV or resolution, knee-anchored crops), flip TTA with a medial/lateral swap and MCL taken from the plain pass, and per-finding weights. Each is worth about one tick (0.001) and is close to public-split noise (SD ≈ 0.003).

### Gaps
- The public/private fraction (30% vs ~50%) is not confirmed by a host statement I could find. Site stratification of the split is unknown.
- The opening post of 749359 (nartaa S6) is complete. Several newer posts with 0 comments (749570, 748925) have pending updates.
- I could not see the images in hengck23's image-only comments: hyperparameter tables, "segmentation label", "trainability".

## Q2. What members of teams at ≥ 0.960 said publicly, and their prior RSNA winning recipes

### Takeaway
- Explicit public hints come only from a handful of top-team members.
  - Tucker Arrants (#2): simple CNN at 224/288, simple Qwen extraction, validate on extracted labels, student beats teacher, external data could matter a lot.
  - azamat1ch (#3): labels, image-aware and AI-made; no external data; ConvNeXt-T 5-fold at 0.959.
  - tennogh (YOKKOISHO): 288 px, OOF pseudo-labels.
  - YYama (knee-deep): no external data; details after the deadline.
  - colum2131, datnt114 and CoolinLai at 0.960–0.962 add similar detail.
- Several top teams include prior RSNA winners whose recipes are public:
  - wadakoki (joinT fusion): RSNA 2024 lumbar 1st.
  - Nischay Dhankhar: RSNA 2023 abdominal trauma 1st.
  - Theo Viel (GM-Enhanced Kaggle Agent): RSNA 2023 2nd and RSNA 2025 aneurysm 6th.
  - tomoon33: RSNA 2025 aneurysm 1st.
  - Claudio Verdú Ruiz: RSNA 2024 lumbar 13th.
- Their common recipe: localize or crop the ROI with an auxiliary model, then a 2.5D 2D-CNN (CoAtNet/ConvNeXt/EfficientNetV2/MaxViT) over adjacent-slice triplets, then sequence aggregation (attention-MIL, GRU/LSTM or transformer), auxiliary segmentation or location losses, multi-seed ensembles.
- I found **no public statements** from CentralyDS (#1), Sergey Bryansky, Ilya Kolb, MEJAL, Team Kangaroo, monnu, Zhengru Li or Pa3฿aJluHа.

### Cited Findings

**Team #2, RB & Matheus & Tucker & Tony (0.967, 325 submissions)**
- **Tucker Arrants, Aug–Oct:**
  - "Single model 0.934 @ 224px"; 0.943 later as a 5-fold ensemble (Aug 16–17).
  - "around 25m for preprocessing and inference… including TTA" (Aug 17).
  - "I only validate against the report generated labels, never the provided labels… CV is around 0.87–0.9… all leading to the same ~0.94" (Aug 31).
  - "Single prompt LLM extraction has not been beaten, outside of noise"; "Image model 'student' seems to consistently outperform the label extractor 'teacher'" (Aug 31).
  - "224 / 288 is the sweet spot… only tried ResNets and EffNets" (Sep 3).
  - Gold-58 used in training (~1.3% of loss mass), not for validation (Sep 5).
  - Sources: [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)
- **Tucker's recipe advice (2026-09-10):**
  - "Start with 224px / 288px and a simple CNN encoder, like a ResNet34 or EffNetB0. No attention mechanisms, just use simple pooling… You can score in the 0.94+ range with this alone".
  - Validate on extracted labels with a noise threshold of ~0.003; then ablate standard and MRI-specific augmentation from past competitions; then backbones and resolution; then attention and new encoders.
  - Low prior that DINO inits beat ImageNet init.
  - Source: [740610](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740610)
- **Tucker on labels and data:**
  - Readers use stricter thresholds, so "the reports over-call" (2026-08-08) — [733836](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733836)
  - His cache is 50 GB at 384 px (2026-09-25) — [743303](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743303)
  - "external data, even small amounts of it, can substantially increase score in this competition" (2026-09-26) — [743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)
  - On OAI he moved from "It is a no" (21 Sep) to posting an NDA reply that it is open to all (8 Oct) and calling the situation "annoyingly grey" (9 Oct) — [741819](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819); [743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)
- **Other team #2 members:** "RB" (likely Rashmi Banthia; mapping unverified) posted only meta comments about AI-written forum posts and a BirdCLEF+ 2026 paper link (Sep 28 – Oct 13) — [743916](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743916); [749038](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749038). A participant named "Dung Duong (Tony) Huynh" asked how stevenleehans reduces each series to 9 slices; whether this is team #2's "Tony" (tonylica) is unverified (2026-08-16) — [735154](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154)

**Team #3, azamatt (0.967)**
- azamat1ch: 5-fold single architectures ConvNeXt-T 0.959, EfficientNetV2-S 0.956, CoAtNet 0.952 (Oct 9). No external data, "Short answer is labels", "AI, not by hand. And don't rely on reports alone." (Oct 12–13) — [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304); [748295](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748295)

**joinT fusion (0.966): wadakoki won RSNA 2024 lumbar spine as "NANACHI"**
- **Recipe (2024-10-12):**
  - Two stages: localize (3D ConvNeXt for instance number; 2D ConvNeXt-base/EfficientNetV2-L for coordinates), then crop.
  - Severity from **attention-based MIL + bi-LSTM** with an auxiliary attention loss, on ConvNeXt-small and EfficientNetV2-S encoders.
  - Attention-MIL took public LB 0.37 → 0.35; bi-LSTM, aux losses and ensembling took it 0.35 → 0.33 (that metric is lower-is-better).
  - "Random shift of instance_number was crucial for robustness" to first-stage error.
  - Sources: [lumbar 1st place](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/540091); [RSNA winners page](https://www.rsna.org/artificial-intelligence/ai-image-challenge/lumbar-spine-degenerative-classification-ai-challenge)

**Nischay and Sho, Aillis CEO (0.966): Nischay Dhankhar won RSNA 2023 abdominal trauma**
- **Recipe:**
  - 3D segmentation for study-level crops.
  - 96 equidistant slices reshaped to (32, 3, H, W) adjacent-slice triplets at 384 px.
  - 2D CNN + GRU (CoaT-Lite Medium/Small, EfficientNetV2-S).
  - **Auxiliary segmentation loss gave +0.01 to +0.03**.
  - Slice-level soft targets derived from organ visibility.
  - Source: [Nischaydnk/RSNA-2023-1st-place-solution](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution)

**GM-Enhanced Kaggle Agent (0.965): Theo Viel, Max Jeblick, Jiwei Liu, dott1718, arc144**
- **Theo Viel, RSNA 2025 aneurysm 6th (with Ian Pan and Brendan Artley):**
  - 2.5D pipeline: ROI crop, then 2D frame classifier, then a sequence model.
  - Backbones `coatnet_rmlp_2_rw_384` and `maxvit_rmlp_base_rw_384`; "The Relative Position Encoding MLP (rmlp) helps a lot".
  - Custom pooling keeps left/right information.
  - About +0.01 CV each from: ROI crop, strong ShiftScaleRotate and colour augmentation, CutMix/Mixup, **horizontal flip that also flips the targets**, adjacent frames as channels, spacing-aware sampling.
  - Max aggregation 0.87, then a sequence model 0.88, then a 3-model ensemble 0.895 CV.
  - Sources: [aneurysm 6th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/611925); [TheoViel/RSNA-Intracranial-Aneurysm](https://github.com/TheoViel/RSNA-Intracranial-Aneurysm)
- **Theo Viel, RSNA 2023 abdominal trauma 2nd:** "The key… is cleverly sampling frames to feed meaningful information and reduce label noise". 2D models + RNN with organ-crop models; heavy augmentation + CutMix; maxvit_tiny_tf_512 was best — [TheoViel/kaggle_rsna_abdominal_trauma](https://github.com/TheoViel/kaggle_rsna_abdominal_trauma)
- **Theo Viel, RSNA 2024 lumbar (NVSpine, 6th):** CoAtNet for coordinates, then crops, then CoAtNet + RNN — [TheoViel/kaggle_rsna_lumbar_spine](https://github.com/TheoViel/kaggle_rsna_lumbar_spine)
- **Agent-led workflow:** the team name suggests one. Chris Deotte (not on this team, 0.957) describes his own "minimal human in the loop" agent workflow (2026-09-27) — [743916](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743916)

**Claudio Verdú Ruiz (0.964; #1 on the efficiency board with 0.964)**
- **RSNA 2024 lumbar 13th:**
  - Keypoint localization, then fixed-spacing 96×96 patches.
  - 2D backbone (regnetz_b16, hgnet_tiny) + encoder-decoder transformer with learnable per-disease query vectors and XYZ metadata.
  - Patch-model pretraining before the sequence model.
  - Left and right sides trained as separate rows of one model ("Leveraging anatomical symmetries").
  - TTA.
  - Sources: [lumbar 13th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539510); [efficiency LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb)

**tomoon33 (0.962; RSNA 2025 aneurysm 1st)**
- **Aneurysm recipe:** coarse-to-fine nnU-Net vessel segmentation, then ROI, then a 3D classifier on an nnU-Net encoder pretrained for segmentation, with mask-guided pooling and an auxiliary sphere task. Left-right mirroring disabled for the fine models "to preserve anatomical asymmetry". He is #8 on the knee efficiency board (0.960) — [aneurysm 1st](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/611846); [efficiency LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb)

**Other ≥ 0.960 voices**
- **tennogh (YOKKOISHO, 0.964):**
  - "0.942 raw, 0.943 with TTA at 288x288. OOF pseudo-labels have been pretty well correlated with LB" (2026-09-03).
  - Own labels "not measurably better than the public ones"; "not sure resolution is such a big driver".
  - "clever 'fixes' end up overfitting to the 58 set" (2026-08-31).
  - Source: [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)
- **YYama (knee-deep, 0.964):** has not used any external dataset ("assuming… any external dataset could potentially be prohibited"). The team will share details after the competition (2026-10-07) — [743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)
- **Cà Rốt Production (team Vinmec - VSF, 0.960 on 15 Oct):**
  - "0,95+ with single fold single model… RESNET" (2026-10-09).
  - Advises against external data: "many top-scorer wont try it" (2026-10-12).
  - Earlier geometry: 384 px, 12 slices, 140 mm, about 16 random windows out of 60 (2026-09-25).
  - Sources: [747613](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747613); [743148](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743148)
- **Optimo (0.962):** "no code" approach; did not write a line himself (2026-09-28) — [743916](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743916)
- **colum2131, datnt114, CoolinLai:** covered under Q1.
- **Cody_Null (0.958):** "OAI… made very little difference in my pipeline" (2026-10-14) — [749145](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749145). "a number of teams… taking effnet b0 to 0.94+ on LB with 5 fold ensemble + TTA with 0.88 AUC on extraction labels" (2026-09-25) — [743148](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743148)

**Social and off-Kaggle searches**
- Web searches for the competition, team names, X/LinkedIn and Rednote found only news, launch posts and one early blog. That blog: 18th of 792 teams on 9 Aug (public 0.903), Qwen3.6-35B labels, EfficientNet-B3 MIL — [HF blog](https://huggingface.co/blog/bishnoiyash/rsna-competetion); [RSNA news](https://www.rsna.org/news/2026/august/ai-challenge-knee-mri)
- Top-member GitHub accounts (TheoViel, Nischaydnk, claverru, wadakoki, rashmibanthia, TuckerArrants, ilyakolb, sggpls) show **no knee repo** among their 6 most recently pushed public repos as of 16 Oct — [GitHub API user repos](https://api.github.com/users/TheoViel/repos)

### Inferences
- Every prior-winner recipe on these teams uses some form of **localization-driven cropping plus an auxiliary task**: segmentation/coordinates, aux seg loss, mask-guided pooling. In the knee competition the public analogue is the SKM-TEA anatomy V-Net anchor crops (S6/R6/R7, nartaa S6) and fastMRI+ boxes as an aux task (R7). This makes "knee/compartment ROI + aux localization" a likely ingredient of the top private recipes (inference).
- **Flip with target swap:** Theo Viel trains with "horizontal flip that also flips the targets", and heliosli uses flip TTA with a medial/lateral output swap. For the knee this means mirroring together with swapping the MM↔LM and MedOA↔LatOA labels and leaving MCL, ACL and other unpaired labels unchanged (inference). This differs from the plain mirror TTA our team tried.
- Top teams' long runtimes and low efficiency ranks suggest large ensembles at inference: CentralyDS's best efficiency rank is #3909, Nischay's #5115. Claudio shows 0.964 is reachable with a fast pipeline. Both are inferences from the efficiency board.

### Gaps
- No public posts, notebooks or repos from CentralyDS, Sergey Bryansky, Ilya Kolb, MEJAL, Team Kangaroo, kosuke123, monnu, TomKroumov, Zhengru Li or Pa3฿aJluHа.
- The deepios "Rednote" post is second-hand. Rednote content is not indexed by the search tools available.
- No X or LinkedIn posts by top-team members about this competition were found.
- Mapping forum display names ("RB", "Dung Duong (Tony) Huynh") to team usernames is unverified.

## Q3. GitHub repositories for this competition: what they implement and claim

### Takeaway
About 190 public repos were pushed since 1 Aug (unauthenticated search). **None is from a team ≥ 0.960.** The most useful are:
- **darcywjoseph** (0.951): five families, a measured "what moved the score" table and a per-finding public probe.
- **A-b29** (0.956 fork + teacher→student denoising).
- **NTejas-1** (Claude Opus 4.5 prompt and labels).
- **zoliseress** (0.942 single EffNet-B0; Qwen/vLLM labelling package).
- **gokdenizby** (0.950–0.954 routing blends, LB-tuned).
- Prior-RSNA-winner repos from current top-team members, listed under Q2.

### Cited Findings
- **darcywjoseph/Knee-Abnormality-Detection (pushed 2026-10-14):**
  - Public **0.951**: 47 checkpoints, 5 families.
  - Efficiency entry 0.948 in about 4 minutes: two series-bag twins, 16 slices at 288 px, GPU-side crop and resize. Matching the training cache pixel-for-pixel gave +0.013.
  - Labels: Claude extraction (0.90 vs gold); prompt v3 drops rules the gold contradicts (e.g. "avascular necrosis → contusion") and is stricter on mild OA. Then a 3-round pseudo-label loop: 50% LLM, 50% held-out ensemble, quantile-mapped.
  - Cache: 140 mm crop, 0.5–99.5 percentile clipping, right knees mirrored to left.
  - Model: series-bag model (every series, 98% kept vs 85% for six slots) with attention-MIL (Ilse) heads. The per-finding head lost on six slots and "paid only with the bag".
  - Loss: BCE on soft targets weighted 0.25 + 0.75·2|p−0.5|; gold rows hard at 3× weight; no flips.
  - **What moved the public score:**
    - Every series in a bag: +0.007
    - Fifth family on public LLM labels: +0.003. "The diversity came from labels, not architectures".
    - Labels v3 + pseudo-label loop: +0.005, but a family without the loop scored the same, 0.948.
    - 384 vs 320 px: level. ConvNeXt-S vs T: level. 176 mm crop: level.
    - PF compartment crops: +0.000 on PF OA.
    - EMA/heavy augmentation: +0.0017 offline, −0.003 public.
  - Source: [SOLUTION.md](https://github.com/darcywjoseph/Knee-Abnormality-Detection/blob/HEAD/SOLUTION.md)
- **A-b29/RSNA-Knee-Competition (status 2026-10-15):**
  - Public 0.956 (reproduction of R7 + Raptor). Students on denoised labels were "running now".
  - Teacher→student denoising: 4-fold teacher, OOF on 4,349 studies, mapped to the label scale, mixed 50/50. Gold agreement rose 0.903 → 0.927 (Effusion .880 → .964; Fracture .901 → .942); gold never used for training.
  - 96-slice ConvNeXt attention-MIL at 0.90–0.91 gold.
  - Engineering: `nn.DataParallel` leaked +30 MB per step and was 2× slower than one T4, so one process per GPU; torch.compile cut step time 19%.
  - Source: [README](https://github.com/A-b29/RSNA-Knee-Competition)
- **gokdenizby/rsna-knee-abnormality:**
  - Tiers: nartaa CoAtNet-384 (0.949), then + goodpjw ConvNeXt reader routing (0.950), then "Routing 2.0 + anti-tie + re-rank" (0.954).
  - Its own limitation notes: "Tuned on public LB"; "Gold-58-derived weights overfit (tested: 0.949)".
  - Source: [README](https://github.com/gokdenizby/rsna-knee-abnormality)
- **zoliseress/Kaggle_2026_knee_MRI:**
  - Best public **0.942** from a single EfficientNet-B0 at 320 px.
  - Labels from Qwen served with vLLM, as soft targets with supporting quotes. "Label quality made the largest difference… most architecture changes made little difference".
  - Claims a 450-study "radiologist" reference set held out of training (unverified; it is not stated whether radiologists read images or reports).
  - Source: [README](https://github.com/zoliseress/Kaggle_2026_knee_MRI)
- **NTejas-1/RSNA-Knee-Abnormality-Detection:** public 0.941 (0.940 public stack + two RadImageNet ResNet50 arms). Includes the Claude Opus 4.5 extraction prompt and labels, which hengck23 cites as gold AUC 0.8919 (2026-10-10) — [repo](https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection); [747658](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747658)
- **Forks and reproductions:**
  - Deveshu04: Qwen3-8B labels and RadImageNet CNNs; 0.938 single family and 0.940 best ensemble — [repo](https://github.com/Deveshu04/RSNA-Knee-Abnormality-Detection)
  - sinhaaditya5, willmurray9, TranBaDat2607, david-pilny, SummerLinda: forks of the 0.94x stack plus a reader (0.941–0.944); TranBaDat2607's best is 0.942 with its own member at w = 0.10; own pipelines 0.905–0.918 — [sinhaaditya5](https://github.com/sinhaaditya5/kaggle-rsna-knee-abnormality); [willmurray9](https://github.com/willmurray9/kaggle-rsna-knee-abnormality-detection); [TranBaDat2607](https://github.com/TranBaDat2607/RSNA-Knee-Abnormality-Detection); [david-pilny](https://github.com/david-pilny/RSNA-Knee-Abnormality-Detection)
  - S-Riku-tus and Msmile-shiny: reproduction logs of R6/R7 blends at 0.956–0.957, in Japanese and Chinese — [S-Riku-tus](https://github.com/S-Riku-tus/pj-RSNA-Knee-Abnormality-Detection); [Msmile-shiny](https://github.com/Msmile-shiny/CVproject_RSNA-knee-detection)
  - cubres: engineering harness (hash-pinning, watchdog) behind prvsiyan's mirror-reader notebook — [cubres](https://github.com/cubres/rsna-knee-abnormality-detection)
- **Repos named in the brief:**
  - konradcinkusz/rsna-knee-abnormalities-efficiency-lb: Polish-language project documentation (stated as of 10 Aug) plus a skeleton 2.5D + multilingual text-branch `model.py`; no scores — [repo](https://github.com/konradcinkusz/rsna-knee-abnormalities-efficiency-lb)
  - Karansaini260/RSNA-Knee-Abnormality-Detection: generic multimodal framework (2.5D CNN + BiLSTM text branch, keyword weak labels); no scores — [repo](https://github.com/Karansaini260/RSNA-Knee-Abnormality-Detection)
- **External data with public code or labels flagged by participants:**
  - 3DReasonKnee: Rajpurkar Lab, CC BY 4.0, about 512k Q&A rows with MOAKS-style grades and per-subregion bounding boxes on knee DESS MRI, arXiv 2510.20967. nartaa used it to match MOAKS rows to OAI series; hengck23 flagged it on 15 Oct. The images remain OAI data under NDA terms (nartaa) — [HF dataset](https://huggingface.co/datasets/rajpurkarlab/3DReasonKnee); [745861](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745861); [746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792)
  - SGMRI-VQA and Hulu-Med-14B (a medical MLLM that takes 3D NIfTI) were suggested by hengck23 for VLM labelling (2026-10-13/14) — [747658](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747658)

### Inferences
- The only GitHub-documented, measured gains above the 0.94x stack come from **label lineage diversity** and **using every series**, both from darcywjoseph. A-b29's OOF-mix denoising shows a large gold gain, but darcywjoseph found the same kind of loop did not transfer to the LB (0.948 with or without it). This matches our team's failed OOF relabelling.
- Repos that claim 0.95+ mostly reproduce public weights. Usable code worth reading: darcywjoseph (series-bag MIL, judge and attention tools, fast GPU loader), A-b29 (staged inference with fallbacks), and the TheoViel and Nischaydnk prior-RSNA repos.

### Gaps
- `gh` is not installed, so GitHub code search (which needs authentication) was impossible. `gh search code "rsna-knee-abnormality-detection"` and topic search were not run. Repos without "rsna knee" or "knee abnormality" in name or description may be missed.
- I did not clone or audit any repo's code. Score claims come from READMEs only.

## Q4. Per-finding difficulties (Lateral OA, PF OA, Synovitis, MCL, others) and how people addressed them

### Takeaway
The weakest findings at the top are PF OA, Lateral OA, Synovitis and Lateral Meniscus. One public probe put a 0.948 model at 0.890 / 0.896 / 0.932 / 0.956 on these, against 0.963 for the other eight. Diagnoses and fixes:
- **Lateral OA:** a report-vs-image *definition* conflict that no prompt can fix.
- **PF OA:** looks *image-limited* (text 0.96, image models ~0.89).
- **Synovitis:** a *silent-report* problem. Fixes in use: effusion-based imputation, OOF/pseudo-label filling, image-aware LLM labels, MOAKS Hoffa-synovitis from OAI and a Raptor-heavy blend weight.
- **MCL:** too few gold positives to judge, and it has no lateral twin when mirroring.

### Cited Findings
- **Per-finding public AUC by inversion probe (darcywjoseph; probe: 0.5 + 6 × score drop, ±0.006):**

  | Finding | Public AUC |
  |---|---|
  | PF OA | 0.890 |
  | Lateral OA | 0.896 |
  | Synovitis | 0.932 |
  | Lateral meniscus | 0.956 |
  | Other eight, mean | 0.963 |

  - Lateral OA: "every report-based extractor tried, ours and three public ones, makes the same five false positives on the expert set… No prompt can correct that."
  - PF OA: "the extractor scores 0.96 from the text, every image model scores about 0.89, and neither higher resolution nor compartment crops changed that."
  - Source: [SOLUTION.md](https://github.com/darcywjoseph/Knee-Abnormality-Detection/blob/HEAD/SOLUTION.md)
- **Stuck at 0.954 (Kabir_Verma01, 2026-10-11):** "PF OA, lateral OA and Synovitis are showing poor results consistently" — [748295](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748295)
- **Gold-58 OOF on these findings:**
  - architkonde (0.950 single model, 2026-09-18): Synovitis 0.797, Lateral OA 0.816, PF OA 0.874 — [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)
  - Will (2026-09-10): image model vs report target: Lateral OA 0.782 vs 0.829; Lateral Meniscus 0.858 vs 0.894; Synovitis 0.693 vs 0.676; Effusion 0.901 vs 0.696; MCL 0.980 vs 0.921 (9 positives). 10 of 11 gold-positive Lateral OA studies are mentioned in their reports — [740610](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740610)
- **Blend weights reflect strengths (heliosli, via goodpjw2008, 2026-10-16):** the Raptor (CoAtNet) half gets Synovitis 0.8, Lateral OA 0.2, PF OA 0.6, Baker's 0.35, else 0.5. "R6 is strong on the lateral compartment and weaker on synovitis" — [goodpjw2008 notebook](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958)
- **OAI MOAKS targets for the three hardest OA/synovitis findings (nartaa, 2026-10-13):**
  - PF OA = 1 if MOAKS full-thickness cartilage loss ≥ 1 in any PF subregion (PM, PL, FMA, FLA).
  - Lateral OA: the same rule over lateral tibiofemoral subregions (FLC, FLP, TLA, TLC, TLP).
  - Synovitis = Hoffa-synovitis (MSYIC) ≥ 1.
  - Hard 0/1, other cells masked. Counts: PF OA 2,396, LatOA 2,397, Synovitis 1,659.
  - OAI took 0.942 → 0.945 in his sequence (not compute-matched).
  - Source: [746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792)
- **Classes with the most label error (hengck23, 2026-10-09):** "there are some classes that has more error than others. and these classes have external data with external label". If using external labels, rank them "kaggle annotation (highest score) > external annotation > negative (least score)" (2026-10-10) — [747613](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747613); [745861](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745861)
- **Synovitis silence:**
  - 25.4% of all cells are "not addressed"; 83.7% for Synovitis, gold AUC 0.678.
  - P(syn | eff) = 0.63 vs 0.22 without effusion.
  - The Effusion field predicts gold Synovitis better than the Synovitis field (0.7115 vs 0.6780). Filling only undecided Synovitis cells from Effusion took the column 0.678 → 0.790 and the key 0.8780 → 0.8873.
  - Blanket learned imputation for all 12 findings was worse (Baker's −0.029, Contusion −0.031). Silence means absence for Baker's (3% positive when silent) but not for Synovitis (34%).
  - Source: stevenleehans, 2026-08-09 — [733932](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932)
- **Synovitis encoders (starkhushi, 2026-08-26):** a RadImageNet encoder reached 0.78 on Synovitis vs 0.62–0.72 for ImageNet models — [737566](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737566)
- **Synovitis and effusion fixes (2026-10-05):** colum2131's image-aware LLM labels reached Synovitis 0.84–0.86 and the pseudo-label image model 0.859 (table in Q1); training on soft probabilities keeps uncertain cells at 0.3–0.6 — [745949](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745949)
- **Synovitis OOF filling (architkonde, 2026-09-21):** used OOF predictions to fill silent Synovitis cells; "50/50 already jumped, going heavier on the model added a bit more". In-fold predictions "just parrot the labels back" — [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)
- **Effusion under-calling:**
  - Reports call fluid positive less often than experts (colum2131).
  - Tom Aindow saw gold positives where reports say "small" effusion despite the moderate/large rubric, and warns LLM confidence measures rule-application confidence, not image uncertainty (2026-08-18) — [735826](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735826)
  - A-b29's OOF mix took gold Effusion .880 → .964 — [A-b29 README](https://github.com/A-b29/RSNA-Knee-Competition)
- **Fracture:**
  - In 7% of reports but 31% of gold positives (gchauhan) — [741310](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741310)
  - SOFT targets helped Fracture +0.044 on gold in a controlled DINOv2 run (FHZ982) — [734105](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734105)
  - Image models rank some subchondral insufficiency-fracture patterns high where reports are silent (darcywjoseph) — [SOLUTION.md](https://github.com/darcywjoseph/Knee-Abnormality-Detection/blob/HEAD/SOLUTION.md)
- **MCL:**
  - Only 9 gold positives, too few to rank label sources (karttikjangid05) — [747540](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747540)
  - DINOv2-B gave MCL its largest per-label gain (+0.020, weak-label OOF; stevenleehans) — [735154](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154)
  - In flip TTA, heliosli takes MCL from the plain pass only — [heliosli 0.9576 notebook](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9576)
  - The rubric counts only *acute* high-grade tears — [733343](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)
- **Lateral Meniscus routing:** routed 100% to CoAtNet in the public 0.940–0.943 lineage (renta0426). Called unvalidated by Shenzai25 (2026-10-15) — [749570](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749570)
- **When silence means absence (Amil Gentili, radiologist, 2026-08-27):** an unmentioned fracture, meniscal or ligament tear means "I did not see" it. OA is often described indirectly (cartilage loss, chondromalacia, osteophytes) — [737650](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737650)

### Inferences
- The four hard findings need different fixes:
  - **Lateral OA:** image-side or compartment-specific information, or labels that ignore report over-calls. R6's knee-anchored compartment crops carry 80% of the blend weight for Lateral OA. OAI MOAKS lateral cartilage-loss targets are the cleanest external signal.
  - **PF OA:** more image signal, such as axial or knee-anchored crops and MOAKS PF; darcywjoseph's native-resolution PF crops gave +0.000.
  - **Synovitis:** filling silent cells. Effusion-informed priors, OOF or pseudo-labels, image-aware VLM labels and MOAKS Hoffa-synovitis are all in use.
  - **MCL:** rank-averaging across diverse models rather than per-label tuning, because gold has 9 positives.
  - These are syntheses, not measured ablations.
- Our team's failed per-finding specialists fit darcywjoseph's result that PF crops did not move PF OA. Per-finding *blend weights* across genuinely different models (heliosli) remain the only public per-finding lever that scored, at +0.001 (0.957 → 0.958).

### Gaps
- No public per-finding breakdown exists from any team ≥ 0.960.
- The "5 Lateral-OA false positives" are counted on gold-58 only.
- No public evidence shows that the OAI MOAKS targets specifically raised Lateral OA, PF OA or Synovitis AUC; only the macro LB change (0.942 → 0.945) was reported.
