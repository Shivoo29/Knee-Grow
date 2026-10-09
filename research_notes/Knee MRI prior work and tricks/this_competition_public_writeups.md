# RSNA Knee Abnormality Detection 2026: public write-ups by participants (Jul 30 to Oct 8, 2026)

Scope and method. I pulled notebook sources and metadata with the Kaggle CLI (`kaggle kernels list --competition rsna-knee-abnormality-detection --sort-by scoreDescending`, `kaggle kernels pull -m`). I read discussion threads through Kaggle's public read-only discussion API, the same JSON the forum page renders. I also read GitHub READMEs. All of this was collected on 2026-10-08. Unless a line says otherwise, every score below is a **public LB** number **claimed by the author**. I did not re-submit anything. The CLI listing sorts by score but does not print the scores, so rank order in that listing counts as evidence and is not a measured value. "Gold-58" means the 58 training studies that carry official radiologist labels.

Leaderboard context (Kaggle CLI, 2026-10-08): #1 Ilya Kolb 0.964, then Pa3฿aJluHа, CentralyDS and rhoskeri at 0.963. Zhengru Li, YOKKOISHO and dalab are at 0.962. Wassim Dobbi, knee-deep, Volkopat, joinT fusion, NMarkS and TomKroumov are at 0.961. Teams from #14 to about #18 sit at 0.960. Source: [Kaggle leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard).

---

## Q1. What are the best public notebooks and scores (above 0.943), and what do they add beyond the d4-blend / Speedy Raptors CoAtNet lineage?

### Takeaway
The public frontier moved on Oct 7. It is no longer the 25+ model DINO/A5/RadImageNet/Raptor/CoAt stack, which tops out at 0.943 to 0.944. It is now **one CoAtNet checkpoint (nartaa / Danial Zakaria, 0.949)** trained with **OAI external knees plus JEV and other blended weak labels**, run with a native 320 px crop and anatomical-mirror TTA. Forks that rank-fuse it with goodpjw2008's independent 2.5D ConvNeXt reader using per-target weights claim **0.950** ("Apex Grandmaster Stack"). Everything at 0.943 to 0.944 is the same stack with blend-constant changes, or that stack plus one weak extra reader.

### Cited Findings
**Public lineage of the "fork wall" (0.937 → 0.943), as summarised by evgendvorkin's RSNA_Versia_5 (2026-09-26):**
- 0.937, prvsiyan "Head and shoulders, knees and toes": 4-arm CoAtNet (v5, v10, v5-reverse, v8), global outer CoAtNet weight 0.60 — [RSNA_Versia_5](https://www.kaggle.com/code/evgendvorkin/rsna-versia-5); [prvsiyan](https://www.kaggle.com/code/prvsiyan/head-and-shoulders-knees-and-toes)
- 0.940, renta0426 "Hybrid Raptor Lateral Meniscus R100": the only change is Lateral Meniscus routed 100% to Raptor — [RSNA_Versia_5](https://www.kaggle.com/code/evgendvorkin/rsna-versia-5)
- 0.941, mattiaangeli "Bend the Knee to the Dinosaurs": adds the residual-gated CoAtNet reader (384 px adjacent-slice triplets, CoAtNet RMLP-2, finding-specific gated spatial head plus attention pooling across windows). Per-target outer routing is ACL 0.75, MedMen 0.80, LatMen 1.00, LatOA 0.75, Fracture 0.75 — [Versia_5](https://www.kaggle.com/code/evgendvorkin/rsna-versia-5); [Dinosaurs](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs)
- 0.942, maverickss26 "0942-restructured": no new models. It raises a5_w 0.45→0.52 and rad_alpha 0.50→0.55, sets the second-pass E13 weight 0.15→0.20, and raises raptor_k_eval to 94 windows — [Versia_5](https://www.kaggle.com/code/evgendvorkin/rsna-versia-5)
- 0.943, Mattia Angeli "Speedy Raptors" v34 and jiweiliu "Fast 2xT4" v9: the CoAt family grows from 2 to 4 readers (residual-gated, D4 depth-zone SWA3, Global96, Repair-v1), each weighted 0.25. The family is reduced as the rank of the mean of probabilities, then blended CoAt 0.40 / Raptor 0.60 — [jiweiliu](https://www.kaggle.com/code/jiweiliu/rsna-knee-fast-2xt4-inference); [Speedy Raptors](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-speedy-raptors-the-original)
- The full 0.943 stack contains 20 DINOv2 slot-attention members, 5 A5 attention-pooling folds, RadImageNet E10/E13/E11 heads, 4 Raptor CoAtNet views and 4 CoAt readers. skarin independently reproduced it at 0.943 on 2026-10-01 (submission 56741757) — [skarin](https://www.kaggle.com/code/skarin/rsna-knee-reproducing-the-0-943-public-stack)
- Names that carry a score are unreliable. "rsna-knee-d4-public0946" scores 0.943. The parent (ryokucha "d4-blend-0946-ours10") reached 0.946 only by blending a **private** ConvNeXt ensemble at 10% (statement dated 2026-09-28) — [sushanthtiruvaipati](https://www.kaggle.com/code/sushanthtiruvaipati/rsna-knee-d4-public0946-no-correction); [yamadan96](https://www.kaggle.com/code/yamadan96/rsna-knee-d4-public0946)

**Readers added on top of the 0.943 stack (0.944 to 0.945):**
- goodpjw2008, "Stack + 2.5D ConvNeXt MIL" (scored 2026-10-04; last run 2026-10-06). Stack alone 0.943; reader alone (3 folds) 0.929; 70/30 rank blend **0.944**. OWN_W 15% and 30% both scored 0.944; 45% scored 0.942. The same pipeline has scored 0.944, 0.944 and 0.943 on different runs (non-determinism). The reader takes canonical volumes at 0.4 mm/px with a fixed 154 mm FOV and orientation from ImageOrientationPatient, slices sorted along the slice normal. It uses up to 6 series (3 planes × fat-sat yes/no), 12 three-slice windows per series, a shared ConvNeXt-Tiny, tokens tagged with series type and slice position, a 2-layer transformer and per-finding attention pooling. Training: 5 folds grouped by report text, soft targets = **mean of four public label tables** (stevenleehans, pilkwang, lixin73, laymond), BCE, 14 epochs at 256 px. Gold-58 was never trained on. The stack needed 6.5 to 8 h of the 9 h limit — [goodpjw2008](https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944); weights at [goodpjw2008/rsna-knee-2-5d-convnext-reader](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader)
- bliverrigpdoge, "Raptor student blend" (2026-10-04): Versia_5 (0.943) blended 0.55/0.45 with an extra CoAtNet-RMLP-2 fine-tuned from the Fine Spacing Raptor checkpoint (80 slices, 2–98% span, 140 mm crop, 336 px). The extra model scores 0.942 alone; the blend scores **0.944** — [bliverrigpdoge](https://www.kaggle.com/code/bliverrigpdoge/rsna-knee-raptor-student-blend-lb-0-944)
- sujanmajhisuzan, "Tri-Specialist Superstack": romantamrazov's DINOsaur V5 says it reports **0.945**. It adds goodpjw's reader and renta0426's meniscus bag to the stack. DINOsaur V5 (2026-10-07) itself only adds a faster DICOM decode, dual-T4 sharding, "AUC-aligned fold-rank consensus" and "agreement-gated target routing". Its own score is not stated in the markdown — [DINOsaur V5](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v5); [Tri-Specialist](https://www.kaggle.com/code/sujanmajhisuzan/rsna-knee-tri-specialist-superstack)

**The new frontier: nartaa single CoAtNet (posted 2026-10-07):**
- Score history, all public LB. Own all-data CoAtNet with 80 slices / 78 windows: 0.939. Then 96 slices, native 384 cache, 94 windows: 0.940. Then blend plus **JEV labels**: 0.942. A 224 crop with SWA dropped to 0.940. **Adding 2,399 OAI knees with masked labels: 0.945**. The 384 full-field external-data recipe: **0.948**. A native 320 crop at inference: 0.948. **Anatomical-mirror probability averaging: 0.949** — [discussion 746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792)
- Training recipe: trained on 4,349 report-labelled studies, with gold-58 excluded from training but reused for selection. OAI supervised only PF OA, Lateral OA and Synovitis, and missing cells were *masked* rather than treated as negatives. Settings: 16 epochs, backbone LR 4.01e-5, head LR 1.95e-3, wd 0.0588, 12 windows per study in training, batch 1 with gradient accumulation over 16 studies. The checkpoint averages 3 selected epochs (14/11/12) — [746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792)
- Inference: 96 slices over 5 sagittal/coronal/axial slots and 94 three-slice windows, with crop [32:352] on native 384 and no resize. The mirror flips sagittal slice order (channel reversal); for coronal and axial it flips width. The two sigmoid outputs are averaged. Submit-to-score turnaround was 26.1 min — [accuracy notebook](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror); weights at [nartaa/rsna-knee-publication-swa-weights-20261007](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007)
- Efficiency variant: a 307 px crop resized to 224, single view, fp16, **0.945**, about 7.2 min turnaround — [0945 efficient](https://www.kaggle.com/code/nartaa/rsna-knee-0945-efficient-224crop)
- Label sources named: stevenleehans LLM labels, pilkwang LLM labels and [riadmohamed42 JEV labels + verification](https://www.kaggle.com/code/riadmohamed42/jev-knee-labels-verification), plus their own report-derived labels — [nartaa](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror)

**Forks of nartaa (Oct 7–8):**
- sujanmajhisuzan, "Apex Grandmaster Stack": the 0.949 CoAtNet plus goodpjw's ConvNeXt reader with **per-target rank weights** for the ConvNeXt share: Baker's 0.35, Contusion 0.30, MedOA 0.25, ACL 0.25, MedMen 0.20, MCL 0.20, LatMen/Fracture/LatOA/PF OA 0.08, Effusion 0.04, Synovitis 0.03. Code comments justify the weights with per-finding numbers ("ConvNeXt 0.976 vs 0.940" on Baker's), and the source of those numbers is unclear. It ranks first in the CLI score-sorted listing — [Apex](https://www.kaggle.com/code/sujanmajhisuzan/rsna-knee-apex-grandmaster-stack)
- haideptry, "V2 vs Apex Comparative Study": states that Apex V3 and V4 both scored **0.950** (unverified). V5 grid-searches per-target weights in [0, 0.55] on gold-58 and claims a target of "0.952–0.955+" (also unverified, and an overfitting risk) — [haideptry](https://www.kaggle.com/code/haideptry/rsna-knee-v2-vs-apex-comparative-study)
- kozykappa, "Gold-Gated Triple Reader": CoAtNet-384 plus ConvNeXt at a gold-selected 0.22 weight (gold 0.92802 vs 0.92294), with an optional 224 crop CoAtNet. Candidates must beat the incumbent in a paired bootstrap on gold before they are adopted. The author says explicitly that this is "not independent validation" — [kozykappa](https://www.kaggle.com/code/kozykappa/rsna-knee-gold-gated-triple-reader)
- matterhorn3838 "V2 Velciraptor DINOsaur Speed" and xianhan "0949 Anatomical Mirror Candidate" are both near-copies of the nartaa inference. haideptry calls Velciraptor 0.949 — [matterhorn3838](https://www.kaggle.com/code/matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed); [haideptry](https://www.kaggle.com/code/haideptry/rsna-knee-v2-vs-apex-comparative-study)

**Dread Development (origin of the Raptor CoAtNet):**
- The published Raptor set was 0.924 (64 slices, 140 mm crop, span 0.06–0.94, 336 px, 42 windows). Identical weights scored 0.927 at 62 windows. A solo SWA checkpoint reached 0.932 (Fine Spacing: 80 slices over 2–98%, 140 mm, 336 px, 78 windows). One CoAtNet-384 run on the full labelled set scored 0.936–0.938 with no fold ensemble. Statements dated 2026-08-27 to 2026-09-15 — [discussion 737696](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737696)
- Gold-panel selection notes (2026-08-19). On a 45-study gold set: CoAtNet-384 0.9025, Swin-B-384 0.8825, EffV2-L-480 0.8716; the 3-arm blend scored 0.9068. On the LB, CoAtNet alone scored 0.914 and the blends 0.914–0.915, so "the ~+0.010 it showed on the old 45-study gold set was gold-set noise". Expanding the corpus from 3,155 to 4,349 studies raised gold from 0.8923 to 0.9054, with the biggest gains on LatMen (+0.071), Fracture (+0.057) and LatOA (+0.048) — [Dread inference notebook](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model)
- 2026-09-19: Dread's own score was 0.944 with 13 min inference ("CoAtNet, paired with a ConvNeXt now"), after training about 40 models in two weeks — [discussion 742050](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742050)

### Inferences
- The CLI score-sorted order (Apex > haideptry > kozykappa > Velciraptor > nartaa-0949 > xianhan > pjmathematician d4-blend > d4-lite > aastikrajan s75-w50 > Tri-Specialist > nartaa-0945 > Raptor-student 0.944 > goodpjw 0.944 > ...) suggests pjmathematician's d4-blend and d4-lite sit between 0.945 and 0.949. I could not read their actual numbers (inference only).
- Beyond the lineage, the best public material adds three things: (a) **one strong, retrained single model** instead of 25+ legacy members; (b) **external OAI data and blended label tables**; (c) **anatomy-aware TTA and crop choices**. Rank-fusing two genuinely independent readers per target adds about +0.001.

### Gaps
- I could not see exact public scores for pjmathematician d4-blend/d4-lite, aastikrajan s75-w50, romantamrazov DINOsaur V5 or Velciraptor. The CLI does not print them and the Kaggle pages are JS-rendered.
- No public notebook above 0.950 exists. No team above 0.955 has published code or a write-up.
- I found no Hugging Face cards and no READMEs from raahimnawaz or pjmathematician. GitHub `gh` was unauthenticated, and the repos I guessed returned nothing.

---

## Q2. What do high scorers say separates them from the 0.943 "fork wall"?

### Takeaway
Teams at 0.950–0.964 consistently describe **strong single models (often 5-fold or single-fold ResNet/EfficientNet/CoAtNet/DINOv2 at 224–384 px)** trained on **their own LLM-extracted soft labels**. Several add **teacher→student pseudo-labelling or label/prediction mixing**, and nobody reports fine-grained blend tuning. The efficiency leaderboard shows public scores of 0.959–0.962 from fast submissions. So the top of the board is not giant ensembles.

### Cited Findings
- Single-model claims in the "Best single-model score" thread ([735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)):
  - miyamotodaiya: 0.955, single model, 5-fold (2026-10-06).
  - coolinlai: 0.954, 5-fold ResNet-50 at 224 px (2026-09-28).
  - musaturfarazi: 0.952, 224 px single model (2026-10-06).
  - andy2709 (GM): 0.950, single fold, from fine-tuning **Qwen3.5-2B** VLM at 384 with LoRA on the LM jointly with the vision encoder, about 3.5 h (2026-09-21/23).
  - architkonde: 0.950, single-fold 2.5D CoAtNet at 224. Gold OOF 0.930; weakest classes Synovitis 0.797, LatOA 0.816, PF OA 0.874 (2026-09-16/18).
  - laymond: 0.949, single full-train CoAtNet at 288 (2026-09-15/23).
  - scottgwillis: 0.949, single fold single model, about 5 min scoring, around 0.930 on gold (2026-09-05).
  - caoyupeng: 5-fold DINOv2 0.942, EffNet-B4 0.940, ResNet50 0.940; a fold-0 DINOv2 at 0.945.
- **Labels: teacher→student pseudo-labelling.** laymond uses "multiple weak-label sources (public, api, local) to train teacher models, then using the teacher models to generate pseudo-labels and eventually training a final model on them… Different teachers seem to help different targets", and reports total API cost under $5 with minimal difference between expensive and cheap LLMs — [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)
- **Labels: mixing OOF predictions into the extracted labels.** architkonde found that masking noisy labels did nothing, replacing them with model predictions was worse, and **mixing** the extracted label with *out-of-fold* predictions helped. The gain was largest on "silent" cases such as Synovitis (in-fold predictions "just parrot the labels back"). Rounding to 0/1 hurt and soft labels were better. pejophylo tried 50/50 soft bootstrapping, saw a gold and CV lift, and got no LB gain. This is mixed evidence — [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)
- cdeotte (GM, 2026-09-25): "If we use pseudo labels, then every single model can match our best ensemble CV score" — [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)
- Tucker Arrants (Master; team "RB & Tucker & Tony" is at 0.960 on the LB as of 2026-10-08). "Single prompt LLM extraction has not been beaten, outside of noise"; "the image model student consistently outperforms the label-extractor teacher"; "224 / 288 is the sweet spot". CV on extracted labels is 0.87–0.90 and correlates strongly with LB (2026-08-31). On 2026-10-02: "simple extraction process with Qwen… most of my models are in the low 0.95 range on the LB. I am doubtful that 'improving labels' matters much" — [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304); [745214](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745214)
- scottgwillis (efficiency #1, public 0.959): extracted labels locally with **Gemma 4**, about 0.89 on gold, no public labels; uses a smaller ResNet and EfficientNet; later a 5-fold reached 0.947 (Aug 22) — [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304); [efficiency LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb)
- Geometry: taindow used 392 px with a 150 mm center crop (0.383 mm/px) and random bags of 32 slices. cody11null (GM) went from a 100 mm to a 140 mm FOV, which "helped a ton", and 160 mm was similar. Dread and nartaa use 140 mm, goodpjw 154 mm — [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)
- Window coverage: 94 windows beat 24 (0.940 vs 0.934 for nartaa's older model). Dread: 42→62 windows gave +0.003, and 48→72 gave about +0.001 at the 0.944 level. Another team's curve flattens after about k=31 — [746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792); [737696](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737696)
- Mattia Angeli (2026-09-19): "basically no one besides DreadDevelopment and me has released genuinely new models… Repeat those minute-scale changes across dozens or hundreds of submissions and you are simply overfitting the public LB" — [742050](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742050)
- starkhushi (2026-09-19): about 200 teams sat at exactly 0.941. Adding own 2.5D members with holdout 0.85–0.87 at 15% dropped the score to 0.936: "Below ~0.90 solo, a member costs more than its diversity buys" — [starkhushi notebook](https://www.kaggle.com/code/starkhushi/rsna-knee-why-public-forks-stop-at-0-941)
- NTejas-1 GitHub: two ResNet50 arms (gold 0.8955 and 0.8892, 50/50 blend 0.9078) took the public 0.940 reference only to 0.941. Both arms used RadImageNet ResNet50, which was "already in ensemble". The lesson they drew: measure correlation against the *baseline*, not between your own arms. Label "repair" dropped Synovitis from 0.840 to 0.802 — [NTejas-1 repo](https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection)
- aditbytes GitHub: a weak reader (v1 alone 0.860) blended at 0.2 into a 0.944 stack *dropped* it to 0.937 — [aditbytes repo](https://github.com/aditbytes/kaggle-rsna-knee-abnormality)
- Gold-58 vs report labels: the 58 were annotated from images, not reports. Report-reader ceilings are around 0.88–0.90 on gold, and Synovitis is under-reported — [sadamtorres](https://www.kaggle.com/code/sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee); [goodpjw2008](https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944); [745861 comment, 2026-10-07](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745861) ("final set of labels was done completely blind to the report", unverified participant claim)

### Inferences
- Across these accounts, 0.95+ comes from training your own model end to end on all 4,349 report-labelled studies, with soft labels and good geometry. Blending public checkpoints does not get there. For pseudo-labels, the out-of-fold / teacher-student distinction matters.
- The 0.943 stack is a runtime sink (6.5–8 h). The top teams' fast runtimes (5–30 min) leave room for 5-fold and multi-architecture ensembles of their own strong models. That is probably how 0.955 single models become 0.96+ (inference, consistent with prateekgroverin's 2026-09-21 forecast in [742327](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742327)).

### Gaps
- No top-10 team has disclosed its full recipe. yosukeyama says the team will share after the deadline ([743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)).
- I found no public per-finding breakdown from a 0.955+ model.

---

## Q3. Pitfalls: decode failures, leakage, site effects, gold-copy label tables, public-LB overfitting

### Takeaway
The pitfalls that hurt most are silent ones: wrong slice ordering, bf16 on T4, missing attached sources, time-guard fallbacks to constants, DataParallel RAM OOM, and decode wheels. Selection on gold-58 and per-label LB-tuned routing are the main overfitting risks. "Single model" scores that use pseudo-labels can hide ensembling.

### Cited Findings
- **Slice order.** A shared baseline sorted slices by filename (SOP UID), which is effectively random: median |Spearman ρ| was 0.164 against ImagePositionPatient order. Fixing it gave +0.0276 on one fold, about 25× seed noise. Cache the IPP sort per series; scanning all headers in probe() cost about 7 min per run (2026-10-04) — [745759](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745759)
- **bf16 on T4.** A model worth about 0.851 scored 0.710 because emulated bf16 made each pass about 50× slower. The time guard then wrote a constant 0.5 for more than half the hidden studies. Use fp16 and check `torch.cuda.is_bf16_supported(including_emulation=False)` (2026-09-29) — [744230](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744230)
- **Silent member drops.** The public stack with missing `model_sources`/`kernel_sources` ran to completion with all 20 DINOv2 members dropped and scored 0.937 instead of 0.940 — [742050](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742050)
- **OOM and hangs.** DataParallel caused a system-RAM OOM on the full hidden test, so the fix is sequential arms or one replica per GPU. A recursive glob over the competition DICOM tree hung a commit for more than 20 minutes (2026-09-26) — [Dread notebook](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model); [Apex source comments](https://www.kaggle.com/code/sujanmajhisuzan/rsna-knee-apex-grandmaster-stack)
- **Decoding.** Bring pylibjpeg/gdcm wheels as insurance, because "a study that fails to decode is a study scored at prevalence". Sampled train files were uncompressed Explicit VR LE — [sadamtorres](https://www.kaggle.com/code/sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee). kozykappa runs a pixel-data smoke test and drops the optional reader if decoding fails. It also treats a constant output column on the hidden set as a sign of failed decoding — [kozykappa](https://www.kaggle.com/code/kozykappa/rsna-knee-gold-gated-triple-reader)
- **Hidden-test composition.** The hidden test has about 1,300 studies (1,322 per kishormorol). The visible test has only 3, so failures appear only at scale — [744230](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744230). Whether every hidden study has all planes and a fluid-sensitive series is still **unanswered by the host** as of 2026-10-06 — [744519](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744519)
- **Laterality and flips.** "No horizontal flip augmentation: forbidden — would silently swap medial/lateral labels" — [NTejas-1](https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection). Dread uses no flips in training. nartaa found that an all-plane width flip TTA hurt (0.940→0.939), while a proper *anatomical* mirror helped (+0.001) — [746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792)
- **Gold-58 overfitting.**
  - caoyupeng: one label version gave gold 0.945 on EffNet but LB 0.910 [745214](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745214).
  - nartaa: epoch 14 improved gold but lost on the LB, and a crop-trained model with "encouraging gold diagnostics" lost 0.949→0.946 [746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792).
  - Dread: a +0.010 gold blend gain was "gold-set noise" [Dread notebook](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model).
  - Per-class gold has only 9–12 positives for some findings [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304).
- **Public-LB overfitting.**
  - The per-label map (LatMen = 1.00) was tuned on the public split; starkhushi advises keeping a flat-0.60 version as the second final pick [starkhushi](https://www.kaggle.com/code/starkhushi/rsna-knee-why-public-forks-stop-at-0-941).
  - returnofsputnik (GM): "The public notebook is overfit blend scoring 0.945" [742327](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742327).
  - goodpjw: stack constants were tuned on the public LB and gold-58, so "expect the two parts to behave differently on the private leaderboard" [goodpjw2008](https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944).
- **Local CV vs LB direction.** OOF on extracted labels understates real gains: Stage A→B moved OOF +0.005 and LB +0.017 — [sadamtorres](https://www.kaggle.com/code/sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee). Others see LB about 0.02 above extracted-label CV — [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)
- **Validating on repaired labels.** "You certainly can't validate against labels you repair or your CV will improve by definition" — Tucker Arrants, [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)
- **Label table quality.** truenikita reports that their Claude Haiku/Sonnet re-extraction matches public "mm3" labels on gold macro-AUC (±0.01) but is far better calibrated. Calibration matters for soft targets — [743374](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743374)

### Inferences
- "Gold-copy label tables": every public label table I checked is described as LLM- or report-derived. nartaa and goodpjw both exclude gold-58 from training. Any table or checkpoint trained on "alldata" could still include the 58, which would make gold evaluation of that model optimistic. Examples are nartaa's checkpoint name `raptor_ft_alldata_...` (the author says gold was excluded) and Dread's "full labelled set with a small fully labelled holdout". Treat gold numbers on public checkpoints as contaminated unless the author excludes the 58 explicitly.
- Site effects: the data covers 16 sites and 12 languages. Per-language lexicon coverage is very uneven ("one language is covered far better than the other eight") — [pilkwang baseline](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1). Grouped folds by report text (goodpjw) are the only site-aware CV I saw.

### Gaps
- I found no public analysis of per-site or per-vendor performance, or of site-stratified CV.
- I found no confirmed instance of a public label table copying the gold-58 labels. The concern stays plausible but unverified.

---

## Q4. External data, efficiency track and team merging

### Takeaway
External data is a live grey area. The host allows click-through non-commercial datasets but not institution-gated ones. On OAI the host gave a conditional answer (2026-09-21), and participants still dispute it. KneeCoT is explicitly banned. The public 0.949 frontier depends on OAI, so anything forked from it inherits that rule risk. The efficiency board is led by 0.959–0.962 public scores. Several 0.94–0.95 solo competitors are advertising merges ahead of an Oct 15 merger deadline (date cited by a participant).

### Cited Findings
- **Host on LLMs (2026-08-09):** commercial LLM APIs may be used on report text and do not count as prohibited private sharing, provided the service is reasonably accessible and of minimal cost — [733965](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965)
- **Host on non-commercial datasets (2026-08-27):** datasets accessible "through a straightforward registration or click-through data-use agreement would generally satisfy" accessibility. Those requiring "institution-specific approvals, negotiated written agreements, IRB approval, lengthy credentialing" may not. Licence compliance is each team's responsibility — [733965](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965)
- **Host on OAI (2026-09-21):** OAI "does not violate the… rules if it is generally accessible to all participants without an escalated (i.e. institutional) review and sign-off… and is free of charge". Tucker Arrants read this as "a no, as it requires institutional sign off"; Optimo: "It's difficult to be less clear" — [741819](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819)
- **KneeCoT** was ruled out because it needs a formal agreement with the hospital — [discussion 734109, as cited in 743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)
- **OAI dispute on the nartaa thread (2026-10-07/08):**
  - nartaa says OAI's own terms permit the use and that the work was registered as an NDA study.
  - Others report that login.gov registration needs only an email, and one shared a UMass reply confirming the use.
  - A participant in China reports failing to download OAI, and NIH bars some countries.
  - [746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792)
- **Open questions in 743416 (2026-10-08, unanswered):** whether OAI is allowed, and whether **public checkpoints trained partly on OAI** are allowed for prize-eligible submissions. bloodaxe recalled DeepFake Detection, where the top two teams were removed for disallowed external data — [743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)
- Other disputed datasets: MRNet (no-derivatives clause), SKM-TEA, fastMRI/fastMRI+, KneeMRI (CC BY-NC-ND) and KneeXNet-2.5D. KJD-25K (MICCAI 2026) was also mentioned — [743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416); [745861](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745861)
- **Measured effect of external data:** OAI with masked labels gave +0.005 (0.942→0.945), and the follow-on 384 full-field recipe reached 0.948 (not compute-matched) — [746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792)
- **Efficiency LB** (Kaggle staff notebook, output 2026-10-07): 5,290 teams listed. Rank 1 Scott Willis (public 0.959), 2 Pa3฿aJluHа (0.962), 3 Claudio Verdú Ruiz (0.957), 4 Wassim Dobbi (0.958), 7 Optimo (0.958), 10 TomKroumov (0.957), 12 tomoon33 (0.956), 20 Danial Zakaria/nartaa (0.945), 22 joinT fusion (0.961). 53 entries have public ≥ 0.955 — [efficiency LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb)
- **Speed tricks** (prateekgroverin, CoAtNet inference from about 40 min to under 10, 2026-09-25):
  - one model copy per GPU instead of DataParallel;
  - a single threaded DICOM pass (59 to 372 MB/s with 16 threads);
  - raw header parsing (0.307 to 0.058 ms/file) and `np.frombuffer` pixels (1.22 to 0.33 ms/slice);
  - strided percentile computation;
  - fp16 weights without autocast (2.08×);
  - uint8 to GPU with on-device normalisation.
  - Source: [743374](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743374)
- **Merging:**
  - kishormorol mentions "the 15 Oct merge deadline" and offers an independent DINOv2 ViT-S arm [744230](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744230).
  - charlessavas (0.943, DINOv2-MIL heavy) proposes checking prediction correlation before merging. Their one external arm (corr 0.83) was worth +0.004 [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304).
  - kmatsu01 (0.940 single CoAtNet), miyamotodaiya (0.955 single, seeking 0.950+ partners) and a "0.942 solo, own 5-fold models + GPU" poster ([745091](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745091)) are open to merging.
- **Final ceiling guesses:** cdeotte (2026-09-26): "Gold threshold is probably 0.96 and first place is probably 0.975" — [735767](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735767)

### Inferences
- A team aiming for prizes should treat OAI-derived checkpoints (nartaa's) as a disqualification risk until the host answers 743416. For a medal-only goal the risk is lower, but it is not zero.
- Merging with an 0.95+ single-model holder whose predictions correlate weakly with yours is the cheapest path to a big jump. Do this before Oct 15 (unverified date; check the Rules tab).

### Gaps
- I did not confirm the exact team-merger deadline from the rules page.
- I could not find the efficiency score formula (the linked evaluation page is JS-rendered).

---

## Q5. Concrete actions for a team at 0.943 aiming for 0.961

### Takeaway
Stop tuning the public stack. Build or merge into **your own strong single-model family** (target ≥0.950 alone) using soft labels from several LLM extractions and OOF-teacher pseudo-label mixing. Get the geometry right (IPP slice order, 140–160 mm FOV, 0.4 mm/px, 80–96 slices, dense windows, anatomical-mirror TTA). Then ensemble 2–3 *decorrelated* architectures by rank, with the folds as members. Choose final submissions with flat weights rather than per-label LB-tuned routing.

### Cited Findings
- Single models at 0.950–0.955 exist with ResNet50 at 224, CoAtNet at 224/288 and Qwen3.5-2B VLM fine-tunes — [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)
- The public stack costs 6.5–8 h of the 9 h budget — [goodpjw2008](https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944)
- Weak members (below about 0.90 solo) hurt the stack — [starkhushi](https://www.kaggle.com/code/starkhushi/rsna-knee-why-public-forks-stop-at-0-941); [aditbytes](https://github.com/aditbytes/kaggle-rsna-knee-abnormality)
- Independent 0.929 reader + 0.943 stack gives only 0.944 — [goodpjw2008](https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944)
- nartaa recipe details (OAI masked labels, 384 training / 320 crop inference, mirror TTA, 94 windows, top-3 epoch weight averaging) — [746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792)

### Inferences (action list, ordered by expected value)
1. **Audit your pipeline for silent failures first** (1 day):
   - IPP-sorted slices;
   - fp16, not bf16;
   - every source attached and logged ("dropped" count = 0);
   - a fallback counter on the hidden set;
   - a constant-column check;
   - decode wheels;
   - run your exact inference path on held-out *train* studies and score them against gold.
   See [745759](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745759), [744230](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744230), [742050](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742050).
2. **Build labels as soft targets from several sources.** Average at least 3–4 LLM tables (public stevenleehans, pilkwang, lixin73, laymond, JEV, plus your own Qwen or Gemma pass); goodpjw's averaging is the template. Use graded severity rather than term-present → 1 ([pilkwang](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)). Keep the soft values; do not binarise.
3. **Teacher→student round.** Train 5-fold teachers and mix each study's *OOF* prediction into its label, 30–60% toward the model, especially for Synovitis, PF OA and LatOA, where reports are silent. Retrain the student on all data. Validate against original extracted labels and the LB, never on the repaired labels (architkonde, laymond, Tucker in [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)).
4. **Geometry and model:**
   - CoAtNet-RMLP-2 or ConvNeXt 2.5D MIL with per-finding attention pooling;
   - 140–154 mm physical crop at 0.4 mm/px, 5–6 series slots (plane × fat-sat), 80–96 slices over a wide 2–98% span;
   - train at 288–384, evaluate with dense windows (≥62, up to 94);
   - average the top-3 epochs;
   - anatomical-mirror TTA only (sagittal slice reversal plus coronal/axial width flip), never a naive all-plane flip.
5. **Diversity.** Add a second family that does not share the CoAtNet inductive bias, such as ResNet50/EffNet at 224 (fast, 0.95+ reported) or a DINOv2 fine-tune. Measure correlation **against your main model**, not between candidates (NTejas-1 lesson), and rank-average with at most a few global weights.
6. **External data decision (team call):** OAI masked labels were worth about +0.005 for nartaa, but rule status is unresolved for prize eligibility. If you avoid it, do not fork nartaa's checkpoint either ([743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)).
7. **Use runtime for members, not legacy stack.** Drop the 0.943 stack, or keep it as a small-weight leg only if you have runtime left, and use the 9 h for your own 5-fold × 2–3 architectures. Apply the speed tricks from [743374](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743374).
8. **Merge before the deadline** (reportedly Oct 15) with a team holding an independent ≥0.950 single model. Agree on a correlation test on shared OOF or gold predictions first, as charlessavas proposed ([735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)).
9. **Final selection.** One submission should be the best LB with flat or global weights. The second should be the most CV-robust (no per-label routing tuned on public). Treat ±0.001 public moves as noise ([742050](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742050); [746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792)).

### Gaps
- No public source confirms that any of these steps alone reaches 0.961. The 0.96+ teams have not disclosed their recipes. The ordering above is my synthesis, not a measured ablation.
