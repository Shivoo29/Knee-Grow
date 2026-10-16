# Multi-scale inputs and error-decorrelation for medical image ensembles

## Single multi-scale model vs separate global/local models + late fusion: which wins, by how much?

### Takeaway
The measured evidence consistently favors multi-view/multi-scale fusion over the best single view/scale by ~+1 point AUC, and favors joint/weighted feature (early) fusion over naive label averaging — but no located study directly compares a single joint global+local-crop network against separate global/local models with late fusion on knee MRI, so that head-to-head is a gap.

### Cited Findings
- MRNet (Bien et al., peer-reviewed, PLoS Medicine 2018): per-exam logistic-regression fusion of 3 series-specific AlexNet+maxpool models achieved validation AUCs of 0.937 (abnormality), 0.965 (ACL), 0.847 (meniscus) on 120-exam internal validation set — [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/)
- MRNet architecture detail: one network per (task × series) = 9 MRNets (sagittal T2, coronal T1, axial PD); slice features via AlexNet (ImageNet-pretrained) + global average pool + max-pool across slices + FC + sigmoid; exam-level output via logistic regression over the 3 series probabilities; axial PD most beneficial for abnormality/meniscus, coronal T1 for ACL — [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/)
- Li et al. (peer-reviewed, Front. Bioeng. Biotechnol. 2023): 3D weighted multi-view CNN (shared-weight 3D ResNet encoder + MLP/softmax view-weighting + feature concatenation + 2-layer MLP) achieved ACL AUC 97.00% on in-house MRI-ACL (100 cases, 4 views) and 92.86% on MRNet-v1.0 (1,250 cases, 3 views) — [Source](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2023.1268543/full)
- Li et al. single-view ablation: best single view AUC 96.00% (MRI-ACL) and 91.78% (MRNet-v1.0), so multi-view gain was +1.00 and +1.08 points respectively; AUC rises monotonically with number of views — [Source](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2023.1268543/full)
- Li et al. vs baselines (same LR, image size): Ours beat 3D-VGG16 by +3.00/+2.61, 3D-ResNet-10 by +1.00/+1.08, MVCNN by +2.00/+0.92, and MRNet by +1.00/+0.28 on (MRI-ACL / MRNet-v1.0) — [Source](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2023.1268543/full)
- Li et al. fusion ablation: feature fusion beat label fusion (mean and probability-weighted averaging); weighted concatenation beat plain concatenation; their weighted feature fusion was best at 97.00%/92.86% — [Source](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2023.1268543/full)
- Harman (peer-reviewed, Journal of AI 2026): on MRNet, best single-plane ACL AUC was 0.976 (sagittal), while three-plane Hybrid Fusion scored 0.949; meniscus AUCs 0.791 (sagittal) and 0.826 (axial); i.e. naive 3-plane fusion did NOT beat the best single plane for ACL in that setup — [Source](https://dergipark.org.tr/en/pub/jai/article/1926635)
- Harman: Mamba-based architecture showed lower mean AUC and higher variance than CNNs for plane-specific knee MRI, attributed to localized findings in adjacent slices; CNNs more stable in limited-data setting — [Source](https://dergipark.org.tr/en/pub/jai/article/1926635)
- Deng et al. CS-MIL (peer-reviewed, Med. Image Anal. 2024): single joint cross-scale attention MIL beat all late/naive multi-scale fusions — Crohn's disease AUC 0.8774 vs mean-vector 0.8147, concatenation (DS-MIL) 0.8470, MS-DA-MIL additive 0.8565, HAG-MIL 0.8374; TCGA-GBMLGG AUC 0.7737 vs mean-vector 0.7425, concatenation 0.7362 — [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC11016375/)
- Deng et al.: single-scale DeepAttnMISL models scored 0.7775 (20×), 0.7900 (10×), 0.8288 (5×) on Crohn's — so multi-scale CS-MIL gain over best single scale was +4.86 points AUC (0.8774 vs 0.8288); AP 0.8486 vs best single-scale 0.7314 — [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC11016375/)
- Deng et al. complexity: CS-MIL added only ~2k params over single-scale DeepAttnMISL (137,475 vs 135,362) and used slightly less GPU memory (1493 vs 1513 MiB); MS-DA-MIL 135,362/1601 MiB; HAG-MIL 406,086/1397 MiB — [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC11016375/)
- Fusion-strategy review (Alshohoumi et al., Applied Sciences 2026, citing Steyaert et al. 2022): late fusion reported best at AUROC 0.947, outperforming single modality — primary source not verified, treat as second-hand — [Source](https://www.mdpi.com/2076-3417/16/4/1964)
- Liu et al. 2019 two-stage ACL pipeline (localize ACL with 2 CNNs, then classify): reported AUC 98%, above Li et al.'s 97.00% — but requires radiologist box annotations for localization, dependent on localization quality, higher train cost — as summarized in Li et al. discussion; original Liu paper not directly fetched — [Source](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2023.1268543/full)
- Astuto et al. 2021 (Radiology: AI): single-view 3D CNN for knee abnormality grading achieved ACL AUC 90.00%, below multi-view MRNet-family results — as summarized in Li et al. discussion; original not directly fetched — [Source](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2023.1268543/full)

### Inferences
- For knee MRI, expect multi-view fusion to add roughly +0.3 to +1.1 AUC points over the best single view when fusion is learned/weighted; plain concatenation or averaging can erase the gain (Harman 3-plane 0.949 < single-plane 0.976 is a cautionary example).
- Early/weighted feature fusion (cross-scale attention, view-weighting) beats late label averaging in both knee MRI (Li) and pathology MIL (Deng) with only marginal extra parameters — so a single multi-scale model with learned per-crop/per-view weights is the better default than separate models + mean blend.
- The 140 mm global + 100 mm knee-anchored close-up design maps directly onto the CS-MIL pattern: shared encoder + cross-scale attention over co-centered global/local instances + clustering/bagging; Deng's +4.9/+3.0 point gains over single scale are the closest measured analogue, though on pathology not knee MRI.

### Gaps
- No located paper directly compares ONE joint global+local-crop network vs SEPARATE global-only and local-only models fused late on the same knee-MRI dataset with the same backbone — the key head-to-head number is missing.
- No located measurement of 140 mm vs 100 mm (or any mm-anchored) close-up crop gain on knee MRI; all cited crop/localization gains use view-count or keypoint-ROI ablations, not physical-size ablations.
- Per-finding (12-label) breakdown of global-vs-local value (e.g. ACL needs global, cartilage needs close-up) not found with numbers.

## Resolution/crop/view changes as decorrelation levers: measured correlation-vs-gain pairs

### Takeaway
Resolution/view/backbone diversity is widely asserted as a decorrelation lever, but no located medical-imaging study reports a within-class prediction-correlation coefficient alongside its ensemble gain; the only hard resolution-blend numbers found are classification accuracies, not correlations and not knee MRI.

### Cited Findings
- MedRxiv 2025 preprint snippet: for multi-resolution ensembles (224×224, 384×384), images were processed at each model's trained resolution independently and results combined — full text blocked (403), so no gain number verified — [Source](https://www.medrxiv.org/content/10.1101/2025.10.25.25338784v1.full-text)
- Domínguez et al. (peer-reviewed, Comp. Meth. Prog. Biomed. 2023): multi-resolution ensemble PR-224-384-512 was "more beneficial in the multi-class task" for retinal images; 384×384 inputs useful with multiple classes — snippet only, exact gain numbers not extracted — [Source](https://www.sciencedirect.com/science/article/pii/S0169260722006836)
- Tummala et al. (peer-reviewed, 2022, brain tumor MRI): ensembling models at 224×224 reached 97.71% overall accuracy; 384×384 ensemble accuracy reported separately — snippet truncated, cross-resolution blend gain not verified — [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC9600395/)
- Diversity-promoting ensembles for medical segmentation (Xu et al., arXiv 2022): proposes leveraging architectural diversity to generate ensembles — conceptual support for backbone-diversity decorrelation, no knee-MRI correlation-vs-gain pair extracted — [Source](https://arxiv.org/html/2210.12388v2)
- General ensemble-diversity principle: combining diverse predictors reduces error only with prediction variety; weak-to-modest positive prediction correlation is typical — textbook-level claim, MongoDB/TowardsDataScience-level sources only, no medical-imaging measurement — [Source](https://towardsdatascience.com/understanding-the-importance-of-diversity-in-ensemble-learning-34fb58fd2ed0/)
- Indirect decorrelation evidence (knee): MRNet logistic-regression view weights selected different "most beneficial" series per task (axial PD for abnormality/meniscus, coronal T1 for ACL), implying views carry non-redundant signal per finding — [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/)
- Indirect decorrelation evidence (pathology toy sets, Deng et al.): micro-anomaly (20×-only white cross) vs macro-anomaly (5× ellipse) — single-scale models each collapsed on the wrong scale (e.g. 20× model AUC 1.0000 micro but 0.7374 macro; 10× model 0.4898 micro but 1.0000 macro), while CS-MIL held 0.9627/0.9982 — demonstrates scale-specific error patterns, but authors report no correlation coefficient — [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC11016375/)

### Inferences
- Resolution (224 vs 384), crop (global 140 mm vs anchored 100 mm close-ups), view (sagittal/coronal/axial; patellar/posterior/medial/lateral anchors), backbone (CNN vs transformer), and label-set (12-findings vs ACL-only) are all plausible decorrelation levers by analogy to the view/task-specificity above — but the correlation-vs-gain table requested cannot be filled from located sources.
- The cheapest decorrelation test available in 2–3 days (UNVERIFIED estimate): train global-only and close-up-only models, compute per-finding Pearson/Spearman of out-of-fold predictions plus pairwise conditional accuracy, then greedy forward blend by macro AUC — no literature number predicts its outcome.

### Gaps
- Zero located (correlation, gain) pairs of the form "224px+384px models correlate at r=X, blend gains +Y AUC" — neither for knee MRI nor any medical image task searched.
- No located within-class prediction correlations between any two knee-MRI models differing only in resolution, crop, view, backbone, or label set.
- Backbone-diversity (e.g. ConvNeXt vs ViT vs Mamba) ensemble gain on knee MRI not found with numbers; Harman only reports single-model Mamba underperformance, not ensemble contribution.

## Kaggle RSNA knee 2026 specifics: close-up views, anchor localisation, per-finding blend weights

### Takeaway
The 2026 RSNA Knee Abnormality Detection competition (12 abnormalities, macro-AUC-style evaluation per public discussion) is confirmed to exist, but none of the three named forum threads (735304, 746792, 747658) nor any public close-up/anchor/blend-weight report could be verified from fetched sources — all competition-specific claims below are competition-anecdote grade at best.

### Cited Findings
- Competition exists: "RSNA Knee Abnormality Detection" on Kaggle, tasking models to detect a defined set of clinically important abnormalities on knee MRI — [Source](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)
- RSNA announcement: 2026 challenge to develop models detecting/classifying knee abnormalities on MRI — [Source](https://www.rsna.org/news/2026/august/ai-challenge-knee-mri)
- Forum activity confirmed: discussion threads exist (e.g. thread 733343 defining the challenge; thread 742926 reporting a 0.940 public AUC data point and asking for next experiments) — [Source](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343); [Source](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742926)
- Prior-RSNA analogue, lumbar spine 2024 (competition anecdote): 4th-place "spine-chart" solution detected keypoints then classified surrounding crops at each level — [Source](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/spine-chart-4th-place-solution) (fetch returned empty body; title/method confirmed via search index)
- Lumbar 2024 codebases: hengck23 one-stage training repo and lhwcv two-stage (localize-then-classify) repo document the keypoint/crop-then-classify pattern used in that competition — [Source](https://github.com/hengck23/solution-rsna-2024-lumbar-spine); [Source](https://github.com/lhwcv/solution-rsna-2024-lumbar-spine)
- Lumbar dataset scale: 2,600+ lumbar MR scans — [Source](https://registry.opendata.aws/rsna-lumbar-spine-degenerative-classification-dataset/)
- Goodpjw2008 / nartaa / heliosli 2026 knee notebooks: named in the assignment but NOT located or verified in any search performed — no link, no number.

### Inferences
- The lumbar-2024 keypoint→crop→classify pattern is the closest public RSNA precedent for the proposed 140 mm/100 mm knee-anchored pipeline, but it is an analogy, not evidence that close-ups help knee findings.
- Any claim about discussions 735304/746792/747658 contents, per-finding blend weights, or anchor-localisation methods in the 2026 knee competition is UNVERIFIED and should not be cited in the final report.

### Gaps
- Could not verify contents of Kaggle discussions 735304, 746792, 747658 (login-gated; fetches fail) — no close-up, anchor, or blend-weight numbers obtained.
- No public 2026 knee solution/notebook with measured close-up gain located.
- Official 2026 metric (macro AUC over 12 findings?) and test size (~1,300 studies) taken from assignment brief, not independently verified — confirm against competition overview/rules pages before citing.

## Inference cost of multi-crop models under a 9 h 2xT4 budget for ~1,300 studies

### Takeaway
No located source measures multi-crop knee-MRI inference time on 2×T4 against a 9-hour limit; arithmetic from public baselines suggests the budget is generous for a 2.5D MIL model but the multi-crop multiplier and DICOM loading are the risks — all timing conclusions are UNVERIFIED estimates.

### Cited Findings
- MRNet training cost anchor (peer-reviewed, 2018 hardware): ~6 hours per MRNet on a single GTX 1070 8GB for 50 passes over 1,130 exams — inference cost not reported — [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/)
- CS-MIL inference-side complexity: 137k MIL-head params, 1493 MiB GPU — backbone (ResNet-50 SimSiam) dominates; MIL head itself is negligible — [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC11016375/)
- Kaggle notebook environment: 2× Tesla T4 GPUs available; CPU+GPU sessions allow up to ~12 h execution (notebook docs) — the "9 h" figure itself is UNVERIFIED against 2026 knee rules — [Source](https://www.kaggle.com/docs/notebooks); [Source](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/391055)
- 2.5D cost pattern: hybrid 2.5D (center + adjacent slices, e.g. 5-slice stacks) reuses 2D backbones with stacked-channel inputs — per-volume cost scales ~linearly in number of sampled slice-stacks, not cubically like 3D — method pattern from Jin et al. 2025 (glioma hybrid 2.5D, center+adjacent slices) — [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC12596209/)
- 2.5D slice-count sensitivity: supplementary analysis in a multiparametric-MRI ensemble study found 3- vs 5-slice 2.5D inputs "comparable" — snippet only, numbers not extracted — [Source](https://www.academicradiology.org/article/S1076-6332(26)00602-1/pdf)
- Knee data volume anchor: MRNet series have 17–61 slices (mean ~31) at 256×256 — [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/)

### Inferences
- UNVERIFIED back-of-envelope (do not cite as fact): 1,300 studies × 3 series × ~31 slices ≈ 120k slices; a 2.5D model sampling ~16 stacks/series × 5 crops (1 global + 4 close-ups) ≈ 300 backbone forwards/study ≈ 400k forwards total; at ~5–15 ms/forward on T4 (FP16, 224–384px) that is ~0.5–1.7 h single-GPU compute plus I/O — plausibly inside 9 h on 2×T4, but DICOM decode, resampling to mm-space, and test-time augmentation are unmodeled and routinely dominate.
- Cost-reduction levers if over budget (UNVERIFIED, standard practice): single global pass to select top-K slices/anchors then close-up only there; FP16 + batch inference; 224px global + 384px close-up only; distill ensemble into one student; cache preprocessed arrays.

### Gaps
- No measured seconds-per-study for any 140 mm/100 mm multi-crop 2.5D attention-MIL pipeline on T4 — the requested budget check cannot be answered with a cited number.
- 9 h limit and ~1,300-study test size both UNVERIFIED from assignment brief; verify against official 2026 knee competition rules.
- No located T4 throughput benchmark for the specific backbones under consideration at 224 vs 384px on knee MRI.

## Appendix — 2–3-day implementation sketch (UNVERIFIED proposal, no literature source)

### Takeaway
This sketch is a proposal synthesized from the cited patterns above, not a cited result; every gain/timing claim in it is UNVERIFIED.

### Cited Findings
- No cited finding; sketch only. Component precedents are cited in sections above (Li weighted fusion; Deng CS-MIL attention + clustering; MRNet per-series models + LR stack; lumbar keypoint→crop pattern).

### Inferences
- Day 1 (UNVERIFIED plan): mm-space resampling + knee-center heuristic (or lumbar-style keypoint net if time); build 140 mm global + 4× 100 mm anchored close-up crops (patellar/posterior/medial/lateral) from sagittal/coronal/axial; 2.5D 5-slice stacks (center ±2); train global-only baseline, log per-finding macro AUC.
- Day 2 (UNVERIFIED plan): shared-encoder attention-MIL over (global + 4 close-ups) instances with learned cross-scale weights (Deng pattern); compare against separate global/local models + late mean/logistic blend (Li pattern); record per-finding prediction correlations and greedy blend weights.
- Day 3 (UNVERIFIED plan): FP16 batched inference rehearsal on a 100-study subset extrapolated to 1,300; keep top-K slice/anchor shortlist if over budget; freeze per-finding blend weights by out-of-fold macro AUC.

### Gaps
- Whether 100 mm anchored crops beat full-slice inputs per finding; anchor definitions; optimal K; 224 vs 384 operating point — all to be measured, none predicted by literature found here.
