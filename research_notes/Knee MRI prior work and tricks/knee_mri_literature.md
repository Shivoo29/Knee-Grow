# Knee MRI abnormality detection: published deep-learning literature and what transfers

Context: RSNA Knee (2026). 4,407 train studies, about 5.5 series each, multi-site. Labels come from free-text reports, plus 58 expert-labelled studies. The metric is macro AUC over 12 findings. The team's 2.5D CoAtNet attention-MIL model scores 0.8925 on gold-58 and 0.943 on the public LB.
Legend: **[V]** = I checked the number in the primary full text or abstract. **[S]** = the number comes from a search snippet or a secondary summary and I did not check the primary table. **[U]** = unverified, or from memory.

---

## Q1. MRNet and its follow-ups: what improved AUC, and by how much?

### Takeaway
On MRNet (1,370 exams, 3 planes) the gains came from ImageNet pretraining, a residual backbone and tuned augmentation (+~0.04 average AUC). Fusing per-plane models with logistic regression added +0.02–0.03 over the best single plane. Pooling/attention over small features (FPN, spatial attention) helped mainly the meniscus. Most follow-up gains are 0.005–0.06 on a 120-exam validation set, so they are inside the noise.

### Cited Findings
**MRNet (Bien et al., PLOS Med 2018).**
- 1,370 exams.
- Each series is a 2D AlexNet slice encoder with max-pool over slices. One model per plane and task, combined by logistic regression.
- Test AUC: abnormal 0.937, ACL 0.965, meniscus 0.847 [S] — [PMC6258509](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6258509/), [summary](https://astro.paperswithcode.com/paper/deep-learning-assisted-diagnosis-for-knee)
- External KneeMRI (Croatia) ACL AUC 0.911 [V, as quoted in ELNet] — [ELNet PDF](https://arxiv.org/pdf/2005.02706)
- Sequences: sagittal T2, coronal T1, axial PD [U, from memory] — [PMC6258509](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6258509/). This matters because MRNet's per-plane conclusions are confounded with sequence: coronal is T1, which is not fluid-sensitive.

**ELNet (Tsai et al., MIDL 2020).**
- Lightweight CNN (~0.2M params, <1 MB), trained from scratch. Single series per task: coronal for meniscus, axial for ACL and abnormal.
- On the MRNet validation set, MRNet → ELNet: meniscus 0.826→0.904, ACL 0.956→0.960, abnormal 0.936→0.941 [V] — [arXiv 2005.02706](https://arxiv.org/pdf/2005.02706)
- KneeMRI ACL: 5-fold mean 0.913 (best fold 0.924) [V] — [ELNet PDF](https://arxiv.org/pdf/2005.02706)
- Ablation, meniscus/ACL/abnormal AUC:

  | Normalization | Pooling | Meniscus | ACL | Abnormal |
  |---|---|---|---|---|
  | Multi-slice norm | BlurPool | 0.904 | 0.960 | 0.941 |
  | BatchNorm | BlurPool | 0.751 | 0.871 | 0.841 |
  | Multi-slice norm | MaxPool | 0.848 | 0.923 | 0.943 |
  | BatchNorm | MaxPool | 0.797 | 0.906 | 0.880 |

  [V] — [arXiv 2005.02706](https://arxiv.org/pdf/2005.02706)
- Weights were chosen by "best random seed 0–4", so the results include seed selection on validation [V].

**Azcona et al. 2020 (MRNet).**
- ResNet18, ImageNet-pretrained, one model per plane and task. Slices interpolated to a fixed count of 15. Logistic-regression fusion.
- Per-plane validation AUC [V] — [arXiv 2010.01947](https://arxiv.org/pdf/2010.01947):

  | Task | Axial | Coronal | Sagittal | Fused |
  |---|---|---|---|---|
  | ACL | 0.934 | 0.926 | 0.897 | 0.956 |
  | Meniscus | 0.882 | 0.852 | 0.781 | 0.908 |
  | Abnormal | 0.949 | 0.913 | 0.950 | 0.938 |

- Ablations [V]:
  - Same per-plane design trained from scratch: average 0.898, vs 0.934 pretrained (−0.036). Meniscus dropped most (0.908→0.821).
  - Single from-scratch network on 45 stacked slices from all planes: 0.858.
  - Single multi-task from-scratch network: 0.828.
  - Authors' lessons: transfer learning regularizes, tuned augmentation matters, residual backbones help [V].

**Optimising with spatial attention + MPFuseNet (2021).**
- Pretrained multi-view CNN with a spatial attention block and a learned multi-plane fusion network.
- ACL AUC 0.977, abnormal 0.957 [S]. I did not retrieve the meniscus AUC or the ablation table (the PDF was too large to fetch) — [arXiv 2108.08136](https://arxiv.org/abs/2108.08136)

**MRPyrNet (Dunnhofer et al., MIDL 2021 / CMIG 2022).**
- Adds a Feature Pyramid Network plus "Pyramidal Detail Pooling" on top of MRNet and ELNet. The motivation is that lesions are small and in predictable locations.
- The authors claim gains for both ACL and meniscus on MRNet and KneeMRI [S]. I could not retrieve the exact deltas (PDF 403) — [PMLR](https://proceedings.mlr.press/v143/dunnhofer21a.html), [GitHub](https://github.com/matteo-dunnhofer/MRPyrNet)

**Other MRNet-scale results.**
- SB-SSL: slice-based self-supervised transformer, <1,000 cases, no external pretraining data. ACL AUC 0.954, accuracy 89.2% [S] — [arXiv 2208.13923](https://arxiv.org/abs/2208.13923)
- KneeXNet (graph conv + contrastive learning) reports abnormal 0.985, ACL 0.972, meniscus 0.968 on MRNet [S]. **Treat as suspect**: a jump of +0.12 on meniscus over every other method, no external test, and the split was not verified — [Frontiers 2025](https://public-pages-files-2025.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2025.1590962/text)
- Tran et al., Eur Radiol 2022: 19,765 scans (Incepto/Keros lineage). ACL AUC 0.939 internally. After retraining on external sets, MRNet 0.962 and KneeMRI 0.922 [S] — [EurekaMag abstract](https://eurekamag.com/research/080/453/080453529.php), [Springer](https://link.springer.com/10.1007/s00330-022-08923-z)

### Inferences
- The MRNet meniscus task is the hardest. It responds most to pretraining (+0.09), plane fusion (+0.03) and small-feature pooling.
- ACL is near ceiling (~0.95–0.97) at every scale, including with 15× more data (Tran). The team's ACL of 0.933 on gold-58 is consistent with that.
- The ELNet BatchNorm ablation applies to tiny from-scratch nets with batch=1 volumes. It is mostly irrelevant to a pretrained CoAtNet, except as a warning that normalization over a variable-length slice bag matters.

### Gaps
- Exact MRPyrNet deltas and the attention/MPFuseNet ablations (PDF access blocked).
- MRNet's sequence-per-plane assignment is from memory [U].

---

## Q2. Multi-abnormality knee MRI models: datasets, inputs, architectures, per-finding AUC

### Takeaway
The closest published analogue to this competition is Rost/Nebelung-group style work, Eur Radiol 2025:
- 3,121 routine studies; axial, coronal and sagittal PD-FS plus sagittal T1.
- Labels extracted from reports.
- ResNet18 slice encoder with 3D residual blocks plus a transformer (TransMed-style).
- Median AUC 0.85 internal and 0.78 external.

The team's gold-58 scores already match or exceed it on most shared findings. ROI-based pipelines (Astuto, eClinMed 2025, Liu ACL) reach similar or higher AUC but need segmentation labels.

### Cited Findings
**Eur Radiol 2025 (doi 10.1007/s00330-025-12052-8), "Comprehensive deep learning-assisted multi-condition analysis of knee MRI".**
- Data: 3,121 studies from 3,018 adults at one practice (1.5T Philips + 3T Siemens). External test: 448 studies on 4 Philips scanners.
- Labels: report-derived, extracted manually by trained medical students [V].
- Input: each sequence (sag/cor/ax PD-FS, sag T1) histogram-normalized per sequence, clipped at the 99.9th percentile, then resized to 256×256×32 slices [V].
- Architecture: TransMed-like. An ImageNet ResNet18 encodes each slice into a token, with residual 3D conv blocks inserted to mix adjacent slices. An encoder-only transformer runs over the tokens from all sequences. **The slice encoder is shared across sequences**, which the authors credit, together with ImageNet init, for limiting overfitting [V].
- Training: one multi-label network, 5-fold CV [V] — [Europe PMC full text PMC13035746](https://europepmc.org/article/PMC/PMC13035746), [Springer](https://link.springer.com/article/10.1007/s00330-025-12052-8)
- Per-condition AUC, internal CV / external (prevalence internal) [V]:

  | Condition | Internal CV | External | Prevalence (internal) |
  |---|---|---|---|
  | MCL tear | 0.95 | 0.88 | 2.7% |
  | Baker's cyst | 0.93 | 0.85 | 27% |
  | Effusion | 0.91 | 0.94 | 31% |
  | ACL tear | 0.89 | 0.89 | 5.2% |
  | Medial meniscus tear | 0.88 | 0.83 | 29% |
  | Lateral meniscus tear | 0.84 | 0.77 | 8.5% |
  | Tibial cartilage | 0.85 | 0.74 | |
  | Retropatellar cartilage | 0.85 | 0.80 | |
  | Femoral cartilage | 0.84 | 0.77 | |
  | Bone marrow edema | 0.83 | 0.82 | 41% |
  | Medial meniscus extrusion | 0.89 | 0.86 | |
  | Lateral meniscus extrusion | 0.85 | 0.73 | |
  | Non-tear MCL pathology | 0.72 | 0.67 | |
  | Medial meniscus degeneration | 0.61 | 0.58 | |

  Mean absolute internal→external drop: 0.05 ± 0.03. Effusion was the only condition to improve externally [V].
- Authors' explanation: low performers (cartilage, BME, degeneration, non-tear ligament) have "subtle imaging features, lack unequivocal diagnostic criteria, or rely on subjective grading" and low inter-reader agreement. Report labels are "not always a high-quality standard" [V].
- No fracture, synovitis or OA-compartment label. Cartilage is split femoral/tibial/retropatellar rather than medial/lateral [V].

**Astuto et al., Radiology: AI 2021 (OAI-style, UCSF).**
- Data: 1,435 studies from 294 patients, 3D FSE CUBE only [S].
- Pipeline: hierarchical. Segment tissue ROIs, then a 3D CNN per region detects the lesion, then grades it with simplified WORMS/MOAKS. Covers cartilage (6 compartments), bone marrow (6), meniscus (4) and ACL [S].
- AUC 0.83–0.93 across tissues; sensitivity 70–88%, specificity 85–89% [S]. BME AUC 0.83 [V, as quoted in Eur Radiol 2025].
- Sources: [RSNA](https://pubs.rsna.org/doi/full/10.1148/ryai.2021210022), [PMC8166108](https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108), [Eur Radiol 2025 discussion](https://europepmc.org/article/PMC/PMC13035746)
- The per-compartment (medial vs lateral vs PF) AUC table was not retrieved.

**eClinicalMedicine 2025 (SMU/HKUST), nine abnormalities.**
- Data: 14,962 exams from 13,419 patients, multi-centre, two external sets [V].
- Input: sagittal, coronal and axial PD-FS [V].
- Pipeline:
  - A 3D U-Net segments the meniscus on sagittal (Dice 0.74), and the image is cropped to a 256×256 ROI around it [V].
  - **Nine separate binary classifiers**, each with an attention-guided object localization module (AOLM, coarse-to-fine re-crop) [V].
  - Separate per-plane feature extraction, then fusion [V].
- AUC, primary findings (meniscus, cartilage, ACL): 0.898 internal, 0.852 external I, 0.812 external II.
- AUC, secondary findings (PCL, MCL, LCL, IFP, plica, cysts): 0.815, 0.744, 0.774 [V].
- External II showed AUC swings >0.15 across age subgroups for some pathologies [V].
- Sources: [Europe PMC PMC12508577](https://europepmc.org/article/PMC/PMC12508577), [HKUST summary](https://smartx.cse.ust.hk/2025/09/29/ecmknee/)
- The paper reports no ablation of crop vs no-crop in what I read.

**Single-finding pipelines that use ROI isolation.**
- Liu et al., Radiology: AI 2019 (ACL): sagittal PD and FS-T2. Two CNNs isolate the ACL, then a classifier runs on the cropped ligament. 350 subjects (175/175, full-thickness tears only). AUC 0.98, sensitivity and specificity 0.96 [S] — [RSNA](https://pubs.rsna.org/doi/full/10.1148/ryai.2019180091). The case mix (full tears vs intact) inflates AUC relative to routine data.
- Namiri et al., Radiology: AI 2020 (ACL grading): 1,243 images from 224 patients, 4 classes. 3D CNN 89% accuracy vs 2D CNN 92%, both κ=0.83. **2D ≥ 3D at this scale** [S] — [arXiv 2003.09089](https://arxiv.org/pdf/2003.09089)
- Pedoia et al., JMRI 2019 (meniscus): 1,478 studies. 2D U-Net segmentation, then 3D CNN. AUC 0.89, sensitivity 89.8%, specificity 82.0% [S] — [Tack 2021 related work](https://europepmc.org/article/PMC/PMC8675251), [ResearchGate PDF](https://researchgate.net/profile/Valentina-Pedoia/publication/325763627)
- Fritz et al., Skeletal Radiol 2020 (meniscus; 100 patients, arthroscopy reference):
  - Medial meniscus: DCNN sensitivity/specificity/accuracy 84/88/86% vs radiologists 93–96/86–91/92% [S] — [PMC7299917](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7299917/)
  - Lateral meniscus tear prevalence 24% vs 57% medial [S]. Lateral numbers not retrieved.
- SFR/JFR 2018 meniscus challenge:
  - Couteaux et al. (1st place): Mask R-CNN explicitly localizes normal and torn menisci on sagittal slices, then ensembled, then a shallow orientation CNN. Weighted AUC 0.906 [S] — [DIII 2019](https://www.em-consulte.com/article/1286640/automatic-knee-meniscus-tear-detection-and-orienta)
  - Roblot et al.: Faster R-CNN horn detection, then tear classification. AUC 0.92 for tear detection [V, as quoted in Tack 2021] — [em-consulte](https://www.em-consulte.com/article/1286641/artificial-intelligence-to-diagnose-meniscus-tears)

**Datasets.**
- fastMRI+: 16,154 bounding boxes plus 13 study-level labels across 22 pathology categories. Single coronal PD or T2 series per knee [S] — [arXiv 2109.03812](https://arxiv.org/pdf/2109.03812)
- SKM-TEA: I did not research it (it is 3D qDESS with segmentation and detection boxes; [U]).

### Inferences
- The team's gold-58 numbers vs Eur Radiol 2025 internal/external:

  | Finding | Team (gold-58) | Eur Radiol internal / external |
  |---|---|---|
  | ACL | .933 | .89 / .89 |
  | Medial meniscus | .975 | .88 / .83 |
  | Lateral meniscus | .837 | .84 / .77 |
  | Effusion | .952 | .91 / .94 |
  | Baker's | .995 | .93 / .85 |
  | Contusion vs BME | .954 | .83 / .82 |
  | MCL | .859 | .95 / .88 |

  MCL is the one finding where the published analogue is clearly higher. Their MCL tear was partial/complete with 2.7% prevalence, and non-tear MCL pathology scored only 0.72, so the label definition decides it.
- ROI-cropped pipelines are what the strongest single-finding papers use, but none gave a clean same-data crop vs no-crop delta except Tack (see Q4).

### Gaps
- No published model I found covers synovitis, fracture, or medial/lateral/PF OA as study-level labels on routine 2D multi-sequence MRI. Astuto's per-compartment cartilage AUCs exist but were not retrieved.

---

## Q3. Which plane or sequence matters per finding? Does per-finding routing beat a shared model?

### Takeaway
On MRNet, axial (PD) was the best single plane for both ACL and meniscus, and fusing all planes always beat the best single one. Per-task models beat one multi-task model only in a from-scratch small-data setting. The best recent multi-condition model is a single shared multi-label network over all sequences. I found no evidence that hard plane routing beats a shared model with learned fusion.

### Cited Findings
- Azcona (pretrained ResNet18, per plane):
  - ACL: axial 0.934 > coronal 0.926 > sagittal 0.897 [V]
  - Meniscus: axial 0.882 > coronal 0.852 > sagittal 0.781 [V]
  - Abnormal: sagittal 0.950 ≈ axial 0.949 > coronal 0.913 [V]
  - Logistic-regression fusion beat the best single plane by +0.022 (ACL) and +0.026 (meniscus) [V] — [arXiv 2010.01947](https://arxiv.org/pdf/2010.01947)
- ELNet chose coronal for meniscus and axial for ACL/abnormal "based on experiments", using one series only [V] — [arXiv 2005.02706](https://arxiv.org/pdf/2005.02706)
- Azcona, from scratch: separate per-task network 0.858 average vs one multi-task network 0.828 [V] — [arXiv 2010.01947](https://arxiv.org/pdf/2010.01947)
- Eur Radiol 2025 used one multi-label model over all four sequences with a shared slice encoder and got good results on 23 conditions [V] — [PMC13035746](https://europepmc.org/article/PMC/PMC13035746)
- eClinMed 2025 used nine separate binary models with per-plane features and fusion [V]. Its Grad-CAMs showed ACL, PCL and cyst lesions found across multiple planes, while meniscus, cartilage, MCL, LCL, IFP and plica were found "in specific planes" (planes not listed in what I read) [V] — [PMC12508577](https://europepmc.org/article/PMC/PMC12508577)
- Tack 2021 (OAI): 3D DESS (sagittal, high-res) beat IW-TSE (coronal-ish, thicker slices) for meniscus. Full-volume medial meniscus subregion AUC 0.74–0.85 on DESS vs 0.82–0.87 on TSE; with bbox loss, 0.93–0.94 vs 0.84–0.88. The authors blame thicker mediolateral slices for missed horizontal body tears [V] — [PMC8675251](https://europepmc.org/article/PMC/PMC8675251)
- Systematic review (52 studies; 3D-MRI-focused): 40% of studies used sagittal only. 2D-data models averaged higher than 3D. Pathology-specific models averaged AUC 0.927 vs 0.898 for general-abnormality models. These are confounded pooled averages [V] — [Eur Radiol systematic review, PMC12021734](https://europepmc.org/article/PMC/PMC12021734)

### Inferences
- MRNet's "axial best for meniscus" is likely a sequence effect (axial = PD; coronal = T1, which is fluid-insensitive) [U]. In a fluid-sensitive multi-sequence protocol, use radiology priors:
  - coronal PD-FS for MCL, meniscus body, and medial vs lateral tibiofemoral OA
  - sagittal for horns, ACL, effusion and Baker's
  - axial for patellofemoral OA, effusion and MCL
- Soft routing (per-finding attention over series, or per-finding heads with series-type embeddings) is the low-risk version. Hard routing has no published win.
- For weak findings (MCL, lateral OA, PF OA), a per-finding specialist model restricted to the most relevant series plus crop is the one place a dedicated model may add value (eClinMed-style). This is an inference without a direct A/B in the literature.

### Gaps
- No paper found with a controlled shared-vs-routed comparison on a pretrained modern backbone at a few-thousand-study scale.

---

## Q4. Slice attention/MIL vs 3D CNN vs 2.5D at few-thousand-study scale; does ROI cropping help?

### Takeaway
At this scale, a pretrained 2D slice encoder plus attention/transformer aggregation beats 3D CNNs, in one case by a large margin (knee meniscus 0.85 vs 0.69). ROI localization gave the largest measured single gain in the literature: Tack's auxiliary bbox-regression loss lifted medial meniscus subregion AUC by 0.08–0.20 over uncropped full volumes, and explicit cropping gave about half that.

### Cited Findings
- Medical Slice Transformer (DINOv2 2D encoder + transformer over slices) vs 3D ResNet [S] — [arXiv 2411.15802](https://arxiv.org/abs/2411.15802):
  - Knee meniscus tear (1,199 patients): **0.85 ± 0.04 vs 0.69 ± 0.05** (p=0.001)
  - Breast MRI: 0.94 vs 0.91
  - Chest CT: 0.95 vs 0.92
- Eur Radiol 2025 deliberately used 2D pretrained ResNet18 slice tokens with light 3D residual mixing between adjacent slices (i.e., 2.5D) plus a transformer [V] — [PMC13035746](https://europepmc.org/article/PMC/PMC13035746)
- Namiri 2020: 2D CNN 92% vs 3D CNN 89% accuracy for ACL staging [S] — [arXiv 2003.09089](https://arxiv.org/pdf/2003.09089)
- Azcona: ImageNet-pretrained 2D beats from-scratch by 0.036 average AUC [V] — [arXiv 2010.01947](https://arxiv.org/pdf/2010.01947)
- **ROI ablation, Tack et al. 2021** (2,399 OAI DESS scans, 3D ResNet/DRN), AUC for medial meniscus anterior/body/posterior horn and lateral meniscus anterior/body/posterior [V] — [PMC8675251](https://europepmc.org/article/PMC/PMC8675251):

  | Sequence | Approach | Medial meniscus | Lateral meniscus |
  |---|---|---|---|
  | DESS | Full-scale (uncropped) | 0.74 / 0.84 / 0.85 | 0.94 / 0.92 / 0.91 |
  | DESS | BB-crop | 0.87 / 0.89 / 0.89 | 0.95 / 0.93 / 0.91 |
  | DESS | BB-loss (full volume + auxiliary bbox regression head) | **0.94 / 0.93 / 0.93** | 0.96 / 0.94 / 0.91 |
  | IW-TSE | Full-scale | 0.82 / 0.87 / 0.82 | 0.88 / 0.85 / 0.85 |
  | IW-TSE | BB-crop | 0.84 / 0.89 / 0.86 | 0.92 / 0.90 / 0.90 |
  | IW-TSE | BB-loss | 0.84 / 0.88 / 0.86 | 0.95 / 0.91 / 0.90 |

  The auxiliary localization loss was as good as or better than hard cropping, with no crop at inference.
- Detection-first pipelines won or topped their benchmarks:
  - Couteaux Mask R-CNN won the SFR challenge, 0.906 [S] — [DIII](https://www.em-consulte.com/article/1286640/automatic-knee-meniscus-tear-detection-and-orienta)
  - Liu ACL isolation, AUC 0.98 [S] — [RSNA](https://pubs.rsna.org/doi/full/10.1148/ryai.2019180091)
  - eClinMed: meniscus-centred 256×256 crop plus AOLM attention re-crop [V] — [PMC12508577](https://europepmc.org/article/PMC/PMC12508577)

### Inferences
- The team's 2.5D CoAtNet attention-MIL is already the literature-favoured family. Switching to 3D CNN is not supported.
- Cheap version of the Tack result: add an auxiliary localization target without box labels. Options:
  - predict the slice index of max lesion attention
  - a medial/lateral half-image target
  - compartment-masked pooling: split coronal/axial feature maps into medial and lateral halves using DICOM orientation and laterality, and feed the medial/lateral OA and meniscus heads only from their half
- These are inferences, not tested in the literature. Tack's effect was largest where the baseline was weakest (medial meniscus anterior horn, +0.20), which suggests the gain concentrates in the team's weak findings (lateral meniscus, lateral OA, PF OA).

### Gaps
- No published crop vs no-crop ablation for OA compartments, MCL, synovitis or fracture.

---

## Q5. Realistic per-finding AUC ceilings

### Takeaway
Published routine-protocol, report-labelled ceilings:

| Finding | Published AUC (int / ext) |
|---|---|
| ACL | ~0.89–0.96 |
| Medial meniscus | ~0.83–0.93 |
| Lateral meniscus | ~0.77–0.84 |
| MCL tear | 0.88–0.95 (non-tear MCL pathology ~0.7) |
| Effusion | ~0.91–0.94 |
| Baker's | ~0.85–0.93 |
| BME | ~0.82–0.83 |
| Compartment cartilage | ~0.74–0.85 |

Synovitis and fracture have no good published MRI deep-learning benchmark. On gold-58, a single positive can move a finding's AUC by several hundredths.

### Cited Findings
- Eur Radiol 2025 internal/external: lateral meniscus tear 0.84/0.77, tibial cartilage 0.85/0.74, retropatellar cartilage 0.85/0.80, femoral cartilage 0.84/0.77, MCL tear 0.95/0.88, BME 0.83/0.82 [V] — [PMC13035746](https://europepmc.org/article/PMC/PMC13035746)
- Astuto 2021: all tissues 0.83–0.93 with 3D CUBE and segmentation [S] — [RSNA](https://pubs.rsna.org/doi/full/10.1148/ryai.2021210022)
- eClinMed 2025: minor lesions (incl. MCL, cysts) 0.744–0.815 even with 15k exams [V] — [PMC12508577](https://europepmc.org/article/PMC/PMC12508577)
- Tack 2021: on OAI 3D DESS, the lateral meniscus is *easier* than the medial (0.91–0.96) [V] — [PMC8675251](https://europepmc.org/article/PMC/PMC8675251). In routine clinical 2D data it is the opposite (0.77–0.84 in Eur Radiol). Prevalence (8.5% vs 29%) and 2D slice thickness likely explain this [inference].
- Fritz 2020: even radiologists vs arthroscopy reach only 86–92% accuracy on the medial meniscus [S] — [PMC7299917](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7299917/)
- Synovitis proxy: a Siamese CNN for knee pain on OAI sagittal IW-TSE (AUC 0.808) drew ~86% of correct predictions from effusion-synovitis regions [S] — [bioRxiv 463497](https://www.biorxiv.org/content/10.1101/463497v4). This is not a synovitis detector.
- Effusion/cyst detection: SE-YOLOv5 effusion mAP 0.945, cyst mAP 0.942 (a detection metric, not AUC) [S] — [Front Med 2022](https://www.frontiersin.org/articles/10.3389/fmed.2022.928642/full)
- Cartilage lesion detection (segmentation + CNN, Liu/Kijowski): AUC 0.914 [S] — [ISMRM 2018](https://cds.ismrm.org/protected/18MProceedings/PDFfiles/0200.html)

### Inferences
- Team scores against these ceilings:

  | Finding | Team | Assessment |
  |---|---|---|
  | Lateral OA | .766 | Published compartment-cartilage range ~0.74–0.85 |
  | PF OA | .798 | Retropatellar 0.80–0.85 |
  | Lateral meniscus | .837 | At the published routine-data ceiling |
  | MCL | .859 | ~0.03–0.09 below published (definition-dependent) |
  | Synovitis | .827 | No benchmark |

  The realistic upside is about +0.03–0.07 on lateral OA, PF OA and MCL. Other findings are at or above published values.
- Synovitis without contrast is reported as "effusion-synovitis" or Hoffa-synovitis (per MOAKS). It is likely highly correlated with effusion labels, so modelling it jointly with effusion (shared head, label co-occurrence) is plausible [inference].
- On gold-58, prevalence per finding is probably <15 positives for MCL, fracture and lateral OA. AUC SE is then ~0.04–0.07, so differences under ~0.05 on gold-58 are not significant. Public LB (larger) is the better selector [inference; standard Hanley-McNeil reasoning].

### Gaps
- Fracture: no knee MRI fracture-classification deep-learning paper with AUC found within budget.
- Synovitis: no non-contrast MRI deep-learning synovitis AUC found.
- Medial vs lateral vs PF OA as study-level labels: no routine-MRI paper found. Astuto per-compartment numbers were not retrieved.

---

## Q6. What transfers to our setting (ranked by expected macro-AUC gain ÷ cost)

### Takeaway
Ranked list (gains are inferred from the evidence cited above; no paper measured them on this exact setup):

1. **Laterality/compartment-aware pooling or crops for the weak lateral and PF findings.** Expected +0.01–0.02 macro; cost low to medium.
   - Evidence: Tack ROI gains of +0.03 to +0.20 on the weakest subregions, and ROI-first designs in eClinMed, Liu and Couteaux.
   - Implementation: split coronal/axial maps into medial/lateral halves (using knee laterality from DICOM) for the medial/lateral OA and meniscus heads. Crop the patellofemoral region on axial for PF OA.
2. **Auxiliary localization loss instead of hard crops.** Expected +0.005–0.015; cost low if a weak target is available.
   - Evidence: Tack BB-loss ≥ BB-crop, and it needs no crop at test time.
   - Implementation: weak targets from report text (side/compartment) or from attention maps of the current model (self-distilled boxes).
3. **Ensemble of per-plane/per-series-type models, or multiple seeds/backbones, fused by logistic regression or a mean.** Expected +0.01–0.02 on the hardest findings.
   - Evidence: Azcona fusion +0.022/+0.026 over the best single plane; meniscus most sensitive.
   - Cost: inference time inside the 9 h 2×T4 limit.
4. **Shared slice encoder across sequences plus series-type embedding plus transformer/attention across all series** (the Eur Radiol 2025 recipe). Likely already close to the team's MIL. Small gain only if the current model fuses series late or non-learnably.
5. **Per-finding specialist (crop + relevant series) only for MCL, lateral OA and PF OA.** Expected +0.02–0.05 on those three, i.e. +0.005–0.012 macro. Cost: one training run each.
   - Evidence: eClinMed separate binary models with ROI; systematic review specialised > general by ~0.03 (confounded).
6. **Stronger pretrained 2D encoders (DINOv2-style) with slice transformer.** Uncertain gain; medium cost.
   - Evidence: MST 0.85 vs 0.69 over 3D ResNet, but that is vs 3D, not vs CoAtNet-MIL.
7. **Not worth it.** 3D CNNs (Namiri, MST, Tack all show no advantage at this scale). ELNet-style from-scratch tiny nets (pretraining is worth +0.036). A single shared multi-task model trained from scratch (Azcona −0.03).

### Cited Findings
- All supporting numbers are in Q1–Q5:
  - Tack ROI table [V] — [PMC8675251](https://europepmc.org/article/PMC/PMC8675251)
  - Azcona fusion and pretraining [V] — [arXiv 2010.01947](https://arxiv.org/pdf/2010.01947)
  - Eur Radiol 2025 architecture and AUCs [V] — [PMC13035746](https://europepmc.org/article/PMC/PMC13035746)
  - MST vs 3D ResNet [S] — [arXiv 2411.15802](https://arxiv.org/abs/2411.15802)
  - eClinMed crop + AOLM [V] — [PMC12508577](https://europepmc.org/article/PMC/PMC12508577)

### Inferences
- The gap to the leaders (0.943 vs 0.961–0.964 LB) is ~0.02 macro. The literature suggests it is concentrated in the subtle, label-noisy findings (cartilage/OA compartments, lateral meniscus, MCL, synovitis). Localization priors and ensembling are the evidence-backed levers there.
- Report-derived labels cap achievable AUC on subjective findings (Eur Radiol 2025 authors' own explanation). Label cleaning or soft labels for OA, synovitis and BME may matter as much as architecture. Covered in the sibling note learning_from_reports.md.

### Gaps
- Exact MRPyrNet and attention/MPFuseNet ablation deltas.
- SKM-TEA and fastMRI+ baselines, and whether their weights or data are permitted as external data in this competition.
- Fracture and synovitis benchmarks.
