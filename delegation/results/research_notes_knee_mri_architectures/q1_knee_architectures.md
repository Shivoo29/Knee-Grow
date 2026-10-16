# Knee-MRI Architectures With Measured Gains Over Simple 2.5D CNNs

## Which knee-MRI architectures beat simple 2.5D CNN baselines, by how much, on which datasets?

### Takeaway
The reference baseline is MRNet (Bien et al., PLOS Medicine 2018): validation AUC 0.937 abnormal / 0.965 ACL / 0.847 meniscus on 120-exam internal validation. Within 2021–2026 literature, MRPyrNet (+2.2% ACL AUC over MRNet), CoPAS (avg AUC 0.812 over 12 findings on a 1748-patient multi-center set), KneeXNet (graph+multi-scale, AUCs 0.968–0.985 on MRNet with a custom split — treat with caution), and SB-SSL transformers (ACL AUC 0.954 with <1000 cases) all report gains, but cross-dataset comparisons are confounded by non-standard splits.

### Cited Findings
- MRNet baseline (2018, pre-window reference): AUC 0.937 abnormal (95% CI 0.895–0.980), 0.965 ACL (0.938–0.993), 0.847 meniscus (0.780–0.914) on 120-exam internal validation; architecture = per-slice AlexNet + max-pool across slices + logistic-regression fusion of 3 planes; each MRNet trains ~6h on a GTX 1070 — [PLOS Medicine](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/)
- MRPyrNet (MIDL 2021, peer-reviewed; journal extension CMIG 2022): Feature Pyramid Network + Pyramidal Detail Pooling; "for the ACL tear, the ROC-AUC improvement of MRPyrNet over the MRNet baseline is 2.2%, and 2.1% over the ELNet baseline" — [PMLR PDF](https://proceedings.mlr.press/v143/dunnhofer21a/dunnhofer21a.pdf); journal version — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0895611122001124); code — [GitHub](https://github.com/matteo-dunnhofer/MRPyrNet)
- ELNet (MIDL 2020, just outside window but the comparator baseline): trains from scratch, single plane (axial or coronal) input, <1MB model size; reported MRNet-dataset AUCs 0.904 meniscus / 0.960 ACL / 0.941 abnormality — [PMLR](https://proceedings.mlr.press/v121/tsai20a.html); numbers as cited in — [PMC review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12326743)
- CoPAS (Nature Communications 2024, peer-reviewed): co-plane attention across sequences, 1748 patients / 12 abnormality types, average AUC-ROC 0.812, beat MRNet/MPFuseNet/ELNet adaptations in 8 of 12 findings; external-center generalization dropped avg AUC to 0.721–0.726 — [Nature](https://www.nature.com/articles/s41467-024-51888-4)
- KneeXNet (Frontiers Bioeng. Biotechnol. 2025, peer-reviewed): GCN over anatomical landmarks + multi-scale 3D-conv fusion + contrastive loss; reports AUC 0.985 abnormal / 0.972 ACL / 0.968 meniscus on MRNet, vs SENet 0.956/0.934/0.928 and 3D CNN 0.937/0.912/0.905 on the same split; ablation: removing contrastive loss drops AUC by ~0.003–0.005 — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12088959)
- SB-SSL (MICCAI-MLCN workshop 2022, peer-reviewed workshop): slice-based self-supervised transformer; ACL accuracy 89.17%, AUC 0.954 with <1000 cases, no external pretraining data — [arXiv](https://arxiv.org/abs/2208.13923)
- KneeDINO (OpenReview preprint, year UNVERIFIED): per-task cross-plane attention fusion, multi-label; reports AUC 0.942 ACL / 0.871 meniscus / 0.955 abnormal on MRNet — [OpenReview PDF](https://openreview.net/pdf?id=NPNray7bzh)
- Azcona et al. (IDSTA 2020): replacing MRNet's AlexNet with pretrained ResNet-18/50/152 reached combined AUC 0.934 — [DCU PDF](https://doras.dcu.ie/25068/1/David_Azcona_MRNet.pdf)
- SKM-TEA (NeurIPS Datasets track 2021, peer-reviewed dataset): 155 qMRI scans (86/33/36 split) with tissue segmentations + 16 pathology bounding boxes — enables supervised anatomy-guided work but pathology-classification gain numbers vs 2.5D CNNs were not found — [GitHub](https://github.com/StanfordMIMI/skm-tea)
- fastMRI+ (Scientific Data 2022, peer-reviewed dataset): 16,154 knee bounding-box annotations, 22 knee pathology categories, 13 study-level labels — [Nature](https://www.nature.com/articles/s41597-022-01255-z); measured classification gains using fastMRI+ vs 2.5D baselines: not found

### Inferences
- The most trustworthy within-window deltas on MRNet are MRPyrNet's +2.2% ACL AUC (same dataset, same baselines re-run) and SB-SSL's transformer result under limited data.
- KneeXNet's near-perfect AUCs use a custom 1130/120/120 split with entropy-based slice selection to 25 slices, not the official MRNet validation labels/split, so its +0.03–0.05 deltas over 3D CNN/SENet likely overstate real-world gain; treat as peer-reviewed but non-comparable.
- No 2021–2026 paper found reports a clean macro-AUC delta of a CoAtNet/window-attention-MIL pipeline vs a matched 2.5D CNN + attention-MIL baseline; that controlled comparison is a gap the team would have to run itself.

### Gaps
- Exact MRPyrNet meniscal-tear AUC delta (source sentence truncated at "For meniscal tear, using…"); per-finding numbers for KneeDINO's baselines; MPF-KANSC (arXiv 2025) numbers — [arXiv listing](https://arxiv.org/html/2508.06157v1) not fetched.
- Kaggle RSNA 2026 discussion topics 735304/746792: not accessible in this environment; competitor-reported deltas unverified.
- Compute cost for KneeXNet/CoPAS/SB-SSL in GPU-hours or FLOPs: not reported in sources retrieved (only qualitative: CoPAS "relatively large number of parameters… prone to overfitting"; ELNet <1MB; MRNet 6h/GTX 1070 per plane-task model).

## Cross-plane fusion: early vs late, cross-attention between plane token sets, plane-specific heads; which findings benefit?

### Takeaway
Late fusion (per-plane model + logistic regression) is the MRNet default; attention-based cross-plane/sequence fusion (CoPAS, KneeDINO, MPFuseNet, Hybrid Fusion, Kunitomi 2025) consistently beats it, with cruciate-ligament and multi-slice-spanning findings gaining most; single-plane best varies by finding (sagittal for ACL/meniscus, coronal for collaterals).

### Cited Findings
- MRNet late fusion: 9 separate AlexNet models (3 planes × 3 tasks), per-exam logistic regression over 3 plane probabilities; most beneficial series per fitted coefficients: axial PD for abnormality/meniscus, coronal T1 for ACL — [PLOS Medicine](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/)
- CoPAS cross-plane attention (Nature Comms 2024): three plane branches, ResNet3D-18 weight-shared encoder, cross-attention where main-plane features query rotated cross-plane volumes (Eq. 1 in paper), plus co-plane cross-sequence attention and a plane–abnormality probability matrix; model plane preferences correlated with radiologists' plane preferences (high positive correlation except CONT and PLICA); strongest on cruciate ligaments (ACL/PCL/MCL/LCL) attributed to cross-slice spatial reasoning — [Nature](https://www.nature.com/articles/s41467-024-51888-4)
- MPFuseNet (Belton et al., MIUA 2021, peer-reviewed): compared multi-planar fusion strategies including simple-concatenation late fusion; equal-contribution concatenation is vulnerable to noise as sequences increase — as characterized in — [Nature](https://www.nature.com/articles/s41467-024-51888-4); exact MPFuseNet AUC numbers: not found
- Multi-Plane Hybrid Fusion (Harman, J. Artif. Intell., peer-reviewed): best single plane for ACL was sagittal (AUC 0.976); three-plane Hybrid Fusion beat single-plane — exact hybrid number not retrieved — [DergiPark](https://dergipark.org.tr/en/pub/jai/article/1926635); [ResearchGate](https://www.researchgate.net/publication/408600914_Multi-Plane_Hybrid_Fusion_for_Robust_Knee_MRI_Diagnosis)
- KneeDINO (preprint, UNVERIFIED): per-task cross-plane attention fusion = plane-specific heads with task queries attending over plane token sets; AUC 0.942 ACL / 0.871 meniscus / 0.955 abnormal — [OpenReview PDF](https://openreview.net/pdf?id=NPNray7bzh)
- Kunitomi et al. 2025 (peer-reviewed per index): clinically-prioritized attention fusion of coronal/sagittal/axial; details and numbers not retrieved — [Elsevier Pure](https://waseda.elsevierpure.com/en/publications/clinically-prioritized-attention-based-fusion-of-multi-plane-knee/)
- Sequence-dependence evidence: CoPAS external-set ablation (fewer sequences) dropped bone-contusion AUC 0.821→0.698, showing T1W/T2W dependence for CONT; IFP surprisingly robust — [Nature](https://www.nature.com/articles/s41467-024-51888-4)

### Inferences
- For the team's 2.5D-window attention-MIL design, the evidence favors: per-plane token sets → per-finding query cross-attention over planes (KneeDINO/CoPAS pattern) over logistic-regression late fusion; keep plane-specific heads since optimal plane is finding-dependent.
- Expect biggest fusion wins on ACL/PCL/meniscal findings spanning slices; effusion/cyst-like findings may suffer negative transfer in shared multi-task heads (CoPAS discussion notes MRNet single-task beat it on MCL/EFFU).

### Gaps
- No head-to-head early-vs-late vs cross-attention numbers on a single fixed dataset with CIs; MPFuseNet and Hybrid Fusion exact deltas not retrieved.
- Which fusion helps rare RSNA findings (e.g., loose bodies, PLC): no evidence found.

## Anatomy-guided attention and region graphs: measured gains, supervision needed

### Takeaway
Anatomy guidance helps but always costs extra supervision: CoPAS needs only a coarse meniscus mask for ROI cropping; KneeXNet needs a landmark detector (claims 95% detection accuracy); Astuto 2021 needs full compartment/bounding-box grading annotations; SKM-TEA and fastMRI+ exist precisely to supply such supervision (segmentations, 16 pathology boxes; 16k boxes / 22 categories).

### Cited Findings
- CoPAS preprocessing: U-Net meniscus segmentation as upstream task, explicitly "coarse segmentation is adequate because we only need a rough ROI to identify the center of the meniscus"; images cropped to ROI before the classifier — [Nature](https://www.nature.com/articles/s41467-024-51888-4)
- KneeXNet graph: nodes = key anatomical landmarks from a pre-trained detector (reported 95% detection accuracy on a separate test set), edges by Euclidean distance threshold τ tuned by grid search, node features = local P³ patches; full pipeline (GCN + multi-scale fusion + contrastive) reaches 0.985/0.972/0.968 vs 3D CNN 0.937/0.912/0.905 on their split — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12088959)
- Astuto et al. 2021 (Radiology: AI, peer-reviewed): full-knee 3D pipeline for detection AND grading of cartilage/meniscus/ligament abnormalities — anatomy-localized grading paradigm; exact AUCs not retrieved here — [PubMed](https://pubmed.ncbi.nlm.nih.gov/34142088/)
- SKM-TEA supervision available: 6-tissue segmentations + 16 pathology 3D bounding boxes with confidence scores on 155 scans — [AIMI](https://aimi.stanford.edu/datasets/skm-tea-knee-mri); [GitHub DATASET.md](https://github.com/StanfordMIMI/skm-tea/blob/main/DATASET.md)
- fastMRI+ supervision available: knee set 16,154 bounding boxes, 13 study-level labels, 22 pathology categories — [arXiv](https://arxiv.org/abs/2109.03812)
- Grad-CAM interpretability overlays used in KneeXNet and CoPAS (EigenCAM/Grad-CAM++) to validate anatomical focus — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12088959); [Nature](https://www.nature.com/articles/s41467-024-51888-4)

### Inferences
- For a 2–3-day build without new annotations, only the CoPAS-style option is feasible: reuse an existing/public meniscus or bone segmentation (or SKM-TEA-trained segmenter) for ROI cropping + bone-mask-guided attention; full landmark-graph (KneeXNet) requires training/finding a landmark detector first — out of scope.
- Region-token supervision cheaper than boxes: bone/compartment masks can generate region tokens with zero extra labeling if a public knee segmentation model is available; box-level supervision (fastMRI+/SKM-TEA style) is unavailable for RSNA-2026 labels.

### Gaps
- Isolated ablation of "anatomy guidance on/off" with matched backbones: KneeXNet's ablations isolate contrastive loss only, not the graph itself; CoPAS has no no-crop baseline reported — true marginal gain of anatomy guidance is UNVERIFIED.
- Whether bone-mask guidance helps cartilage/meniscus findings specifically: not found.

## Multi-scale crop fusion (global + local close-ups in one model): measured gains vs single scale

### Takeaway
The only within-window measured multi-scale gain for knee-MRI classification is MRPyrNet's FPN + Pyramidal Detail Pooling (+2.2% ACL AUC over MRNet, +2.1% over ELNet), motivated by anomalies localizing in small knee regions; explicit global-plus-close-up-crop fusion papers with single-scale ablations were not found for knee-MRI classification.

### Cited Findings
- MRPyrNet design rationale: "anomalies are localized in small-sized knee regions that appear in particular areas of MRI scans"; FPN enhances small-appearing features, Pyramidal Detail Pooling preserves them; improves ACL and meniscal diagnosis of two SOTA pipelines on two public datasets — [PMLR](https://proceedings.mlr.press/v143/dunnhofer21a.html)
- MRPyrNet ACL numbers: +2.2% ROC-AUC over MRNet baseline, +2.1% over ELNet baseline — [PMLR PDF](https://proceedings.mlr.press/v143/dunnhofer21a/dunnhofer21a.pdf)
- KneeXNet multi-scale module: 3D convs with kernels 3³/5³/7³ and dilations 1/2/4 plus learnable scale weights; contributes to overall result but no standalone multi-scale-vs-single-scale ablation reported — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12088959)
- CoPAS implicitly multi-scale via full-volume + ROI-cropped inputs and cross-plane rotated volumes, but no crop-fusion ablation — [Nature](https://www.nature.com/articles/s41467-024-51888-4)

### Inferences
- For the team's pipeline, the cheapest faithful implementation of this idea: FPN-style lateral connections on the CoAtNet encoder and/or a two-stream global + meniscus-centered-crop encoder with cross-attention fusion; MRPyrNet's result suggests small-lesion findings (meniscal tears, cartilage defects) should gain most.
- Implementation sketch (2–3 days, 2.5D-window attention-MIL with CoAtNet encoder, 3-slice windows, per-finding queries): (1) Day 1: CoAtNet (e.g., coatnet-0/1, ImageNet-pretrained) per 3-slice window → window tokens; reproduce MRNet-style max-pool MIL baseline first. (2) Day 2: replace max-pool with per-finding learnable queries cross-attending to window tokens; add plane-specific query heads with a learned plane–finding weight matrix (CoPAS pattern). (3) Day 3: add FPN laterals on CoAtNet stages + a global+ROI-crop two-stream fusion; train with asymmetric/focal loss. All sketch choices are UNVERIFIED estimates, not literature prescriptions.

### Gaps
- No knee-MRI study found that ablates global-only vs global+close-up-crop fusion; expected magnitude for RSNA findings is UNVERIFIED.
- Optimal crop source (meniscus-centered vs bone-mask vs learned ROI) for each finding: no comparative evidence found.
- 2.5D-window (3-slice) tokenization specifically: no 2021–2026 knee-MRI paper retrieved uses exactly this; window-size ablations would need to be run in-house.
