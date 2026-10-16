# Slice-Sequence Models and MIL Variants for ~100–200-Instance Bags with Noisy Labels

## Do sequence models (transformer/Mamba with positional encodings) beat permutation-invariant attention-MIL? By how much, where?

### Takeaway
Sequence-aware models (TransMIL transformer, MambaMIL/S4MIL state-space) beat plain ABMIL by ~2–5pp AUC on pathology WSI benchmarks, with the gap largest on multi-class subtyping (BRACS) and smallest on binary tumor detection with strong SSL features; TransMIL can underperform ABMIL-family models when features are weak (ImageNet ResNet) or cohorts small, and Mamba matches transformer gains at linear complexity with less overfitting.

### Cited Findings
- TransMIL (NeurIPS 2021) models instance correlation via self-attention + positional encoding under a "correlated MIL" framework, with convergence proof — [NeurIPS paper](https://proceedings.neurips.cc/paper/2021/file/10c272d06794d3e5785d5e7c5356e9ff-Paper.pdf); [arXiv:2106.00908](https://arxiv.org/abs/2106.00908); [code](https://github.com/szc19990412/transmil).
- TransMIL achieved AUC 93.09% on CAMELYON16 binary tumor classification, "at least 5% higher than attention-based methods like ABMIL, PT-MTA, and CLAM" — [review of TransMIL](https://liner.com/review/transmil-transformer-based-correlated-multiple-instance-learning-for-whole-slide).
- In a matched 2024 reproduction (ACMIL paper, Table 1) with ImageNet ResNet-18 features, TransMIL scored CAMELYON16 AUC 0.706 vs ABMIL 0.790 and ACMIL 0.841; with SSL ViT-S/16 features TransMIL reached AUC 0.943 vs ABMIL 0.945, DTFD-MIL 0.980, ACMIL 0.974 — i.e., TransMIL underperforms ABMIL-family with weak features but is competitive with strong features — [ACMIL arXiv HTML, Table 1](https://arxiv.org/html/2311.07125v4).
- MambaMIL (MICCAI 2024) adds selective state-space (Mamba) sequence modeling with linear complexity plus a Sequence-Reordering Mamba (SR-Mamba) branch; on BRACS subtyping AUC 80.4% and NSCLC AUC 95.9%, beating same-aggregator ABMIL by +3.9pp and +2.1pp AUC respectively — [MambaMIL summary](https://smartx.cse.ust.hk/2024/07/15/mambamil); [arXiv:2403.06800](https://arxiv.org/abs/2403.06800); [code](https://github.com/isyangshu/MambaMIL).
- MambaMIL mean C-index 0.680 vs TransMIL 0.654, S4MIL 0.641, DTFD-MIL 0.638, DSMIL 0.638, ABMIL 0.635 across 7 TCGA survival datasets (ResNet-50 features); with PLIP features MambaMIL 0.693 vs TransMIL 0.649 — [MambaMIL arXiv HTML, Table 1](https://arxiv.org/html/2403.06800v1).
- MambaMIL subtyping (ResNet-50): BRACS-7 AUC 0.804 vs TransMIL 0.699, ABMIL 0.765, DSMIL 0.768, DTFD-MIL 0.758; NSCLC AUC 0.959 vs TransMIL 0.937, ABMIL 0.938; TransMIL was worst of all aggregators on BRACS-7 (0.699) — [MambaMIL arXiv HTML, Table 2](https://arxiv.org/html/2403.06800v1).
- SR-Mamba ablation: SR-Mamba mean C-index 0.680 vs vanilla Mamba 0.662 vs Bidirectional Mamba 0.665 — reordering adds ~+1.5–1.8pp — [MambaMIL arXiv HTML, Table 3](https://arxiv.org/html/2403.06800v1).
- During training on BRACS validation, TransMIL shows rising validation loss with falling ACC/AUC (overfitting) while MambaMIL stays stable; authors attribute this to multi-ordering representations acting like data augmentation — [MambaMIL arXiv HTML, Sec 3.4](https://arxiv.org/html/2403.06800v1).
- S4MIL (MICCAI 2023) introduced structured state-space models as MIL instance-sequence learners before MambaMIL, but without WSI-specific ordering design and with sub-optimal results vs MambaMIL (S4MIL BRACS-7 AUC 0.760 vs MambaMIL 0.804) — [MambaMIL paper refs](https://arxiv.org/html/2403.06800v1).
- Mamba core method: selective state spaces with linear-time modeling — [Mamba arXiv:2312.00752](https://arxiv.org/abs/2312.00752); S4 predecessor — [arXiv:2111.00396](https://arxiv.org/abs/2111.00396).
- DSMIL (CVPR 2021 oral) dual-stream aggregator with trainable distance measurement + multiscale pyramidal fusion: single-scale AUC 0.8944 / acc 0.8682 vs ABMIL 0.8653 AUC on CAMELYON16 (+2.9pp); DSMIL-LC multiscale 0.9165 AUC / 0.8992 acc (+5.1pp over ABMIL); localization FROC 0.4371 — [DSMIL review](https://liner.com/review/dualstream-multiple-instance-learning-network-for-whole-slide-image-classification); [paper PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8765709/); [arXiv:2011.08939](https://arxiv.org/abs/2011.08939); [code](https://github.com/binli123/dsmil-wsi).
- DSMIL repo expected performance: Camelyon16 5-fold CV acc 94.9% / AUC 0.961; standalone test acc 92.4% / AUC 0.915; TCGA Lung 5-fold acc 93.78% / AUC 0.981 — [DSMIL GitHub](https://github.com/binli123/dsmil-wsi).
- Positional-encoding background: vanilla transformers are permutation-equivariant and need explicit position signal; sinusoidal (Vaswani 2017), learned absolute, relative, RoPE, ALiBi are the standard options; TransMIL adds spatial/positional info to instance tokens — [IBM positional encoding explainer](https://www.ibm.com/think/topics/positional-encoding); [ICLR 2025 positional embeddings survey](https://iclr-blogposts.github.io/2025/blog/positional-embedding).

### Inferences
- For knee-MRI slice bags (~100–200 windows/study), slices have a natural anatomical order, so sequence models are better motivated than in pathology (where patch order is arbitrary raster order). Expect transformer/Mamba + slice-index positional encoding to add +2–4pp AUC over ABMIL, most on multi-finding subtyping-like heads.
- TransMIL-style full self-attention over 100–200 instances is cheap (200² = 40k attention entries) — no need for linear-complexity SSM at this bag size; Mamba's advantage (linear scaling to 5k–60k instances) is largely irrelevant here, though its overfitting robustness may still help.
- TransMIL's collapse with weak features (0.706 vs ABMIL 0.790) warns: sequence models need good instance embeddings (ImageNet-pretrained 2.5D CNN or SSL) or they overfit small cohorts.

### Gaps
- No direct MSK-radiology slice-sequence-vs-ABMIL ablation found; all deltas are pathology WSI. Effect size on knee MRI is UNVERIFIED extrapolation.
- No anatomical (slice-position-in-joint) vs sinusoidal vs learned positional encoding comparison found for medical slice bags — which encoding is best is UNVERIFIED.
- Attention-supervision (e.g., CLAM clustering loss) vs plain attention-MIL delta on slice bags not quantified in sources found.

## Which MIL aggregator is best for 100–200 instances with noisy bag labels? (TransMIL / DSMIL / DTFD-MIL / ACMIL / additive MIL)

### Takeaway
ACMIL (ECCV 2024) is the current best-documented aggregator under matched conditions (10/12 metric wins across 3 datasets × 2 backbones), DTFD-MIL wins on CAMELYON16-with-strong-features by virtual bag expansion (best for small cohorts), DSMIL is the best cheap baseline (+3–5pp over ABMIL), and additive MIL trades a small accuracy cost for exact Shapley-value interpretability.

### Cited Findings
- ACMIL (ECCV 2024, [arXiv:2311.07125](https://arxiv.org/abs/2311.07125), [code](https://github.com/dazhangyu123/ACMIL), [ECCV poster](https://eccv.ecva.net/virtual/2024/poster/804)) = Multiple-Branch Attention (MBA, M=5 branches with semantic + diversity loss) + Stochastic Top-K Instance Masking (STKIM, K=10, p=0.6–0.8, removed at inference).
- ACMIL motivation measurement: sum of top-10 attention values exceeds 0.85 on all three datasets (i.e., 10 instances dominate the bag); STKIM cuts CAMELYON16 top-10 cumulative attention 0.87 → 0.6 — [ACMIL arXiv HTML, Sec 3.3/4.5](https://arxiv.org/html/2311.07125v4).
- ACMIL Table 1 (ResNet-18 features): CAMELYON16 F1 0.798 / AUC 0.841 vs runner-up +2.1pp F1 / +2.6pp AUC; LBC AUC 0.821 vs next 0.816; BRACS comparable to MHIM-MIL — [ACMIL arXiv HTML, Table 1](https://arxiv.org/html/2311.07125v4).
- ACMIL Table 1 (SSL ViT-S/16 features): CAMELYON16 F1 0.954 / AUC 0.974 vs DTFD-MIL 0.948/0.980 (DTFD wins AUC by 0.6pp); BRACS F1 0.722 / AUC 0.888 beats 2nd-best by +4.2pp F1 / +1.7pp AUC; LBC F1 0.662 / AUC 0.901, best on both — [ACMIL arXiv HTML, Table 1](https://arxiv.org/html/2311.07125v4).
- ACMIL localization: FROC 0.4233 vs ABMIL 0.3987 (+2.5pp) on CAMELYON16 — [ACMIL arXiv HTML, Table 2](https://arxiv.org/html/2311.07125v4).
- ACMIL diversity-loss ablation: removing diversity loss drops AUC by up to 6.4pp (LBC SSL: 0.901→0.837), 3.1pp (CAMELYON SSL), 5.5pp (CAMELYON ResNet) — diversity regularization is load-bearing — [ACMIL arXiv HTML, Table 3b](https://arxiv.org/html/2311.07125v4).
- ACMIL STKIM-at-test ablation: keeping masking on at inference hurts by 0.4–2.7pp — always disable at test — [ACMIL arXiv HTML, Table 3a](https://arxiv.org/html/2311.07125v4).
- ACMIL masking comparison: STKIM (mask minority K=10, p<1) beats WENO (masks 95 instances, needs teacher-student warmup) and MHIM-MIL (masks 1%, needs pretrained teacher); STKIM needs no teacher and no pretraining — [ACMIL arXiv HTML, Sec 3.3](https://arxiv.org/html/2311.07125v4).
- DTFD-MIL (CVPR 2022, [arXiv:2203.12081](https://arxiv.org/abs/2203.12081), [CVPR page](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_DTFD-MIL_Double-Tier_Feature_Distillation_Multiple_Instance_Learning_for_Histopathology_Whole_CVPR_2022_paper.html), [code](https://github.com/hrzhang1123/DTFD-MIL)): pseudo-bags virtually enlarge bag count + two-tier distillation; reported best AUC 96.1%, "at least 4% better in AUC than other existing methods" — [CS proceedings summary](https://www.computer.org/csdl/proceedings-article/cvpr/2022/694600s8780/1H1nhNJhmbm).
- DTFD-MIL design point: built for small-slide-cohort + huge-instances-per-slide regimes; Tier-1 trains on pseudo-bags, Tier-2 distills via MaxMinS (min+max derived-probability instances per pseudo-bag), which beats MaxS-only distillation — [DTFD-MIL supplement](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Zhang_DTFD-MIL_Double-Tier_Feature_Distillation_Multiple_Instance_Learning_for_Histopathology_Whole_CVPR_2022_supplemental.pdf).
- In matched ACMIL reproductions DTFD-MIL is the strongest rival: SSL CAMELYON16 AUC 0.980 (best overall), but ResNet-18 CAMELYON16 only 0.815 vs ACMIL 0.841 — DTFD's edge depends on strong features — [ACMIL arXiv HTML, Table 1](https://arxiv.org/html/2311.07125v4).
- In MIL-Dropout paper's matched WSI runs, DTFD-MIL(AFS) baselines: CAMELYON16 ImageNet AUC 88.2 / SimCLR 89.6; TCGA-NSCLC ImageNet 94.4 / SimCLR 93.1 — [MIL-Dropout arXiv HTML, Table 2](https://arxiv.org/html/2504.14783v1).
- Additive MIL (NeurIPS 2022, [arXiv HTML](https://arxiv.org/html/2206.01794v2), [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/82764461a05e933cc2fd9d312e107d12-Paper-Conference.pdf)): additive aggregation provably equals exact per-instance Shapley values with no approximation; authors report "similar predictive performance" to attention-MIL (interpretability at ~parity, not a gain) — [Additive MIL paper](https://arxiv.org/html/2206.01794v2).
- CLAM-SB (Nature Biomed Eng 2021) attention-MIL with instance clustering supervision: matched SSL CAMELYON16 F1 0.925/AUC 0.969; ResNet CAMELYON16 0.742/0.763 — strong supervised-attention baseline behind ACMIL/DTFD — [ACMIL arXiv HTML, Table 1](https://arxiv.org/html/2311.07125v4); [CLAM](https://pmc.ncbi.nlm.nih.gov/articles/PMC8765709/) context.
- MHIM-MIL and IBMIL are second-tier in matched runs (MHIM-MIL SSL LBC AUC 0.872 vs ACMIL 0.901; IBMIL SSL BRACS AUC 0.871 vs ACMIL 0.888) — [ACMIL arXiv HTML, Table 1](https://arxiv.org/html/2311.07125v4).

### Inferences
- For ~4k-study knee-MRI bags with noisy report-derived labels: default choice = ABMIL + STKIM-style top-K stochastic masking + MBA multi-branch (i.e., ACMIL-lite), because attention-concentration overfitting is the documented failure mode under weak supervision and both components have isolated gains (diversity loss up to +6pp; STKIM consistent +1–3pp-class improvements in combination).
- DTFD pseudo-bags are worth adding if any finding has <200 positive studies (virtual bag expansion); with 4k studies and common findings it adds complexity for little gain — UNVERIFIED judgment, needs ablation.
- Per-finding queries (one attention branch per finding) map naturally onto MBA: use one branch per finding with shared instance encoder + diversity loss across findings; this is an architectural inference, UNVERIFIED empirically.
- Additive MIL is the right head when localization credit-assignment must be exact (radiologist-facing heatmaps), accepting parity-level accuracy rather than SOTA.

### Gaps
- No head-to-head aggregator benchmark on knee-MRI or any MSK radiology bag task found — "best for 100–200 instances" ranking is extrapolated from 5k–60k-instance pathology bags (MambaMIL reports bag lengths 26–62k, avg ~7–14k — [MambaMIL arXiv HTML, Table 4](https://arxiv.org/html/2403.06800v1)). Behavior at 100–200 instances is UNVERIFIED.
- Optimal pseudo-bag count/size for 100–200-instance bags not reported anywhere found.
- No label-noise-robustness curve (AUC vs % flipped bag labels) for ACMIL/DTFD found.

## Instance dropout / stochastic pooling under label noise: measured effect? + Cost: what trains in hours on a single 12–24 GB GPU for ~4k studies?

### Takeaway
Dropping top-K salient instances is the single best-supported anti-noise regularizer: MIL-Dropout (ICML 2025) gives +1–4pp accuracy/AUC across 4 aggregators × 2 WSI datasets with negligible compute, and ACMIL's STKIM independently confirms the mechanism; all aggregator-stage methods train in minutes–hours on one GPU once features are pre-extracted, and a 2.5D-window knee pipeline (~4k studies × ~150 windows) fits 12–24 GB easily with frozen-encoder + cached features.

### Cited Findings
- MIL-Dropout (ICML 2025, [ICML poster](https://icml.cc/virtual/2025/poster/43917), [arXiv HTML](https://arxiv.org/html/2504.14783v1), [code](https://github.com/ChongQingNoSubway/MILDropout)): key empirical claim — "dropping the top-k most important instances within a bag leads to better performance and generalization even under noise attack" — [MIL-Dropout abstract](https://arxiv.org/html/2504.14783v1).
- Mechanism evidence: only top-K DropInstance reduces gradient-direction error (GDE) during training; random and bottom-K dropout increase GDE; DropInstance also yields flatter minima (lower ε-sharpness) and better robustness to Gaussian/Poisson/Salt&Pepper input-noise attacks than DropNeuron — [MIL-Dropout arXiv HTML, Sec 3.2–3.4](https://arxiv.org/html/2504.14783v1).
- MIL benchmark gains (ABMIL + MIL-Dropout vs ABMIL): MUSK1 0.964 vs 0.892 (+7.2pp), MUSK2 0.954 vs 0.858 (+9.6pp), FOX 0.789 vs 0.615 (+17.4pp), TIGER 0.917 vs 0.839 (+7.8pp), ELEPHANT 0.934 vs 0.868 (+6.6pp); mean +9.7pp — [MIL-Dropout arXiv HTML, Table 1](https://arxiv.org/html/2504.14783v1).
- WSI gains averaged over 4 aggregators (ABMIL, DSMIL, TransMIL, DTFD-MIL) × 2 feature types: CAMELYON16 +2.3pp accuracy / +2.46pp F1 / +2.48pp AUC; TCGA-NSCLC +2.05pp / +2.44pp / +1.80pp — [MIL-Dropout arXiv HTML, Sec 6.2](https://arxiv.org/html/2504.14783v1).
- Largest single WSI deltas: ABMIL CAMELYON16/ImageNet AUC 86.0→90.1 (+4.1pp); ABMIL-Gated CAMELYON16/ImageNet AUC 86.2→90.7 (+4.6pp); DTFD-MIL(AFS) TCGA/SimCLR AUC 93.1→96.1 (+3.0pp) — [MIL-Dropout arXiv HTML, Table 2](https://arxiv.org/html/2504.14783v1).
- "ABMIL + MIL-Dropout achieves performance comparable to or even surpasses SOTA methods" (i.e., plain ABMIL + dropout ≈ TransMIL/DTFD-class) — [MIL-Dropout arXiv HTML, Sec 6.2](https://arxiv.org/html/2504.14783v1).
- Complexity: MIL-Dropout adds no learnable parameters, only K and G hyperparameters; per-iteration cost O(N·D) since K(≈20) ≪ N(≈5–10k); "negligible computational cost" — [MIL-Dropout arXiv HTML, Sec 4.0.1](https://arxiv.org/html/2504.14783v1).
- DSMIL's own implementation already randomly drops input instances (DropInstance) as undocumented regularization, cited as partial evidence dropout helps MIL — [MIL-Dropout arXiv HTML, Sec 1](https://arxiv.org/html/2504.14783v1).
- ACMIL STKIM independently converges on the same recipe (stochastic top-K masking, train-only): MBA alone and STKIM alone each beat ABMIL; combined wins in 39/45 hyperparameter settings; optimal M=5 branches, p=0.6–0.8, K=10 (K insensitive, <1pp across values) — [ACMIL arXiv HTML, Sec 4.4](https://arxiv.org/html/2311.07125v4).
- Mask-all-top-K (p=1.0) hurts (worse than no masking on some datasets) — stochasticity is required, not hard removal — [ACMIL arXiv HTML, Sec 4.4](https://arxiv.org/html/2311.07125v4).
- Bag-size context: WSI bags run 26–62k instances (avg ~7–14k) per MambaMIL's dataset table — two orders of magnitude above knee-MRI slice bags — [MambaMIL arXiv HTML, Table 4](https://arxiv.org/html/2403.06800v1).
- Standard MIL recipe (all cited papers): freeze pretrained backbone, cache per-instance features, train only aggregator + shallow layers; aggregation is a tiny MLP/attention over cached vectors — implies minutes–hours training, though exact GPU-hours are rarely reported (ACMIL has a computational-cost appendix section not extracted here).
- MambaMIL used lr 2e-4 with dataset-specific tuning (2e-5–1e-5 on some) to tame SR-Mamba gradient noise — extra tuning cost vs ABMIL — [MambaMIL arXiv HTML, Sec 3.2](https://arxiv.org/html/2403.06800v1).

### Inferences
- Cost estimate (UNVERIFIED, arithmetic): 4k studies × ~150 windows × 512-d features ≈ 300M floats ≈ 1.2 GB cached — fits RAM/VRAM trivially; aggregator-only training (ABMIL/ACMIL-lite/DSMIL, batch = 8–32 bags) should train in ~1–4 h on a single 12–24 GB GPU; end-to-end 2.5D CNN finetuning dominates cost and should be staged after aggregator convergence.
- Recommended 2–3-day implementation sketch (UNVERIFIED plan, composed from cited components):
  - Day 1: build 2.5D windows (3 adjacent slices → RGB, ~100–200/study), ImageNet-pretrained encoder (frozen), cache 512-d features; ABMIL baseline with per-finding query vectors (one gated-attention head per finding, shared encoder).
  - Day 2: add STKIM/MIL-Dropout (K=10, p=0.6–0.8, train-only, renormalize) + MBA-lite (2–5 branches on the hardest finding + diversity loss); add slice-index sinusoidal positional encoding + ablation (with/without) to test sequence-vs-permutation-invariant.
  - Day 3: pseudo-bag augmentation for rare findings (DTFD-style split of each study into 2–4 pseudo-bags inheriting bag label); run ablations (ABMIL → +dropout → +MBA → +positional) and report per-finding AUC deltas; disable all masking at inference.
- Skip full TransMIL encoder stack on day 1–3 unless ABMIL+dropout underfits: MIL-Dropout paper shows ABMIL+dropout ≈ SOTA, so dropout-first is the cheaper path.
- Mamba/SR-Mamba is deprioritized for 100–200-instance bags: linear-complexity payoff needs 5k+ sequences; keep as stretch only if overfitting persists after STKIM+MBA.

### Gaps
- No measured "noise attack" numbers (AUC at X% label flip) were extracted — the paper claims robustness under noise attack but exact curves were not pulled; needs follow-up.
- No published GPU-hours for any method found — "hours on single GPU" is inference from architecture size (aggregator ≈ thousands of parameters over cached features), marked UNVERIFIED.
- No stochastic-pooling (e.g., wildcat/top-k+ average) vs attention comparison with numbers found.
- Knee-MRI-specific: optimal K for 100–200-instance bags unknown (pathology uses K=10–20 at N=5–10k, i.e., ~0.2–0.4% of bag ≈ <1 window at N=150 — scaling rule UNVERIFIED; start K=2–5).
