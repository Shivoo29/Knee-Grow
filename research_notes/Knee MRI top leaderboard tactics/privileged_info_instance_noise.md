# Privileged information (PI) and instance-dependent label noise (IDN): evidence-graded catalogue and recipes for RSNA Knee 2026

Scope: methods (2016–2026) that use information available only at training time, such as the 31 LLM label tables, to absorb label noise, plus instance-dependent-noise and small-trusted-set methods. The first research round already covered class-conditional corrections, aggregation of the 31 tables, GLC/L2RW/MW-Net and robust losses ([`noisy_labels_auc_ensembling.md`](noisy_labels_auc_ensembling.md)). They are referenced here but not repeated.

Evidence grades (same legend as the earlier note):
- **[PR]**: peer-reviewed, venue confirmed in this session from the paper header or proceedings page.
- **[PR\*]**: peer-reviewed according to my knowledge, venue not re-checked this session.
- **[PP]**: preprint.
- **[team]**: our own measurement, from [`DEEP_RESEARCH_PROMPT.md`](../../DEEP_RESEARCH_PROMPT.md) §3 or the earlier note.

Qualifiers:
- **[snippet]**: from a search summary, not checked against the full text.
- **[computed]**: my arithmetic.
- **[judgement]**: my estimate.
- **[unverified]**: a claim I could not confirm.

Every number below was read from the paper PDF (arXiv) unless it is marked [snippet].

Rule flags:
- **NEEDS-TEXT**: the method needs the raw report column, which our team does not open.
- None of the recommended recipes needs report text or any external API.

---

## Q1. TRAM and its successors (ImageNet-PI, TRAM++, Pi-DUAL), generalized distillation/LUPI, heteroscedastic heads: recipes and results

### Takeaway
PI methods help by giving the network a cheap "shortcut" to memorise mislabeled examples from the PI, so the image features learn only from the examples the image can explain. Pi-DUAL is the strongest published variant:
- ImageNet-PI, where the noisy labels come from 16 model "annotators" (the closest public analogue to our LLM tables): 62.1% vs 55.3% for the best earlier PI method and 47.2% for plain cross-entropy (CE) at high noise; 71.6 vs 68.2 (CE) at low noise.

Two failure modes matter for us:
- With coarse PI, TRAM gains are about 0.
- If the PI encodes the label itself, PI methods collapse (TRAM 38.3% vs 55.0% with no PI).

Other results:
- Generalized distillation with a PI teacher is no better than self-distillation in every benchmark found, including a 2025 radiology-report study. This matches our failed teacher distillation.
- Heteroscedastic (HET) heads are drop-in output layers. They help on large noisy datasets, are mixed on small ones, and stack with TRAM.

### Cited Findings

#### TRAM: "Transfer and Marginalize" (Collier, Jenatton, Kokiopoulou, Berent) [PR, ICML 2022, PMLR 162]
- **Architecture.** One network, two heads, shared feature extractor φ(x):
  - The PI head q(y|x,a) = q(y|ψ(φ(x),a)) gets image features plus PI.
  - The no-PI head is π(y|x) = softmax(W·stop_gradient(φ(x))).
  - Loss: L₂(y, π(y|x)) + β·L₁(y, q(y|x,a)), with β = 1 (folded into the learning rate).
  - φ is updated only by gradients from the PI head.
  - At test time the PI head is dropped and only the no-PI head is used: zero test-time overhead, "minimal training time overhead".
  - Justification: p(y|x) minimises E[KL(p(y|x,a) ‖ q(y|x))].
  - Source: [Collier et al. 2022](https://arxiv.org/abs/2202.09244) ([PMLR](https://proceedings.mlr.press/v162/collier22a.html))
- **PI tower** ([same paper, App. D](https://arxiv.org/abs/2202.09244)):
  - ImageNet: ψ is a 2-layer MLP, 128 units, ReLU. The first layer takes only a; the second takes [layer-1 output, φ(x)]; its output is concatenated with φ(x) before the output layer.
  - CIFAR-10H: 64 units on a MobileNet + Dense(64) feature.
  - PI preprocessing: annotator ID one-hot; reaction time and annotator experience as one-hot decile IDs.
  - One-step and two-step training perform equivalently on ImageNet (App. E).
- **Results**, NLL / accuracy (%):

  | Setting | No PI | Full marginalisation | TRAM | Het-TRAM | Distillation (Lopez-Paz) | Distillation without PI | Distilled-TRAM |
  |---|---|---|---|---|---|---|---|
  | CIFAR-10H (MobileNet pretrained; 10k train images, ~1.64 labels/image, labels ≈79% correct; 20 runs) | 1.058 / 67.0 | 1.119 / 70.3 | 0.980 / 70.1 | 0.972 / 70.4 | 1.121 / 70.2 | 1.118 / 70.1 | 0.941 / 71.8 |
  | ImageNet relabelled by 16 pretrained models + a 10% random "malicious" annotator (labels 68.3% correct; PI = model ID + probability of the sampled label; ResNet-50 from scratch) | 1.264 / 71.7 | 1.217 / 72.6 | 1.225 / 72.5 | 1.207 / 72.8 | 1.216 / 72.7 | – | 1.154 / 73.8 |

  - CIFAR-10H, other baselines: mean imputation 0.963 / 70.1.
  - ImageNet, imputation baselines were *worse* than no PI: zero imputation 63.5%, mean imputation 65.1%.
  - Source: [Collier et al. 2022, Tables 2–3](https://arxiv.org/abs/2202.09244)
- **Multi-label evidence.** Civil Comments Identities: 7 binary toxicity labels, PI = 24 identity attributes, text encoder.
  - NLL / mean accuracy: no PI 0.085 / 97.8; TRAM 0.064 / 98.2; Het-TRAM 0.062 / 98.1.
  - Lopez-Paz distillation 0.089 / 97.8 (no gain). Full marginalisation 0.065 / 97.8.
  - Marginalisation was poor here because the PI is derived from x, which violates the p(a|x) ≈ p(a) assumption.
  - Source: [Collier et al. 2022, Table 4](https://arxiv.org/abs/2202.09244)
- **Stated limits** ([App. F, H](https://arxiv.org/abs/2202.09244)):
  - TRAM needs PI that is predictive of y given x. Removing the confidence feature shrinks the gain.
  - It needs a model with enough capacity to overfit the noisy samples. Gains drop in an underfitting (¼-capacity) ResNet-50.
  - Gains fall as I(y;a|x) falls.
- **Variants:**
  - Het-TRAM: heteroscedastic heads.
  - Distilled-TRAM: L₁ uses a PI teacher's soft labels; two models, two steps.
  - Approximate Full Marginalisation (AFM): at test, average the PI head over S PI vectors sampled from the training set. Cost O(S) heads. Assumes p(a|x) = p(a).
  - Source: [Collier et al. 2022, §3.3, Table 1](https://arxiv.org/abs/2202.09244)

#### "When does Privileged Information Explain Away Label Noise?" and ImageNet-PI (Ortiz-Jimenez, Collier, et al.) [PR, ICML 2023, PMLR 202]
- **ImageNet-PI construction:**
  - ILSVRC-2012 relabelled by 16 tf.keras pretrained models (ResNet50V2 … Xception).
  - Each label is *sampled* from a temperature-scaled predictive distribution. The temperature is drawn from Gamma(α = 1, β); β = 0.1 gives high noise, β = 0.5 low noise.
  - PI = each model's confidence on its sampled label, the model's parameter count, and its clean test accuracy.
  - Agreement with clean labels: 16.2% (high noise), 51.9% (low noise).
  - Released files include 16 labels and 16 confidences per image, plus annotator features.
  - Data at [google-research-datasets/imagenet_pi](https://github.com/google-research-datasets/imagenet_pi) (link from the paper, not opened). Code in [uncertainty-baselines](https://github.com/google/uncertainty-baselines).
  - Source: [Ortiz-Jimenez et al. 2023, §2.1, App. A](https://arxiv.org/abs/2303.01806) ([PMLR](https://proceedings.mlr.press/v202/ortiz-jimenez23a.html))
- **Protocol:**
  - One label sampled per example. Early stopping and tuning on a *noisy* held-out split; 5 seeds.
  - Agreement with clean labels: CIFAR-10H "worst" 35.4% (uniform 95.1%); CIFAR-10N worst 59.8% (uniform 82.6%); CIFAR-100N 59.8%.
  - CIFAR-N PI is coarse: annotation times and IDs averaged over batches.
  - Source: [same paper, §2](https://arxiv.org/abs/2303.01806)
- **Table 1 (high noise): accuracy by PI type.** Rows are methods; columns are datasets. Each cell gives Original / Indicator / Labels / Near-optimal PI. The no-PI baseline is one number per dataset (top row).

  | Method | CIFAR-10H worst | CIFAR-10N worst | CIFAR-100N | ImageNet-PI high |
  |---|---|---|---|---|
  | No PI | 55.0 | 80.6 | 60.4 | 47.7 |
  | TRAM | 64.9 / 63.3 / **38.3** / 67.8 | 80.5 / 87.9 / **48.9** / 89.3 | 59.7 / 62.4 / **34.9** / 67.4 | 53.3 / 53.6 / **41.0** / 56.5 |
  | AFM | 64.0 / 66.7 / **29.5** / 74.4 | 82.0 / 91.2 / **22.6** / 92.0 | 60.0 / 66.4 / **20.1** / 70.2 | 55.6 / 55.3 / **0.8** / 58.3 |
  | Distillation (PI) | 45.4 / 49.9 / 44.5 / 48.2 | 80.2 / 80.1 / 80.7 / 80.2 | 61.1 / 61.8 / 60.5 / 61.5 | 51.0 / 50.6 / 39.0 / 27.5 |

  - "Labels" means the PI is the one-hot noisy label itself.
  - "Near-optimal" means PI = [noise indicator, the annotator's label only on mislabeled examples].
  - Source: [Ortiz-Jimenez et al. 2023, Table 1](https://arxiv.org/abs/2303.01806)
- **Low-noise reproduction** (App. C, Table 3):

  | Dataset | No PI | TRAM (Original) | AFM (Original) | TRAM (Indicator) | AFM (Indicator) | TRAM / AFM (Labels) |
  |---|---|---|---|---|---|---|
  | CIFAR-10N uniform (~17% noise, the closest to our noise level) | 88.3 | 88.7 | 88.6 | 92.2 | 93.0 | 54.2 / 37.4 |
  | CIFAR-10H uniform | 85.8 | 85.1 | 85.9 | – | – | – |
  | ImageNet-PI low | 68.3 | 69.7 | 70.6 | – | – | 55.9 / 24.9 |

  - Distillation (PI) was below no-PI in all three: 81.0, 88.1, 67.2.
  - Source: [same paper, App. C](https://arxiv.org/abs/2303.01806)
- **Clean validation set** for early stopping and tuning (App. D.2, Table 5): most methods improve and the findings hold.
  - CIFAR-10H worst: no PI 53.2, TRAM 67.7, AFM 70.6.
  - ImageNet-PI: 48.3 / 53.3 / 55.6.
  - With *no* early stopping, the relative PI gains are larger.
  - Source: [same paper, App. D](https://arxiv.org/abs/2303.01806)
- **Mechanistic findings** ([§3–4](https://arxiv.org/abs/2303.01806)):
  - PI is most helpful when it (i) separates clean from mislabeled examples *and* (ii) offers a shortcut to memorise the mislabeled ones.
  - PI that is too predictive of the label starves φ(x): the PI head fits everything.
  - Wider PI towers help.
  - Test accuracy is U-shaped in feature-extractor size: bigger backbones overfit with x instead of using the PI.
  - A unique random vector per example ("random PI") enables positive shortcuts.
- **TRAM++** = TRAM + random PI appended to the original PI + a tuned no-PI loss weight λ ([App. B, F](https://arxiv.org/abs/2303.01806)):
  - Loss: ℓ(π(φ,a), ỹ) + λ·ℓ(ψ(φ), ỹ).
  - CIFAR sweep: λ ∈ {0.1, 0.5}, PI tower width ∈ {512, 1024, 2048, 4096}, random-PI length ∈ {8, 14, 28}.
  - ImageNet: λ = 0.5, length 30, width 2048.
  - PI tower: [preprocessed PI → Dense+ReLU] concatenated with φ(x) → Dense+ReLU → residual → concatenated with φ(x).
  - A sweet spot exists at λ < 1, where the no-PI head fits clean examples but not all noisy ones.
- **Combinations** (Table 2, accuracy):

  | Method | CIFAR-10H worst | CIFAR-10N worst | CIFAR-100N | ImageNet-PI high |
  |---|---|---|---|---|
  | No PI | 55.0 | 80.6 | 60.4 | 47.7 |
  | TRAM | 64.9 | 80.5 | 59.7 | 53.3 |
  | TRAM++ | 66.8 | 83.9 | 61.1 | 53.9 |
  | SOP | 59.2 | 87.9 | 65.3 | – |
  | **TRAM+SOP** | **70.9** | **88.5** | **66.1** | – |
  | HET | 50.8 | 81.9 | 60.8 | 51.5 |
  | TRAM+HET | 67.7 | 82.0 | 62.1 | 55.8 |

  - TRAM+SOP recipe: pre-train with TRAM++, then fine-tune the no-PI side with the SOP loss *without* stop-gradients, 120 epochs. Their own SOP implementation, below the original SOP paper's numbers.
  - SOP was not run on ImageNet-PI ("cannot be easily scaled").
  - Source: [same paper, §5, App. I–J](https://arxiv.org/abs/2303.01806)

#### Pi-DUAL (Wang, Ortiz-Jimenez, Jenatton, Collier, Kokiopoulou, Frossard) [PR, ICML 2024, PMLR 235]
- **Model.** Logits: h(x,a) = [1 − γ_ψ(a)]·f_θ(x) + γ_ψ(a)·ε_φ(a).
  - f_θ is the prediction network (image only); ε_φ is the noise network (PI only); γ_ψ is a sigmoid gate (PI only).
  - Single loss: L(softmax(h), ỹ). f is *never* forced to fit every ỹ.
  - PI is augmented with a random per-sample ID; its length is the *only* extra hyperparameter.
  - At inference only f_θ(x) is used.
  - Source: [Wang et al. 2024, §3.2](https://arxiv.org/abs/2310.06600) ([PMLR](https://proceedings.mlr.press/v235/wang24bb.html))
- **Architecture details** ([App. B](https://arxiv.org/abs/2310.06600)):
  - Noise network: 3-layer ReLU MLP, width 1024 (CIFAR-10H), 2048 (CIFAR-N, ImageNet-PI).
  - Gate: shares the first layer with the noise network, then two FC ReLU layers and a sigmoid.
  - No weight decay on noise and gate networks on CIFAR.
  - Random-PI length searched in {4, 8, 12, 16} on CIFAR; 30 on ImageNet-PI.
- **Results** (Table 2; early stopping on noisy validation):

  | Method | CIFAR-10H worst | CIFAR-10N worst | CIFAR-100N | ImageNet-PI low | ImageNet-PI high |
  |---|---|---|---|---|---|
  | Cross-entropy | 51.1 | 80.6 | 60.4 | 68.2 | 47.2 |
  | ELR | 48.5 | 86.6 | 64.0 | – | – |
  | HET | 50.8 | 81.9 | 60.8 | 69.4 | 51.9 |
  | SOP | 51.3 | 85.0 | 61.9 | – | – |
  | TRAM | 64.9 | 80.5 | 59.7 | 69.4 | 54.0 |
  | TRAM++ | 66.8 | 83.9 | 61.1 | 69.5 | 53.8 |
  | AFM | 64.0 | 82.0 | 60.0 | 70.3 | 55.3 |
  | **Pi-DUAL** | **71.3 ± 3.3** | 84.9 | **64.2** | **71.6** | **62.1** |

  ELR and SOP were not run on ImageNet-PI: they would need about 1B extra per-sample parameters. Source: [Wang et al. 2024, Table 2](https://arxiv.org/abs/2310.06600)
- **Ablations** (Table 4):

  | Variant | CIFAR-10H worst | CIFAR-10N worst | CIFAR-100N | ImageNet-PI low | ImageNet-PI high |
  |---|---|---|---|---|---|
  | No gate | 61.5 | 84.5 | 59.0 | 67.9 | 47.8 |
  | No noise network | 59.7 | 82.4 | 59.7 | 71.6 | 62.3 |
  | Gate in probability space | 62.2 | 81.6 | 59.4 | 71.0 | 60.4 |
  | **Only random PI** | 53.5 | 83.7 | 61.8 | 68.4 | 47.0 |

  - With only random PI, the result is ≈ CE on ImageNet-PI and CIFAR-10H, with +2–3 points on CIFAR-N.
  - Training Pi-DUAL with TRAM's loss: CIFAR-10H 61.6 vs 71.3; CIFAR-10N 81.6 vs 84.9 (Table 13).
  - Replacing a growing share of PI with random values steadily lowers accuracy (App. C.8; numbers not extracted).
  - Source: [Wang et al. 2024, §5.3, App. C.7–C.8](https://arxiv.org/abs/2310.06600)
- **Noise detection** (AUROC of the prediction network's confidence on the observed label, Table 3):
  - Pi-DUAL: 0.954 (CIFAR-10H), 0.962 (CIFAR-10N), 0.911 (CIFAR-100N), 0.953 / 0.986 (ImageNet-PI low / high).
  - CE: 0.810 / 0.951 / 0.883 / 0.935.
  - Gate thresholding is better with rich PI: 0.982 on CIFAR-10H.
  - It is worse with coarse PI: about 0.81 / 0.73 on CIFAR-10N / 100N [values parsed from a garbled PDF table; treat as approximate].
  - Source: [Wang et al. 2024, §4.3](https://arxiv.org/abs/2310.06600)
- **Cost** on ImageNet-PI with ResNet-50 (Table 16): CE 26M parameters, 0.510 s/step; TRAM++ 32M, 0.541 s; Pi-DUAL 36M, 0.566 s (+11%). It does not need early stopping. Source: [Wang et al. 2024, App. D](https://arxiv.org/abs/2310.06600)
- **Pi-DUAL+** adds label smoothing and a prediction-consistency (KL) regulariser. Source: [App. C.5](https://arxiv.org/abs/2310.06600)
- **No public code** repository was found. Source: [search, Oct 2026](https://proceedings.mlr.press/v235/wang24bb.html)

#### Generalized distillation / LUPI and related [PR]
- **Generalized distillation** (Lopez-Paz, Bottou, Schölkopf, Vapnik) [PR, ICLR 2016]:
  - Teacher f_t is trained on the PI x\*. Soft labels s_i = σ(f_t(x\*_i)/T).
  - Student: argmin Σ (1−λ)·ℓ(y_i, σ(f(x_i))) + λ·ℓ(s_i, σ(f(x_i))), with imitation λ ∈ [0,1] and temperature T > 0.
  - Synthetic results, privileged / regular / distilled accuracy (%): 96 / 88 / 95; 90 / 68 / 70; 98 / 89 / 97; 96 / 55 / 56. These are noiseless-label sample-efficiency gains.
  - Source: [Lopez-Paz et al. 2016](https://arxiv.org/abs/1511.03643)
- **Distillation with a PI teacher ≈ self-distillation** in TRAM (Tables 2–4, above) and in the ICML 2023 study (Table 1, above). Source: [Collier 2022](https://arxiv.org/abs/2202.09244); [Ortiz-Jimenez 2023](https://arxiv.org/abs/2303.01806)
- **SVM+** (Vapnik & Vashist 2009): the slack variables are a function of the PI. Here "slack" means each training sample's permitted constraint violation, i.e. a per-sample allowance for misfitting. So PI models each sample's difficulty or noise; it does not predict y. As restated in [TRAM §4](https://arxiv.org/abs/2202.09244).
- **Heteroscedastic dropout** (Lambert, Sener, Savarese) [PR\*, CVPR 2018]:
  - The Gaussian-dropout variance is a function of the PI (bounding boxes); dropout is removed at test.
  - ImageNet subsets: +5% single-crop top-1 at 75K images, equal at 200K, no gain at 600K. All 25K runs diverged.
  - Source: [Lambert et al. 2018](https://arxiv.org/abs/1805.11614)

#### Heteroscedastic heads (HET, HET-XL) [PR]
- **HET** (Collier, Mustafa, Kokiopoulou, Jenatton, Berent) [PR, CVPR 2021 oral]:
  - Logits u = μ(x) + ε, with ε ~ N(0, Σ(x)) and Σ(x) = V(x)V(x)ᵀ + diag(d(x)²) (low rank R).
  - p_c = (1/S)·Σ_s softmax(u^(s)/τ). Multi-label: temperature-scaled sigmoid per class.
  - Drop-in replacement for the last dense layer (`MCSoftmaxDenseFA`).
  - Hyperparameters:

    | Setting | Rank R | τ | MC samples (train and eval) |
    |---|---|---|---|
    | ImageNet | 15 | 0.9 | – |
    | Multi-label (ImageNet-21k, JFT) | 50 | sigmoid τ = 0.15 | 1,000 |

  - Results: ImageNet 79.3% (+2.6 over a homoscedastic head; +0.6 over a diagonal-only head); ImageNet-21k 47.0% (+1.1); JFT 64.7% (+1.6); WebVision 76.6% top-1 (state of the art then).
  - The diagonal model at τ = 1 (Kendall & Gal) was significantly worse than at τ\* = 0.15.
  - Source: [Collier et al. 2021](https://arxiv.org/abs/2105.10305)
- **HET-XL** [PR, ICLR 2023]:
  - Noise is injected in pre-logit space, W(φ(x) + ε′(x)), so extra parameters do not scale with the number of classes: 14× fewer extra parameters.
  - τ is learned as an ordinary training parameter. This was validated on 12.8M–4B-image, few-epoch regimes where overfitting τ is unlikely.
  - Source: [Collier et al. 2023](https://arxiv.org/abs/2301.12860)

#### Medical / radiology evidence for PI
- **TRAM on chest X-ray ARDS with multi-annotator scores as PI** (Gao, Wittrup, Najarian) [PR\*, Bioengineering 2024]:
  - Data: 3,055 CXRs; 14 physicians, 2–4 reads per image, 1–8 scale, κ ≈ 0.37.
  - Architecture: ResNet-50; PI branch = 64-unit encoder + 2×128 predictor with an ordinal head; TRAM stop-gradient; loss L1 + 0.5·L2.
  - Key trick: **PI set to zero unless the case's uncertainty is ≥ 2** (training median).
  - Test AUROC: fine-tuning 0.856; "Confusion Estimation" crowd baseline 0.871; **original TRAM 0.842 (worse than fine-tuning)**; thresholded TRAM 0.872; proposed 0.873.
  - A threshold sweep was best at 2.5. Test labels are mean-aggregated (no gold).
  - The abstract's "0.850" appears to be an error (it equals the linear-probe baseline).
  - Source: [Gao et al. 2024, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10885868/)
- **Radiology reports as PI** (Bergström, Yue, Johansson) [PP, Oct 2025]:
  - Setup: generalized distillation from an image + report teacher (λ = 0.75).
  - It *did not beat self-distillation* when the label is well represented in the text (MIMIC 5×1200 diagnostic labels).
  - It helped prognostic tasks (12-month readmission).
  - The authors tie this to "PI too predictive of the label".
  - **NEEDS-TEXT.**
  - Source: [Bergström et al. 2025](https://arxiv.org/abs/2510.24385)
- **LLM report labels vs a rule-based labeler** (Khosravi et al.) [PR\*, J Imaging Inform Med 2025]:
  - Fracture sensitivity: Llama-3 95% vs CheXpert labeler 51%.
  - ConvNeXt trained on LLM labels: external fracture AUC 0.84 against human labels.
  - Classifiers were "fairly tolerant" of label noise; the choice of test-label scheme changed measured AUC materially.
  - Source: [Khosravi et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12920854)

### Inferences
- **The mechanism is memorisation routing, not information transfer.** Every positive result comes from a PI path that absorbs the hard, wrong labels so the image features do not. Distillation-style transfer of PI never beat self-distillation. This explains our failed teacher distillation: it was a transfer-style method.
- **Our noise is milder than the benchmarks.** Our report-vs-gold disagreement is roughly 10–20% of cells per finding ([team] prevalence gaps), versus 40–84% noise in the "worst"/high-noise benchmarks.
  - The closest benchmark is CIFAR-10N uniform (~17% noise). There, TRAM with real, coarse PI gave +0.4; with a good noise indicator it gave +3.9; and labels-as-PI collapsed it.
  - Expect PI gains to depend almost entirely on whether our PI flags the gold-vs-report errors.
- **The medical TRAM paper is the most transferable data point.**
  - Raw multi-rater PI made TRAM *worse* than plain fine-tuning (0.842 vs 0.856 AUROC).
  - Giving PI only on uncertain cases made it better (0.872).
  - This is the practical version of the "near-optimal PI" finding.
- **None of these papers report ROC-AUC under label noise**, except Gao 2024, which used mean-aggregated test labels. All gains are top-1 accuracy. The transfer to a rank metric is unproven.
- **HET is cheap for us** (K = 12) but mixed on small data (50.8 vs 55.0 on CIFAR-10H worst). Treat it as an add-on to TRAM or Pi-DUAL, not a standalone fix.

### Gaps
- No 2025–2026 peer-reviewed successor to Pi-DUAL or TRAM was found. The search index may be incomplete for 2026.
- No PI method has been benchmarked with soft (probabilistic) targets or with multi-instance (MIL) inputs.
- The Pi-DUAL corrupted-PI table (App. C.8, Table 14) and the HET multi-label gAP table were not extracted numerically.
- The ImageNet-PI repository and uncertainty-baselines code were not opened. TRAM, HET and HET-XL are TF/JAX implementations. A PyTorch port is needed [unverified whether one exists].

---

## Q2. Our PI candidates, how to feed them, and which method fits when PI explains report noise rather than true signal

### Takeaway
- **Best fit: Pi-DUAL**, then TRAM++ (both adapted to 12 sigmoid outputs). Pi-DUAL is the only method that models "this cell's label is explained by the annotation process" and trains the image head on the clean distribution.
- **What not to feed: the 31-table vector or its mean, for all cells.** That is exactly the "labels as PI" setting, where TRAM and AFM collapse (e.g. AFM 0.8% on ImageNet-PI).
- **What to feed instead:**
  - label-agnostic noise indicators: table disagreement and a sign-free dissent pattern;
  - reporting-style metadata;
  - a random per-study ID;
  - optionally, votes only on ambiguous cells, as in the thresholded TRAM of Gao et al. 2024.
- **Run one gold-58 diagnostic first:** does the PI predict report-vs-gold disagreement? Our hard-mask result already hints that disagreement alone is a weak indicator.

### Cited Findings
- **Labels as PI collapses TRAM and AFM in every dataset, at high and low noise:**
  - TRAM: 38.3 vs 55.0 (CIFAR-10H worst); 48.9 vs 80.6 (CIFAR-10N worst); 54.2 vs 88.3 (CIFAR-10N uniform).
  - AFM: 0.8 vs 47.7 (ImageNet-PI high).
  - Distillation (PI) does not collapse but does not gain (80.7 vs 80.6).
  - Source: [Ortiz-Jimenez et al. 2023, Tables 1 and 3](https://arxiv.org/abs/2303.01806)
- **A clean/noisy indicator as PI gives large gains; "near-optimal" PI gives more:**
  - Near-optimal = indicator + the label shown only on mislabeled examples.
  - CIFAR-10N uniform: TRAM 92.2 (indicator) and 93.7 (near-optimal), vs 88.3 no-PI.
  - Source: [Ortiz-Jimenez et al. 2023, App. C](https://arxiv.org/abs/2303.01806)
- **Thresholded multi-rater PI** (zero vector for low-uncertainty cases) turned TRAM from −0.014 to +0.016 AUROC vs fine-tuning on chest X-ray. Source: [Gao et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10885868/)
- **PI quality drives Pi-DUAL:**
  - Random-only PI gives ≈ CE on ImageNet-PI (47.0 vs 47.2) and +3.1 on CIFAR-10N worst.
  - Gate detection AUROC is about 0.98 with rich PI vs about 0.73–0.81 with coarse PI.
  - Source: [Wang et al. 2024, Tables 3–4](https://arxiv.org/abs/2310.06600)
- **AFM assumes p(a|x) ≈ p(a).** It did worst on Civil Comments, where the PI is derived from x. Source: [Collier et al. 2022, §5.3](https://arxiv.org/abs/2202.09244)
- **ImageNet-PI's useful PI** was per-annotator confidence on the sampled label plus annotator quality (accuracy, parameter count). In TRAM's ImageNet, dropping the confidence feature shrank but did not erase the gain. Source: [Ortiz-Jimenez et al. 2023, App. A](https://arxiv.org/abs/2303.01806); [Collier et al. 2022, App. H](https://arxiv.org/abs/2202.09244)
- **[team] facts** that shape the design ([`DEEP_RESEARCH_PROMPT.md` §3](../../DEEP_RESEARCH_PROMPT.md); [earlier note, Q1–Q2](noisy_labels_auc_ensembling.md)):
  - **Prevalence shift, gold vs train-report positive rate (%):**
    - Under-called by reports (omissions): ACL 41 / 23, lateral meniscus 40 / 23, fracture 31 / 18, effusion 60 / 50, synovitis 47 / 39, contusion 33 / 27.
    - Over-called against the severity threshold: medial OA 26 / 37, lateral OA 19 / 29, Baker's 21 / 31, PF OA 36 / 44.
  - **Gold-58 AUC:** LLM-table mean 0.899; public image teacher 0.932; 0.6 image + 0.4 report 0.939–0.941; training on those mixed targets 0.918.
  - **Masking ablations:** masking cells with mean 0.3–0.7 was neutral; masking teacher/report disagreements was neutral.
  - **Pipeline:**
    - per-finding attention-MIL with 12 queries;
    - 12 random windows per step;
    - BCE with soft labels, OneCycle, 16 epochs;
    - SWA of the top 3 epochs, selected on gold-58;
    - DICOMs stripped to 86 tags (sex present, age removed);
    - multilingual reports.
- **Effectively one noisy observation.** When every annotator reads the same report, aggregation removes LLM parsing errors but not report-level errors (omissions, hedges, threshold mismatch). [Earlier note, Q2](noisy_labels_auc_ensembling.md)

### Inferences

#### Mapping our PI onto the published taxonomy [judgement]

| Our candidate PI | Analogue in the papers | Label leakage | Needs report text? | Use |
|---|---|---|---|---|
| G1. Per-cell disagreement d_ik = 4·m(1−m) or table std; share of NaN/"uncertain" outputs if tables have them | "Indicator" (imperfect proxy) | Low: symmetric in m, carries no sign | No | Yes: gate input |
| G2. Sign-free dissent pattern: which tables (or LLM families) disagree with the majority, per finding; plus per-table gold-58 accuracy as annotator features | Annotator ID + annotator accuracy (CIFAR-10H, ImageNet-PI) | Low–medium: magnitude, not sign | No | Yes: gate input |
| G3. Reporting-style metadata: site/institution, vendor, field strength, sex (from the 86 DICOM tags); report language only if it is a provided column | Annotator/process features | Low | Language only, if not a column (**NEEDS-TEXT** otherwise) | Yes: noise and gate input |
| G4. Random per-study ID r_i ∈ ℝ^L, L ∈ {4, 8, 16, 30}, fixed per run | TRAM++ / Pi-DUAL random PI | None | No | Yes: noise input |
| G5. 31-table votes (or m) for **all** cells | "Labels" | **Total** | No | **Never** feed to the PI head or noise network |
| G6. Votes only on ambiguous cells (d ≥ δ, e.g. m ∈ [0.15, 0.85]), else zeros + indicator | Gao 2024 threshold / "near-optimal" | Partial (only on ambiguous cells) | No | Optional variant; likely ≈ our neutral hard mask |
| G7. Severity words, hedges, report length | Rich process PI | Medium | **NEEDS-TEXT** | Not with the current policy |

- **Metadata is not truly privileged.** Site and scanner tags exist at test time. Routing them to the PI or noise branch, not the image head, is deliberate: site-specific reporting habits (such as terse reports that omit ACL tears) are label noise. If fed to f(x), the model would *reproduce* each site's under-reporting at test.

#### Recipe A (first PI experiment): Pi-DUAL for 12 sigmoid outputs, our attention-MIL
```python
# z: [B,12,D] per-finding pooled embeddings (12 attention queries); m: [B,12] soft labels (mean of 31 tables)
f  = cls(z)                                   # [B,12] existing per-finding heads = prediction network (kept at test)
h0 = relu(pi_in(cat([a_G1_G3, r_G4], 1)))     # a: label-agnostic PI; r: fixed random ID (len L)
e  = noise_mlp(h0)                            # 2 more Dense+ReLU (3-layer total), width W -> [B,12]
g  = sigmoid(gate_mlp(h0))                    # shares first layer; 2 FC+ReLU + Dense -> [B,12] per-finding gates
logit = (1 - g) * f + g * e                   # mix in LOGIT space (probability-space mixing was worse)
loss  = bce_with_logits(logit, m)             # single loss; no stop-gradient
# inference: sigmoid(f) only, TTA unchanged
```
- **Hyperparameters [judgement, starting from the paper's]:**
  - W ∈ {256, 1024}; L ∈ {8, 16, 30}; no weight decay on the gate and noise MLPs.
  - Give the gate and noise MLPs about 10× the head learning rate. Reason: we train 16 epochs, so each random ID is seen only about 16 times, versus 90 epochs in Pi-DUAL.
- **Guardrails [judgement]:**
  - Log the mean gate per finding. If it exceeds about 0.3, or the train AUC of f vs m collapses, the PI is leaking the label.
  - Fix: remove G2/G6, or add a small penalty β·mean(g) [my addition, unverified].
- **Diagnostics on gold-58:**
  - AUROC of g_ik, and of f's confidence on the observed label (Pi-DUAL §3.3), for the cell noise indicator 1[(m_ik ≥ 0.5) ≠ gold_ik].
  - Expect higher gates on ACL / lateral meniscus / fracture if omissions are being absorbed.

#### Recipe B: TRAM++ for 12 outputs
```python
pi_logits   = pi_tower(a, r, z)      # Dense+ReLU(a,r) -> concat z_k -> Dense+ReLU (+residual) -> concat z_k -> 1 logit per finding
nopi_logits = cls_nopi(z.detach())   # test-time head on stop-gradient features
loss = bce_with_logits(pi_logits, m) + lam * bce_with_logits(nopi_logits, m)   # lam in {0.1, 0.5}
```
- The tower is shared across findings, with a finding embedding.
- The no-PI head still targets the *noisy* marginal, but on noise-protected features.
- Only after the TRAM++ run: fine-tune the no-PI side with SOP and no stop-gradient (TRAM+SOP).

#### Which method fits "PI explains report noise, not true signal" [judgement]
1. **Pi-DUAL.** It explicitly assigns noise to PI and trains f on the clean distribution. Best on model-annotated ImageNet-PI. One extra hyperparameter. About +11% step cost.
2. **TRAM++.** Protects features, but its test head still fits p(ỹ|x).
3. **AFM.** Avoid: our PI (disagreement) depends on x (mild findings), which violates its independence assumption.
4. **Distillation from a report teacher.** Avoid: it is our failed experiment, there is no gain in any benchmark, and it needs report text if the teacher reads text.

#### What these methods cannot fix [judgement]
- **Severity-threshold over-calls** (the OA findings and Baker's) are *systematic*: a mild-OA image is easy to fit as "positive", so neither the gate nor random-ID memorisation will route it to the noise path. PI methods mainly help *omission* noise, where the image contradicts the label.
- The threshold shift needs target remapping, e.g. ambiguous threshold-finding cells mapped to about 0.2, or severity grades ([earlier note, Q1](noisy_labels_auc_ensembling.md)). The severity-grade option is **NEEDS-TEXT**.
- **Our neutral hard-mask** (cells with mean 0.3–0.7) is close to "Pi-DUAL with a fixed indicator gate". So gains, if any, must come from the random-ID path and metadata, which act on *confident* cells. That is where the omission noise lives (all 31 tables say 0).

### Gaps
- Unknown whether the 31 tables contain "uncertain" or NaN outputs, or probabilities (which would be ImageNet-PI-style confidences).
- Unknown whether report language exists as a non-text column.
- Unknown which DICOM tags identify the site.
- PI quality (AUROC of G1–G3 for report-vs-gold disagreement on gold-58) has not been measured. This decides whether Pi-DUAL is worth a run.
- No published test of PI methods with soft targets or with per-finding gates.

---

## Q3. Instance-dependent-noise methods: which work with soft labels, 12 binary outputs, 2 studies × 12–24 windows per batch, and 4.3k studies?

### Takeaway
- **SOP** (per-sample sparse noise parameters) is the only IDN method with strong real-noise evidence that is cheap and easy to adapt to binary soft labels: about 104k extra parameters for us. It also combines with TRAM.
- **CORES²** adapts but works by discarding high-loss cells, which risks dropping subtle true positives.
- **PTD and ILFC** need anchor points or per-instance confidence scores.
- **CausalNL** needs a generative model of the image (infeasible for 3D MRI).
- **SEAL** is a self-distillation loop, the family we already tried.
- **Identifiability:** no method can separate report error from image truth using the 31 tables alone.

### Cited Findings
- **Part-dependent noise, PTD** (Xia et al.) [PR, NeurIPS 2020]:
  - The instance transition T(x) is a weighted combination of part-level transition matrices; parts come from a feature decomposition; matrices are learned with anchor points.
  - Clothing1M: PTD-R-V 71.67 vs CE 68.88, Forward 69.91, T-Revision 70.97.
  - Source: [Xia et al. 2020](https://arxiv.org/abs/2006.07836)
- **CausalNL** (Yao et al.) [PR\*, NeurIPS 2021]:
  - Structural causal model in which Y and a latent Z cause X; a VAE-style generative model of X with networks q(Z|Y,X), p(Ỹ|Y,X), p(X|Y,Z) (5-layer conv encoder/decoder on Clothing1M).
  - Clothing1M 72.24 vs CE 68.88.
  - At synthetic IDN-50%, the authors state it is "at least over 10% higher than the best baseline". The paragraph covers F-MNIST, SVHN and CIFAR-10; CIFAR-100 is discussed separately.
  - Source: [Yao et al. 2021](https://arxiv.org/abs/2109.02986)
- **Confidence-scored IDN, ILFC** (Berthon et al.) [PR\*, ICML 2021]:
  - Each pair carries r_x = P(Y = ȳ | Ȳ = ȳ, X = x).
  - Instance-level forward correction with T_ii(x) = r_x·β_i(x), where β_i(x) = P(Ȳ=i|x)/P(Y=i|x); off-diagonal terms are α_ij(x)·(1 − T_ii(x)).
  - Needs anchor points.
  - Clothing1M with r_x from a ResNet-18 trained and calibrated on the *curated clean subset* (~50K): ILFC 73.35 vs Forward 60.62, LQ 67.65, Co-teaching 70.11 (all ResNet-18).
  - Source: [Berthon et al. 2021](https://arxiv.org/abs/2001.03772)
- **CORES²** (Cheng, Zhu et al.) [PR, ICLR 2021]:
  - CE plus confidence regulariser ℓ_CR = −β·E_{Ỹ}[ℓ(f(x), Ỹ)], with a sample sieve that drops examples whose regularised loss exceeds a closed-form threshold.
  - Assumes Bayes-optimal labels equal clean labels.
  - Clothing1M 73.24 vs CE 68.94.
  - CIFAR-10N: aggregate 95.25, worst 91.66. CIFAR-100N: 55.72 (vs CE 55.50).
  - Source: [Cheng et al. 2021](https://arxiv.org/abs/2010.02347); CIFAR-N numbers from [SOP Table 4](https://arxiv.org/abs/2202.14026)
- **SOP: sparse over-parameterisation** (Liu, Zhu, Qu, You) [PR\*, ICML 2022]:
  - Per sample: s_i = u_i⊙u_i⊙y_i − v_i⊙v_i⊙(1 − y_i), with u, v ∈ [−1, 1]^K, separate learning rates α_u, α_v, and no weight decay.
  - θ and u are trained with CE(φ(f(x) + s_i), y_i); v is trained with MSE (CE cannot train v); f is projected to one-hot in the MSE term.
  - Assumes sparse noise that is incoherent with the clean-data function.
  - Results:

    | Benchmark | SOP | CE / comparison |
    |---|---|---|
    | Clothing1M | 73.5 | ELR 72.9, CORES² 73.2 |
    | WebVision / ILSVRC12 | 76.6 / 69.1 | – |
    | CIFAR-10N worst (SOP+) | 93.24 | CE 77.69 |
    | CIFAR-100N (SOP+) | 67.81 | CE 55.50 |

  - Fastest training time among the baselines.
  - Source: [Liu et al. 2022](https://arxiv.org/abs/2202.14026)
- **SEAL** (Chen et al.) [PR\*, AAAI 2021]:
  - Proves real-world noise (Clothing1M) is unlikely to be class-conditional.
  - Retrains on epoch-averaged softmax predictions ("self-evolution average labels").
  - Clothing1M 70.63 vs their CE 69.07.
  - Source: [Chen et al. 2021](https://arxiv.org/abs/2012.05458)
- **Universal probabilistic IDN model** (Wang et al.) [PR\*, AAAI 2021]:
  - Instances are "confusing" (label drawn from an instance-dependent confusion distribution) or "unconfusing" (label = truth); alternating optimisation. A no-PI cousin of Pi-DUAL's gate.
  - Clothing1M: 74.02; 77.55 with +50k clean images in training; 80.68 with clean fine-tuning (vs Forward + fine-tune 80.38).
  - Source: [Wang et al. 2021](https://arxiv.org/abs/2101.05467)
- **Identifiability** (Liu, Cheng, Zhang) [PR\*, venue not re-checked]:
  - With i.i.d. noisy labels, three informative labels that are *conditionally independent given (X, Y)* are sufficient and necessary to identify an instance-level T(X).
  - Successful IDN methods replace that requirement with extra assumptions.
  - Source: [Liu et al. 2022/23](https://arxiv.org/abs/2202.02016)

### Inferences

#### Applicability to our setting [judgement]

| Method | Soft targets | 12 sigmoids | Small batch | N = 4.3k | Extra compute | Verdict |
|---|---|---|---|---|---|---|
| SOP (binary adaptation) | Yes (sign rule below) | Yes | Yes, per-sample parameters | Yes, 104k parameters | ≈0 | **Try (rank 1–2)** |
| TRAM+SOP / Pi-DUAL+SOP | Yes | Yes | Yes | Yes | ≈ +10% | Try after the single runs |
| CORES² | Awkward: the sieve assumes hard labels | Yes | Yes | Yes | ≈0 | Low: drops subtle positives |
| ILFC (r_x from a gold-calibrated model) | Via E over ȳ ~ Bernoulli(m) | Yes, binary T is 2×2 | Yes | Yes | ≈0 | Low: r_x would rest on 58 gold |
| PTD | Possible | Yes | Yes | Anchors uncertain | Moderate | Low: anchor points under report noise are unreliable |
| CausalNL | – | – | – | – | Generative 3D MRI model | **Skip** |
| SEAL | Yes | Yes | Yes | Yes | 2+ trainings | **Skip**: same family as our failed pseudo-label loops |
| Wang 2021 confusing/unconfusing | Yes | Yes | Yes | Yes | ≈0 | Pi-DUAL does the same, with PI |

#### SOP for binary soft labels (my adaptation, unverified)
- For study i and finding k: s_ik = u_ik²·[m_ik ≥ 0.5] − v_ik²·[m_ik < 0.5]; q = clip(σ(f_k) + s_ik, 1e-4, 1 − 1e-4).
  - The positive part absorbs report over-calls; the negative part absorbs omissions.
- Train θ and u with BCE(q, m); train v with MSE(round(σ(f)).detach() + s, m), following SOP's split.
- Initialise u, v ~ N(0, 1e-8), clamp to [−1, 1].
- Use SGD for u, v with learning rate α·lr, α ~ 10–100, because each study is visited only 16 times. SOP's own runs used 120–300 epochs.
- **Per-finding restriction:**
  - Enable only the negative (omission) part, and only on the under-called findings: ACL, lateral meniscus, fracture, effusion, contusion, synovitis.
  - The OA findings and Baker's errors are systematic threshold shifts, not the "sparse, incoherent" noise SOP assumes.
- **What changes at test:** nothing. u and v are discarded.

#### Why identifiability matters here
The 31 tables are conditionally *dependent* given the image truth: they share the report. They are effectively one observation of "the report". By Liu et al., the report→image error then cannot be identified from the tables, and only an image signal or gold can anchor it. PI and SOP-type methods do not need identifiability: they rely on memorisation dynamics. That is why they are the realistic IDN tools here.

### Gaps
- No IDN method has published results on report-derived medical labels, AUC metrics or MIL inputs.
- The SOP and CORES² binary/soft-label adaptations are untested.
- SOP's behaviour with only 16 visits per sample is unknown.
- CausalNL on 2.5D features (a VAE on frozen embeddings instead of pixels) is untested and speculative.

---

## Q4. Using the 58 gold studies as a tiny trusted set for instance-dependent noise without overfitting

### Takeaway
Every published instance-dependent trusted-set method used far more clean data than we have:
- 100 to about 50,000 clean examples, and usually ≥ 10 per class: Veit 40k, MLC 1,000, IEG 10 per class, ILFC about 50K, Wang about 50K.
- We have 58 studies and 9–35 positives per finding.

So use gold for three things only:
1. a pre-registered diagnostic: does the PI predict report-vs-gold disagreement?
2. choosing among at most 2–3 pre-declared variants, with the LB as tie-break;
3. at most a ~10–20-parameter, instance-dependent residual "cleaner" (hengck23's "~10 parameters" hint), with leave-one-study-out validation.

Meta-learned reweighting (L2RW/MW-Net) and meta label correction are too high-variance at this size.

### Cited Findings
- **Label-cleaning network** (Veit et al.) [PR\*, CVPR 2017]:
  - Cleaned labels: ĉ = clip(y + g′(y, f(I)), [0, 1]). This residual on the noisy multi-label vector is conditioned on visual features, so it is instance-dependent; it starts as the identity.
  - Trained with an L1 loss against verified labels. The image classifier h trains on ĉ, with ĉ held constant.
  - Open Images: ~9M images, 6k classes, about 40k clean images.
  - AP_all 83.82 → 87.68; MAP 61.82 → 62.36 (62.38 jointly). Fine-tuning on clean data alone gave 85.88 / 61.53.
  - Source: [Veit et al. 2017](https://arxiv.org/abs/1701.01619)
- **Meta Label Correction, MLC** (Zheng et al.) [PR\*, AAAI 2021]:
  - Label-correction network y_c = g_α(h(x), y′), meta-learned so that a model trained on y_c does well on the clean set.
  - Clean-set sizes: 1,000 (CIFAR-10: 10 × 100; CIFAR-100: 100 × 10).
  - Mean accuracy on CIFAR-10: MW-Net 65.12, GLC 86.62, MLC 86.81.
  - Clothing1M: MLC 75.78, GLC 73.69, MW-Net 73.72, Forward 69.84.
  - Source: [Zheng et al. 2021](https://arxiv.org/abs/1911.03809)
- **Learning to Reweight** (Ren et al.) [PR\*, ICML 2018] takes one meta-gradient step on per-example weights (initialised at 0) to minimise loss on a clean validation set. Source: [Ren et al. 2018](https://arxiv.org/abs/1803.09050)
- **MW-Net** (Shu et al.) [PR, NeurIPS 2019] learns a 1-hidden-layer MLP mapping loss → weight from "a small amount of unbiased meta-data". Source: [Shu et al. 2019](https://arxiv.org/abs/1902.07379)
- **IEG** (Zhang et al.) [PR, CVPR 2020] [snippet]: with **10 trusted examples per class**, CIFAR-100 at 40% noise reached 80.2%, and 75.5% at 80% noise (previous best 48.2%). Source: [Zhang et al. 2020](https://arxiv.org/abs/1910.00701)
- **ILFC** derived r_x from a classifier trained and calibrated on Clothing1M's clean subset and validation set (about 50K curated). Source: [Berthon et al. 2021](https://arxiv.org/abs/2001.03772)
- **A clean validation set for early stopping and tuning** helps most methods and preserves PI gains. Source: [Ortiz-Jimenez et al. 2023, App. D.2](https://arxiv.org/abs/2303.01806)
- **[team]** ([`DEEP_RESEARCH_PROMPT.md`](../../DEEP_RESEARCH_PROMPT.md); [earlier note](noisy_labels_auc_ensembling.md)):
  - Gold-58 noise is about ±0.02 macro and ±0.06 per finding.
  - The 0.6/0.4 OOF-mix targets scored 0.938 "quality" on gold, but the trained model scored 0.918.
  - A forum hint (hengck23): "use 58 samples to train a model with only ~10 parameters … unlikely to overfit".
  - The earlier note judged GLC, L2RW and MW-Net on gold high-variance and recommended skipping them.

### Inferences

#### Gold-58 decision noise [computed, Hanley–McNeil approximation, AUC = 0.93]

| Gold positives per finding | 3 | 5 | 10 | 20 |
|---|---|---|---|---|
| SE of one finding's AUC | 0.104 | 0.080 | 0.057 | 0.041 |
| SE of the 12-finding macro, if independent | 0.030 | 0.023 | 0.017 | 0.012 |

Any gold-based selection among many variants overfits. A difference below about 0.02 macro is noise.

#### Allowed uses of gold [judgement]
1. **PI-quality diagnostic** (pre-registered, one shot):
   - On 696 cells, compute the AUROC of d_ik (G1), the dissent score (G2) and metadata (G3) for noise_ik = 1[(m_ik ≥ 0.5) ≠ gold_ik]. Split it by under-called findings vs threshold findings.
   - Also report the share of noisy cells that are *confident* (m < 0.1 or m > 0.9). If most are, disagreement-based PI cannot flag them.
   - Calibration: in Pi-DUAL a useful gate reached about 0.95–0.98 AUROC and a coarse one about 0.73–0.81. Below about 0.65, expect ≈0 from the PI part.
2. **Residual cleaner with ~15 parameters** (Veit structure, hengck23 budget):
   - c_ik = clip(m_ik + tanh(b_k + w₁·logit(m_ik) + w₂·d_ik + w₃·o_ik + w₄·d_ik·o_ik), 0, 1).
   - o_ik is a *cross-fitted* image logit: 4-fold OOF, never in-sample.
   - Shared weights w plus 12 finding biases. L2 toward the identity. Fit with BCE on gold.
   - Accept only if leave-one-study-out log-loss beats the identity *and* the AUC of c vs gold beats m by more than 0.02. Then train on c.
   - Risk: o_ik carries the report concept, which is exactly why our 0.6/0.4 mix failed.
3. **Instance-level forward correction (ILFC-style):** r_ik is the cleaner's own gold-fit output, P(gold = ȳ | PI, o). Use BCE on T(x)ᵀσ(f). This is more principled than relabelling, but rests on the same ~15 gold-fitted parameters.

#### Not recommended
L2RW, MW-Net, MLC and IEG at 58 studies: their meta-gradients are dominated by a few gold positives per finding. The earlier note reached the same conclusion for the class-conditional variants.

### Gaps
- No paper validates any trusted-set IDN method below about 100 clean samples, or with multi-label medical gold.
- The hengck23 hint is a forum claim [unverified].
- Unknown whether gold-58 is an enriched or random sample. This affects the cleaner's intercepts.

---

## Q5. Multi-annotator learning when the annotators are LLMs reading the same report (strongly correlated errors): crowd models vs PI

### Takeaway
- **Crowd models learn annotator-specific confusions and need an anchor for the shared one.** This covers the crowd layer, Doctor Net, trace-regularised confusions, CoNAL, MaDL, TAIDTM and Max-MIG. Some also assume labels are independent of x given the truth (Max-MIG).
- **Our dominant error is the shared one.** It is the report-vs-image disagreement, common to all 31 tables and instance-dependent. Without gold or image anchoring, these models cannot recover it. Expect ≈ the soft mean, as the earlier note found: mean of 17 tables 0.899 vs best single table 0.904 on gold.
- **PI methods need no identifiability, only a memorisation shortcut and a decent noise indicator.**
- **The only medical head-to-head found is a tie:** thresholded-PI TRAM 0.872–0.873 vs a confusion-estimation crowd baseline 0.871 AUROC.

### Cited Findings
- **Crowd layer** (Rodrigues & Pereira) [PR\*, AAAI 2018]: per-annotator transformation layers on top of the softmax, trained end to end; dropped at test. Source: [Rodrigues & Pereira 2018](https://arxiv.org/abs/1709.01779)
- **Doctor Net** (Guan, Gulshan, Dai, Hinton) [PR\*, AAAI 2018]:
  - Setting: **31 doctors** (diabetic retinopathy), with a separate softmax head per doctor on a shared Inception-v3.
  - Averaging the modelled doctors beat modelling the average doctor:
    - 5-class test error 23.83% → 21.86%;
    - binary AUC 97.11 → 97.28;
    - specificity at 97% sensitivity 79.60 → 81.81.
  - Learned (and image-dependent) averaging weights are added on top: WDN / IWDN.
  - The test set is adjudicated gold.
  - Source: [Guan et al. 2018](https://arxiv.org/abs/1703.08774)
- **Trace-regularised confusions** (Tanno et al.) [PR\*, CVPR 2019]: per-annotator confusion matrices estimated jointly, with tr(Â) minimised. Works with one label per image. Source: [Tanno et al. 2019](https://arxiv.org/abs/1902.03680)
- **CoNAL** (Chu, Ma, Wang) [PR\*, AAAI 2021]:
  - p(ŷ_ir | x_i) = ω_ir·p_{W_g}(·|f(x_i)) + (1 − ω_ir)·p_{W_r}(·|f(x_i)), with ω_ir = σ(u_rᵀv_i) from annotator and instance embeddings.
  - "Common" confusion is shared by all annotators.
  - LabelMe accuracy: 87.12 vs crowd layer 83.27, Max-MIG 85.33, Doctor Net 82.12, majority vote 79.83.
  - Music: 84.06 vs crowd layer 81.46.
  - Column order was parsed from PDF text, so the per-baseline mapping is not fully certain. CoNAL is best in both rows either way.
  - LabelMe has 1,000 AMT-labelled training images, about 2.5 annotators per image, and mean annotator accuracy 69.2%.
  - Source: [Chu et al. 2021](https://arxiv.org/abs/2012.13052)
- **Max-MIG** (Cao, Xu, Kong, Wang) [PR, ICLR 2019]: designed for crowds in which some members make "highly correlated mistakes". It still assumes "the crowdsourced labels and the data are independent conditioning on the ground truth". Source: [Cao et al. 2019](https://arxiv.org/abs/1905.13436)
- **MaDL** (Herde et al.) [PR, TMLR 2023]: annotator embeddings estimate annotator density as a proxy for correlated annotations; uses a weighted loss and optional instance-dependent annotator performance. Source: [Herde et al. 2023](https://arxiv.org/abs/2304.02539)
- **Annotator- and instance-dependent transition matrices**:
  - TAIDTM (Li, Xia, Deng, Ge, Liu) [PP v3 2024; venue not checked]: parameterised by DNNs and transferred from a mixture model to individual annotators. Source: [Li et al.](https://arxiv.org/abs/2306.03116)
  - Sample-wise label fusion (Gao et al. 2022): label-error parameters depend on annotator *and* sample. Source: [Gao et al. 2022](https://arxiv.org/abs/2207.11327)
- **CCEM / GeoCrowdNet** (Ibrahim, Nguyen, Fu) [PR, ICLR 2023]: identifiability guarantees for coupled cross-entropy, plus regularised variants. Source: [Ibrahim et al. 2023](https://arxiv.org/abs/2306.03288); survey: [Ibrahim et al. 2025](https://arxiv.org/abs/2407.06902)
- **Pixel-wise (instance-dependent) annotator confusion in medical segmentation** (Zhang, Tanno et al.) [PR, NeurIPS 2020]: two coupled CNNs; annotators are pushed to be "maximally unreliable" while matching the noisy labels. Source: [Zhang et al. 2020](https://arxiv.org/abs/2007.15963)
- **Correlated labelling functions** (Bach et al., Snorkel structure learning) [PR\*, ICML 2017]: ℓ1-regularised marginal pseudolikelihood learns dependencies between labelling functions; 100× faster, with a quarter of the spurious dependencies. Source: [Bach et al. 2017](https://arxiv.org/abs/1703.00854)
- **Instance-level identifiability** needs three conditionally independent labels. Source: [Liu et al.](https://arxiv.org/abs/2202.02016)
- **Medical PI vs crowd head-to-head:** thresholded TRAM 0.872 / 0.873 vs "Confusion Estimation" 0.871 AUROC (that baseline's exact method was not checked). Source: [Gao et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10885868/)
- **[team]:** mean of 17 tables 0.899, best single table 0.904. The earlier note expected about 0 from DS, Snorkel, GLAD or crowd layers. Source: [earlier note, Q2](noisy_labels_auc_ensembling.md)

### Inferences
- **CoNAL's common-confusion branch is structurally the PI "noise path".** Its ω_ir is an instance gate. With LLM tables, the common branch would model "what reports say vs what images show". But its training signal is only the 31 tables, which agree with each other, so it cannot tell common report error from truth. Pi-DUAL replaces this with a PI-driven gate plus a memorisation path [judgement].
- **Doctor-Net-style per-table heads** (31 × 12 sigmoid outputs, averaged at test) are a cheap auxiliary to try only if a slot is free. Guan's gain came from *independent* doctors grading images. Our annotators are near-duplicates reading one report, so expect a much smaller gain [judgement].
- **Use LLM identity as PI, not as a crowd layer.** That is, the sign-free dissent pattern (G2). It is the ImageNet-PI analogue: annotator ID plus quality features.
- **Max-MIG's assumption** (labels independent of x given the truth) is violated by design: whether a report omits a tear depends on how subtle the tear looks.

### Gaps
- No study trains crowd models on multiple LLM annotators that read the same document, then evaluates against image-based gold.
- The CoNAL, Tanno and Max-MIG numbers come from AMT datasets with about 2.5 annotators per item. Behaviour with 31 near-identical annotators is untested.

---

## Q6. Concrete design and ranked plan for 16–22 Oct 2026 (A100/H100 rentable)

### Takeaway
- **Step 0: cheap diagnostics on gold-58**, about one hour. Is the PI informative, and is the noise in confident cells?
- **Run 1:** SOP-binary, omission-only (no PI).
- **Run 2:** Pi-DUAL with label-agnostic PI and a random ID.
- **Combinations:** only if a single run shows ≥ +0.003 on LB or does not hurt gold: TRAM++→SOP, or a HET head on the better run.
- **Skip:** report-teacher distillation, AFM, CausalNL, PTD, SEAL, crowd layers.
- **Expected gain:** 0 to +0.004 macro per technique [judgement]. Evidence for AUC gains is indirect: accuracy benchmarks and one medical TRAM study.

### Cited Findings
- **Cost per run [team]:** about 28 min per CoAtNet epoch on the 4070 Super, 38–40 min on the L4, 16 epochs. Source: [`DEEP_RESEARCH_PROMPT.md` §3.4](../../DEEP_RESEARCH_PROMPT.md)
- **Method overhead:**
  - Pi-DUAL +11% step time; TRAM++ +6% (ImageNet-PI). Source: [Wang et al. 2024, Table 16](https://arxiv.org/abs/2310.06600)
  - SOP was the fastest of its baselines. Source: [Liu et al. 2022](https://arxiv.org/abs/2202.14026)
  - HET adds MC sampling only at the head. Source: [Collier et al. 2021](https://arxiv.org/abs/2105.10305)
- **Best evidence for each ranked item:**
  - Pi-DUAL: ImageNet-PI with model annotators, +14.9 / +3.4 over CE at high / low noise. Source: [Wang et al. 2024](https://arxiv.org/abs/2310.06600)
  - TRAM+SOP: best combination on all three CIFAR PI datasets. Source: [Ortiz-Jimenez et al. 2023](https://arxiv.org/abs/2303.01806)
  - SOP: best on CIFAR-N, Clothing1M and WebVision. Source: [Liu et al. 2022](https://arxiv.org/abs/2202.14026)
  - Thresholded PI in a medical TRAM: +0.016 AUROC. Source: [Gao et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10885868/)

### Inferences

#### Ranked plan [judgement]
- **0. Diagnostics** (no training; Q4 item 1). Also measure ρ̄ (table error correlation) as in the earlier note.
  - Continue if the PI's AUROC for disagreement is ≥ 0.65 on under-called findings, *or* if ≥ 40% of noisy cells are confident (then the random-ID/SOP path is the lever).
  - Kill the PI-feature part of Pi-DUAL if the AUROC is below 0.6. Keep its random ID.
- **1. SOP-binary, omission-only, on 6 under-called findings** (Q3 recipe). One seed on the 192-slice corpus, same schedule.
  - Accept if gold macro does not drop by more than 0.01 and the gains on ACL / lateral meniscus / fracture point in the right direction. Then submit a blend.
  - Why first: no PI-leakage risk, ≈0 cost, and it targets the largest measured shift.
- **2. Pi-DUAL-ML** (Q2 Recipe A), with PI = G1 + G2 + G3 + G4 and L = 16, W = 512.
  - Log gate statistics. One seed.
  - Variant 2b adds G6 only if 2 is neutral and the diagnostics show that ambiguous cells are noisy.
- **3. Combination of the best of 1 and 2:**
  - TRAM++ pre-train, then SOP fine-tune for 3–4 epochs at low LR (the published TRAM+SOP recipe, shortened [unverified]);
  - or Pi-DUAL + SOP, where SOP acts on f's no-gate path [my construction].
- **4. HET-ML head** on the best run:
  - R = 4 factors over 12 findings, S = 256–1,000 MC samples, τ ∈ {0.15, 0.5}; mean-pool z for V(x), d(x).
  - Score both the MC-mean sigmoid and σ(μ), since rank order can differ.
  - Expect a small gain, or blend diversity.
- **5. Gold-calibrated residual cleaner, or ILFC** (Q4 items 2–3): only if diagnostic 0 shows that PI + OOF predicts gold beyond m. Strict leave-one-study-out.
- **6. Doctor-Net per-table auxiliary heads:** only if a slot is free.
- **Never:**
  - feeding all 31 table votes or m as PI (collapse);
  - report-text teachers (NEEDS-TEXT, and no gain in the literature);
  - AFM (dependence of the PI on x);
  - CausalNL; meta-reweighting on 58 gold.

#### Blend logic [judgement]
Even when a noise-handling model does not beat the baseline alone, it errs differently. So:
- add it to the per-finding rank blend at weight 0.2–0.35;
- weight under-called findings toward the SOP / Pi-DUAL models, and the OA findings toward the baseline;
- decide on the LB (SD about 0.003), not gold-58.

#### How this answers the #3 team's hint [judgement]
"Labels … AI, not by hand. And don't rely on reports alone" is consistent with letting an image model decide *which report labels to trust*. That is what the Pi-DUAL gate, TRAM's representation protection and SOP's per-cell noise do. Mixing image predictions into targets, which we tried, does something else.

### Gaps
- None of the ranked methods has been tested on report-derived MRI labels or with ROC-AUC as the metric. Effect sizes are judgement.
- The A100/H100 speed-up over the L4 for our CoAtNet was not measured here [unverified].
- No public PyTorch reference implementation of Pi-DUAL was found. It is about 30 lines on top of our head; TRAM++ is similar.
