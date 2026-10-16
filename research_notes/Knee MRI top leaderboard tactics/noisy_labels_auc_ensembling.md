# Noisy report-derived labels, AUC-oriented training and ensembling for macro ROC-AUC (RSNA Knee 2026, 6-day push to 0.969)

Scope: what could add at least +0.003 macro AUC in 6 days. Setting:
- Train labels: 31 LLM label tables read from the same multilingual reports.
- Gold and test labels: image-based, with severity thresholds (borderline counts as negative).
- Trusted set: 58 gold studies (macro noise about ±0.02, per finding about ±0.06).
- Metric: macro ROC-AUC over 12 findings.
- Our position: public LB 0.956; LB top is 0.968.

This note does not repeat what `../Knee MRI prior work and tricks/learning_from_reports.md` already covers. That note covers:
- VisualCheXbert and Jain 2021 numbers;
- the CheXpert U-policies and Pham LSR;
- the BoMD benchmark (ELR, NVUM, DivideMix);
- Karimi, Bernhardt, Gold Loss Correction (GLC), Dunnmon/Snorkel;
- Hanley–McNeil standard errors.

Those results are pointed to below, not restated.

Evidence grades used on every bullet:
- **[PR]** peer-reviewed paper
- **[PP]** preprint
- **[KW]** Kaggle winning or top write-up, or a top team's repository
- **[AN]** anecdote: non-winning repository, blog or secondary summary
- **[team]** our own measurement, from `DEEP_RESEARCH_PROMPT.md` §3

Qualifiers:
- **[snippet]** taken from a search-engine summary, not checked against the full text
- **[computed]** my arithmetic or simulation; script in the session scratchpad, `calc/binormal.py`
- **[derivation]** a short mathematical argument given here
- **[unverified]** a claim I could not confirm

---

## Q1. Report-to-image label shift: what newer evidence says, how to encode severity, and whether PU learning helps "not mentioned"

### Takeaway
Reports mostly *under-call* findings. They omit minor findings, and a reader who has found one or two findings tends to stop looking ("satisfaction of search"). Under the host's severity thresholds, reports also *over-call* mild disease.

A classical result (Menon et al. 2015) sets the frame. Label noise that depends only on the true class, including Positive–Unlabeled (PU)-style "not mentioned" noise, **does not change the AUC-optimal ranking**. So only *instance-dependent* noise can be fixed to gain AUC.

That points to two instance-level levers:
1. **Severity- and definition-aware soft targets** for the threshold findings: the three OA findings, effusion, Baker's cyst and the meniscus definition.
2. **Cautious large-loss rejection** of "not mentioned" negatives for the findings reports under-call: ACL, lateral meniscus and fracture.

Blanket PU or soft-negative schemes lost 5–7 mAP on real partial labels.

### Cited Findings

#### Why report labels differ from image labels (beyond VisualCheXbert)
- **Neuroradiology validation study** [PP]: MRI head reports, 126,556 extracted, 5,000 manually labelled, with a 500-image gold standard.
  - Report-derived *binary* normal/abnormal labels against image review: sensitivity 98.7%, specificity 96.6%.
  - *Granular* categories: macro sensitivity **85.1%**, specificity 96.0%, F1 82.8%. The lowest sensitivities were extracranial 60.0%, haemorrhage 69.2% and hydrocephalus 70.0%.
  - Reasons the authors give for report–image disagreement:
    - minor findings are omitted when more important ones are present;
    - reports are tailored to the referrer's question;
    - "satisfaction of search";
    - incidental findings are deliberately left out.

  Source: [Wood et al., arXiv 2007.04226 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2007.04226)
- **Chest X-ray (CXR) studies** [PP]: on 1,000 CXR studies there is "a surprisingly large discrepancy between what radiologists visually perceive and what they clinically report". With flawed reports as ground truth, state-of-the-art NLP "fails to produce high-fidelity labels". The abstract gives no numbers. — [Olatunji et al. 2019, arXiv 1905.02283](https://arxiv.org/abs/1905.02283)
- **Hard 0/1 report labels** push image models toward overconfidence where the reports are weak. The authors propose label correction. — [Olatunji & Yao, arXiv 1910.00673](https://arxiv.org/abs/1910.00673) [PP][snippet]

#### Theory: which label noise can hurt AUC at all
- **Menon et al. 2015** [PR] treat label noise and PU learning as instances of "mutually contaminated distributions". In that setting "one can optimise balanced error and AUC without knowledge of the corruption parameters". — [Menon et al. 2015, ICML/PMLR](https://proceedings.mlr.press/v37/menon15.html)
- **Why the ranking is preserved** [derivation]. Define the flip rates:
  - a = P(report-negative | image-positive)
  - b = P(report-positive | image-negative)

  If a and b do not depend on the study x, then P(report+ | x) = b + (1 − a − b)·P(image+ | x). This is strictly increasing in P(image+ | x) whenever a + b < 1, so the best ranking under report labels is the best ranking under image labels.
  - Corollary: per-finding transition-matrix corrections (GLC, forward or backward correction) and SCAR-type PU class-prior corrections cannot raise *asymptotic* AUC. SCAR ("selected completely at random") means every true positive has the same chance of being mentioned.
  - Only noise whose rate varies with x changes the ranking. Examples: mild OA called positive in the report but negative on the image; a subtle tear left unreported.
- **Symmetric losses** (e.g. sigmoid; the convex "barrier hinge" is symmetric on [−1, 1]) are "advantageous in … BER minimization and AUC maximization from corrupted labels". The paper proves an AUC-consistency condition. — [Charoenphakdee, Lee & Sugiyama 2019, ICML](https://proceedings.mlr.press/v97/charoenphakdee19a) [PR]

#### "Not mentioned" = assumed negative: evidence from weakly-supervised multi-label learning
- **Large Loss Matters (LL)** [PR] treats unobserved labels as negatives ("Assume Negative", AN), which turns the problem into noisy-label learning. Recall that in BCE a confident wrong prediction gives a *large loss*; the method acts on the assumed-negative cells with the largest loss.
  - Variants:
    - LL-R rejects those cells (weight λ = 0).
    - LL-Ct temporarily corrects them (λ = log f / log(1 − f), which flips the BCE target to 1).
    - LL-Cp permanently flips them.
  - Only assumed-negative cells are touched; observed labels are never changed.
  - Schedule: nothing is rejected in epoch 1. The rejection fraction then grows by Δrel per epoch, and Δrel = 0.2 was best.
  - The network learns true positives and negatives first and memorises false negatives "in the middle of the training phase". Most false-negative cells peak in loss after the warm-up epoch (VOC 72.2%, COCO 82.7%).
  - Precision of the modified labels *falls* as epochs pass. Larger Δrel "wrongly modifies the true negatives".

  Source: [Kim et al. 2022, CVPR (arXiv 2206.03740)](https://arxiv.org/abs/2206.03740)
- **LL results on artificial single-positive partial labels (ResNet-50, mAP)** [PR]:
  - COCO end-to-end: Naive AN 64.1 → LL-R 71.0 / LL-Ct 70.5 (full labels 78.0).
  - VOC: 85.1 → 89.2 / 89.0 (full 90.2).

  Source: [Kim et al. 2022](https://arxiv.org/abs/2206.03740) (Table 2)
- **LL results on real partial labels (OpenImages V3, all groups, mAP)** [PR]:

  | Method | mAP |
  |---|---|
  | Naive AN | **82.0** |
  | Weighted-AN (down-weight negatives) | **77.0** |
  | Label-smoothed AN | **75.0** |
  | Curriculum | 77.1 |
  | IMCL | 78.1 |
  | LL-R | 82.5 |
  | LL-Ct | 82.6 |
  | LL-Cp | 82.5 |

  Blanket softening or down-weighting of "unlabelled = negative" lost 5–7 mAP. Selective large-loss handling gained about +0.5. Source: [Kim et al. 2022](https://arxiv.org/abs/2206.03740)

#### Severity / ordinal soft targets
- **Ordinal severity estimation with noisy labels** (ulcerative-colitis endoscopy and diabetic retinopathy) [PR]: soft labels derived from noisy hard ordinal labels, plus clean-sample selection with two networks. The paper "outperforms various state-of-the-art methods"; the abstract gives no numbers. — [Takezaki et al., WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Takezaki_Self-Relaxed_Joint_Training_Sample_Selection_for_Severity_Estimation_with_Ordinal_WACV_2025_paper.html)
- **Ordinal regression with soft labels built from the distribution of annotator ratings** (vocal-fold severity) "approach[es] clinical experts". — [arXiv 2511.09702](https://arxiv.org/abs/2511.09702) [PP]
- **SIIM-ISIC 2020 1st place** trained on the granular diagnosis instead of the binary target because it "could give the model more information" [KW].
  - No binary-vs-granular ablation is reported.
  - Their 9-class and 4-class models were within noise of each other: cv_all 0.9771 vs 0.9765.

  Source: [Ha et al., arXiv 2010.05351 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2010.05351)
- **Earlier note:** soft LSR values gave +1.3/+1.4 mean AUC on CheXpert. Image-aligned labels gave +0.04 mean AUROC, and +0.14/+0.16 on hedge-prone findings. See `learning_from_reports.md` Q1–Q2.

#### Where the shift sits in our data [team] + [computed]
- **Gold vs train-report prevalence**, as z-scores assuming gold is a random sample [computed from team §3.2]. **Caveat:** gold may be an enriched sample.
  - Gold higher (under-called by reports):
    - ACL 41 vs 23% (z = +3.3)
    - Lateral meniscus 40 vs 23% (+3.1)
    - Fracture 31 vs 18% (+2.6)
    - Effusion 60 vs 50% (+1.5)
    - Synovitis 47 vs 39% (+1.2)
    - Contusion 33 vs 27% (+1.0)
  - Gold lower (over-called against the threshold):
    - Medial OA 26 vs 37% (−1.7)
    - Lateral OA 19 vs 29% (−1.7)
    - Baker's 21 vs 31% (−1.6)
    - PF OA 36 vs 44% (−1.2)
  - No shift: medial meniscus (−0.2), MCL (0.0).
- **Gold AUC, image teacher vs LLM-label mean** [team]: Lat OA .793/.845, PF OA .869/.934, Synovitis .842/.802. These are the three weakest teacher findings. For both OA findings the report labels rank gold better than the image model does.

### Inferences
- **Effusion fits the omission pattern.** The gold threshold is moderate/large, yet gold prevalence is *higher* than in reports. Omissions dominate, consistent with the forum example: a gold effusion-positive study whose report says "no effusion".
- **Encode severity per finding, from definitions, not from gold.** The recipe below is my design; the evidence base is the VisualCheXbert, LSR and ordinal-soft-label results above.
  1. **Extract grades with a local LLM** (allowed; no external API, no report text leaves the machine). Run it only on (study, finding) cells whose 31-table mean ≥ 0.2, so presence comes from the consensus and the LLM only grades.
     - Use a closed vocabulary written from the host definitions.
     - OA per compartment: none / mild (partial-thickness, fissuring, "early") / moderate / high-grade or full-thickness.
     - Effusion: none / trace-physiologic / small / moderate / large.
     - Meniscus: definite surface-reaching tear / "possible" or fraying / intrasubstance signal only.
     - ACL and MCL: complete or high-grade / low-grade or sprain / mucoid.
     - Fracture: acute / old or healed / insufficiency.
     - Contusion: traumatic bone bruise / degenerative subchondral oedema.
  2. **Map grades to soft targets** as target = consensus mean × g(grade). Prior values, borrowing Pham's LSR ranges:

     | Grade group | Examples | g(grade) |
     |---|---|---|
     | Below threshold | mild, small, trace, intrasubstance, sprain, old fracture, degenerative oedema | 0.1–0.3 |
     | Ungraded "present" | — | 0.6–0.7 |
     | At or above threshold | moderate, severe, full-thickness, definite surface-reaching | 0.9–1.0 |

     Keep the consensus mean wherever no grade is extracted.
  3. **Optionally add an ordinal auxiliary head** (cumulative logits). Rank-average P(grade ≥ threshold) with the main head.
  4. **Do not tune g on gold-58** (±0.06 per finding). Gold is a harm check only (reject if macro drops by more than about 0.02). Decide on the LB.
- **Where the gain is.** The expected gain is concentrated where the teacher is weakest and the shift is severity-type: Lat OA, PF OA, Medial OA and Baker's. Raising Lat OA alone from 0.79 to 0.85 is worth about +0.005 macro [computed: 0.06/12]. My range for the whole recipe is +0.003 to +0.010 macro (judgement; nothing measured on knee).
- **"Not mentioned" for under-called findings.** For ACL, lateral meniscus and fracture (optionally contusion), use LL-Ct or LL-R on assumed-negative cells only (31-table mean < 0.1).
  - Start rejecting at epoch 2. Grow the rejected fraction by about 0.2% of those cells per epoch, so at most about 3% after 16 epochs.
  - Never touch positives.
  - Expected: small, about +0.001–0.003 macro (judgement, scaled from +0.5 mAP on real partial labels).
- **What not to do with "not mentioned".**
  - Do **not** apply nnPU-style class-prior correction or blanket soft or down-weighted negatives. By the derivation they cannot raise AUC, and they lost 5–7 mAP on OpenImages.
  - Do **not** apply rejection to the threshold findings. There a large loss on an assumed negative is often a mild case that gold also calls negative.

### Gaps
- I found no study that trains on graded or severity soft targets and reports *binary-threshold AUC* against binary training. The ordinal soft-label papers report severity metrics only.
- I found no knee- or MSK-MRI study of report-vs-image discordance by finding.
- We do not know whether any of the 31 public tables already contain grade fields. Checking would save the LLM pass.
- We lack the host's full definition text (brief §1), so the grade→target maps above are priors and must be rewritten from whatever definitions the team has.

---

## Q2. Aggregating 31 LLM label tables whose errors are correlated (same report text, often the same model families)

### Takeaway
When every annotator reads the same report, aggregation can only remove *LLM-specific* parsing errors. Report-level errors (omissions, hedges, threshold mismatch) are shared by all 31 tables and survive any aggregator. LLM errors are also correlated across models: when two models both err, they pick the same wrong answer up to 60% of the time, and more accurate models are *more* correlated.

Recent LLM-panel studies are consistent:
- Dawid–Skene (DS) is no better than majority vote (MV).
- Nine judges give about two effective votes.
- The best single judge matches the panel.

Our gold result agrees: the mean of 17 tables scores 0.899 and the best single table 0.904, within ±0.02. Expect about 0 gain from DS, Snorkel, FlyingSquid, GLAD-type models or crowd layers over the soft mean. Keep the soft mean, which carries per-study ambiguity, and spend effort on image-concept corrections (Q1).

### Cited Findings
- **LLM errors are correlated** [PR]. Data: 349 HuggingFace models on 12,032 questions; 71 HELM models on 14,042 questions.
  - When both models err, they agree on the same wrong answer **60%** of the time on HELM (mean; 0.423 on HuggingFace). Random baselines are about 1/3 and about 0.12.
  - Same company adds +0.066 (HuggingFace) / +0.022 (HELM) to error agreement; same architecture adds +0.076 (HuggingFace).
  - "Larger and more accurate models have highly correlated errors, even with distinct architectures and providers."

  Source: [Kim, Garg, Peng & Garg, ICML 2025](https://proceedings.mlr.press/v267/kim25e.html); [arXiv html](https://arxiv.org/html/2506.07962v1)
- **"Nine Judges, Two Effective Votes"** [PP]:
  - Using the Kish effective sample size, "the 9 judges effectively provide only about 2 independent votes' worth of information". Roughly three-quarters of nominal independence is lost.
  - Panel accuracy is 8–22 pp below what independent voting would give.
  - "The best single judge matches or outperforms the full panel across all conditions."
  - "Established aggregation methods close at most 11% of this gap, even with access to correct answers."
  - [snippet] DS 70.7% vs MV 72.0% on MNLI.

  Source: [arXiv 2605.29800](https://arxiv.org/abs/2605.29800)
- **Dependence-aware aggregation** [PP][snippet]: DS and MV assume conditional independence, which LLM judges violate through shared data, architectures and prompts. Ignoring the dependence gives miscalibrated posteriors. The paper proposes Ising-model aggregation; a search summary says ICML 2026 [unverified]. — [arXiv 2601.22336](https://www.alphaxiv.org/abs/2601.22336)
- **Affect-annotation study** [snippet][unverified]: MV, DS and MACE produced near-identical consensus labels, and the authors attribute the error to shared annotator bias rather than the aggregation rule. — [ACL Anthology 2026.cas-1.11](https://aclanthology.org/2026.cas-1.11/)
- **Practitioner tooling** [AN]: posterior aggregation 0.895 vs majority vote 0.897. Weighting helps "when judge quality is uneven, not when errors are shared". — [judgepanel (PyPI)](https://pypi.org/project/judgepanel/)
- **Theory of MV vs DS** [PP]: MV is model-free and "more robust to model misspecification". DS is minimax-optimal when its model holds, and MV can be inconsistent when most workers are spammers. — [Gao & Zhou, arXiv 1310.5764](https://arxiv.org/abs/1310.5764)
- **EBCC** models worker *correlation* (a mixture of intra-class reliabilities). It had the highest mean accuracy against 10 methods on 17 real crowdsourcing datasets. — [Li, Rubinstein & Cohn, ICML 2019](https://proceedings.mlr.press/v97/li19i.html) [PR]
- **CoNAL** splits annotation noise into a *common* confusion matrix shared by all annotators and *individual* matrices, with an auxiliary network deciding which applies per instance–annotator pair. It notes that DS-style methods assume noise comes only from individual expertise. — [Chu, Ma & Wang, AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16730) [PR]; implementation: [crowd-kit CoNAL](https://toloka.ai/docs/crowd-kit/reference/crowdkit.learning.conal.CoNAL)
- **Trace-regularised annotator confusion** is learned jointly with the classifier; the trace term matters most under large label noise. — [Tanno et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Tanno_Learning_From_Noisy_Labels_by_Regularized_Estimation_of_Annotator_Confusion_CVPR_2019_paper.html) [PR]
- **FlyingSquid** gets closed-form label-model parameters from *triplets of conditionally independent* sources (at least three are needed) and is 170× faster than earlier methods. — [Fu et al., ICML 2020](https://proceedings.mlr.press/v119/fu20a.html) [PR]
- **"Who Said What"** models each expert separately, then learns (possibly sample-dependent) combination weights. It improved computer-aided diabetic-retinopathy diagnosis and beat the methods of Welinder & Perona and of Mnih & Hinton. — [Guan, Gulshan, Dai & Hinton, AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11756) [PR]; numbers not extracted.
- **Combination gains are small** [working paper]: gains from optimal combination are often small, and "for gains to be large the best approach is to discard some of the forecasts and average over the remaining ones". — [Elliott & Liao (UCSD)](https://econweb.ucsd.edu/~gelliott/Combination.pdf)
- **Using vote count as label quality (Kaggle HMS 2024, soft vote-distribution targets):**
  - 2nd place [KW, via secondary summary that contains errors]: stage 1 on all data with loss weight = voters/20 for 15 epochs; stage 2 only on samples with ≥ 6 voters for 5 epochs at a 10× lower learning rate; validation on samples with ≥ 6 voters. — [dev.to summary of the 2nd place](https://dev.to/moyuto/kaggle-hms-2nd-solution-explaind-34jj); original post: [Kaggle 492254](https://kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492254) (did not render)
  - A 329th-place repository [AN]: vote-weighted KL loss gave "0.02 boost in CV and 0.02 boost in LB". Two-stage training (1–3 votes, then 4–28 votes) is listed alongside other changes, so it is confounded. — [Cranjis-McB/HMS repo](https://github.com/Cranjis-McB/HMS-Harmful-Brain-Activity-Classification)
- **Ours** [team]: on gold, the mean of 17 public tables scored 0.899; the best single tables were JEV hybrid 0.904, Claude Opus 4.5 0.892 and Qwen-32B 0.869. "Bigger LLMs did not help."

### Inferences
- **Effective number of independent tables** [computed, Kish]: n_eff = 31 / (1 + 30·ρ̄), where ρ̄ is the mean pairwise *error* correlation. Measuring ρ̄ is cheap: on gold-58 (696 cells), correlate the per-cell errors |table − gold| between every pair of tables.

  | ρ̄ | n_eff |
  |---|---|
  | 0.1 | 7.8 |
  | 0.2 | 4.4 |
  | 0.3 | 3.1 |
  | 0.5 | 1.9 |

  If ρ̄ ≥ 0.3, any extra aggregation work is pointless.
- **What survives aggregation.** Aggregation removes LLM-specific misreads (language, negation scope). It cannot remove what is not in the report. The VisualCheXbert gap (in the earlier note) and the neuroradiology omissions (Q1) are this shared component. So 31 tables do *not* approach image truth as tables are added; they converge on "what the report says".
- **Label models over the 31 tables** (DS, Snorkel/MeTaL, FlyingSquid, GLAD, EBCC):
  - Expected ≈ the soft mean, because the shared errors break their independence assumptions.
  - A DS posterior also sharpens toward 0/1 (overconfident under correlation), which destroys the ambiguity information that ranking can use.
  - If one is used, do family-balanced averaging: average within each model family or provider first, then across families. This addresses the same-provider error coupling (Kim 2025).
- **Crowd layer, trace regularisation, CoNAL.** They can absorb *annotator-specific* confusions. The *common* report→image confusion is identifiable only with an image-truth anchor. Gold-58 supports about 2 parameters per finding, i.e. a class-conditional correction, which cannot change AUC (Q1 derivation). Expected: about 0 in 6 days.
- **Masking vs fine-tuning on agreement.** Use agreement as a *weight or second stage*, not a hard mask (the HMS pattern):
  - Stage 1: all cells, soft mean targets.
  - Stage 2: 2–3 epochs at low LR on cells with |mean − 0.5| > 0.4, plus the severity-graded cells of Q1.
  - Expected 0 to +0.002 (judgement).
- **What masking cells with mean 0.3–0.7 removes.** It drops the hedged and borderline reports. For threshold findings these are mostly *gold-negative* by the host's rule, so masking removes informative near-boundary negatives. For under-called findings they are a mix.
  - Predicted effect of the running ablation: neutral overall, perhaps slightly negative on the three OA findings and Baker's.
  - Per-finding alternative: map ambiguous cells to about 0.2 for threshold findings (U-Zeros + LSR style) and mask them only for under-called findings.

### Gaps
- No study found that measures image-model AUC when training on DS/label-model vs mean-of-LLM labels, in radiology or elsewhere.
- I could not open the full "Nine Judges" paper for per-method numbers beyond the abstract and snippet.
- The HMS 1st- and 3rd-place write-ups did not render (Kaggle needs JavaScript). The "≥ 10 votes" detail is unconfirmed.

---

## Q3. Noise-robust training with medical evidence: which methods help, and which discard subtle true positives

### Takeaway
The robust-loss and sample-selection literature (earlier note: ELR +1, NVUM +2.5, DivideMix −12.9 AUC on real CXR report noise) is built for random-like noise. Ours is systematic and shared.

Small-loss selection is hazardous here. Under report labels, the high-loss *positive* cells are disproportionately subtle true positives, and selection-based methods drop them. Supporting evidence:
- LL's correction precision falls as training proceeds.
- Medical papers had to add explicit "hard vs noisy" detectors.
- RSNA 2024 lumbar 2nd place used OOF confident learning only on the last day, with no reported gain.

Small-clean-set methods (GLC, Learning-to-Reweight, Meta-Weight-Net) cannot change AUC asymptotically when per-finding and class-conditional (Q1), and are high-variance with 58 studies. Meta-Weight-Net matched gold-based methods *without any gold*.

Expected gain from a robust-loss swap here: ≤ +0.002 macro (judgement).

### Cited Findings
- **Hard vs noisy samples in histopathology** [PP; published version not checked]: noisy labels and hard samples are confounded, so the authors build an easy/hard/noisy detector from each sample's training history to "save more clean samples". It is then used in self-training label correction plus a noise-suppressing, hard-enhancing scheme. It outperforms the state of the art on synthetic and real noisy datasets. — [Zhu et al., arXiv 2112.03694](https://arxiv.org/abs/2112.03694v1)
- **LL precision analysis** [PR]: the precision of modified labels decreases with epochs, and Δrel above 0.2 "wrongly modifies the true negatives". Direct evidence that loss-based selection eventually hits clean but hard cells. — [Kim et al. 2022](https://arxiv.org/abs/2206.03740)
- **RSNA 2024 lumbar, 2nd place (Ian Pan, Kevin, Yuji Ariyasu, Brendan Bartley)** [KW]:
  - Bartley's part "removed noisy labels using confident learning, based on an out-of-fold ensemble with teammates": "removing data points when the prediction was very different from the given label".
  - It was tried only on the final day, and no gain is reported. He "should have experimented with other denoising strategies sooner".
  - His best submission used 0.5 × loss on labels + 0.5 × loss on pseudo-labels, with 9-rotation TTA.

  Source: [brendanartley/RSNA-2024-Competition](https://github.com/brendanartley/RSNA-2024-Competition)
  - The team's Kaggle write-up exists but did not render: [writeup](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution).
  - **The "+1% from OOF-based filtering (Yuji Ariyasu)" claim is [unverified].**
  - Lumbar labels were radiologist image gradings, not report-derived [not re-verified this session].
- **Meta-Weight-Net (MW-Net)** needs a small clean, balanced meta set (M ≪ N). — [CMW-Net paper restating MW-Net](https://arxiv.org/pdf/2202.05613) [PP][snippet]
  - Trained *without* clean meta samples, using a robust meta-loss (MAE), it beats methods without clean data and is "on par with methods that do". — [Ghosh & Lan, WACV 2021](https://openaccess.thecvf.com/content/WACV2021/html/Ghosh_Do_We_Really_Need_Gold_Samples_for_Sample_Weighting_Under_WACV_2021_paper.html) [PR]
- **GLC at about 1% trusted data gave small gains on CIFAR** (earlier note, Q6). 58/4,349 ≈ 1.3%.
- **Bootstrapping** (Reed et al. 2014) blends the label with the model's own prediction using β ≈ 0.95 (soft) and 0.8 (hard), i.e. 5–20% model weight. — [Reed et al., arXiv 1412.6596](https://arxiv.org/abs/1412.6596) [PR workshop; β values from memory of the paper, **not re-verified this session**]
- **BCE is linear in the target** [derivation]: BCE(p, t) = −[t·log p + (1 − t)·log(1 − p)]. So 0.6·BCE(p, ŷ_OOF) + 0.4·BCE(p, y) = BCE(p, 0.6·ŷ_OOF + 0.4·y). Our "0.6 image + 0.4 report" target is *identical* to a 0.6/0.4 two-loss scheme, and Bartley's 0.5/0.5 is the same family.
- **Ours** [team]:
  - The 4 OOF folds trained on report labels scored 0.904–0.914 on gold.
  - The mixed targets scored 0.938 "quality", but the final model trained on them scored 0.918.
  - Masking the 0.5% of cells where teacher and report disagree was neutral.

### Inferences
- **Risk of discarding subtle true positives, highest to lowest** (judgement grounded in the cited mechanisms):
  1. DivideMix, co-teaching or JoCoR-style small-loss selection over *all* cells. These drop high-loss positives, i.e. the subtle tears and contusions. DivideMix already collapsed on real report noise.
  2. Confident-learning pruning of *positives*. Pruning negatives is the safer half.
  3. OOF-based relabelling with a high model weight.
  4. Bounded losses (GCE/SCE). They softly down-weight every high-loss cell, subtle positives included.
  5. LL-style rejection restricted to assumed negatives.
  6. Label smoothing or soft targets (lowest risk).
- **Why the OOF 0.6/0.4 relabelling failed** (agrees with and sharpens the team's explanation):
  - By BCE linearity it is a 60% self-distillation from *weaker* models: 75% of the data, gold 0.904–0.914.
  - Those models were trained on report labels, so they carry the *report concept* and cannot move targets toward the image concept. They add their own errors and compress target contrast.
  - Literature-default model weights are 5–20%, not 60%.
  - Bartley's 0.5/0.5 differs in kind. His noise was inter-reader grading noise, not a report-vs-image concept shift, and his teacher was a multi-member team ensemble.
- **OOF variants still worth one cheap run** only if a slot is free:
  - (a) Model weight ≤ 0.2.
  - (b) Instance-selective replacement: only where all 4 folds disagree with an assumed-negative cell by more than 0.7, and only for ACL, lateral meniscus and fracture. This is effectively LL-Cp with cross-fitted losses.
  - Expected 0 to +0.002.
- **Teacher/report-disagreement masking was neutral** because 0.5% of about 52k cells is about 22 cells per finding. It is too few to move a rank metric.
- **GLC, forward/backward correction, and EM with gold-estimated 2×2 confusions per finding:**
  - These are class-conditional corrections, so they leave AUC unchanged (Q1).
  - With 9–35 gold positives per finding, the matrices are also noisy.
  - Learning-to-Reweight or MW-Net on 58 gold: high variance, and Ghosh & Lan suggest gold is not the active ingredient.
  - Skip all of these.

### Gaps
- No published comparison of robust losses on *report-derived MRI* labels.
- No evidence on robust losses inside attention-MIL heads.
- The Ariyasu "+1%" figure and the exact lumbar 2nd-place denoising effect are unverified (Kaggle pages did not render).

---

## Q4. AUC-oriented training (LibAUC AUC-margin, compositional AUC, pairwise losses)

### Takeaway
The closest public analogue to our setting is the CheXpert competition: train on report-derived labels, test on expert image-consensus labels, mean AUC over 5 findings. There, deep AUC maximisation (DAM) with the AUC-margin loss beat the best cross-entropy ensemble by only **+0.0006** (0.9305 vs 0.9299).

Gains are large only at extreme class imbalance (1% positives) and modest at about 2% (Melanoma single model: +0.005–0.007 private AUC). Our prevalences are 16–60%. BCE is already rank-consistent, so expect 0 to +0.003 per model. Use AUC-margin only as a cheap last-stage fine-tune that may add ensemble diversity.

### Cited Findings
- **LibAUC on CheXpert** [PR] ([Yuan, Yan, Sonka & Yang, ICCV 2021, arXiv 2012.03173](https://arxiv.org/abs/2012.03173)):
  - Training: 224,316 images labelled from reports. Test: 500 patients with a 5-radiologist consensus.
  - Test mean AUC on 5 findings, the DAM ensemble of 5 models trained with AUC-M:
    - DAM: **0.9305** (rank 1)
    - Hierarchical learning (Pham et al., cross-entropy): **0.9299** (rank 2)
    - YWW: 0.9289
    - Stanford baseline: 0.9065
    - An AUC-square-loss DAM: 0.922
- **LibAUC recipe on CheXpert** [PR]:
  - Stage 1: cross-entropy with Adam, lr 1e-5, weight decay 1e-5, 2 epochs, batch 32, 320 px, label smoothing for uncertain labels.
  - Stage 2: replace the last classifier layer with random weights, then optimise **all** layers with AUC-margin using the PESG optimiser.
  - Stage 2 settings: γ ∈ {1/300, 1/500, 1/800}, weight decay 0, initial lr 0.1 divided by 3 at 2,000 and 8,000 iterations, 2 epochs.
  - Margin m tuned in {0.3, 0.5, 0.7, 1.0} in other experiments.
  - The authors say direct AUC optimisation from scratch struggles on complex tasks, hence the two-stage design.

  Source: [arXiv 2012.03173](https://arxiv.org/abs/2012.03173)
- **LibAUC on Kaggle Melanoma (EfficientNet-B5, 384 px; about 1.8% positives)** [PR], private LB:

  | Loss | Without TTA | With TTA |
  |---|---|---|
  | Cross-entropy | 0.9285 | 0.9345 |
  | AUC-M | 0.9357 (+0.0072) | 0.9393 (+0.0048) |

  The authors' 10-model ensemble scored public 0.9685 / private 0.9438 (42nd/33rd of 3,314). The winner scored private 0.9490. Source: [arXiv 2012.03173](https://arxiv.org/abs/2012.03173)
- **LibAUC on other medical sets** [PR], with AUC fine-tuning of 1 epoch from the last cross-entropy iterate:
  - DDSM+ (13% positives): cross-entropy 0.9392 → AUC-M 0.9544.
  - PatchCamelyon (1%): 0.8394 → 0.8896.
  - Synthetic CIFAR-2 (1%): 0.718 → 0.809.
  - The claimed robustness to noisy labels (1%/5% flips) is against the *AUC-square* loss, not against cross-entropy.

  Source: [arXiv 2012.03173](https://arxiv.org/abs/2012.03173)
- **Compositional training** treats the objective as an outer AUC loss over an inner gradient step on cross-entropy. It is reported to beat the two-stage approach; I did not extract numbers. — [Yuan et al., ICLR 2022](https://iclr.cc/virtual/2022/poster/6071) [PR]
- **AUC-consistency of symmetric losses under corrupted labels.** — [Charoenphakdee et al. 2019](https://proceedings.mlr.press/v97/charoenphakdee19a) [PR]

### Inferences
- **Theory.** BCE is a proper loss, so its population minimiser is monotone in the posterior, and that ranking is AUC-optimal. AUC losses can help only through finite-sample and optimisation effects: focusing on mis-ranked pairs and on imbalance. That explains the near-zero CheXpert gain against a strong cross-entropy ensemble.
- **Recipe if tried** (2–4 GPU-hours on cached windows):
  1. Start from our best BCE checkpoint and re-initialise the 12 attention-MIL output logits.
  2. Apply 12 per-finding AUC-margin losses on study-level logits, with imratio = per-finding train prevalence.
  3. Use PESG with lr about 0.05–0.1, γ = 1/500, weight decay 0, 1–2 epochs, margin 1.0 (one alternative at most: 0.5).
  4. AUC-margin needs **binary** labels. Binarise soft targets at 0.5, or keep only cells with |t − 0.5| > 0.3. Either way the soft information is lost, which is a real cost in our setting.
  5. Use it as an *extra ensemble member*, not a replacement.
- **Expected effect:** 0 to +0.003 macro, mostly via diversity (judgement).
- **Pairwise ranking losses:** no evidence found that they beat AUC-margin or BCE on medical multi-label data.

### Gaps
- No AUC-loss result on 3D MRI or MIL heads.
- Compositional-AUC CheXpert numbers not extracted.
- No study found on AUC losses trained with *soft* targets.

---

## Q5. Ensembling for macro AUC and choosing the final submissions

### Takeaway
- **How to blend.** Rank-average (or average z-scored logits) per finding, with global or heavily shrunk weights. Never hill-climb or fit per-finding weights on gold-58. The forecast-combination literature and our own split-half test (plain averaging +0.0075 vs best-per-finding +0.0064) both say averaging wins at this sample size.
- **The binding constraint is decorrelation.** Under a binormal model, a 0.948 model adds +0.003 to a 0.956 base only if its *within-class* correlation is ≤ 0.85, roughly overall Spearman ≤ 0.90–0.93. Our candidates (Spearman 0.94–0.96) sit in the ≤ +0.001 zone, which is exactly what we observed.
- **Final picks.** Leaderboard-reuse overfitting is usually small across Kaggle. With about 390 public studies, though, the best of 10 near-equal submissions is inflated by about +0.0046. Apply the optimizer's-curse discount.

### Cited Findings
- **SIIM-ISIC 2020 1st place** [KW]:
  - Final ensemble: 18 models, combined as "a simple average of each model's probability ranks" (each model mapped to uniform [0, 1]).
  - "We did not use public LB score feedback in any way". The public LB is "so noisy that its correlation with cv_all is essentially 0 for the single models".
  - Spread across single models: cv_all 0.0012, cv_2020 0.0043, private 0.0060, public 0.0093.
  - Final: CV 0.9600 (cv_2020), public 0.9442, private **0.9490** (1st).
  - Metadata models were weaker alone "but added useful diversity".

  Source: [Ha et al., arXiv 2010.05351](https://ar5iv.labs.arxiv.org/html/2010.05351)
- **Contrast with LibAUC's Melanoma ensemble** [PR]: it had the much higher *public* score (0.9685) but a lower *private* one (0.9438) than the winner. — [arXiv 2012.03173](https://arxiv.org/abs/2012.03173)
- **Ensemble selection** [PR]: Caruana et al. introduced forward stepwise selection from model libraries. — [Caruana et al., ICML 2004](https://mlanthology.org/icml/2004/caruana2004icml-ensemble)
  - "Especially with small validation datasets, the ensembling process itself can overfit." In a later replication, selection with replacement did not reduce overfitting, and bootstrapping the validation points helped only marginally (accuracy) or not at all (cross-entropy). — [Deep Ensembles for Low-Data Transfer Learning, arXiv 2010.06866](https://arxiv.org/pdf/2010.06866) [PP][snippet]
- **Forecast combination puzzle:**
  - Simple averages often beat estimated optimal weights; high variance of the estimated weights is the main cause. — [Qian et al., "On the forecast combination puzzle"](https://experts.umn.edu/en/publications/on-the-forecast-combination-puzzle/) [PR][snippet]; [Chan & Pauwels (MODSIM paper)](https://www.mssanz.org.au/modsim2015/E5/chan.pdf) [snippet; attribution from search summary]
  - Optimal-combination gains are often small. — [Elliott & Liao](https://econweb.ucsd.edu/~gelliott/Combination.pdf) [working paper]
- **Optimizer's curse** [PR]: picking the option with the highest estimate systematically overstates its true value, even with unbiased estimates. The remedy is Bayesian shrinkage. A 2007 correction made the disappointment figures larger. — [Smith & Winkler 2006, Management Science](https://fds.duke.edu/db/aas/stat/faculty/bob.winkler/publications/278471); [correction](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/pds_correction.pdf)
- **Kaggle LB overfitting** [PR]: over 100 Kaggle competitions (a reviewer counts 112) show "little evidence of substantial overfitting" from public-LB reuse. — [Roelofs et al., NeurIPS 2019](https://papers.nips.cc/paper/9117-a-meta-analysis-of-overfitting-in-machine-learning)
- **Binormal link**: d′ = √2·Z(AUC), so AUC = Φ(d′/√2) for equal-variance Gaussians. — [Wikipedia: Sensitivity index](https://en.wikipedia.org/wiki/Sensitivity_index)
- **Ours** [team]:
  - Student vs teacher Spearman 0.958; DINOv2 vs teacher 0.942. Adding either at weight 0.15–0.3 moved gold by ≤ 0.001.
  - Split-half: best-per-finding +0.0064 vs plain average +0.0075.
  - Public LB noise SD for one submission is about 0.003 (participant simulation).

### Inferences
- **Binormal ensemble arithmetic** [computed]. Assumptions: base AUC 0.956; candidate AUC A2; within-class correlation ρ; optimal linear weights.
  - Gain is (D − d1) through Φ(·/√2), where D² = (d1² − 2ρ·d1·d2 + d2²)/(1 − ρ²).
  - The ρ needed is:

    | Candidate AUC | ρ for +0.003 | ρ for +0.005 |
    |---|---|---|
    | 0.956 | ≤ 0.924 | ≤ 0.874 |
    | 0.948 | ≤ 0.848 | ≤ 0.796 |
    | 0.94 | ≤ 0.790 | ≤ 0.736 |
    | 0.93 | ≤ 0.730 | ≤ 0.673 |
    | 0.90 | ≤ 0.591 | ≤ 0.532 |

  - Mapping to overall Spearman for a 0.948 candidate, by simulation:

    | Within-class ρ | p = 0.2 | p = 0.35 |
    |---|---|---|
    | 0.85 | 0.90 | 0.93 |
    | 0.80 | 0.87 | 0.91 |
    | 0.90 | 0.935 | 0.954 |

  - 50/50 averaging of a much weaker model *hurts*: a 0.93 model at ρ = 0.85 gives −0.005. Weak members need small weights. At ρ = 0.85 the optimal weight for a 0.948 member is about 0.35 of the total.
- **Our teacher + student** [computed]: Spearman 0.958 at p = 0.3 implies within-class ρ ≈ 0.925. The binormal model then predicts +0.003 at best, but we observed ≤ 0.001. Real errors cluster on shared hard cases (definition and threshold cases), which the Gaussian model ignores. Treat binormal gains as **upper bounds**.
- **Diversity diagnostic to run before spending an LB slot**, per finding:
  - (i) Within-class Spearman, positives and negatives separately, on gold-58 (noisy), and on the 4,349 train studies against consensus labels wherever *both* models are out-of-sample there. Public models are in-sample on train, so their train predictions are unusable.
  - (ii) Pair-fix rate: among positive–negative pairs the base mis-orders, the share the candidate orders correctly, minus the pairs it breaks.
  - A candidate is worth a slot only if (i) ≤ about 0.85 and (ii) is clearly positive.
- **Weighting protocol:**
  1. Rank or z-score each model per finding.
  2. Set global weights from priors: equal among strong models; 0.2–0.35 for a decorrelated weaker model.
  3. Shrink any per-finding deviation ≥ 70% toward the global weight. With per-finding gold SE about ±0.06, weights are not estimable per finding.
  4. Never hill-climb on gold-58. Hill-climb only on ≥ 1,000 out-of-fold studies, with replacement and bagging.
- **Final 2 submissions:**
  1. The "robust" blend: prior weights, the most diverse members, passed the gold harm check.
  2. The best public-LB blend, but only if its margin over (1) is ≥ about 0.003 *and* is reproduced by ≥ 2 neighbouring variants. Otherwise use a blend that hedges the main model risk (for example, the OAI-trained teacher's leakage risk).
- **Winner's curse figures** [computed, iid approximation]:

  | Submissions compared (K) | Expected public-LB inflation of the best |
  |---|---|
  | 5 | +0.0035 |
  | 10 | +0.0046 |
  | 20 | +0.0056 |

  - Correlated variants inflate less (about half, if the paired-difference SD is about 0.0015).
  - The private split (about 910 studies) has SD about 0.002.
  - A +0.003 true gain is about 1 public-LB SD: detectable only through consistency across several submissions.

### Gaps
- Caruana's own hill-climb-set-size numbers could not be extracted (PDF text is font-encoded).
- No study found that measures rank vs logit vs probability averaging for AUC. The melanoma winners used ranks; differences are expected to be ≤ 0.001 [judgement].
- No published shake-up estimate for this competition beyond the participant's 0.003 simulation.

---

## Q6. Transductive and test-time options inside a 9-hour offline notebook

### Takeaway
I found no verified case of a medical Kaggle winner gaining from test-time batch-norm (BN) adaptation or from pseudo-labelling the hidden test set inside the notebook. BN re-estimation can fail catastrophically under *label* shift, and the host warns that prevalence differs between train, public and private. Recent MedMNIST evidence shows standard test-time augmentation (TTA) can *hurt* badly. That matches our null result for mirror TTA. Skip transductive tricks in this push.

### Cited Findings
- **TTA on MedMNIST v2** (3 benchmarks, 4 architectures with 21K–11M parameters) [PP]:
  - Standard TTA lowered accuracy, by up to **31.6 pp** (ResNet-18, pathology).
  - The only gain was +1.6% (ResNet-18, dermatology).
  - Proposed causes: augmentation-induced distribution shift, amplified by mismatched BN statistics.
  - Intensity-only augmentations hurt less than geometric ones; including the original image only partly mitigates the drop.

  Source: [arXiv 2604.09697](https://arxiv.org/abs/2604.09697)
- **Recomputing BN statistics on test batches** is vulnerable to label-distribution shift, "sometimes causing catastrophic failure". Tested on CIFAR10-C, ImageNet-C and a fatty-liver diagnosis task. — [Channel-Selective Normalization, arXiv 2402.04958](https://arxiv.org/html/2402.04958v2) [PP][snippet]
- **Entropy-minimising test-time adaptation** fails to improve significantly with a single test image; BN statistics are "highly important yet unstable" (segmentation). — [InTEnt, arXiv 2402.09604](https://arxiv.org/pdf/2402.09604) [PP][snippet]
- **Code competitions re-run the notebook on hidden data**, so any test pseudo-labelling must fit inside the time limit. — [fast.ai forum](https://forums.fast.ai/t/kaggle-code-competitions/103660) [AN][snippet]
- **BirdCLEF 2024 2nd place** used "pseudo labels from target domains" under a CPU-only, 120-minute limit. It is unclear whether this happened inside the notebook. — [zenn summary](https://zenn.dev/yuto_mo/articles/85eee84a753159) [AN][snippet]
- **Ours** [team]: mirror TTA gave no gain. Host warning on prevalence shift (brief §1).

### Inferences
- **BN-TTA:** ConvNeXt uses LayerNorm, so BN adaptation does not apply to it. CoAtNet's convolutional stages use BN, so the label-shift risk applies. Expected value negative to zero.
- **Test pseudo-labelling:**
  - Full retraining on about 1,300 test studies within 9 hours on 2× T4, after the ensemble inference, is not feasible with CoAtNet-384 or ConvNeXt.
  - Head-only self-training on cached MIL features is feasible, but I found no evidence and it risks confirmation bias.
  - Expected about 0.
- **Better use of notebook time:** more diverse members (Q5).

### Gaps
- No medical-classification benchmark found that quantifies BN-TTA under label shift.
- No verified example of in-notebook test pseudo-labelling in an RSNA competition.

---

## Q7 (synthesis). Ranked, evidence-graded plan for +0.003 or more macro AUC in 6 days, with failure analysis of our experiments

### Takeaway
Two levers have both theory and evidence behind them for our failure pattern:
1. **Instance-level, image-concept label fixes**: severity and definition-aware soft targets for the threshold findings, and assumed-negative large-loss handling for the under-called findings.
2. **A genuinely decorrelated ensemble member**, plus a disciplined blend and selection protocol.

Several popular tools are predicted to add about 0 here, with theory backing the prediction:
- aggregation models over the 31 tables;
- class-conditional corrections (GLC, PU, EM with 2×2 confusions);
- more teacher distillation;
- heavy OOF self-mixing;
- AUC losses at our prevalences;
- test-time tricks.

### Cited Findings
- Class-conditional (including PU) noise preserves the AUC-optimal ranking. — [Menon et al. 2015](https://proceedings.mlr.press/v37/menon15.html) [PR]
- Image-aligned labels gave +0.04 mean AUROC; soft LSR targets gave +1.3–1.4. See `learning_from_reports.md` (Jain 2021; Pham 2021) [PR].
- Selective large-loss handling of assumed negatives: +0.5 mAP on real partial labels. Blanket soft or weighted negatives: −5 to −7. — [Kim et al. 2022](https://arxiv.org/abs/2206.03740) [PR]
- LLM panels: about 2 effective votes from 9; DS ≈ MV; the best single judge ≥ the panel. — [arXiv 2605.29800](https://arxiv.org/abs/2605.29800) [PP]; [Kim et al. 2025](https://proceedings.mlr.press/v267/kim25e.html) [PR]
- AUC-margin vs the best cross-entropy ensemble on CheXpert: +0.0006. — [LibAUC](https://arxiv.org/abs/2012.03173) [PR]
- Rank-averaged diverse ensembles, chosen by CV and not by public LB, won Melanoma. — [Ha et al.](https://ar5iv.labs.arxiv.org/html/2010.05351) [KW]

### Inferences
**Ranked list.** Expected gains are my judgement unless marked; none has been measured on knee MRI.

1. **Severity/definition-aware soft targets** for Med/Lat/PF OA, effusion, Baker's and the meniscus definition (Q1 recipe).
   - Evidence: VisualCheXbert [PR], LSR [PR], the Menon theory [PR], neuro report omissions [PP].
   - Expected gain: **+0.003 to +0.010**.
   - Cost: about 0.5 day of local-LLM grading (only cells with consensus ≥ 0.2) plus one training run.
   - Main risk: local-LLM grading errors; maps not fitted to gold.
   - How to test: gold harm check, then 1 LB slot.
2. **Decorrelated ensemble member, blended at rank weight 0.25–0.4.** The model from item 1 is the natural candidate, as are 192-slice corpus models.
   - Evidence: binormal [computed]; Melanoma 1st place [KW].
   - Expected gain: **+0.003 to +0.006** if within-class ρ ≤ 0.85; about 0 otherwise.
   - Cost: inference minutes.
   - Main risk: correlation higher than hoped.
   - How to test: run the Q5 diagnostic *before* submitting.
3. **LL-Ct/LL-R on assumed negatives for ACL, lateral meniscus, fracture** (Δrel about 0.2%/epoch, from epoch 2; positives untouched).
   - Evidence: LL [PR].
   - Expected gain: +0.001 to +0.003.
   - Cost: hours (loss change).
   - Main risk: corrupting true negatives late in training.
   - How to test: cap the rejected fraction; compare per finding on LB-blend.
4. **Two-stage agreement fine-tune** instead of hard masking; ambiguous cells set to about 0.2 for threshold findings.
   - Evidence: HMS [KW/AN].
   - Expected gain: 0 to +0.002.
   - Cost: hours.
   - Main risk: removing near-boundary negatives.
   - How to test: compare with the running mask ablation.
5. **AUC-margin fine-tune (LibAUC two-stage) as an extra member.**
   - Evidence: LibAUC [PR].
   - Expected gain: 0 to +0.003.
   - Cost: 2–4 GPU-hours.
   - Main risk: binary labels discard the soft information.
   - How to test: diversity diagnostic.
6. **Blend and selection protocol** (rank-average, global weights, ≥ 70% shrinkage, no gold hill-climbing, two-pick rule).
   - Evidence: forecast combination [PR], optimizer's curse [PR], Roelofs [PR], Melanoma [KW].
   - Expected effect: avoids −0.002 to −0.005 of winner's-curse loss.
   - Cost: none.

**Do not do in this window** (each with the reason):
- **DivideMix, co-teaching/JoCoR, small-loss selection over positives:** −12.9 AUC on real report noise (earlier note), and LL precision decays.
- **Blanket PU, weighted-AN, label-smoothed AN for "not mentioned":** −5 to −7 mAP, and ranking-invariant in theory.
- **GLC, forward/backward correction, EM with per-finding 2×2 confusions from gold:** these do not change AUC under class-conditional noise, and 9–35 gold positives per finding make them noisy.
- **Learning-to-Reweight or MW-Net on gold-58:** high variance; gold not required per Ghosh & Lan.
- **DS, Snorkel, FlyingSquid, crowd layer over the 31 tables:** ≈ mean under correlated LLM errors.
- **More in-sample teacher distillation:** within-class ρ about 0.925 gives at most +0.003 theoretically and ≤ 0.001 observed.
- **OOF self-mixing with model weight ≥ 0.4:** see the failure analysis below.
- **BN-TTA, test pseudo-labelling, heavier TTA:** see Q6.

**Failure analysis of our experiments:**

| Experiment | Observed | Explanation (literature-backed) | What could still work |
|---|---|---|---|
| Distillation from the in-sample public teacher | LB 0.948 alone; ≤ 0.001 in blend | The student copies the teacher: within-class ρ about 0.925 caps the gain at about +0.003 in theory, less in practice (Q5) | Change the *information* (severity targets, other inputs or ROI), not the loss |
| 0.6 OOF image + 0.4 report targets | gold 0.918 | BCE linearity makes this a 60% self-distillation from weaker report-concept models; defaults are 5–20%; the 0.938 "quality" used report information absent at test | Model weight ≤ 0.2; instance-selective, assumed-negative-only replacement for the under-called findings |
| Masking teacher/report disagreement | neutral | About 22 cells per finding: too few to move AUC | — |
| Consensus mask (mean 0.3–0.7), running | prediction: neutral, slightly negative on OA/Baker's | Removes near-boundary *gold-negative* hedged cases for threshold findings | Map ambiguous cells to about 0.2 for threshold findings; mask only for under-called ones |

**Validation protocol:**
- Gold-58 cannot resolve +0.003 (macro SE about 0.02) and is a harm check only.
- One public-LB submission has SD about 0.003.
- Decide on priors plus consistent LB moves across related submissions.
- Analyse per finding only for large effects, such as Lat OA or PF OA moving ≥ 0.05.

### Gaps
- None of the ranked effects has been measured on knee MRI or on multilingual report labels. All are extrapolations.
- No direct evidence that severity-graded soft targets raise binary-threshold AUC.
- Whether the public tables carry grades is unknown.
- The host's full label definitions are unknown.
- Kaggle top write-ups for HMS 1st/3rd place and RSNA lumbar 2nd place (Ariyasu section) could not be read. The "+1%" OOF-filtering claim stays unverified.
