# Learning image classifiers from free-text radiology reports: labels, noisy-label training, text as privileged info / pretraining, and tiny gold sets

Scope: RSNA Knee Abnormality Detection 2026. 4,407 knee-MRI studies with free-text reports in about 12 languages, 58 expert-labelled studies, image-only test set, macro ROC-AUC over 12 binary findings. The public LLM label table agrees with gold at 0.893 macro-AUC. The team's local Qwen3.5-9B labeller agrees at 0.850. The image model scores 0.943 on the public LB; the leaders score 0.961–0.964. Only local open-weight LLMs are allowed.

Legend:
- **[unverified]**: taken from a search-engine snippet, a press release, secondary news or an abstract-level summary. I did not check it against the full paper in this session.
- **[computed]**: my own arithmetic from the cited numbers.
- **[extracted]**: numbers pulled from paper tables by an automated page reader. They are probably right, but check them before quoting them in public.

---

## Q1. Turning reports into labels: how accurate are rule-based, BERT, graph and LLM labellers, and does better label accuracy raise image-model AUC?

### Takeaway
Open-weight LLMs of about 70B (or a well-prompted mid-size model) now match GPT-4-class labellers on chest X-ray (CXR) reports in English and German, with macro-F1 around 0.92. They beat rule-based labellers. Voting across several LLMs is the best-documented single improvement. However, report labels are an imperfect proxy for image labels: radiologists reading reports disagree with radiologists reading the images by 0.12–0.21 F1. Image AUC only rose when the better labeller moved labels toward image truth. VisualCheXbert added about +0.04 mean AUROC. CheXbert had better report-level F1 than the rule-based labeller but gave about 0 image-AUC gain.

### Cited Findings

#### Classic labellers (English CXR)
- **CheXpert labeler** (rule-based: mention extraction, then negation/uncertainty classification, then aggregation). On the report test set its macro-F1 was 0.948 for mention, 0.899 for negation and 0.770 for uncertainty, and it beat the NIH labeler. It was used to label 224,316 CXRs of 65,240 patients for 14 observations. — [Irvin et al. 2019, AAAI (ar5iv)](https://ar5iv.labs.arxiv.org/html/1901.07031)
- **CheXbert** is a biomedical BERT. It is trained first on the CheXpert labeler's outputs, then fine-tuned on a small expert-annotated set augmented with back-translation. It beats the rule-based CheXpert labeler with statistical significance (p<0.001). On that metric the CheXpert labeler scores 0.743 (95% CI 0.719–0.764). CheXbert's own score was not captured in this session. — [Smit et al. 2020, EMNLP](https://arxiv.org/abs/2004.09167) [CheXpert number from search snippet of the same arXiv PDF]
- **VisualCheXbert** maps report text straight to *image* labels. Its supervision comes from a CXR vision model. "Radiologists labeling reports significantly disagree with radiologists labeling corresponding chest X-ray images." VisualCheXbert agrees better with image-reading radiologists than report-reading radiologists do, by an average F1 of 0.12 (0.09–0.15) to 0.21 (0.18–0.24). It beats an approach built on an existing report labeler by an average F1 of 0.14 (0.12–0.17). — [Jain et al. 2021, ACM CHIL](https://arxiv.org/abs/2102.11467)
- **RadGraph** (entity/relation schema). Its development set has 500 MIMIC-CXR reports (14,579 entities, 10,889 relations) and its test set has 100 reports. The benchmark model reaches relation-extraction micro-F1 0.82 on MIMIC-CXR and 0.73 on CheXpert. It was then run on 220,763 MIMIC-CXR reports. — [Jain et al. 2021, NeurIPS Datasets & Benchmarks](https://arxiv.org/abs/2106.14463)

#### Does better labeller accuracy translate to better image AUC? (the one controlled study)
- Jain et al. 2021 trained image classifiers on 224,316 CXRs using labels from each labeller. They evaluated on a 500-study radiologist-labelled test set over 8 conditions with ≥10% prevalence. Average AUROC was 0.84 (0.82–0.85) with CheXpert-labeler labels (U-Zeros) and 0.88 (0.86–0.89) with VisualCheXbert labels. Weighted averages were 0.83 and 0.87. — [Jain, Smit, Ng, Rajpurkar 2021, NeurIPS DCAI workshop](https://arxiv.org/abs/2104.00793); numbers from [ar5iv Table 2](https://ar5iv.labs.arxiv.org/html/2104.00793) [extracted]
- Per-condition AUROC in the same table [extracted], listed as CheXpert U-Zeros / CheXpert U-Ones / CheXbert U-Zeros / CheXbert U-Ones / VisualCheXbert:
  - Atelectasis: 0.70 / 0.75 / 0.76 / 0.77 / **0.84**
  - Enlarged Cardiomediastinum: 0.69 / **0.30** / 0.73 / 0.55 / **0.85**
  - Edema: 0.88 / 0.91 / 0.88 / 0.89 / 0.89
  - Pleural Effusion: 0.92 / 0.90 / 0.91 / 0.90 / 0.90
  - Support Devices: 0.89 / 0.89 / 0.85 / 0.86 / 0.87

  Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2104.00793)
- [computed] Mean AUROC across the 8 conditions, from the same per-condition table: CheXbert U-Zeros ≈ 0.84 and CheXbert U-Ones ≈ 0.81. CheXpert U-Ones ≈ 0.79 and CheXpert U-Zeros ≈ 0.84, which matches the reported 0.84. So the labeller with better report-level F1 (CheXbert) gave **no** average image-AUC gain. The image-aligned labeller (VisualCheXbert) gave +0.04, concentrated in hedge-prone findings: Atelectasis +0.14 and Enlarged Cardiomediastinum +0.16 vs CheXpert U-Zeros labels (+0.08 and +0.12 vs CheXbert U-Zeros). Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2104.00793)
- **Cross-modal data programming** (Snorkel-style labelling functions over reports, a generative label model, then an image/EEG model):
  - Tasks: CXR (50k training examples), extremity radiographs "EXR" (knee abnormality, 30k), head-CT haemorrhage (4k) and EEG (30k).
  - Average of 14 labelling functions per task, a 200-example dev set per task, and under 8 hours of clinician time per task.
  - Models came within 1.75 ROC-AUC points of large hand-labelled sets (physician-years of labelling) and were 10.25 points better than medium hand-labelled sets. They gained up to 6 AUC points over baseline weak supervision.
  - For the CXR text model, the generative label model was +8.5 AUC points over majority vote.
  - On EXR, the data-programming model was statistically *better* than the large hand-labelled baseline (DeLong p<0.05).
  - On head CT, with only 4,000 examples, it was not different from large hand-labelled sets (p>0.38).

  Source: [Dunnmon et al. 2020, Patterns (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7413132/) [extracted]
- **Upper-extremity radiographs with German reports.** GPT-4o (zero-shot, hosted) extracted labels for 1,170 clavicle, 3,755 elbow and 1,978 thumb radiographs.
  - Label-level accuracy was 98.6% (60,618/61,488), but report-level accuracy (every label in a report correct) was only 72.7–85.5%.
  - Classifiers trained on these labels reached macro-AUC of about 0.76–0.81.

  Source: [Kreutzer et al. 2025 (arXiv 2510.05664)](https://arxiv.org/html/2510.05664v1)
- **LLM relabelling of a big public 3D dataset.** In CT-RATE (24,434 reports, 439,812 label instances), the original report-derived labels agreed with GPT-5.4 labels on 96.4% of instances (κ 0.884). Radiologist adjudication sided with GPT-5.4 in 74.2% of discordant instances, and in 91.9% for lymphadenopathy. A multi-LLM majority vote gave the highest label-macro-F1 and κ against radiologist reference labels. The abstract reports no downstream image-AUC numbers. — [Yamagishi et al., June 2026 (arXiv 2606.22382)](https://arxiv.org/abs/2606.22382)
- Cleaning noisy labels "mitigates their negative impact on model training, evaluation, and selection". Active label cleaning corrects labels "up to 4× more effectively" than random selection (CXR benchmark NoisyCXR, 26.6k images, about 12.7% noise). — [Bernhardt et al. 2022, Nature Communications](https://pmc.ncbi.nlm.nih.gov/articles/PMC8897392/)

#### LLM labellers (local / open-weight)
- **Vicuna-13B, run locally, zero-shot, on CXR reports.**
  - Median κ on MIMIC-CXR was 0.57 against the CheXpert labeler and 0.64 against CheXbert. On NIH it was 0.52 and 0.55.
  - It was at par with both labellers on 9 of 11 findings (median AUC 0.84).

  Sources: [Mukherjee et al. 2023, Radiology](https://pubs.rsna.org/doi/10.1148/radiol.231147); numbers via [HealthDay summary](https://www.healthday.com/healthpro-news/health-technology/large-language-model-good-for-chest-radiography-2665852753.html) [unverified, secondary]
- **Open vs commercial on CXR reports**, micro-F1 on ImaGenome (500 reports) and institutional MGH (540 reports):
  - ImaGenome: Llama2-70B 0.972 zero-shot and 0.970 few-shot; GPT-4 0.975 and 0.984.
  - Institutional: QWEN1.5-72B 0.952 zero-shot rising to 0.965 few-shot; GPT-4 0.975 and 0.973.
  - Mistral-7B, Mixtral-8x7B, Llama2-13B, CheXbert and the CheXpert labeler were also compared.

  Source: [Dorfner et al. 2024 (arXiv 2402.12298)](https://arxiv.org/abs/2402.12298)
- **17 open-weight LLMs vs OpenAI models, zero-shot, on CXR findings.**
  - English: 3,927 reports. Macro-F1 was GPT-4o 92.4%, Mistral-Large 92.6%, Llama-3.1-70B 92.2% and Llama-3.1-405B 90.3%.
  - German: Mistral-Large reached 91.6% macro-F1.
  - Open models beat the rule-based CheXpert labeler.

  Source: [Nowak et al. 2025, Radiology](https://pubs.rsna.org/doi/10.1148/radiol.240895) (abstract via search snippet) [unverified]

  The size of the German set conflicts between sources: "18,500 reports, 16,844 patients" in one abstract snippet versus "2,300 German reports" in another summary. Possibly the 2,300 are a test split; this is unresolved.

  Same study, per the Bonn press release:
  - With in-context examples, "larger, open LLMs were better than smaller, open LLMs".
  - After training on more than 3,500 structured reports, a very large model and a model "1,200 times smaller" showed no meaningful accuracy difference.

  Source: [Univ. Bonn press release, Jan 2025](https://www.uni-bonn.de/en/news/from-text-to-structured-information-securely-with-ai)
- **15 open-weight LLMs across six pathology/radiology extraction use cases in the Netherlands, UK and Czech Republic (multilingual).**
  - Prompting strategies compared: zero-shot, one-shot, few-shot, chain-of-thought, self-consistency and prompt graph.
  - "Small-to-medium general-purpose models performed comparably to large models, while tiny and specialised models performed worse."
  - "Prompt graph and few-shot prompting improved performance by ~13%."
  - The best models approach inter-rater agreement.

  Source: [Spaanderman et al. 2025 (arXiv 2511.10658)](https://arxiv.org/abs/2511.10658)
- **Prompted LLMs as labelling functions + Snorkel label model** (general NLP benchmark WRENCH): an average 19.5% error reduction versus zero-shot prompting, and accuracy comparable to or better than hand-engineered rules. — [Smith, Fries, Hancock, Bach 2022/2023, ACM JDS](https://arxiv.org/abs/2205.02318)
- **Probability outputs.** LLM softmax probabilities for structured extraction are poorly calibrated. On free-text radiology reports the models are *overconfident*, and calibration varies with document structure, extraction category and model. The paper motivates per-domain (conformal) calibration. — [arXiv 2603.00924 (2026)](https://arxiv.org/abs/2603.00924)

#### Knee-specific report NLP
- **Classical ML on knee MRI reports.** 3,593 knee MRI reports with bag-of-n-grams logistic regression / SVM / random forest.
  - Report-classification F1 was 0.93–0.94 for medial meniscus tear but only 0.86–0.88 for lateral meniscus tear.
  - For radiology–arthroscopy mismatch detection on 108 studies, sensitivity was 79% and specificity 87%.

  Source: [Li et al. 2021, Academic Radiology (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8355247/)
- **KneeTex** (open-source rule-based extraction from knee MRI reports): precision 98.00%, recall 97.63%, F 97.81%. — [Cardiff ORCA record](https://orca.cardiff.ac.uk/id/eprint/75959/) [unverified, search snippet]
- **Knee OA, whole-organ MRI scores (WORMS) from reports.** WORMS-structured reports were 93.9% accurate with GPT-4o and 76.2% with GPT-4o-mini. For OA-severity prediction, the best accuracy was 98.1% (GPT-4o) against 68.7% (GPT-4o-mini). Both models are hosted. — [PMC12078906 (2025)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12078906/) [unverified, search snippet]

#### Multilingual
- **Locally hosted Llama-2-70B-chat** structuring 202 English (MIMIC-CXR) and 197 German reports. Performance was comparable to humans, with MCC differences within expected ranges for both languages. — [TU Dresden record, "Automatic structuring of radiology reports with on-premise open-source LLMs"](https://fis.tu-dresden.de/portal/en/publications/automatic-structuring-of-radiology-reports-with-onpremise-opensource-large-language-models(06f061da-3732-498d-b5f7-c3c170ae772e).html) [unverified, snippet]
- **LLaMA 3.1 on 947 Dutch brain-MRI reports**, compared with their English translations. It is robust on categorical and mention-detection tasks. Numeric extraction needs few-shot prompting or post-processing. — [news summary](https://letsdatascience.com/news/llama-31-extracts-structured-information-from-brain-mri-repo-3f367de7) [unverified, secondary; native-vs-translation numbers not obtained]
- **German reports**: open-weight Mistral-Large reached 91.6% macro-F1 against 92.6% on English (different datasets) — [Nowak et al. 2025](https://pubs.rsna.org/doi/10.1148/radiol.240895) [unverified]. GPT-4o on German reports reached 98.6% label-level accuracy — [Kreutzer et al. 2025](https://arxiv.org/html/2510.05664v1).

### Inferences
- The decisive lever is how far the labels move toward **image** truth, not report-level F1. In the one controlled study, a better text labeller with no image grounding (CheXbert) gave about 0 image-AUC gain. The image-grounded labeller (VisualCheXbert) gave +0.04 mean, and +0.08 to +0.16 on hedge-prone findings (Atelectasis, Enlarged Cardiomediastinum).
  - In our setting the weak findings (synovitis, fracture, lateral OA, contusion) are exactly the kind where report–image discordance is likely: under-reported, hedged, or graded inconsistently.
  - Better prompting alone may therefore cap out. Image-informed label refinement (Q3) is the complement.
- The 0.893 vs 0.850 gap between the public table and the 9B labeller is consistent with the literature on model capacity:
  - Knee OA grading collapses with small models (WORMS accuracy 93.9% → 76.2%).
  - Open models reach parity with GPT-4o only at about 70B+ (Nowak, Dorfner).
  - Mid-size general models can match large ones when well prompted (Spaanderman, with few-shot or prompt-graph prompting worth about +13%).
- Combining labellers is the most consistently supported trick. Evidence: multi-LLM majority vote was best on CT-RATE; in Dunnmon, data-programming image models were about 4 AUC points above majority-vote labels, and the generative text label model was +8.5 points over majority vote; prompted-LF + label model cut error by 19.5%. The public table, the team's own LLM, a second LLM family and simple per-language keyword rules are cheap, partly independent sources.
- Raw LLM token probabilities are informative for *ranking* but overconfident. Use them as soft scores, not as calibrated probabilities (see Q6).
- Method cards for Q1 (cost and the 4k column are my judgement):

| Method | Setting / scale | Reported gain vs baseline | Our impl. cost | Helps at ~4k pairs? |
|---|---|---|---|---|
| Rule-based labeller (CheXpert-style) | English CXR, 224k | report macro-F1 0.948/0.899/0.770 ([Irvin](https://ar5iv.labs.arxiv.org/html/1901.07031)); beaten by open LLMs ([Nowak](https://pubs.rsna.org/doi/10.1148/radiol.240895)) | High for 12 languages | Only as extra cheap voters (keyword rules) |
| CheXbert-style BERT labeller | English CXR + ~expert set | significant over rules ([Smit](https://arxiv.org/abs/2004.09167)); **≈0 image-AUC gain** ([Jain 2021](https://ar5iv.labs.arxiv.org/html/2104.00793)) [computed] | Moderate; needs English data | Low |
| VisualCheXbert-style (report → image-model output) | CheXpert, 224k | +0.14 F1 vs image labels; image AUROC 0.84→0.88 ([Jain](https://arxiv.org/abs/2102.11467), [Jain 2021](https://ar5iv.labs.arxiv.org/html/2104.00793)) | Moderate (multilingual text model on OOF image preds) | Plausible; untested at 4k |
| Open-weight LLM zero/few-shot | CXR EN/DE, 0.5k–3.9k eval | macro-F1 ≈0.92 ≈ GPT-4o ([Nowak](https://pubs.rsna.org/doi/10.1148/radiol.240895)); few-shot +0.013 micro-F1 for QWEN1.5-72B ([Dorfner](https://arxiv.org/abs/2402.12298)) | Hours of local inference | Yes (already in use) |
| Multi-LLM vote / label model | CT-RATE 24k; WRENCH; 4–50k medical | best F1/κ ([Yamagishi](https://arxiv.org/abs/2606.22382)); −19.5% error ([Smith](https://arxiv.org/abs/2205.02318)); about +4 AUC image model / +8.5 text label model over MV ([Dunnmon](https://pmc.ncbi.nlm.nih.gov/articles/PMC7413132/)) | Low–moderate | **Yes**: head CT at 4k matched large hand-labelled sets |

### Gaps
- No study found that measures image-model AUC as a function of labeller AUC on **knee MRI**, or on any multilingual report corpus.
- No head-to-head result found for *native-language* vs *translate-to-English-then-label* on a multilingual corpus. The Dutch study did this comparison, but its numbers were not obtained.
- No quantitative radiology result found for self-consistency sampling as a labelling method. Spaanderman compared it, but the abstract gives no per-strategy number.
- Not found: whether token-probability soft labels from an LLM beat hard LLM labels for training an image model in radiology.
- CheXbert's own headline score, and the exact German dataset size in Nowak et al., were not verified.

---

## Q2. Uncertain / hedged mentions: U-Ones, U-Zeros, U-Ignore, label-smoothing regularisation (LSR) and soft labels

### Takeaway
The right uncertainty mapping is **finding-specific** and can matter a lot: on one CXR finding, U-Ones gave 0.30 AUROC against 0.69 for U-Zeros. On CheXpert, soft LSR targets beat hard U-Ones/U-Zeros by +1.3 to +1.4 mean AUC. With an LLM labeller on MSK radiographs, inclusive vs exclusive handling of uncertain labels made no significant difference.

### Cited Findings
- **CheXpert validation AUROC by uncertainty policy** (U-Ignore / U-Zeros / U-Ones / U-SelfTrained / U-MultiClass):
  - Atelectasis 0.818 / 0.811 / **0.858** / 0.833 / 0.821
  - Cardiomegaly 0.828 / 0.840 / 0.832 / 0.831 / **0.854**
  - Consolidation 0.938 / 0.932 / 0.899 / **0.939** / 0.937
  - Edema 0.934 / 0.929 / **0.941** / 0.935 / 0.928
  - Pleural Effusion 0.928 / 0.931 / 0.934 / 0.932 / **0.936**

  The validation set is small (200 studies). Source: [Irvin et al. 2019](https://ar5iv.labs.arxiv.org/html/1901.07031) [extracted]
- **Pham et al. (DenseNet-121, CheXpert, 5 findings), mean validation AUC:**
  - U-Ignore 0.863, U-Zeros 0.858, U-Ones 0.860.
  - **U-Zeros+LSR 0.871** (uncertain labels mapped to uniform random values in [0, 0.3]).
  - **U-Ones+LSR 0.874** (uniform random values in [0.55, 0.85]).
  - Adding conditional training (exploiting the label hierarchy) gave U-Ones+CT+LSR 0.894 for a single model.
  - The ensemble reached 0.940 on validation and 0.930 on the 500-study test set, first on the CheXpert leaderboard at the time.

  Source: [Pham et al. 2021, Neurocomputing (arXiv 1911.06475)](https://ar5iv.labs.arxiv.org/html/1911.06475) [extracted]
- **The choice of uncertainty policy can be catastrophic for one finding.** On Enlarged Cardiomediastinum, CheXpert-labeler U-Ones labels gave 0.30 AUROC against 0.69 for U-Zeros. With CheXbert labels the figures were 0.55 vs 0.73. — [Jain et al. 2021 (ar5iv Table 2)](https://ar5iv.labs.arxiv.org/html/2104.00793) [extracted]
- **LLM-extracted labels from German reports for clavicle, elbow and thumb radiographs.** The "inclusive" model (uncertain → true) and the "exclusive" model (uncertain → false) gave macro-AUC 0.80 vs 0.81, 0.80 vs 0.80 and 0.76 vs 0.78. "No significant differences were observed across labeling strategies" (p≥0.15). — [Kreutzer et al. 2025](https://arxiv.org/html/2510.05664v1)
- Label smoothing "is competitive with loss-correction under label noise". A smoothing parameter much larger than the true noise rate can further help. — [Lukasik et al. 2020, ICML](https://proceedings.mlr.press/v119/lukasik20a.html)

### Inferences
- In our setting the hedged categories ("possible", "suspected", "cannot be excluded", "mild/early", "small") correspond to soft targets. LSR-style values fit them well: about 0.55–0.85 for hedged-positive and about 0.0–0.3 for hedged-negative or "unlikely". The team already trains on soft labels, so the remaining gain is in choosing *per-finding* values rather than global ones.
- The evidence on how much this matters is mixed: +1.3–1.4 mean AUC in Pham et al.; 0 in Kreutzer et al.; huge on a single finding in Jain et al. The safest procedure is to have the LLM output an explicit certainty level (e.g. present / possible / absent / not mentioned) and map each level per finding.
  - Pick the mapping by out-of-fold (OOF) image-model AUC against *consensus* labels, not against the 58 gold alone (see Q6).
  - Default to U-Zeros-like (low) values for findings where hedging usually means "absent on images". Fracture and contusion "cannot exclude" plausibly fall here; that is my judgement.
- Cost: trivial once the labeller emits certainty levels. Helps at 4k: yes, cheaply. The expected gain is small on average but protects against single-finding disasters.
- Method cards for Q2:

| Method | Setting / scale | Reported gain | Cost | At ~4k? |
|---|---|---|---|---|
| U-Ignore / U-Zeros / U-Ones (hard) | CheXpert 224k | best policy varies per finding; spread up to 0.04–0.05 AUROC ([Irvin](https://ar5iv.labs.arxiv.org/html/1901.07031)); 0.30 vs 0.69 worst case ([Jain 2021](https://ar5iv.labs.arxiv.org/html/2104.00793)) | Trivial | Yes, per finding |
| LSR soft targets | CheXpert 224k | +1.3 (U-Zeros) / +1.4 (U-Ones) mean AUC ([Pham](https://ar5iv.labs.arxiv.org/html/1911.06475)) | Trivial | Yes |
| Inclusive vs exclusive with LLM labels | MSK radiographs, 1–4k per region | no significant difference ([Kreutzer](https://arxiv.org/html/2510.05664v1)) | Trivial | Neutral |
| U-SelfTrained / U-MultiClass | CheXpert | best on 1–2 of 5 findings ([Irvin](https://ar5iv.labs.arxiv.org/html/1901.07031)) | Low–moderate | Possibly; marginal |

### Gaps
- No evidence found on hedge handling for MSK MRI findings specifically (synovitis, contusion or OA grade boundaries).
- No multilingual study found on how hedging vocabulary in different languages maps to label error.

---

## Q3. Noisy-label training (GCE, symmetric CE, bootstrapping, ELR, co-teaching, DivideMix, small-loss selection, label smoothing): which give real gains on report-derived labels?

### Takeaway
On real report-derived multi-label CXR labels evaluated against radiologist-relabelled test sets, gains are real but modest:
- ELR: about +1 AUC.
- NVUM: about +2.5.
- BoMD: about +3.9.

DivideMix, a sample-selection/semi-supervised method, *collapsed* (about −13 AUC). Cleaning or refining labels and self-supervised (SSL) initialisation are more reliable than sophisticated loss tricks. Systematic (non-random) report errors call for label cleaning rather than robust losses.

### Cited Findings
- **Benchmark on real report-derived noise.** Train on NIH ChestX-ray14 (86,524 images, NLP labels) or CheXpert (170,958). Test on the clean OpenI (3,818), PadChest (14,714) and NIH-Google (1,962, radiologist-adjudicated) sets.
  - Mean AUC on OpenI after NIH training: CE 85.65, ELR 86.62, **DivideMix 72.76**, NVUM 88.17, BoMD 89.57.
  - On PadChest: ELR 85.24, DivideMix 75.49, NVUM 85.49, BoMD 86.45.
  - The authors report comparable trends when training on CheXpert.

  Source: [Chen et al. "BoMD", ICCV 2023 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2203.01937) [extracted]; [CVF page](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_BoMD_Bag_of_Multi-label_Descriptors_for_Noisy_Chest_X-ray_Classification_ICCV_2023_paper.html)
- **Karimi et al. survey with experiments on three medical datasets:**
  - Systematic annotator error (lesions missed about 18% of the time): iterative label cleaning raised lesion-detection F1 from 0.747 to 0.819 and DSC from 0.584 to 0.605. Data re-weighting gave F1 0.802.
  - Inter-observer variability (prostate pathology, κ 0.40–0.60): a single pathologist's labels gave 0.80 accuracy, STAPLE 0.84, STAPLE + iMAE robust loss 0.93, and annotator-confusion modelling 0.92.
  - Algorithm-generated noisy labels at low noise: DSC 0.889 against 0.878 for training on fewer clean labels, i.e. more noisy data beat less clean data.
  - Recommendation: match the remedy to the noise type, using label cleaning for systematic errors and confusion modelling or robust losses for inter-rater noise.

  Source: [Karimi et al. 2020, Medical Image Analysis](https://ar5iv.labs.arxiv.org/html/1912.02911) [extracted]
- **Active label cleaning on CXR.** The NoisyCXR benchmark (26.6k images, about 12.7% noise) showed that noisy labels harm training, evaluation *and model selection*. SSL pre-training improved noisy-sample detection over scratch and ImageNet initialisation. A vanilla selector reached 90% correct labels with 3× fewer re-annotations than random selection. Per Table 1 [extracted], cleaning moved a vanilla classifier from 64.1% to 68.4% accuracy and an SSL-initialised one from 78.7% to 80.3%. — [Bernhardt et al. 2022, Nature Communications](https://pmc.ncbi.nlm.nih.gov/articles/PMC8897392/)
- **SSL initialisation improves robustness to noisy labels** on histology (NCT-CRC-HE-100K) and CXR (COVID-QU-Ex) with synthetic noise. — [Khanal et al. 2023](https://deepai.org/publication/improving-medical-image-classification-in-noisy-labels-using-only-self-supervised-pretraining)
- **Deep networks tolerate massive *random* label noise given enough data.** On MNIST, test accuracy stays above 90% even with 100 randomly labelled examples per clean example. — [Rolnick et al. 2017](https://arxiv.org/abs/1705.10694) [abstract-level]
- **Label smoothing** is competitive with loss correction under label noise. — [Lukasik et al. 2020, ICML](https://proceedings.mlr.press/v119/lukasik20a.html)
- **Primary method papers.** I found no quantitative results for these on report-derived *medical* labels other than the BoMD benchmark rows above.
  - Generalized cross-entropy (GCE), which interpolates between CE and MAE: [Zhang & Sabuncu 2018](https://arxiv.org/abs/1805.07836)
  - Symmetric CE (CE + reverse CE): [Wang et al. 2019](https://arxiv.org/abs/1908.06112)
  - Bootstrapping, which blends targets with the model's own predictions: [Reed et al. 2014](https://arxiv.org/abs/1412.6596)
  - Early-learning regularization (ELR): [Liu et al. 2020](https://arxiv.org/abs/2007.00151)
  - Co-teaching (two networks exchange small-loss samples): [Han et al. 2018](https://arxiv.org/abs/1804.06872)
  - DivideMix (GMM loss split + semi-supervised learning): [Li et al. 2020](https://arxiv.org/abs/2002.07394)
  - Confident learning / cleanlab: [Northcutt et al. 2021](https://arxiv.org/abs/1911.00068)

### Inferences
- Our noise is mostly **systematic and finding-specific**: report omissions, hedges, and multilingual phrasing that the LLM misreads. It is not symmetric random noise. Karimi's evidence favours label cleaning or refinement for systematic noise over robust losses.
- Small-loss and sample-selection methods are risky here, for two reasons:
  1. In multi-label data with rare positives (fracture, Baker's cyst), the "hard" high-loss samples are disproportionately true positives.
  2. DivideMix lost about 13 AUC on the real report-noise benchmark.

  Co-teaching also doubles training cost on a 24 GB L4.
- The cheapest robust options are already compatible with soft-label BCE:
  - label smoothing or soft targets (already in use);
  - bootstrapping/ELR-style targets that blend the LLM label with the model's own (or EMA/OOF) predictions;
  - GCE-style BCE as a one-run ablation.

  Expect at most about +0.5–1 macro-AUC (my judgement, scaled from BoMD's +1 for ELR). Run it only after the label-quality levers in Q1, Q2 and Q7.
- "OOF image predictions → refine labels → retrain" is the practical analogue of VisualCheXbert, NVUM and BoMD's "detect and smoothly relabel". It needs K-fold OOF predictions to avoid confirmation bias.
- Method cards for Q3:

| Method | Setting / scale | Reported gain vs CE | Cost on our HW | At ~4k? |
|---|---|---|---|---|
| Label smoothing / soft targets | general ([Lukasik](https://proceedings.mlr.press/v119/lukasik20a.html)) | competitive with loss correction | Trivial | Yes (in use) |
| GCE / SCE / bootstrapping | general benchmarks ([GCE](https://arxiv.org/abs/1805.07836), [SCE](https://arxiv.org/abs/1908.06112), [Reed](https://arxiv.org/abs/1412.6596)) | no medical report-label numbers found | Trivial (loss swap) | Unknown; cheap ablation |
| ELR | NIH→OpenI ([BoMD](https://ar5iv.labs.arxiv.org/html/2203.01937)) | +0.97 AUC | Low | Plausible small gain |
| NVUM | NIH→OpenI ([BoMD](https://ar5iv.labs.arxiv.org/html/2203.01937)) | +2.52 AUC | Moderate | Plausible; untested at 4k |
| BoMD (relabel via language-model descriptors) | NIH→OpenI ([BoMD](https://ar5iv.labs.arxiv.org/html/2203.01937)) | +3.92 AUC | High | Uncertain |
| DivideMix | NIH→OpenI ([BoMD](https://ar5iv.labs.arxiv.org/html/2203.01937)) | **−12.9 AUC** | High (2 nets + SSL) | Avoid |
| Co-teaching / small-loss | general ([Han](https://arxiv.org/abs/1804.06872)); not in our extracted rows | n/a | 2× compute | Risky with rare positives |
| Iterative / active label cleaning | MRI lesions ([Karimi](https://ar5iv.labs.arxiv.org/html/1912.02911)); CXR 26.6k ([Bernhardt](https://pmc.ncbi.nlm.nih.gov/articles/PMC8897392/)) | F1 0.747→0.819; 3–4× relabel efficiency | Moderate | Yes, if done via LLM re-query rather than experts |
| SSL init | histology, CXR ([Khanal](https://deepai.org/publication/improving-medical-image-classification-in-noisy-labels-using-only-self-supervised-pretraining)) | better robustness | Depends on backbone | Yes (pretrained medical encoders) |

### Gaps
- No noisy-label benchmark found on **3D MRI** or on **multilingual** report-derived labels.
- No published GCE, SCE or co-teaching numbers found on report-derived multi-label medical labels (beyond what the BoMD tables may contain but I did not extract).
- The exact meaning of Bernhardt's Table 1 accuracy numbers (64.1→68.4; 78.7→80.3) should be checked against the paper.

---

## Q4. Report text as privileged information: generalized distillation, report→image teacher–student, auxiliary report prediction

### Takeaway
Generalized distillation with a text(+image) teacher helps when the report only *partially* predicts the label (prognostic tasks, +3–7 AUC). It gives roughly nothing (about 0–1 AUC) when the label is *derived from the report*, which is exactly our case. Report-*prediction* objectives (captioning, masked report-token reconstruction) help mainly as **pretraining** at low label fractions.

### Cited Findings
- **Generalized distillation** unifies Hinton-style distillation and Vapnik's learning using privileged information (LUPI). A teacher trained with the privileged modality produces soft targets for a student that sees only the regular modality. — [Lopez-Paz et al. 2016, ICLR](https://arxiv.org/abs/1511.03643)
- **Bergström, Yue & Johansson 2025.** Datasets: MIMIC-CXR (6k–35k) and INSPECT (CT-derived radiographs). Settings: "diagnostic" (label derived from the report), "prognostic" and "auxiliary".
  - Image-only students distilled from an image+report teacher gained about **0–1 AUC** on the diagnostic MIMIC 5×1200 task at 5% data, i.e. nothing beyond self-distillation.
  - On the prognostic readmission task, distillation gained about **3–7 AUC**.
  - Distillation "fails when text is 'too predictive'".
  - Text-aligned pretraining (GLoRIA, BioViL) gave about 2–5 AUC advantage in the low-sample diagnostic regime, which "disappears with >1,000 samples".

  Source: [arXiv 2510.24385 (Oct 2025)](https://arxiv.org/html/2510.24385v2) [extracted summary]
- **Multimodal privileged knowledge distillation (MMPKD)** uses a text teacher for CXR (MIMIC-CXR) and a tabular-metadata teacher for mammography (CBIS-DDSM), each with a ViT student. It improved zero-shot region-of-interest localisation of attention maps, but "this effect does not generalize across domains". The abstract reports no AUC gain. — [Baur et al. 2025 (arXiv 2508.06558)](https://arxiv.org/abs/2508.06558)
- **Report prediction as pretraining (ConVIRT ablations), MURA bone-abnormality fine-tuning AUC:**
  - 1% labels: Caption-LSTM 78.7, Caption-Transformer 75.2, ImageNet init 72.1.
  - 100% labels: 87.8 and 87.6 against 87.0.

  Source: [Zhang et al. ConVIRT (ar5iv Table 1b)](https://ar5iv.labs.arxiv.org/html/2010.00747) [extracted]
- **Masked record modelling (MRM)** jointly reconstructs masked image patches and masked report tokens. It reaches 88.5% mean AUC on CheXpert with 1% labels and is about 3% better than the best counterpart on NIH at small label ratios. — [Zhou et al. 2023, ICLR](https://arxiv.org/abs/2301.13155)
- **Exploiting the label hierarchy (conditional training) on top of LSR** raised CheXpert mean AUC from 0.874 to 0.894 for a single model. This was measured on a 200-study validation set. — [Pham et al.](https://ar5iv.labs.arxiv.org/html/1911.06475) [extracted]

### Inferences
- Our 12 targets are extracted from the reports. A report→image teacher's output is therefore essentially the soft label we already train on, and Bergström's "diagnostic" result predicts about 0–1 AUC. Low priority.
- The useful privileged signal is information the reports contain **beyond** the 12 binary labels: tear location and grade, ligament sprain grade, cartilage grade per compartment, bone-marrow-oedema location, loose bodies, laterality, and so on. Using it as **auxiliary multi-task heads** is cheap: one extra LLM extraction pass plus extra BCE heads.
  - It is closer to the MRM / captioning evidence and to Pham's hierarchy gain.
  - Expected effect: small but positive regularisation at 4k (my judgement, not measured in knee MRI).
- Method cards for Q4:

| Method | Setting / scale | Reported gain | Cost | At ~4k? |
|---|---|---|---|---|
| Generalized distillation (img+text teacher → img student) | MIMIC-CXR / INSPECT 5–35k ([Bergström](https://arxiv.org/html/2510.24385v2)) | ~0–1 AUC diagnostic; ~3–7 AUC prognostic | Moderate (train teacher) | Low value (labels come from text) |
| MMPKD (text teacher) | MIMIC-CXR ([Baur](https://arxiv.org/abs/2508.06558)) | localisation only; not cross-domain | Moderate | Low |
| Captioning / report-generation pretraining | MURA 33k; RIH 48k pairs ([ConVIRT](https://ar5iv.labs.arxiv.org/html/2010.00747)) | +6.6 AUC at 1% labels; +0.8 at 100% | High (needs text decoder) | Low at 4k |
| Masked image + report modelling (MRM) | MIMIC-scale ([Zhou](https://arxiv.org/abs/2301.13155)) | 88.5 AUC CheXpert at 1% labels | High | Low at 4k |
| Auxiliary heads for extra report findings / hierarchy | CheXpert hierarchy ([Pham](https://ar5iv.labs.arxiv.org/html/1911.06475)) | +2.0 mean AUC (200-study val) | Low | Plausible |

### Gaps
- I found no study of report-derived *auxiliary* findings as multi-task heads for 3D MRI classification with a measured AUC gain.
- I found no LUPI/distillation result on MSK MRI.
- I did not fetch TieNet or other "joint classify + generate report" CXR papers, so there are no numbers for that design here.

---

## Q5. Image–report contrastive and generative pretraining (ConVIRT, GLoRIA, BioViL, MedCLIP, MedKLIP/KAD, CT-CLIP, Merlin, RadFM, 3D MRI): data scale needed, and value at ~4k pairs

### Takeaway
Every documented win from image–report pretraining used about 20k–200k+ pairs: MedCLIP 20k; ConVIRT MSK 48k; Merlin 15k CTs; CT-RATE 25k scans; Decipher-MR 22k MRI studies with 200k series. The benefit is largest at ≤1–10% labels and shrinks with more labelled data; one study found the advantage gone beyond about 1,000 labelled samples. I found no evidence that contrastive pretraining on about 4k multilingual pairs beats supervised training on good soft labels from the same reports.

### Cited Findings
- **ConVIRT.** Pretrained on MIMIC-CXR (217k pairs) and a Rhode Island Hospital MSK set of 48k image–text pairs (each pair on average 2.5 images and 8.0 sentences).
  - MURA (33,078 training images) fine-tuning AUC at 1/10/100% labels: ImageNet init 72.1/81.8/87.0, ConVIRT 81.3/86.5/89.0.
  - Linear probe: ImageNet 63.8/74.1/79.0, ConVIRT 81.2/85.1/87.6.
  - CheXpert fine-tuning: ImageNet 80.1/84.8/87.6, ConVIRT 87.0/88.1/88.1.
  - RSNA at 1%: 83.1 → 88.8.

  Source: [Zhang et al. 2020/2022 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2010.00747) [extracted]; [MLHC version](https://proceedings.mlr.press/v182/zhang22a.html)
- **MedCLIP** decouples images and texts and uses a knowledge-based semantic-matching loss to remove false negatives. With **20K** pretraining pairs it beats GLoRIA trained on about 200K CheXpert pairs (zero-shot CheXpert-5×200). — [Wang et al. 2022, EMNLP](https://arxiv.org/abs/2210.10163)
- **GLoRIA** is a global plus local (word ↔ image sub-region attention) contrastive framework that is label-efficient on CXR. — [Huang et al. 2021, ICCV](https://openaccess.thecvf.com/content/ICCV2021/html/Huang_GLoRIA_A_Multimodal_Global-Local_Representation_Learning_Framework_for_Label-Efficient_ICCV_2021_paper.html). I did not verify its 1%-label numbers.
- **MedKLIP** extracts medical triplets (entities) from reports instead of using raw text. — [Wu et al. 2023, ICCV](https://arxiv.org/abs/2301.02228)
- **KAD** (knowledge-graph-guided) reaches zero-shot performance "comparable to fully supervised models", superior to the average of three radiologists on 3 of 5 pathologies. — [Zhang et al. 2023, Nature Communications](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10382552/); [arXiv 2302.14042](https://arxiv.org/abs/2302.14042)
- **RadTex** uses reports to improve classification when labelled data are under 1,000 examples, beating ImageNet-supervised and in-domain supervised pretraining in that regime. — [arXiv 2208.03218](https://arxiv.org/abs/2208.03218) [abstract-level]
- **3D CT:**
  - CT-RATE has 25,692 non-contrast chest CTs from 21,304 patients with reports. CT-CLIP "outperforms state-of-the-art fully supervised models" on multi-abnormality detection. — [Hamamci et al. 2024](https://arxiv.org/abs/2403.17834)
  - Merlin was trained on 15,331 CTs (>6M images), more than 1.8M EHR diagnosis codes and more than 6M report tokens. It was evaluated on 752 tasks, with external testing on 44,098 scans. — [Blankemeier et al. (arXiv 2406.06512)](https://arxiv.org/abs/2406.06512). The arXiv record says it is published in Nature 2026 [unverified].
- **RadFM.** MedMD contains 16M 2D/3D scans with text across more than 5,000 diseases. Fine-tuning used RadMD, about 3M radiology visual–language pairs. — [Wu et al. 2023 (arXiv 2308.02463)](https://arxiv.org/abs/2308.02463)
- **3D MRI: Decipher-MR.** Trained on 200,000 MRI series from more than 22,000 studies across anatomies, combining self-supervised vision learning with report-guided text supervision. It is designed for a *frozen encoder plus lightweight task decoders*. The abstract does not mention knee/MSK specifically, and it gives no AUC numbers. — [Yang et al. 2025/2026 (arXiv 2509.21249)](https://arxiv.org/abs/2509.21249)
- **Bergström et al.** Text-aligned pretrained encoders (GLoRIA, BioViL) were about 2–5 AUC better in the low-sample diagnostic regime, and the advantage disappears beyond about 1,000 labelled samples. Explicit text alignment can *hurt* on labels weakly reflected in text. — [arXiv 2510.24385](https://arxiv.org/html/2510.24385v2)

### Inferences
- At 4,407 pairs, in-domain CLIP-style pretraining from scratch sits roughly 3–50× below every documented success (vs 15k–217k).
  - Text-side problem: our reports span about 12 languages, which needs a multilingual text encoder and splits the effective data further.
  - Image-side problem: 3D MRI encoders are expensive to pretrain on 1×L4.
  - Expected value: low. Cost: high (days of engineering plus GPU time we lack).
- The "knowledge-enhanced" papers (MedKLIP/KAD) effectively replace raw report text with *extracted entities*. That is what the team's LLM soft labels already are; with ~12 languages, LLM-extracted structured labels are the practical form of this idea. Supervised training on high-quality extracted labels already captures most of what report-text pretraining offers for diagnostic targets, consistent with Bergström's finding that text-alignment gains vanish above about 1k labelled samples.
- What can still help: off-the-shelf pretrained medical encoders (2D slice encoders, or 3D MRI encoders like Decipher-MR if weights are available) as initialisation or frozen-feature ensemble members. That is an image-pretraining question outside this note. The weights' availability and licence for Decipher-MR were not checked.
- Method cards for Q5:

| Method | Pretraining scale | Reported gain vs ImageNet/supervised | Cost here | At ~4k pairs? |
|---|---|---|---|---|
| ConVIRT | 217k CXR; 48k MSK pairs ([ConVIRT](https://ar5iv.labs.arxiv.org/html/2010.00747)) | MURA +9.2 (1%), +4.7 (10%), +2.0 (100%) fine-tune AUC [computed] | High | Unlikely |
| MedCLIP | 20k suffices (> GLoRIA 200k) ([MedCLIP](https://arxiv.org/abs/2210.10163)) | zero-shot gains | High | Unlikely below ~20k |
| GLoRIA / BioViL | ~200k CXR | ~2–5 AUC low-sample only ([Bergström](https://arxiv.org/html/2510.24385v2)) | High | Unlikely |
| MedKLIP / KAD | MIMIC-scale ([KAD](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10382552/)) | zero-shot ≈ supervised | High | The idea is already captured by LLM labels |
| CT-CLIP / Merlin / RadFM | 15k–25k CTs; 16M scans | claims ≥ supervised ([CT-CLIP](https://arxiv.org/abs/2403.17834), [Merlin](https://arxiv.org/abs/2406.06512)) | Very high | No (from scratch) |
| Decipher-MR (MRI) | 22k studies / 200k series ([arXiv](https://arxiv.org/abs/2509.21249)) | "consistent improvements"; no numbers | Low *if* weights are available | Possibly as a frozen encoder |

### Gaps
- No paper found that tests image–report pretraining with only about 1–5k pairs.
- No paper found that tests it on knee MRI.
- No multilingual-report VLP evaluation found.
- GLoRIA/BioViL 1%-label numbers were not verified this session.
- Decipher-MR's anatomy mix and weight availability are unknown.

---

## Q6. Using a ~58-study gold set: calibrating noisy labels (Platt/isotonic, Dawid–Skene, Snorkel), choosing checkpoints and blend weights, and overfitting risk

### Takeaway
58 studies are enough to *rank labellers roughly* and to fit very low-dimensional maps: one or two parameters per finding, ideally shared across findings. They are **not** enough to select checkpoints or blend weights. Per-finding AUC standard errors are about 0.05–0.14, so only macro-AUC differences of about ≥0.03–0.05 are detectable.

Isotonic calibration overfits below about 1,000 samples. Gold-loss-correction-style use of trusted data gives only small gains at about 1% trusted (we have 58/4,407 ≈ 1.3%).

Our metric is per-label ROC-AUC, so any monotone per-finding recalibration of the *final predictions* has no effect at all. Calibration only matters for training targets and for blending.

### Cited Findings
- **Cross-validation error bars are large at small n**: "±10% for 100 samples", and standard errors across CV folds underestimate the true error bars. — [Varoquaux 2018, NeuroImage (arXiv 1706.07581)](https://arxiv.org/abs/1706.07581)
- **AUC standard error formula** (depends on AUC, number of positives and number of negatives). — [Hanley & McNeil 1982, Radiology](https://pubs.rsna.org/doi/10.1148/radiology.143.1.7063747). [computed] values for n=58:

| Positives / 58 | SE at AUC 0.85 | SE at AUC 0.90 | SE at AUC 0.95 | 95% CI half-width at AUC 0.90 |
|---|---|---|---|---|
| 3 | 0.142 | 0.121 | 0.089 | ±0.24 |
| 5 | 0.111 | 0.094 | 0.069 | ±0.18 |
| 10 | 0.079 | 0.067 | 0.049 | ±0.13 |
| 20 | 0.059 | 0.049 | 0.035 | ±0.10 |
| 29 | 0.052 | 0.042 | 0.030 | ±0.08 |

  [computed] Macro-AUC over 12 findings, if per-finding errors were independent: SE ≈ 0.014–0.026 for per-finding SEs of 0.05–0.09. Paired comparisons of two models on the same 58 studies have smaller SE than this, but the size of the reduction is unquantified here.
- **Isotonic regression is not advised with too few calibration samples (≪1,000)** because it overfits. Platt (sigmoid) scaling has 2 parameters. — [scikit-learn calibration docs](https://scikit-learn.org/1.4/modules/generated/sklearn.calibration.CalibratedClassifierCV.html), following [Niculescu-Mizil & Caruana 2005, ICML](https://mlanthology.org/icml/2005/niculescumizil2005icml-predicting). Temperature scaling has a single parameter — [Guo et al. 2017](https://arxiv.org/abs/1706.04599) [abstract-level].
- **Gold Loss Correction (GLC)** estimates the noisy-label corruption matrix by averaging a noisy-trained classifier's predicted label distribution over the trusted examples of each true class, then retrains with loss correction.
  - Trusted fractions tested: 1/5/10/25%.
  - CIFAR-10 at 1% trusted [extracted]: GLC 26.9% error, forward correction 28.1%, no correction 28.3%, distillation 30.9%. The gain at about 1% trusted is small.
  - The per-class estimate needs enough trusted examples of each class.

  Source: [Hendrycks et al. 2018, NeurIPS](https://ar5iv.labs.arxiv.org/html/1802.05300)
- **Dawid–Skene**: EM estimation of each annotator's confusion matrix and the latent true labels without gold labels. — [Dawid & Skene 1979, JRSS-C](https://doi.org/10.2307/2346806) [classic, abstract-level]
- **Snorkel / data programming**: a generative label model learns source accuracies and correlations from agreement patterns. — [Ratner et al. 2017](https://arxiv.org/abs/1711.10160) [abstract-level]
  - In medicine, a 200-example dev set per task was enough to build the labelling functions and label model, which came within 1.75 AUC of large hand-labelled sets. — [Dunnmon et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7413132/)
  - With prompted LLMs as the sources: −19.5% error vs zero-shot. — [Smith et al.](https://arxiv.org/abs/2205.02318)
- **Noisy validation labels distort model selection and performance estimates.** Cleaner validation labels restored reliable rankings. — [Bernhardt et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8897392/)
- **Repeatedly querying a holdout or leaderboard overfits it** (adaptive data analysis). Remedies include limiting precision or number of queries. — [Blum & Hardt 2015, "The Ladder"](https://arxiv.org/abs/1502.04585); [Dwork et al. 2015, Science](https://doi.org/10.1126/science.aaa9375) [abstract-level]
- **LLM extraction probabilities on radiology reports are overconfident**, so per-domain calibration is needed. — [arXiv 2603.00924](https://arxiv.org/abs/2603.00924)

### Inferences
- **Metric-level fact.** Macro ROC-AUC is invariant to any strictly monotone transform applied separately to each finding's final scores, so Platt or isotonic "calibration" of the submission is pointless.
  Calibration matters in three places only:
  1. Soft *training* targets. Their scale changes the BCE weighting between samples, a second-order effect.
  2. *Blending* models or labellers with different score scales. Use rank or logit averaging.
  3. *Combining* labellers in a label model.
- **What 58 can do:**
  - Rank labellers per finding when they differ by ≥~0.10 AUC on findings with ≥10 positives. The public 0.893 vs own 0.850 macro gap (0.043) is about 1.7–3 macro SEs, so it is plausibly real [computed, rough; unpaired; a paired comparison would be tighter].
  - Estimate a *shared* temperature or Platt slope across findings with a per-finding intercept tied to prevalence. That is about 13 parameters for 58 × 12 = 696 label instances.
  - Sanity-check that a change doesn't break something.
- **What 58 cannot do:**
  - Pick among checkpoints or epochs.
  - Tune per-finding blend weights. That is 12+ parameters, each fitted on a few positives, which invites the "winner's curse".
  - Fit isotonic maps.
  - Estimate full per-finding corruption matrices for rare findings. With 3–5 positives, GLC-style matrices are dominated by noise.
- **The per-finding gold AUCs of the weakest labels are themselves noisy** (synovitis 0.79, fracture 0.79). If fracture has about 3–5 gold positives, its 95% CI is about ±0.18–0.24. A large-sample proxy for label difficulty is the **disagreement rate between labellers on all 4,407 reports**, for example the public table vs own LLM vs a second LLM. Use it to decide where to spend LLM re-labelling effort.
- **Model selection.** Use K-fold OOF AUC against the **consensus soft labels on 4,407 studies**: noisy, but low variance. Use gold only as a paired-bootstrap check, and log how many times gold was looked at.
  - For blend weights, prefer equal weights or rank averaging unless OOF shows a large, consistent difference.
- **Dawid–Skene or Snorkel over labellers** (public table, own LLM(s), keyword rules) is cheap: minutes on CPU. Their weights are learned from 4,407 unlabelled agreement patterns, not from the 58, which avoids gold overfitting. The 58 then validate the combined labels.
  - Caveat: the sources share errors (all read the same report), so conditional independence is violated. Snorkel's correlation modelling or simple averaging may then be as good.

### Gaps
- No study found that quantifies checkpoint-selection error with about 50–60 labelled multi-label medical cases.
- No empirical comparison found of Dawid–Skene vs Snorkel vs simple averaging for *LLM* radiology labellers.
- The GLC numbers are from CIFAR, not medical data, and the extraction should be checked against the paper's tables.

---

## Q7 (synthesis). What to try first in our setting: ranked

### Takeaway
The largest documented levers are (1) better and *combined* report labels for the weakest findings and (2) moving labels toward image truth with out-of-fold image predictions. Next come cheap per-finding hedge handling and auxiliary report-derived heads. Robust-loss tricks are worth at most a small ablation. Contrastive pretraining and text-privileged distillation are poor bets at 4.4k multilingual pairs in 14 days. Use the 58 gold only for coarse decisions with confidence intervals.

### Cited Findings
- Combining labellers or using a label model gives the best label F1/κ and +4–8.5 AUC over majority vote. Models trained on such labels reach near hand-labelled performance even at 4k examples. — [Yamagishi 2026](https://arxiv.org/abs/2606.22382); [Dunnmon 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7413132/); [Smith et al.](https://arxiv.org/abs/2205.02318)
- Moving labels toward image truth raised image AUROC by +0.04 mean (+0.14 Atelectasis and +0.16 Enlarged Cardiomediastinum vs CheXpert-labeler labels), whereas a better text-only labeller did not help. — [Jain et al. 2021](https://ar5iv.labs.arxiv.org/html/2104.00793)
- Model capacity matters for MSK grading from reports (WORMS accuracy 93.9% vs 76.2%), and open models reach GPT-4o parity at about 70B+. Mid-size models with few-shot or prompt-graph prompting gain about 13%. — [PMC12078906](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12078906/) [unverified]; [Nowak 2025](https://pubs.rsna.org/doi/10.1148/radiol.240895) [unverified]; [Spaanderman 2025](https://arxiv.org/abs/2511.10658)
- Robust losses on real report noise: ELR +1, NVUM +2.5, BoMD +3.9, DivideMix −12.9 AUC. — [BoMD benchmark](https://ar5iv.labs.arxiv.org/html/2203.01937)
- Soft targets for hedges: +1.3–1.4 mean AUC (Pham). Hard policies can fail badly on one finding (0.30 vs 0.69). — [Pham](https://ar5iv.labs.arxiv.org/html/1911.06475); [Jain 2021](https://ar5iv.labs.arxiv.org/html/2104.00793)
- Privileged-text distillation gives about 0–1 AUC when labels are derived from reports, and text-alignment pretraining gains vanish above about 1k labelled samples. — [Bergström 2025](https://arxiv.org/html/2510.24385v2)

### Inferences
Ranked list. Expected gains are my judgement, extrapolated from the cited evidence; none were measured on knee MRI.

1. **Build a multi-labeller consensus for the training labels, focused on the weakest findings.**
   - Sources:
     - the public label table;
     - the team's Qwen labeller;
     - one *different-family* open-weight LLM at the largest size that fits (about 30B-class 4-bit on the 24 GB L4, or 70B with CPU offload, since 4,407 reports is a small job);
     - optionally cheap per-language keyword/negation rules.
   - Ask each LLM for a certainty level plus token-probability scores per finding, with definitions and multilingual synonyms in the prompt. Few-shot examples should be synthetic or drawn from non-gold reports.
   - Aggregate by averaging ranks or logits, or with a Dawid–Skene/Snorkel model.
   - Use the gold set only to drop a source for a finding when it is clearly worse (≥~0.1 AUC with ≥10 positives).
   - Where to spend effort: target synovitis, fracture, lateral OA and contusion, but **rank effort by inter-labeller disagreement on all 4,407 reports**, not by the noisy gold AUCs.
   - Cost: about 1–2 days, mostly LLM inference and prompt work.
   - Expected: the largest gain. +0.01–0.02 macro-AUC is plausible if the weakest findings improve.
2. **Use a per-finding hedge mapping (LSR-style soft values) for each certainty level.** For example, "possible" → 0.6–0.7 and "unlikely / cannot exclude" → 0.1–0.3, chosen per finding by OOF AUC against consensus labels.
   - Cost: hours.
   - Expected: small on average; protects against one-finding failures like the 0.30-AUROC case above.
3. **Refine labels with image predictions (VisualCheXbert / ELR / NVUM analogue).**
   - Train K-fold models and get OOF image probabilities.
   - Set the new target to (1−β)·consensus label + β·OOF prediction, with β about 0.2–0.3, or only where confident image predictions disagree with hedged or absent report mentions.
   - Retrain once.
   - Cost: one extra full training cycle (about 1–2 days on L4 + 4070S).
   - Expected: small–moderate, mainly on under-reported findings such as synovitis, effusion grading and contusion.
   - Risk: confirmation bias. Mitigate by keeping β low and using strictly OOF predictions.
4. **Add auxiliary report-derived heads** for granular findings: meniscal tear location/type, ACL partial vs complete, cartilage grade per compartment, marrow-oedema location, loose bodies. Also try hierarchy-aware heads, e.g. contusion conditioned on ACL tear patterns and synovitis on effusion.
   - Cost: one LLM extraction pass plus extra BCE heads.
   - Expected: small positive regularisation. Evidence is indirect (Pham hierarchy +2.0 on a 200-study validation set; MRM/captioning).
5. **Adopt a gold-set protocol.**
   - Paired bootstrap CIs for every gold comparison.
   - Act only on macro differences of about 0.03 or more (smaller differences are noise).
   - No checkpoint picking on gold; never isotonic.
   - Shared-slope Platt/temperature only for *training* targets.
   - Rank or logit averaging for blends.
   - Keep a count of gold "looks".
   - Cost: trivial. Protects against fooling ourselves.
6. **Only if time remains: one cheap robust-loss ablation** (GCE-style BCE or bootstrapped targets) on the best label set. Expect ≤ about +0.5 macro-AUC.
   - Skip DivideMix and co-teaching: they collapsed or are risky in multi-label rare-positive data, and they double compute.
7. **Do not pursue in this window:**
   - In-domain CLIP/ConVIRT-style pretraining on 4.4k multilingual pairs: about 3–50× below the documented data scale.
   - Text-privileged teacher–student distillation: about 0–1 AUC for report-derived labels.
   - Meta-learning on the 58 gold (GLC, learning-to-reweight): too few trusted examples per finding.

### Gaps
- None of the expected gains above has been measured on knee MRI or on multilingual reports. They are extrapolations from CXR, CT and MSK-radiograph studies.
- The competition's 58-study gold prevalence per finding was not available to me, so the SE estimates above assume generic prevalences. The team should plug in actual positive counts.
- Whether the gold labels are image-based expert reads or report-based is unknown to me. If they are image-based, the VisualCheXbert-style refinement (item 3) becomes more important.
