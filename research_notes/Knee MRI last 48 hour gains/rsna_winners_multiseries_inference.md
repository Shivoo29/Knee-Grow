# RSNA Kaggle winners: multi-series handling and last-days inference gains (knee MRI addendum, 20 Oct 2026)

**Scope.** This file adds to `research_notes/Knee MRI top leaderboard tactics/prior_kaggle_winners_tricks.md` (16 Oct) and `research_notes/Knee MRI prior work and tricks/past_kaggle_winners.md` (8 Oct). Those files already cover the general trick rankings, LSDC/aneurysm architecture cards, label-noise handling and runtime engineering. This file covers what they leave out:
- how teams chose between, or combined, several series of the same type, including averaging over alternative series ("series TTA");
- plane- and sequence-specialist models versus joint models;
- ensemble size, blending and post-processing under a 9–12 h code-competition limit;
- full-data retrains versus fold models, and seeds.

**Method.**
- Kaggle pages are JavaScript-rendered, so I read them through the r.jina.ai reader proxy. The links below are the original Kaggle URLs. Write-up URLs came from the [faridrashidi/kaggle-solutions](https://github.com/faridrashidi/kaggle-solutions) index.
- GitHub READMEs were read raw. The melanoma paper PDF was converted to text locally.
- Nothing from this machine was uploaded anywhere. No competition data was downloaded and no code was run.

**Metric directions.**
- LSDC 2024 and trauma 2023 use weighted log loss (lower is better).
- Aneurysm 2025, MGMT 2021 and SIIM-ISIC 2020 use AUC (higher is better).
- A log-loss gain cannot be converted into AUC.

**Tags.**
- "[16 Oct notes]": a fact taken from the earlier file and not re-read today.
- "(unquantified)": the authors gave no number.

## Q1. RSNA 2024 Lumbar Spine (LSDC, ranks 1–15): how teams picked or combined several Sagittal T1/T2 or Axial T2 series, whether anyone averaged over same-type series (and the gain), and how they ensembled within the time limit

### Takeaway
**Duplicate series were common, and teams handled them in five ways:**
1. Use every series that covers the target and aggregate (2nd place, Ian's part: crops from both duplicated axial series; 11th place: inference per series, then a simple average).
2. Merge the series into one stack sorted by ImagePositionPatient (6th).
3. Keep only one series (7th: the highest-confidence axial detection, which the author called a defect; 14th: only one orientation group per stack).
4. Fall back to the other plane when a level is missing (2nd: sagittal-only prediction when the axial lacks the level).
5. Impute missing keypoints with rules (15th).

**No write-up isolates the gain from averaging over same-type series.** The closest evidence is the 2nd place's multi-crop and slice-offset aggregation, which "improved performance significantly" (unquantified).

**Winning ensembles were small.** 1st place: 2 backbones per condition, with "5 fold for each task". 9th: 6 models × 5 folds, simple mean. 3rd: 30 models, simple mean, CV 0.3858 → 0.3643.

**Joint versus plane-specialist models: mixed evidence.**
- Adding multi-view pipelines to a sagittal-only pipeline moved NVSpine (6th) from private 0.420 to 0.401.
- Ian (2nd) ranked "sagittal T2 + axial T2 for spinal" among his top-3 elements.
- The 1st and 11th places found that extra series or an axial model added nothing for some conditions.

### Cited Findings
**Several series of the same type: what the teams did**

- **2nd (Ian's part), use every series.** "Many studies had multiple axial T2 series. Several axial T2 series did not have all of the levels. The subarticular slice localization model was also used to predict whether a level was present… If a level was duplicated across two axial T2 series, then crops from both were used. For spinal axial T2 prediction, if a level was not predicted to be present, the spinal sagittal model prediction was used on its own." — [LSDC 2nd (Ian)](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539452)

- **2nd (Ian's part), multi-crop aggregation.**
  - 27 crops per target: (original slice ± 1 slice) × (original xy + 8 in-plane offsets). "At inference, all of the crop predictions for a target were aggregated using either median or mean. This can be thought of as a form of test-time augmentation (TTA) and improved performance significantly." (Unquantified.)
  - Ian's top 3 elements: "Training and aggregating over multiple crops (especially slices)"; "Using both sagittal T2 and axial T2 series to predict spinal canal stenosis"; "Noise reduction step".

  — [LSDC 2nd (Ian)](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539452)

- **11th (Naoya's part), inference per series, then a simple average.** "For study_id with multiple axial series, a separate dataset was created for each series, and inference was performed on each. The outputs were then ensembled by taking a simple average." (Unquantified.) — [LSDC 11th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539569)

- **11th (kurimats's part), both orientations.** "the nearest axial image was selected. If the second closest image had a different orientation, we obtained axial slices from two orientations." (Unquantified.) — [LSDC 11th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539569)

- **6th (NVSpine, Model 2), merge into one stack.**
  - "Multiple axial series for a study are combined by sorting by ImagePositionPatient." The median of the per-slice x/y predictions defines one 3D box per study.
  - In the 3-view model, "If a view is not available it is replaced with zeros", and "Dropping one of the views with a chance of 10% is used as additional augmentation." (Unquantified.)

  — [LSDC 6th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/541813); [Model 2 repo](https://github.com/darraghdog/kaggle-rsna-2024-6th-place-solution-model2)

- **7th (lhwcv's part), pick one series, and the author calls this a weakness.**
  - "For the axial plane, there may be multiple detections, but we only used the one with the highest confidence (not necessarily the best)."
  - The first stage-2 model "does not handle multiple series on the axial plane very well", which motivated a second model.
  - Final: both models with many backbones, "TTA: hflip for sagittal, vflip for axial", CV 0.373 / LB 0.35 / private 0.40. No ablation of either choice.

  — [LSDC 7th (lhwcv)](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539486)

- **14th, avoid mixing orientation groups.**
  - "I do not do s1 on axial view because multiple groups make the data messy."
  - Axial stacks were built "making sure the group=view angle is the same for all selected images, and replicate images if the number of selected slices is insufficient".

  — [LSDC 14th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539459)

- **15th, rule-based imputation when a series or keypoint is missing.**
  - Missing T1 keypoint: mirror its twin ("if T1 left L1/L2 went missing, mirror T1 right L1/L2 around T2 L1/L2").
  - Missing T2 keypoints: "linearly interpolate all missing T2 keypoints before inferring axial levels".
  - Missing axial keypoints: "This step sadly doesn't exist for us… none of them really work out that well."
  - The model is one backbone with 3 heads, one per series type.

  — [LSDC 15th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539981)

- **1st.** The write-up does not say how duplicate series were chosen. It describes stage 1 per series type, with "5 fold for each task" and median-ensembled slice predictions. — [LSDC 1st](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/540091)

- **5th, "TTA-like ensemble".** "Instead of combining the stage 1 models developed by team members and passing them to stage 2 models, we inferred stage 2 models for each individual stage 1 model and then performed an ensemble." This was "the ensemble method that yielded the highest score on the private leaderboard". It averages over alternative localisations/crops. (Unquantified.) — [LSDC 5th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539472)

**Joint multi-plane versus plane/sequence specialists**

- **1st: joint multi-plane model per condition.** The final severity models were per condition, each fed several planes: `scs_classify_5ch_axsagt2` (axial + sagittal T2), `nfn_classify_5ch_axsagt1` (axial + sagittal T1), `ss_classify_5ch_ax` (axial). Each existed in a ConvNeXt-S and an EffNetV2-S version, so 6 severity models in total.
  - Did not work: "sagt1 image for scs, sagt1 and sagt2 image for ss, sagt2 image for nfn".

  — [LSDC 1st](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/540091)

- **2nd (Ian's part): one specialist per plane, blended.** "4 groups of models: foraminal (sagittal T1), spinal (sagittal T2), subarticular (axial T2), and spinal (axial T2). Spinal was averaged using weight of 1 for sagittal and 0.5 for axial, though there was not much difference from equal weighting." Ian's part scored CV 0.3805 / LB 0.35. — [LSDC 2nd (Ian)](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539452)

- **3rd: one model sees all three series.** "the most effective approach was to input groups of sagittal T1, sagittal T2, and axial T2 slices at arbitrary disc levels".
  - Did not work: "Models specialized for each disc level"; "Models specialized for each side of the body".

  — [LSDC 3rd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453)

- **6th (NVSpine): sagittal-only versus the full multi-view ensemble.**

  | Ensemble | CV | Public | Private |
  |---|---|---|---|
  | Theo's sagittal-only pipeline | 0.394 | 0.370 | 0.420 |
  | + Christof's and Darragh's pipelines (including the 3-view, axial-using model) | 0.382 | 0.355 | 0.401 |

  - Adding the multi-view members gave −0.019 private log loss.
  - Separate per-condition sagittal models lost to a joint one: "handling the two sagittal modalities together worked better".
  - The blend is an MLP that treats each target independently.

  — [Theo's repo](https://github.com/TheoViel/kaggle_rsna_lumbar_spine); [LSDC 6th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/541813)

- **9th: split and global models in one ensemble.**
  - Split models: spinal from Sag T2 + Axial; foraminal from Sag T1; subarticular from Axial.
  - Global model: all ROIs → all 5 conditions.
  - "My final submission was an ensemble of 6 models, combining both split and global models with three different feature extractors ResNet18, Swin-Tiny, and ConvNeXt-Nano. Each of these models was further a 5-fold ensemble, and the final predictions were the simple mean."

  — [LSDC 9th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539690)

- **11th: some specialists added nothing.**
  - Spinal canal from Sagittal T2 only: an axial model "did not show any improvement when ensemble methods were applied, so I did not adopt it".
  - An axial whole-crop subarticular model "did not improve when included in the ensemble".

  — [LSDC 11th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539569)

- **12th: per-disease specialists plus a combined model, weighted average.** "a separate classification model for each disease… and… a combined multi-modal model for the three diseases… The final results are derived from the weighted average of these models." — [LSDC 12th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/540001)

- **14th: per-target weights over specialist "sources".**
  - "I use a simple weighting of each model independent for each condition (apply to logits)."
  - Seven sources (per-series s2 models for Sagittal T2, Sagittal T1 and Axial T2; s12 variants; and a multi-stage single-condition model) are weighted per target.
  - Examples: the Sagittal T1 source gets 0.257 for left foraminal and 0.003 for spinal canal stenosis. The Axial source gets 0.237 for left subarticular. The single-condition source gets 0.37–0.52 for every target.
  - Result: CV 0.379, LB 0.35 / 0.41.

  — [LSDC 14th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539459)

**Ensemble size, blending, TTA counts**

| Place | Members | Blend | Numbers |
|---|---|---|---|
| 1st | 6 severity models (3 conditions × ConvNeXt-S/EffNetV2-S); "5 fold for each task" (stated in the stage-1 section) | Not stated beyond the median for stage-1 slices | Public 0.37 → 0.35 (attention MIL) → 0.33 (BiLSTM + auxiliary loss + ensembling) [16 Oct notes] |
| 3rd | 30 models (15 Center + 15 Side) | "simply averaging"; temperature 0.91 on SCS logits | CV 0.3858 → 0.3643 |
| 4th | Multi-view models + stackers | MLP residual over a Nelder-Mead blend; per-level GBDT | [16 Oct notes] |
| 9th | 6 models × 5 folds | simple mean | — |
| 11th | 4–7 models per condition | weights from Nelder-Mead on CV | CV 0.373 / public 0.35 / private 0.41 |
| 13th | 5 models + TTA 16 (best private) | — | See the table below |
| 7th (hengck23 + lhwcv) | 2-stage + 1-stage | "0.75*2stage + 0.25*1stage" | — |

Sources: [1st](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/540091), [3rd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453), [4th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539443), [9th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539690), [11th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539569), [13th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539510), [7th (hengck23)](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539439)

- **13th: TTA count and model choice against private.**

  | Submission | Backbone | Local CV | Public | Private |
  |---|---|---|---|---|
  | "best public: ensemble 4 (dropped one) TTA 20" | regnet | 0.4027 | 0.3452 | 0.4121 |
  | "best local: ensemble 5 TTA 40" | hgnet | 0.3883 | 0.3612 | 0.4215 |
  | "best private: ensemble 5 TTA 16" | regnet | 0.4076 | 0.3472 | 0.4118 |

  - The CV-best submission (with the most TTA) was the worst of the three on private, by about 0.01 log loss. It is confounded with the backbone change.
  - "The best in the public leaderboard was the 3rd best in the private leaderboard."

  — [LSDC 13th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539510)

- **12th: post-processing.** "I multiplied all the probabilities by a temperature value of 1.3." This is a log-loss calibration; see Q3 for why it does not apply to AUC. — [LSDC 12th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/540001)

### Inferences
**1. Series TTA.** It is the knee analogue of what LSDC 2nd and 11th did, and needs no retraining. About 44% of our studies lose at least one real series to the 5-slot packing.
- **How.** For each slot with more than one candidate of the *same plane and contrast class*, run the trained model once per alternative fill and average the per-finding probabilities or logits. Keep the other slots fixed. Cap at 2–3 passes per study.
- **Same-group rule.** Keep each alternative within its plane/contrast group (14th) rather than mixing contrasts into a slot.
- **Expected gain:** unknown. No LSDC team isolated it. My guess is +0.001–0.004 macro AUC overall, concentrated in the subset that has duplicates. This is not a measured number.

**2. Measure before shipping.** Run the fold models on their own validation folds with alternative fills (inference only). Compare macro AUC over all studies and over the duplicate subset for three options:
- first pick only;
- mean over picks;
- max over picks.

**3. Prefer mean over max.** Max over a varying number of passes pushes scores up for studies with more series. That can reorder studies across the cohort and cost AUC (my inference). LSDC 11th and trauma 7th/8th used means.

**4. Keep truly missing slots as trained.** If no candidate exists, keep the training-time convention (zeros or mask). Don't fill the slot with another contrast unless the model saw such substitutions in training. LSDC 6th zero-filled missing views and used 10% view dropout "as additional augmentation". That keeps missing views in-distribution (my reading).

**5. Cheapest additional TTA: window offsets.** Shift the 2.5D window grid by half a stride and average. This is the knee version of Ian's ±1-slice crops, which "improved performance significantly".

**6. Ensemble choice for the last 2 days.** The LSDC winner shipped only 2 backbones per condition. Most measured ensemble gains came from *different* pipelines or views (NVSpine −0.019 private; 3rd −0.0215 CV). Prefer 1–2 diverse members, such as a different backbone or a plane-specialist head, over more seeds of the same model.

**7. Final selection.** Choose by CV, but use the second final submission as a hedge. LSDC 13th's CV-best was its worst on private, and the public-best was 3rd-best on private.

### Gaps
- No LSDC write-up gives a number for averaging over duplicate same-type series, or for "highest-confidence pick" against "use all".
- The 1st place's duplicate-series rule is not in its write-up. It is probably in the linked inference notebook, which I did not read.
- I could not confirm the LSDC runtime limit: the overview page rendered without the code-requirements block. Trauma 2023 is verified at 9 h and aneurysm 2025 at 12 h (Q2).
- The 2nd place's Yuji and Brendan parts were not returned by the reader proxy. Only Ian's part was readable, so their series handling is unknown.

## Q2. Aneurysm 2025, Abdominal Trauma 2023 and Brain Tumor MGMT 2021: multi-series use at inference, specialist versus joint models, TTA that paid off, and measured gains

### Takeaway
**Trauma** (9 h limit; 1–2 CT series per patient):
- Patient-level fusion across series was a plain mean (7th, 8th) or a max over all slices of both series (1st).
- The 9th place instead *selected* a series per target by contrast phase (aortic HU).
- Nobody reported the gain of fusion versus picking.
- The measured late gains were metric-specific post-processing for log loss (scaling: −0.0105 CV; stacking: −0.0091 CV), which is irrelevant to AUC.
- The 2nd place's best private was a single model; ensembling helped public only.

**Aneurysm** (12 h; scored per series through an evaluation API):
- No cross-series fusion was needed.
- Modality and plane were handled inside one joint model (modality auxiliary heads, reorientation). Specialist models (a dedicated T2 model) did not help.
- Flip TTA was the best-documented inference gain: +0.012 private for 4×, then +0.005 for 8× with left/right label swap.

**MGMT** (AUC; 4 sequences; near-random signal):
- The 1st place's top model used all 4 sequences, one 3D CNN per sequence; its runner-up used T1wCE only.
- Missing sequences were zero-filled channels (2nd).
- Seed × fold averaging gave +0.03 private for the 9th place, confounded with a resolution change.

### Cited Findings
**RSNA 2023 Abdominal Trauma** (weighted log loss incl. an auto-generated `any_injury`; "GPU Notebook <= 9 hours run-time" — [overview](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/overview))

- **1st, max over all slices per patient.**
  - "To get the predictions, we aggregated the model outputs at slice level and simply took the maximum value for each patient."
  - "Within folds of each models, we are doing slice level ensemble. For different architectures & cross data models (theo/ours), we did ensemble after the max aggregation."
  - Scores: best single 4-fold OOF 0.326, best ensemble OOF "0.31x".

  — [Trauma 1st](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447449)

- **7th, mean over series.**
  - "I simply averaged the results of different series of the same patient id." Unquantified.
  - Ensemble members differ only in sequence length (T = 24/32/48) and folds.

  — [Trauma 7th](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447549)

- **8th, mean over series.**
  - "For patient-level prediction, predictions were averaged across the series, if there were 2." Unquantified.
  - Then a square-root rescale of probabilities, "which improved the private LB loss by 0.4". That figure is as written and is implausibly large next to a final OOF CV of 0.375; it may be a typo.

  — [Trauma 8th](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447706)

- **9th, pick a series per target by contrast phase.**
  - Organ models: "For patients with 2 series, only used the one with the lower aortic_hu for training… This decision was based on experimentation rather than a specific rationale."
  - Extravasation: "Used data with higher aortic_hu for patients with 2 series to highlight features."
  - Post-processing: multiplying the low/high/extravasation probabilities by 1.5 took CV 0.3994 → 0.3889 (−0.0105). Almost all of it came from `any_injury` (0.5533 → 0.4925).

  — [Trauma 9th](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447506)

- **10th, organ-specific models plus a stacker.**
  - Dedicated per-organ models. Left and right kidney crops are concatenated side by side, which "enables us horizontal flip augmentation and TTA".
  - A stacker optimises the full metric: CV 0.3827 → 0.3736, with `any_injury` 0.5528 → 0.4777.
  - The second stage combines attention and max pooling of BiGRU outputs into a "series-level prediction".

  — [Trauma 10th](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447450)

- **2nd (Theo Viel), speed and ensembling.**
  - Speed: "1 frame out of 2 to speed up 2D models inference, and limit stack size to 600"; dicomsdl with GPU processing; "The pipeline without ensembling runs in less than 4h."
  - Scores:

    | Submission | Public | Private |
    |---|---|---|
    | Single crop model + RNN | 0.37 | 0.35 (best private) |
    | 3×2D + 8×2.5D ensemble | 0.35 | 0.35 |

  - "Ensembling did not really help on private ultimately and my best sub is my strongest single model." Things tried during the 2-day deadline extension did not help.

  — [Trauma 2nd](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447453)

- **3rd (Yuji Ariyasu), per-target weighted average.**
  - "A simple weighted average was performed for each target." Many models got weight 0.0 for some targets.
  - Stacking "was slightly worse than the simple pp due to overfit". Final CV 0.3316.

  — [Trauma 3rd](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447464)

- **4th, ensembling and scaling (CV / LB).**

  | Model | CV | LB |
  |---|---|---|
  | pvt-b2 | 0.3783 | 0.41 |
  | pvt-b3 | 0.3750 | 0.41 |
  | pvt-b4 | 0.3786 | 0.40 |
  | maxvit_t | 0.3810 | 0.42 |
  | Ensemble | 0.3570 | 0.40 |
  | Ensemble with scaling | 0.3530 | 0.39 |

  The ensemble improved CV by about 0.02 but matched the best single LB (0.40). Only the log-loss scaling moved LB to 0.39. — [Trauma 4th](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447848)

**RSNA 2025 Intracranial Aneurysm**
- **Competition setup.**
  - Metric: AUC per label, with "Aneurysm Present" weighted 13 and each of the 13 locations weighted 1.
  - Submission: "You must submit to this competition using the provided evaluation API."
  - Limit: "GPU Notebook <= 12 hours run-time".

  — [overview](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/overview)

  The write-ups report time "per Series" and predict each series on its own, so no cross-series fusion appears in any write-up. — [1st](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution)
- **1st.** "The final predictions are an average of the models from 4 of the 5 cross-validation folds." TTA: "averaged the predictions from the original volume and a left-right flipped version". Fail-safe: per-class OOF-mean probabilities if segmentation or ROI extraction fails. — [Aneurysm 1st](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution)
- **2nd, ablation (public / private AUC).**

  | Step | Public | Private | Gain (public / private) |
  |---|---|---|---|
  | Baseline, one fold | 0.84407 | 0.81268 | — |
  | + TTA 4× | 0.87056 | 0.82508 | +0.026 / +0.012 |
  | Improved annotations | 0.88832 | 0.85718 | +0.018 / +0.032 |
  | + left/right augmentation with label swap and TTA 8× | 0.89805 | 0.86228 | +0.010 / +0.005 |
  | + 2-checkpoint ensemble (fold 0, epoch 100 + fold 1, epoch 250) | 0.90035 | 0.86727 | +0.002 / +0.005 |

  - The text gives the final private as 0.86732, the table 0.86727.
  - The final used only two single-fold checkpoints, not a 5-fold set.
  - Modality was handled jointly: "modality 4-class heads", plus "a T2-specific orientation classifier for corrections".
  - Inference "encoder + classification head only… ~9 hours for conservative ensembles".
  - "increasing in the public leaderboard led to consistent improvement in the private leaderboard."

  — [Aneurysm 2nd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution)
- **3rd.**
  - Aggregation: "for each class we sorted the `Aneurysm Present` scores across spatial positions and averaged the top-N scores (Top-N mean)".
  - An 11-model ensemble scored public/private 0.86 / 0.84.
  - Modality/plane auxiliary heads did not help [16 Oct notes].

  — [Aneurysm 3rd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/3rd-place-solution)
- **4th.** "Augmentations: HFlip, VFlip, no TTA". The MaxViT + CoaT ensemble scored CV 0.896 and public/private 0.87 / 0.84. — [Aneurysm 4th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/4th-place-solution)
- **5th.**
  - The detector separates "aneurysm (modality CTA, MRA, MRI T1post) and aneurysm_mri_t2 (modality MRI T2)": a modality-specific *class* within one model.
  - "I could only use a single model trained on almost the full dataset, using only 50 series for evaluation, and selecting the best epoch based on the public leaderboard." The team fell from 2nd public to 5th private [16 Oct notes].

  — [Aneurysm 5th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution)
- **6th.**
  - Plane handling: "This model works on coronal and axial data: sagittal stacks are converted to axial during inference, and ignored during training."
  - Speed: "only half of the frames are inferred for stacks of size > 64, and a quarter for stacks > 128".
  - Budget: "Using 2 models instead of 3 in the pipeline would make the pipeline run in 7 hours".
  - CV path: max over frames 0.87 → sequence model 0.88 → 3-model ensemble 0.895.

  — [Aneurysm 6th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/6th-place-solution)
- **Other places [16 Oct notes].**
  - 15th trained on all planes but ran inference only on the finest (least-interpolated) plane ([15th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/15th-place-solution)).
  - 18th: "a dedicated MRI-T2 model" did not help, and false-positive filtering "may remove high-probability predictions that help AUROC" ([18th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/18th-place-solution)).
  - 11th ran 5 folds plus one all-data model ([11th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/11th-place-solution)).

**RSNA-MICCAI 2021 Brain Tumor MGMT** (AUC; 4 sequences: FLAIR, T1w, T1wCE, T2w; private scores around 0.60–0.62)
- **1st.**
  - Model choice: "the top 1 was using all of the 4 different structural multi-parametric MRI data… while the second top model was just using the 'T1wCE' data". The final is "4 backbones 3D CNNs (each for one structural multi-parametric MRI folder)", i.e. sequence specialists fused.
  - Seed noise: "when I trained a EF-b0 more than 100 times (same everything) I got CV scores between 0.53 and 0.62… Training 5 folds 100 times… between 0.52 and 0.56." Ideas were ranked on averages over 100–250 runs.
  - "Ensembling was not really improving the score."

  — [MGMT 1st](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/281347)
- **2nd.** The 4 sequences were fused as channels: "If one (or more) MRI image type is missing for a patient, that image channel(s) is filled with 0." At inference, the mean of the 5 fold models. — [MGMT 2nd](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/280033)
- **3rd.** "four EfficientNet-B3. One for each kind of MRI scans". Patient aggregation was a heuristic choosing between the maximum and minimum of the per-sequence predictions, depending on which differed more from the average. Unquantified. — [MGMT 3rd](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/287713)
- **6th.**
  - One model per MRI type, with patient prediction = "the average of all the probability outputs of those chunks… this kind of emsembling make the model more robust".
  - "TTA Zoom In 1.2" (unquantified).
  - Scores: validation 0.685, public 0.678, private 0.60696.

  — [MGMT 6th](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/280402)
- **9th.** "an ensemble of these 5 trainings (5 folds each one)… scored 0.66701 on public and 0.60186 on private, which is better than previous submission (based on single training on 224 image size) on 0.07 and 0.03 respectively". The input size also changed from 224 to 384, so the gain is confounded. — [MGMT 9th](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/279826)

### Inferences
**1. Mean across series is the field-tested default.** Two trauma gold solutions (7th, 8th) and LSDC 11th used a plain mean across a patient's series. The trauma 1st place's max-over-slices was built into a max-aggregation pipeline, which ours is not (attention-MIL). For our attention-MIL model, average the per-finding outputs of the alternative-fill passes.

**2. A per-finding contrast preference is a weaker option.** The trauma 9th place's phase-based pick is the knee analogue of "prefer the fat-sat series for marrow/oedema findings". It was chosen by experiment and never quantified. In 2 days, a learned per-finding preference is riskier than a plain mean. Use it only if OOF shows the mean losing on specific findings.

**3. Plane/sequence specialists rarely win alone, but help as blend members with per-target weights** (aneurysm 18th; MGMT 1st/3rd; LSDC 14th and 2nd). A knee plan:
- Train cheap head-only specialists on cached window embeddings, for example sagittal slots only and coronal slots only.
- Blend them with the multi-plane model using per-finding non-negative weights fitted on OOF, or a few fixed groups, e.g. cruciate findings → sagittal-heavy, collateral → coronal-heavy. These groups are my anatomical prior, not an evidence-based mapping.

**4. Left/right flip TTA has the most consistent measured AUC gain** (aneurysm 2nd: +0.012 private for 4×, +0.005 more for 8×). For the knee it is label-preserving (see the 16 Oct notes). Per pass, it is cheaper than series TTA because every study benefits.

**5. Ignore the trauma post-processing gains for our AUC metric.** The scaling, sqrt and stacking gains came from the log-loss `any_injury` term. Per-finding monotone transforms do not change AUC (see Q3).

### Gaps
- None of the trauma write-ups I read gives a number for mean-over-series against single-series selection. The 3rd, 4th and 6th place pages I read do not describe how two-series patients were handled.
- In aneurysm 2025 each series was scored separately, so there is no precedent there for fusing multiple series at test time.
- MGMT signal was near random (private around 0.60–0.62), so its multi-sequence conclusions are weak evidence.

## Q3. Which inference-time tricks gave ≥ +0.003 late in a competition, and which were reported useless (including rank vs probability averaging, per-class weights and AUC post-processing)?

### Takeaway
**Measured gains of at least +0.003 from inference-time changes:**
- Flip TTA (aneurysm 2nd: +0.012 private for 4×, +0.005 for 8×).
- Small ensembles of *different* models (aneurysm 2nd +0.005 and 15th +0.019 private; SIIM-ISIC 1st, an 18-model rank-average, +0.008 private over the best single model and +0.020 over the mean single).
- Averaging over alternative localisations/crops (LSDC 2nd and 5th; unquantified but "significant" or "best private").

**Reported as useless or harmful:**
- Large or stacked blends tuned on OOF (trauma 3rd overfit; aneurysm 9th: CV 0.858 vs private 0.823).
- Ensembling when one strong model exists (trauma 2nd; MGMT 1st).
- Adding TTA to a CV-chosen model (LSDC 13th; confounded).
- False-positive filtering under AUC (aneurysm 18th).

**Post-processing under AUC.** Every documented RSNA post-processing gain (temperature, coefficient scaling, sqrt, 1.25× on the max level) was a log-loss calibration. Under per-finding AUC, any per-finding monotone transform has *zero* effect. I found no RSNA write-up that reports a cross-sample re-ranking step that improved AUC.

### Cited Findings
**Inference-time tricks with measured deltas**

| Trick | Where | Metric | Delta | Source |
|---|---|---|---|---|
| 4× flip TTA | Aneurysm 2nd | AUC | +0.026 public / +0.012 private | [2nd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution) |
| 8× TTA + left/right label-swap augmentation | Aneurysm 2nd | AUC | +0.010 public / +0.005 private | [2nd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution) |
| 2-checkpoint ensemble | Aneurysm 2nd | AUC | +0.002 public / +0.005 private | [2nd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution) |
| 2-model weighted average | Aneurysm 15th | AUC | +0.019 private over the best single model [16 Oct notes] | [15th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/15th-place-solution) |
| Sequence head + 3-model ensemble | Aneurysm 6th | AUC (CV) | 0.87 → 0.88 → 0.895 | [6th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/6th-place-solution) |
| 18-model rank average | SIIM-ISIC 2020 1st (non-RSNA, AUC) | AUC | private 0.9408 best single → 0.9490 (+0.008); +0.020 over the mean single model (0.9288, my calculation from their table); public 0.9387 → 0.9442 | [paper](https://arxiv.org/abs/2010.05351) |
| 27-crop slice/xy-offset aggregation | LSDC 2nd (Ian) | log loss | "improved performance significantly" (unquantified) | [LSDC 2nd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539452) |
| Stage 2 run per stage-1 model, then averaged | LSDC 5th | log loss | "highest score on the private leaderboard" (unquantified) | [LSDC 5th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539472) |
| Simple average of 30 models | LSDC 3rd | log loss | CV 0.3858 → 0.3643 | [LSDC 3rd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453) |
| Adding multi-view pipelines to a sagittal-only one | LSDC 6th | log loss | private 0.420 → 0.401 | [repo](https://github.com/TheoViel/kaggle_rsna_lumbar_spine) |
| Seeds × folds (5 × 5) + larger input | MGMT 9th | AUC | +0.07 public / +0.03 private (confounded) | [MGMT 9th](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/279826) |
| Coefficient scaling (log-loss only) | Trauma 9th | log loss | CV 0.3994 → 0.3889 | [Trauma 9th](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447506) |
| Metric-optimising stacker (log-loss only) | Trauma 10th | log loss | CV 0.3827 → 0.3736 | [Trauma 10th](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447450) |
| Ensemble, then scaling | Trauma 4th | log loss | Best single CV 0.3750 / LB 0.41 → ensemble 0.3570 / 0.40 → with scaling 0.3530 / 0.39 | [Trauma 4th](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447848) |

**Reported useless or harmful**
- **Ensembling did not move private.**
  - Trauma 2nd: the ensemble moved public 0.37 → 0.35 but private stayed at 0.35. "my best sub is my strongest single model." — [Trauma 2nd](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447453)
  - MGMT 1st: "Ensembling was not really improving the score." — [MGMT 1st](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/281347)
- **Stacking overfit.**
  - Trauma 3rd: it was "slightly worse than the simple pp due to overfit". — [Trauma 3rd](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447464)
  - Aneurysm 9th: the Nelder-Mead stack scored CV 0.858 against private 0.823 [16 Oct notes]. — [9th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/9th-place-solution)
- **Picking or filtering at inference.**
  - LSDC 7th: picking only the highest-confidence axial series was "not necessarily the best", in the authors' own words. — [LSDC 7th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539486)
  - Aneurysm 18th: false-positive filtering hurt AUROC [16 Oct notes]. — [18th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/18th-place-solution)
- **More TTA on the CV-best model.** LSDC 13th: TTA 40 with the CV-best model was its worst private (0.4215 vs 0.4118 for TTA 16). Confounded with the backbone. — [LSDC 13th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539510)
- **Specialists that added nothing.** LSDC 11th: an axial spinal model and an axial whole-crop subarticular model "did not improve when included in the ensemble". — [LSDC 11th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539569)
- **Top places without TTA.** Aneurysm 4th placed with "no TTA" ([4th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/4th-place-solution)). Aneurysm 7th placed with a single model and no TTA [16 Oct notes] ([7th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution)).

**Rank versus probability averaging; per-class weights**
- **SIIM-ISIC 1st (AUC): rank average.** "The final ensemble is a simple average of the 18 models' probability ranks. In other words, we transform each model's probability predictions to uniform [0,1] before averaging them." No probability-average ablation is given.
  - Single-model score spread (std) by metric, from most to least stable: cv all 0.0012 < cv 2020 0.0043 < private 0.0060 < public 0.0093.
  - "we did not use public LB score feedback in any way when developing models".
  - Diversity came partly from "5 folds are split differently among models".

  — [SIIM-ISIC 1st paper](https://arxiv.org/abs/2010.05351)
- **RANZCR 1st (AUC):** rank averaging on 5 of 11 columns gave +0.00032 CV [16 Oct notes]. — [RANZCR 1st](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/discussion/226633)
- **Per-target weights:**
  - LSDC 14th weights 7 sources independently per target (CV 0.379, LB 0.35 / 0.41) ([LSDC 14th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539459)).
  - Trauma 3rd used per-target weights with many zeros ([Trauma 3rd](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447464)).
  - LSDC 11th used Nelder-Mead per condition ([LSDC 11th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539569)).
  - LSDC 2nd weighted spinal sagittal 1 vs axial 0.5, with "not much difference from equal weighting" ([LSDC 2nd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539452)).
- **Order of aggregation** (trauma 1st): folds were averaged at the slice level *before* the max-over-slices, and different architectures after it. — [Trauma 1st](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447449)

**Post-processing in the write-ups (all for log loss)**

| Place | Post-processing | Source |
|---|---|---|
| LSDC 3rd | Temperature 0.91 on spinal-canal logits | [link](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453) |
| LSDC 12th | Temperature 1.3 | [link](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/540001) |
| LSDC 2nd (Yuji) | ×1.25 on the maximum spinal-severe level [8 Oct notes] | [link](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution) |
| Trauma 9th | ×1.5 coefficients | [link](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447506) |
| Trauma 8th | Square-root rescale | [link](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447706) |
| Trauma 1st | "tuning scaling factors based on CV scores" | [link](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447449) |

**Robustness at inference.** Aneurysm 1st falls back to per-class OOF-mean probabilities when segmentation or ROI extraction fails. — [Aneurysm 1st](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution)

### Inferences
**1. Calibration tricks are no-ops for us.** Macro AUC is computed per finding across studies, so temperature, scaling or power transforms applied to one finding cannot change its AUC. Skip all the LSDC/trauma post-processing tricks. Post-processing can only matter if it re-ranks studies within a finding:
- blending members;
- adding a study-level "abnormal" or co-occurrence signal;
- series TTA.

**2. Rank-average heterogeneous members per finding.** Rank (or z-score) each member per finding before averaging (SIIM-ISIC 1st). This matters most when members differ in calibration, e.g. our ConvNeXt against public components or a specialist head. Between near-identical seeds it makes almost no difference (RANZCR +0.0003).

**3. Ship order for the last 48 h, by measured evidence per GPU-hour:**
1. left/right flip TTA on every member;
2. series TTA on the duplicate subset (validated on OOF first);
3. 1–2 *different* members (backbone, or a plane-specialist head over cached embeddings) blended with equal or few-group weights;
4. more seeds of the same model only if time remains.

**4. Weighting discipline.**
- Per-finding weights fitted on OOF are acceptable only when non-negative and few (LSDC 14th, trauma 3rd).
- Unconstrained Nelder-Mead, or stackers on about 4.4k studies × 12 findings, risk the trauma 3rd / aneurysm 9th overfit.

**5. Final pair.** Keep one CV-best and one robust ensemble as the two final submissions. CV-best picks lost on private in LSDC 13th, and public-best picks lost in aneurysm 5th.

### Gaps
- No RSNA write-up reports a direct rank-vs-probability averaging ablation under AUC. SIIM-ISIC 1st used rank averaging without one.
- No RSNA AUC competition write-up I read reports a post-processing step (beyond aggregation choices) that raised AUC.
- The trauma 8th place's "0.4" sqrt gain cannot be checked and may be a typo.

## Q4. Full-data retrain (no holdout) versus fold models (and seeds versus folds): what winners reported

### Takeaway
**No RSNA winner published a clean ablation of full-data retrains against fold models.** What exists is structural evidence:
- The trauma 1st place's final ensemble mixed 4-fold models with many full-data models trained under several seeds.
- LSDC 11th (one member) and aneurysm 11th (one model next to 5 folds) also shipped all-data models.
- Aneurysm 1st and 2nd shipped *subsets* of fold models (4 of 5 folds; two single-fold checkpoints) plus TTA.

**The one cautionary case confounds two things.** Aneurysm 5th shipped a single near-full-data model whose epoch was chosen on the public LB, and fell from 2nd to 5th.

**Seeds matter most when the data is small or noisy.** In MGMT, identical reruns spanned CV 0.53–0.62, and seed × fold averaging added about +0.03 private (confounded).

### Cited Findings
- **Trauma 1st: full-data, multi-seed models dominate the final ensemble.**
  - "Final ensemble for all organs model includes multiple Coat medium and V2s based models trained on either 4 Folds or Full data." — [Trauma 1st](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447449)
  - The README's reproduction commands list full-data scripts run with several seeds: `train_v2s_try5_v10_fulldata.py` (4 seeds), `train_try11_tf_efficientnetv2_s_in21ft1k_v1_fulldata.py` (4 seeds), `train_fullextracoatsmall384.py` (2 seeds), `train_coatmed384fullseed.py`. Alongside them sit fold-specific runs (`--fold 1`, `--fold 3`). No separate score is given for the full-data members. — [Trauma 1st repo](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution)
- **LSDC 11th (YumeNeko's part).** "For submission, I used the weights trained on all data." Unquantified. The team blended this pipeline with fold-based pipelines using Nelder-Mead weights. — [LSDC 11th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539569)
- **Aneurysm 11th.** 5 folds plus one all-data CoAtNet model in the ensemble [16 Oct notes]. — [Aneurysm 11th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/11th-place-solution)
- **Aneurysm 5th.** "a single model trained on almost the full dataset, using only 50 series for evaluation, and selecting the best epoch based on the public leaderboard". The team dropped from 2nd public to 5th private (−0.046) [16 Oct notes]. — [Aneurysm 5th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution)
- **Aneurysm 1st.** "an average of the models from 4 of the 5 cross-validation folds", plus flip TTA. — [Aneurysm 1st](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution)
- **Aneurysm 2nd.** The final was "Epoch 100, Fold 0 + Epoch 250, Fold 1 (Ensemble)" with 8× TTA. That ensemble added +0.005 private on top of TTA. — [Aneurysm 2nd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution)
- **Fold ensembles elsewhere.**
  - LSDC 9th: 6 models × 5 folds, simple mean ([LSDC 9th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539690)).
  - LSDC 1st: "5 fold for each task" ([LSDC 1st](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/540091)).
  - Trauma 7th: members differ only by sequence length and folds ([Trauma 7th](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447549)).
- **MGMT, seeds versus folds.**
  - 1st: 100 identical reruns of EffNet-B0 gave CV 0.53–0.62; averaging 5 folds narrowed this to 0.52–0.56 ([MGMT 1st](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/281347)).
  - 9th: 5 seeds × 5 folds gave +0.07 public / +0.03 private over a single 5-fold run. The run also changed input size from 224 to 384 ([MGMT 9th](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/279826)).
- **SIIM-ISIC 1st (non-RSNA, AUC).** Diversity came from members whose "5 folds are split differently among models". No full-data retrain is described. — [paper](https://arxiv.org/abs/2010.05351)
- **No RSNA-specific comparison found.** Two web searches for full-data versus fold comparisons in RSNA write-ups returned no relevant result (see Gaps).

### Inferences
**1. Full-data retrain: only with a fixed recipe, and as an added member.**
- **Recipe.** Use exactly the fold recipe and epoch count. Don't pick the epoch on LB: the aneurysm 5th place's failure mixed full-data training with LB-picked epochs.
- **How to use it.** Trauma 1st's pattern is "full-data seeds *added to* fold models". Use a full-data seed as an additional member, not a replacement for the 5 folds. Its gain cannot be checked on OOF, so it rides on faith in the recipe.
- **Timing.** One full-data run costs about as much as one fold, so it fits 2 days only if a GPU slot is free.

**2. If inference time is the bottleneck, prune folds before dropping TTA.** Aneurysm 1st and 2nd shipped 4 of 5 folds or 2 single-fold checkpoints, plus flip TTA. Dropping 1–2 fold checkpoints frees time for flip TTA or series TTA, which had larger measured gains (+0.005–0.012 private) than the extra checkpoints.

**3. Seeds only reduce variance.** With about 4.4k studies and 12 findings, seed noise should be far smaller than in the small, near-random MGMT task. A second seed of the same model is lower-value than a different backbone or a plane-specialist member (my inference from Q1/Q3).

### Gaps
- No RSNA top write-up reports an A/B of full-data retrain against fold ensembles, or a CV-free check of full-data members. The trauma 1st place gives none for its full-data models.
- Two web searches returned only general or non-RSNA material: fast.ai and Zindi forum threads, and slide decks. I did not use any of it as evidence.
- I did not find the epoch-count rule each team used for full-data retrains (for example, the same epochs, or epochs × 1.25).
- The MGMT seed gain is confounded with an input-size change, and that competition's signal was near random.
