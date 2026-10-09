# Past Kaggle medical-imaging winners: tricks that transfer to multi-series knee MRI multi-label classification

Method note: Kaggle discussion and write-up pages are JavaScript-rendered, so a direct fetch returns an empty shell. I read them through a JS-rendering reader proxy (r.jina.ai) and quote the original Kaggle URLs below. The URLs came from the curated index at github.com/faridrashidi/kaggle-solutions (data/competitions.yml). Public and private leaderboard numbers come from the Kaggle API (`kaggle competitions leaderboard -d` gives the full public LB CSV; `-s --page-size 200` gives the private top 200). I joined the two by TeamId and computed the shake-up statistics myself. Those are labelled "computed". "LB" means leaderboard and "CV" means local cross-validation. In LSDC, trauma, C-spine and ICH the metric is a weighted log loss, so lower is better. In RANZCR, aneurysm and brain tumour it is AUC, so higher is better. COVID and VinBig use mAP and mammography uses probabilistic F1.

## Q1. RSNA 2024 Lumbar Spine Degenerative Classification (LSDC): what did 1st–9th do, and what was each component worth?

### Takeaway
Every top-9 team used the same pipeline. Stage 1 localises the level or slice and the keypoint. Stage 2 classifies crops built from a few adjacent slices around each keypoint, using a 2D/2.5D CNN with a sequence or attention aggregator over slices. Teams then fused several series and ensembled small backbones. The biggest reported gains came from four things: (a) attention-MIL/BiLSTM aggregation over slices (1st: public 0.37→0.35→0.33 log loss); (b) robustness to stage-1 errors through jittered training and multi-crop/slice-offset TTA; (c) sharing weights across levels and sides, which gives 5–10× more training samples; (d) cleaning noisy labels with ensemble out-of-fold (OOF) predictions (2nd: about 1% better on both public and private). Small models trained for short schedules beat large ones.

### Cited Findings
**Leaderboard context (computed).** The private top 10 had public ranks [5, 1, 2, 6, 11, 13, 8, 9, 3, 4]. 1st "Avengers" (solo, wadakoki) scored private 0.38894 / public 0.33922 (5th public). 2nd IanPan-Kevin-Yuji-Bartley scored 0.38959 / 0.33191 (1st public). 3rd SonySpine scored 0.39137 / 0.33660. 9th Adam Narai was 3rd on public (0.33761) and fell to 9th. Spearman correlation between private top-200 rank and public rank was 0.918, and the median |rank change| in the private top 50 was 5. Every team's log loss was about 0.05 worse on private than on public. — [LSDC leaderboard](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/leaderboard); prize ranking confirmed by [RSNA news](https://www.rsna.org/news/2024/november/2024-ai-challenge-winners)

**1st (wadakoki, "Avengers").**
- Stage 1 had three parts: (i) a 3D ConvNeXt predicted the instance number (slice) per level, trained with classification + regression heads and ensembled by median. Accuracy was ±0 slices 71.08% / ±1 slice 27.04% for classification and 67.48% / 30.59% for regression. (ii) 2D ConvNeXt-base and EffNetV2-L regressed (x, y). These were pretrained on a community coordinate dataset, which was "slightly better" than ImageNet init. (iii) Axial slices were found by projecting sagittal points (hengck23's 2D→3D method). — [1st place](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/540091)
- Stage 2 cropped 5 slices around the predicted slice, with condition-specific crop windows on a 512² resize (e.g. sagT2 for spinal canal stenosis: 96 px left, 32 px right, ±40 px vertical). Encoders were ConvNeXt-small and EffNetV2-s. — [1st place](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/540091)
- **Ablation (public LB log loss):** attention-based MIL took 0.37 → 0.35. Adding a BiLSTM before attention pooling, an auxiliary attention-score loss (cross-entropy on per-slice attention logits) and ensembling took 0.35 → 0.33. — [1st place](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/540091)
- The "crucial" augmentation was random shifts of the target slice (±2, with probability set to the slice model's measured error rate) plus random coordinate shifts of ±10 px. This made stage 2 robust to stage-1 errors. — [1st place](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/540091)
- What didn't work: Mamba or self-attention in place of the BiLSTM; long training (final runs were 7 epochs for ConvNeXt-s and 14 for EffNetV2-s); larger models ("convnext-large < convnext-base < convnext-small"); ViTs; adding extra series to a condition (e.g. sagT1 for spinal canal stenosis). Everything was trained on Colab T4s. — [1st place](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/540091)

**2nd (Ian Pan, Yuji Ariyasu, Kevin, Brendan Artley).**
- Yuji's part used small per-target models that treat levels and left/right as the same task. YOLOX boxes defined the regions, ConvNeXt-small classified them, and a 5-image MIL model handled sagittal input. — [2nd place (Yuji)](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution)
- **Noise reduction.** Using the team's ensemble OOF (CV 0.3687), they removed training samples where |label − prediction| ≥ 0.8, with class coefficients for moderate and severe, then retrained. Yuji: "This magic improved our score by 1% on both public and private leaderboards". The idea came two days before the deadline. — [2nd place (Yuji)](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution)
- Post-processing: within each study, the maximum spinal-severe prediction across the 5 levels was multiplied by 1.25. Member predictions were blended with a simple weighted average. — [2nd place (Yuji)](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution)
- Ian's part trained a CNN-transformer for slice localisation (smooth-L1 distance + BCE) and 2D keypoint regression. Each target got 27 training and test crops (3 slice offsets × 9 in-plane offsets), aggregated by median or mean, which "improved performance significantly". Training on crops from ground-truth coordinates was slightly better than training on crops from predicted coordinates. Simple 2D CNNs did best (maxvit_tiny_tf_512, coatnet_1_rw_224, NFNet-l0, CSN-ResNet101); attention-pooling and CLAM-MIL models only added diversity. His part scored CV 0.3805 / LB 0.35. — [2nd place (Ian's part)](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539452)
- Ian named the three most important elements: multi-crop training and aggregation; using both sagittal T2 and axial T2 for spinal stenosis (radiologists read axial); and noise reduction. On noise reduction he noted, "I thought the models could learn through the noise… [it] resulted in a notable improvement in our LB score". Duplicated axial series both contributed crops, and the slice model also predicted whether a level was present in a series. — [2nd place (Ian's part)](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539452)

**3rd (SonySpine).**
- Stage 1 used a CenterNet keypoint detector (EffNet-B6 + FPN). It was trained on pseudo-labels for the unlabelled studies, and the team **manually reviewed and corrected every coordinate label**. — [3rd place](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453)
- Stage 2 used a "Center" and a "Side" classifier: a 2D encoder plus attention over slices (15 equally spaced sagittal slices and 10 axial). Inputs came from all three series. **Each level was treated as an independent sample (5× data), and each side too (10× data; right side mirrored).** This "contributed to the improvement in accuracy". Auxiliary losses covered other conditions' severity and slice-level predictions. Class weights were [1, 2, 4]. Flip TTA on axial improved LB. — [3rd place](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453)
- Small encoders at 160–320 px (ResNet18, ConvNeXt-N/T, EffNetV2-S, MaxViT-N); 10–20 epochs; AdamW lr 2.5e-5; drop_path 0.2–0.3; CoarseDropout and optional Mixup. — [3rd place](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453)
- **Ensemble:** best single model CV 0.3858; a simple average of 30 models reached CV 0.3643. Soft pseudo-labels made "not significant" difference and were kept only for diversity. Temperature 0.91 was applied to spinal-canal logits. — [3rd place](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453)
- What didn't work: a one-stage model; multi-level multi-disease models; per-level or per-side specialist models; 3D CNN; 2.5D-CNN + attention; 2D-CNN + LSTM; focal loss; long epochs. — [3rd place](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453)

**4th (yu4u + tattaka, "SPINE CHART").**
- Stage 1: a 2.5D CNN + LSTM assigned axial slices to levels, and UNet keypoints were trained with BCE×0.2 + Dice×0.8.
- Stage 2: a multi-view (sagT1/sagT2/axial crops), multi-condition model combining a transformer encoder and attention pooling. The "key technique" was attention pooling *before* the transformer, with an auxiliary loss on it.
- Stacking used two methods: (i) an MLP that learns the residual over a Nelder-Mead-optimised blend; (ii) per-level LightGBM/XGBoost that takes only predictions of the same target type ("using predictions for different targets was not effective"). — [4th place](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539443)

**5th (siwooyong + ahmedelfazouan, "Two people").**
- Stage 1 used heatmap detection with Gaussian-expanded labels: Gaussian in x/y, weight ×0.5 per slice away in z. This "improved the convergence speed… and the z-axis accuracy". A 2D UNet plus a sequence model beat a 3D UNet on z.
- **Level-wise sequence modelling**, an RNN over only the 5 conditions at the same level instead of all 25, "significantly improved our score".
- **Two-step training:** first train the whole network for AUC with no metric weighting, then freeze the backbone and fine-tune only the head with the weighted metric loss. This "significantly improve[d]" results compared with training on the weighted loss directly.
- **CutMix with p=1.0** was the single most effective augmentation for AUC. ±1 z-jitter, left/right label flips and label smoothing gave no gain.
- "TTA-like ensemble": running each stage-2 model on each stage-1 model's crops separately gave their best private score. — [5th place](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539472)

**6th (Theo Viel, Darragh Hanley, Christof Henkel, "NVSpine").**
- Four folds. "Our experience from previous RSNA competitions led us to expect a small yet reasonable shake-up, and we trusted our CV more than public LB."
- Final scores were CV 0.382 / public 0.355 / private 0.401. That was 13th on public and 6th on private.
- A CoAtNet + RNN model on sagittal data alone reached about 0.42 private. Training the two sagittal series together beat separate models. Adding frames helped, because 5 frames was not enough.
- The study-level 3-view model concatenated per-view crops into a single 288×288×9 input. A missing view was replaced with zeros, and **one view was dropped with 10% probability** as augmentation.
- An MLP stacker optimised the metric, with each target independent and weights shared across levels.
- What didn't help: pseudo-labelling external data, denoising, an auxiliary localisation decoder, and a bi-encoder for axial + sagittal. — [6th place](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/541813)

**7th (HLIP: lhwcv + hengck23 + others).**
- lhwcv's 2-stage keypoint and crop models fused the three views in feature space and used flip TTA, reaching CV 0.373 / LB 0.35 / private 0.40. — [7th place (lhwcv)](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539486)
- hengck23's one-stage model (heatmap + DSNT "point-masking and pool") blended at 0.75 two-stage + 0.25 one-stage improved the ensemble "even with bugged implementation". — [7th place (hengck23)](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539439)

**8th (K_mataro).**
- Instead of cropping, a heatmap from an auxiliary detector weights the feature pooling. A self-supervised multi-view similarity loss "slightly improved the public/private score" but not CV. The author guessed that "the labels of test dataset are cleaner than train dataset". A manual relabelling attempt failed for lack of domain knowledge. — [8th place](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539548)

**9th (Adam Narai).**
- Gaussian-heatmap DeepLabV3+ keypoints; 50×50 mm ROIs with 5 slices as channels; 2.5D ResNet18/Swin-T/ConvNeXt-Nano with a GRU head. Channel shuffle on axial ROIs "significantly improved performance". The final model was the simple mean of 6 models × 5 folds. — [9th place](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539690)

### Inferences
- LSDC is the closest structural analogue to the knee task: multi-series MRI (sag/cor/ax, T1 plus fluid-sensitive), multiple conditions per study, and a 9 h Kaggle code-competition limit. Its recipe was localise, then crop a few slices, then use a 2D CNN with slice attention or an RNN, then fuse series, then ensemble small models. Our 2.5D CoAtNet attention-MIL is the right family.
- LSDC's biggest lever (accurate keypoint crops) depended on provided coordinate labels. If the knee data has no coordinates, the transferable parts are: (i) robustness to localisation errors through jitter during training and offset-TTA; (ii) multi-series feature fusion with view dropout; (iii) OOF-based label-noise cleaning; (iv) short schedules on small backbones.
- The 1st-place +BiLSTM-before-attention + aux-attention-loss change is a cheap upgrade to an existing attention-MIL head.

### Gaps
- None of the LSDC write-ups give per-component deltas in a common metric except the 1st (public LB path), 2nd (≈1% from noise cleaning) and 3rd (single model → ensemble CV). Most other gains are described only as "significant".
- Exactly how the 2nd place's class coefficients worked in the noise filter is not described.

## Q2. RSNA 2025 Intracranial Aneurysm Detection (multi-modality, series handling): what did the winners do?

### Takeaway
The winners used coarse-to-fine ROI pipelines that are robust across modalities (CTA, MRA, T1-post, T2). Their main gains came from (a) cropping an ROI and keeping resolution (the 4th place's ROI crop moved its score from about 0.70 to 0.80; the 3rd place's higher-resolution feature maps moved LB from 0.77 to 0.81); (b) dense or auxiliary supervision (the 1st place's auxiliary segmentation loss: 0.876→0.902; pretraining the backbone on segmentation: 0.794→0.902); (c) corrected labels and annotations (2nd: private +0.032); (d) TTA, especially left–right flips with label swap (2nd: private +0.012 and +0.005); (e) soft pseudo-label distillation (4th: CV +0.03, private +0.01). The most stable solution (1st) had the smallest public→private drop. A team that tuned on the public LB fell from 2nd to 5th.

### Cited Findings
**Leaderboard (computed; a host "perfect_submission" row sits at private rank 1 and is excluded here).**

| Private rank | Team | Private | Public (rank) | Δ |
|---|---|---|---|---|
| 1 | tomoon33 | 0.879 | 0.883 (3) | −0.004 |
| 2 | BraveCoWCoW | 0.867 | 0.900 (1) | −0.033 |
| 3 | BTYND | 0.851 | 0.867 (6) | −0.017 |
| 4 | Harshit Sheoran | 0.848 | 0.872 (5) | −0.024 |
| 5 | "more CV challenge pls" | 0.847 | 0.893 (2) | −0.046 |
| 6 | Ian, Theo & Bartley | 0.840 | 0.845 (9) | −0.005 |

Source: [Aneurysm leaderboard](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/leaderboard)

**1st (tomoon33).**
- Data handling: about 60 series excluded (orientation anomalies, corrupt DICOM, implausible spacing). Within a series, slices were filtered to the majority Rows×Cols/PixelSpacing configuration. dcm2niix was used, with a gdcmconv fallback. Volumes were reoriented to a common orientation and z-scored. Folds were multilabel-stratified 5-fold.
- Segmentation: three nnU-Net v2 models, coarse to fine. A 1 mm coarse model localised a 140 mm ROI using DBSCAN on the largest cluster. Two fine vessel models followed, one balanced and one recall-focused with Tversky + SkeletonRecall loss.
- Classifier: input was a 128×256×256 ROI. The backbone was the nnU-Net pretrained on vessel segmentation. "Vessel Region-Masked Pooling" extracted 13 per-location features, which went through a "Location-Aware Transformer". An auxiliary loss reconstructed a 5-px sphere at each aneurysm. Loss weights were 0.1 (locations), 0.05 (present) and 1.0 (aux).
- Training and inference: EMA weights, 4 of the 5 folds, and left–right flip TTA.
- Fail-safe: if the pipeline hit an anomaly it returned per-class OOF-mean probabilities.
- Inference time on 2×T4 per series: preprocessing 4.0 s, segmentation 10.7 s, classification 3.3 s. — [1st place](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution)
- **1st-place ablation** (folds 0–2, 96×192×192): final model 0.902. Without Location-Aware Transformer 0.896. Dice instead of FocalTversky++ 0.899. All loss weights = 1: 0.884. **Without backbone pretraining 0.794.** Without fine segmentation model 3: 0.891. **Without the auxiliary segmentation loss 0.876.** Full resolution 128×256×256: **0.916**. Author: "Pretraining the backbone on the vessel segmentation task was the most important factor". — [1st place](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution)
- Design history: one end-to-end 3D classifier detected "presence" but failed on the 13 locations. A variant that fed the vessel mask as an input channel needed 14 forward passes and was too slow, so it was replaced by a single pass with masked pooling. — [1st place](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution)

**2nd (BraveCoWCoW).**
- Pipeline: a tri-axial 2D nnU-Net ROI, then a 3D nnU-Net trained for multiple tasks (vessel and aneurysm segmentation + classification) with a modality-4-class head. Everything was resized to 224³. Inference ran only the encoder and classification head, about 9 h on 2×T4.
- Robustness: when test DICOM tags were missing, spacing was mapped by slice count, and a T2-specific orientation classifier corrected orientation. They deliberately ran with no try/except 0.5 fallback.
- External data: TopCoW/TopBrain annotations. Provided labels with left/right and supra/infra-clinoid mix-ups were corrected manually. — [2nd place](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution)
- **2nd-place ablation (public / private):** baseline 0.844 / 0.813 → +TTA 4× 0.871 / 0.825 → +improved annotation data 0.888 / 0.857 → +left–right aug with label swap and TTA 8× 0.898 / 0.862 → +2-model ensemble 0.900 / 0.867. "Increasing in the public leaderboard led to consistent improvement in the private leaderboard". They saw a consistent 3–4% public–private gap and attributed it to a differing distribution of abnormal cases. — [2nd place](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution)

**3rd (BTYND).**
- ROI: YOLOv8 boxes on mid-sagittal and mid-coronal slices gave a 3D ROI (90 or 120 mm). They judged vessel segmentation "not sufficiently robust".
- Model: 3D ResNet-18 (timm-3d) with a 14-class head on *every* feature-map voxel, aggregated by Top-N mean.
- **Reducing stride so the feature map grew from 8³ to 25³, with input 128³→196³, raised single-fold LB from 0.77 to 0.81.** An 11-model ensemble scored 0.86 public / 0.84 private.
- An EffNetV2-S regressor predicted voxel spacing from pixels to cover missing DICOM tags (MAE about 0.015–0.07 mm). It **raised private from 0.84 to 0.85**.
- What didn't work: vessel segmentation, modality/plane auxiliary heads, MIL/LSTM/decoder heads. — [3rd place](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/3rd-place-solution)

**4th (Harshit Sheoran, solo, 14 days).**
- ROI: a DINOv3 ViT regressed crop box (x1, x2, y1, y2) from 48 slices at 128², with MAE 0.027 in normalised coordinates.
- Classifier: CoaT-Lite-Medium at 384 px, with max aggregation over slices. — [4th place](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/4th-place-solution)
- **Ablation (CV, public/private):** ROI crops 0.805 (0.78/0.78) → +rotate ±25° 0.83 (0.80/0.81) → **+2.5D (−2, 0, +2 slices) 0.86 (0.86/0.83) → +soft pseudo-distillation on about 500k remaining "potentially positive" slices 0.89 (0.86/0.84)** → +MaxViT ensemble 0.896 (0.87/0.84). In comments the author said the ROI crop moved the score "from 0.7 to 0.8". Weighted sampling did not help. — [4th place](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/4th-place-solution)

**5th ("more CV challenge pls", longb173).**
- They hand-annotated aneurysm boxes (from the given centroids) and brain boxes. The **brain-crop detector improved accuracy by about 0.03–0.05**. **Horizontal flip with left/right label swap gave about +0.01.**
- Models: ViT-L/EVA-L at 384 (OOF 0.850 / 0.855) and an mit-b4 FPN classification + segmentation model (0.855).
- **Cleaning plus external data:** negatives whose OOF "aneurysm present" score exceeded 0.9 were relabelled positive, and two OpenNeuro TOF-MRA sets were pseudo-labelled. This raised OOF to 0.8558 (ViT), 0.8579 (EVA) and 0.8629 (mit-b4). The final ensemble scored OOF 0.8823 / public 0.89.
- Risk taken: "I could only use a single model trained on almost the full dataset, using only 50 series for evaluation, and selecting the best epoch based on the public leaderboard". The team fell from 2nd public to 5th private, the largest drop in the top 6 (−0.046). — [5th place](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution); [leaderboard](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/leaderboard)

**6th (Ian Pan, Theo Viel, Brendan Artley).**
- Pipeline: skull crop, then a 2.5D frame classifier (coatnet_rmlp_2_rw_384, maxvit_rmlp_base_rw_384), then a sequence model. Max aggregation alone gave CV 0.87, the sequence model 0.88, and a 3-model ensemble 0.895. Public and private were both 0.84, the most stable result in the top 6.
- **Each of these gave about +0.01 CV:** a fixed-ratio extra ROI crop; strong ShiftScaleRotate and colour augmentation; CutMix/Mixup; horizontal flip that also flips targets; 2 adjacent frames as channels; spacing-aware adjacent-frame sampling; manually refined localiser labels.
- External OpenNeuro data "did not really help CV".
- Inference saving: frames were subsampled (½ for stacks over 64 frames, ¼ over 128). — [6th place](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/6th-place-solution)

### Inferences
- Robust handling of series and modalities mattered as much as modelling: orientation canonicalisation, a spacing fallback, filtering inconsistent slices, and a fallback to OOF-mean predictions instead of a crash. The knee test set (about 1,300 studies, roughly 5.5 series each) will likely contain odd series (missing tags, different orientations). A series-type classifier with a fallback is cheap insurance.
- For macro AUC over many sparse labels (13 locations there, 12 findings here), separate binary heads plus dense auxiliary supervision helped more than complex aggregation heads (3rd: MIL/LSTM didn't help; 1st: the transformer added only +0.006).

### Gaps
- A search-engine summary claimed the 1st place's Location-Aware Transformer added "+0.02 macro AUC". The write-up I read shows only +0.006 overall (+0.004 on locations). I treat the +0.02 figure as unverified.
- I did not read the 7th–9th aneurysm write-ups, e.g. MIC-DKFZ ([GitHub](https://github.com/MIC-DKFZ/kaggle-rsna-intracranial-aneurysm-detection-2025-solution)).

## Q3. Abdominal Trauma 2023, Cervical Spine 2022, Intracranial Hemorrhage 2019, Screening Mammography 2023, Brain Tumor MGMT 2021: transferable tricks and shake-up lessons

### Takeaway
Across CT and MRI volumes the dominant recipe is: localise or crop the organ, use a 2.5D CNN on adjacent-slice channels, aggregate with an RNN or attention, and give slices soft labels (study label × organ or vertebra visibility). Segmentation auxiliary losses (+0.01–0.03) and organ crops gave the largest reported gains. Weak or noisy targets produced extreme shake-ups. In MGMT, CV was essentially random, and the private winner was the simple baseline that sat 1040th on public. In mammography, the winner came from 22nd on public. In trauma 2023 Theo Viel's best private submission was his strongest single model, not his ensemble.

### Cited Findings
**Abdominal Trauma 2023** (weighted log loss; private top 10 public ranks [2, 1, 6, 3, 4, 8, 5, 10, 21, 13]; Spearman 0.913; computed from the [leaderboard](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/leaderboard)).
- 1st (Team Oxygen: Nischay Dhankhar, Qishen Ha, Harshit Sheoran). A 3D segmentation model gave a study-level crop. 96 equidistant slices were reshaped into a 2.5D sequence of (32, 3, H, W). **Soft slice labels = study label × normalised organ visibility per slice.** Model: 2D CNN + GRU with max aggregation over slices. **An auxiliary Dice segmentation loss on the last two encoder blocks gave "+0.01 to +0.03".** Backbones: CoaT-lite medium/small and EffNetV2-s. Best single model 4-fold OOF 0.326; ensemble 0.31x. — [1st](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447449)
- 2nd (Theo Viel, solo). An organ-visibility classifier chose which frames to train on. maxvit_tiny_tf_512 was the best 2D model. A 3D ResNet-18 cropped organs, and coatnet_1_rw_224 + RNN classified the crops. A separately trained RNN aggregated everything and optimised the metric. **2D + RNN scored public 0.41 / private 0.39; adding crop models gave 0.37 / 0.35 (his best private); the full ensemble gave 0.35 / 0.35.** "Ensembling did not really help on private ultimately and my best sub is my strongest single model." Loading DICOM with dicomsdl and preprocessing on GPU kept the pipeline under 4 h. — [2nd](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447453)
- 3rd (Yuji Ariyasu). Organ cubes went into 2.5D+3D models. Masking the liver box was "very effective", and a custom sampler let multi-organ models train together. **Ensembling was a per-target weighted average. "I also tried stacking which directly optimizes the metric, but the results were slightly worse… due to overfit."** CV 0.3316. — [3rd](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447464)
- 4th: TotalSegmentator masks with UNet + PVTv2/MaxViT. Ensemble CV 0.3570, or 0.3530 with scaling; LB 0.39–0.40. The author blamed the lack of an RNN for the gap to the top. — [4th](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447848)
- 6th: 3D segmentation crop, then a 2.5D CNN (±2 adjacent slices) + LSTM/GRU. A segmentation head trained on the hosts' bounding-box labels helped the extravasation model. — [6th](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/448208)

**Cervical Spine 2022** (weighted log loss; Spearman 0.811; public #1 Selim Seferbekov finished 4th private; computed from the [leaderboard](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/leaderboard)).
- 1st (Qishen Ha). Only 87 3D masks were available, but they were "sufficient" for a 3D segmentation model at 128³. Each vertebra was cropped, 15 slices × 5 channels (±2 adjacent) were taken, and the vertebra mask was added as a 6th channel. The classifier was a 2.5D CNN + LSTM, with per-vertebra and per-patient variants. "Training a 3D CNN on this data did not give me satisfactory results." Inference took 7.5 h; ensemble CV about 0.20. — [1st](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/362607); [code](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/362787)
- 2nd (RAWE): 2.5D UNet segmentation, then 24 uniformly sampled slices into a CNN + BiGRU + attention, with spatial dropout. — [2nd](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/365115)
- 3rd (Darragh Hanley). Slice labels = vertebra-volume ratio × study fracture label. Model: 2.5D CNN + RNN; "the addition of the attention mechanism over the 1d RNN output helped a lot". The final stage froze the CNN and trained a new RNN + attention on study labels with the metric as loss. Studies were batched as uint8 and normalised on GPU; inference about 4.5 h. — [3rd](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/362643)
- 4th (Selim Seferbekov): ir-CSN-152 pretrained on video (Kinetics/IG65M) for 3D classification. 3D networks overfit unless video-pretrained. — [4th](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/364837)
- 5th ("Speedrun": Philipp Singer, Pascal Pfeiffer, Christof Henkel; 11 days): slice pseudo-labels, 2.5D+3D EffNetV2, heavy regularisation (Mixup) and upscaling ≥1.125 at inference. **A stage-3 MLP on per-study mean/min/max of slice predictions, blended 30:70 with max aggregation, improved scores by "2–3 LB points".** — [5th](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/363232)

**Intracranial Hemorrhage 2019** (two-stage competition; public LB not comparable, so no shake-up statistics).
- 1st (SeuTao): three 2D CNN pipelines (3 windows; 3 adjacent slices; both), then two sequence models: MLP+LSTM on embeddings and 1D-CNN+LSTM on logits plus position. "Sequence model wins… there is no shakeup." — [1st](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/discussion/117210)
- 2nd (Darragh + Dmitry Larko): ResNeXt-101 embeddings with TTA-averaged embeddings went into an LSTM. The LSTM also received the deltas to the previous and next slice embeddings. — [2nd](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/discussion/117228)
- 3rd (takuoko): concatenating neighbouring slices in several patterns was the biggest gain. His stacking-only model would have scored 0.043 private but was not selected; the final scored 0.045. — [3rd](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/discussion/117223)
- 4th and 5th: 2D feature extractor plus 1D/3D CNN over stacked features; balanced per-fold sampling for CV–LB correlation; metadata post-processing (e.g. 0.054→0.048). — [4th](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/discussion/118249); [5th](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/discussion/117232)

**Screening Mammography 2023** (pF1; Spearman 0.705; private top 10 public ranks [22, 54, 3, 4, 2, 1, 21, 9, 13, 87]; computed from the [leaderboard](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/leaderboard)).
- 1st (dangnh0611; 22nd on public). A YOLOX-nano ROI detector was trained on 571 boxes, partly self-annotated. Classifiers were 4 × ConvNeXt-small at 2048×1024. **External datasets improved OOF F1 by about 0.02 and private by about 0.02–0.03** (0.53 → 0.56/0.55). **Soft positive labels (0.8–0.9) showed no clear gain over label smoothing 0.1.** Five folds gave inconsistent CV because positives were few, so he switched to 4 folds and trained with external data. Other choices: positive upsampling so every batch had at least one positive; max pooling; drop_rate 0.5; EMA; checkpoint grid search on OOF. — [1st](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/392449)
- 2nd: pretrained on external data at 1280 px and fine-tuned at 1536 without it. Auxiliary losses on BIRADS, density, view, etc. **External data took public 0.57→0.58 and private 0.51→0.53.** They chose one high-resolution model over an ensemble. — [2nd](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/391676)
- 3rd: YOLOX ROI; 4 external datasets ("significantly enhance the stability of the training"); machine-ID one-hot metadata; CNN+LSTM multi-view MIL. Because of the shake risk they "adopted a relatively conservative" ensemble. — [3rd](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/391725)
- 4th: "it probably helped that we did not start blending the models until the last few days… so did not get distracted by high Public LB scores". An ROI extractor and higher resolution did not help them. — [4th](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/391208)
- 5th: multi-task pretraining on VinDr (BIRADS, density, lesion labels). The best negative:positive sampling ratio differed by backbone (1:7 vs 1:3). Plain BCE was best; weighted BCE and focal loss were worse. — [5th](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/391979)

**RSNA-MICCAI Brain Tumor MGMT 2021** (AUC; Spearman 0.034; median |rank change| in private top 50 = 676; every public top-10 team except one fell below private 200; the private winner was 1040th on public; computed from the [leaderboard](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/leaderboard)).
- 1st (private 0.622): "my final solution was one of the very first baselines… no model ensembling, no complex/big models". "The CV score was almost random." Training the same EffNet-B0 100 times with fixed seeds gave CV from 0.53 to 0.62. — [1st](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/281347)
- 2nd: 10 uniformly subsampled frames, with the 4 MRI sequences as channels (zero-filled if missing), into EffNet-B0 + LSTM with light augmentation. — [2nd](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/280033)
- 3rd: four EffNet-B3 models, one per sequence. — [3rd](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/287713)

**Bonus: RSNA-STR Pulmonary Embolism 2020** (Spearman 0.966).
- 1st (Guanshuo Xu): lung-localisation crop. Three adjacent slices as channels beat three windows. A single held-out split of 1,000 studies was "reliable enough". Embeddings plus neighbour differences went into a BiGRU with attention + max pooling. — [1st](https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection/discussion/194145)
- 2nd (Ian Pan): 2D ResNeSt-50 + transformer. He hand-labelled heart slices in 1,000 CT scans, "probably a full day's worth of work". — [2nd](https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection/discussion/193401)

### Inferences
- The soft slice-label construction (study label × slice relevance) used by trauma 1st, C-spine 3rd and C-spine 5th is a template for our MIL setting. If we can estimate slice relevance per finding (e.g. from attention maps, or from crude anatomy priors such as menisci on peripheral sagittal slices and ACL on intercondylar slices), slice-level auxiliary losses become possible without any annotation.
- MGMT and mammography show that when the label signal is weak, public-LB tuning is worse than useless. Report-derived knee labels are noisy, but knee findings are visually grounded (unlike MGMT), so the expected shake-up is moderate. The closest analogues are RANZCR and LSDC, not MGMT.

### Gaps
- No 5th-place trauma write-up is in the index. I did not retrieve the COVID 5th or mammography 6th–10th write-ups.

## Q4. Teacher–student with privileged information, and pseudo-labelling of external data (RANZCR CLiP 2021, SIIM-FISABIO-RSNA COVID-19 2021, VinBigData 2021): how much did they add?

### Takeaway
Annotation-guided teachers and pseudo-labelling paid off, but by small, single-model-level amounts once strong baselines existed. RANZCR's annotation teacher gave about +0.01 CV in the public 3-stage recipe (hengck23 estimated +2.0–2.5% AUC). Segmentation-derived pseudo-annotations and pseudo-labelled external data each gave about +0.001–0.002 CV per single model. Soft labels beat hard ones. In COVID, pretraining on large external CXR sets "significantly" helped. Pseudo-labelling there was mixed: it didn't improve CV for 3rd place, and 4th place's best-CV submission used none. RANZCR's winners also showed that annotations can be used purely as an auxiliary segmentation target, with no teacher at all (4th: single stage, no external data, CV = public = private best).

### Cited Findings
**RANZCR** (macro AUC over 11 labels; Spearman 0.725; Guanshuo Xu went from 116th public to 5th private; computed from the [leaderboard](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/leaderboard)).
- **Original teacher–student recipe (Y. Nakama, 7th).**
  - Stage 1: train a teacher on annotated images (annotations drawn on the image).
  - Stage 2: train a student on raw images with BCE + w·MSE(student features, teacher features) over the annotated subset.
  - Stage 3: fine-tune the student on all data.
  - "In my experience, there is 0.01 improvement with this approach" (CV 0.95x). — [3-stage training post](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/discussion/207577)
- **hengck23 (idea originator).** "I estimated the addition annotation supervision signals will cause an improvement of about 2.0% to 2.5% for the AUC metric." A teacher trained on rendered-annotation images reached about 96% local CV. He used L2 consistency on features and a frozen teacher. — [hengck23 post](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/discussion/205243)
- **3rd (Preferred CLiP): the cleanest ablation.** Numbers are CV / public / private:

  | Step | CV | Public | Private |
  |---|---|---|---|
  | Vanilla 3-stage teacher–student (resnet200d, TTA) | 0.96606 | 0.97020 | 0.97328 |
  | + segmenter-generated pseudo-annotations for unannotated/external images | 0.96742 | 0.97323 | 0.97389 |
  | (+ ensemble of 4 architectures) | 0.97101 | 0.97430 | 0.97515 |
  | + NIH pseudo-labels (single model) | 0.96918 | 0.97335 | 0.97455 |
  | + ensemble of 8 training setups (no TTA) | 0.97215 | 0.97386 | 0.97611 |
  | + ensemble coefficients from logistic regression | 0.97228 | 0.97434 | 0.97624 |

  Whole 5-fold multi-stage training of resnet200d took about 15 GPU-days on V100s. — [3rd](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/discussion/228775)
- **1st ("All Data Are Ext": Bo Liu, Qishen Ha, Gary).**
  - Segmentation: 2-stage (tube lines + tips; trachea bifurcation from a doctor-contributed set of 5k images), with pseudo masks on about 28k NIH external images (de-duplicated with imagehash, patients kept within folds).
  - Classification: input was 6 channels (3 image + 3 predicted masks) with 12 outputs (an added no_ETT). Stage 1 (official data, 20 models) reached CV 0.97553. **Stage 2 (+28k soft pseudo-labelled external images, 31 models) reached CV 0.97606.** "We used soft pseudo labels… They work better than hard ones."
  - Ensembling: simple averages, no weighting. Rank-averaging only 5 of the 11 columns added about +0.00032 CV.
  - A minimal 2-stage, official-data-only version still reached CV 0.968–0.969 / LB 0.972 on one 16 GB GPU. — [1st](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/discussion/226633); [minimal pipeline](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/discussion/227128)
- **2nd:** a dual-head UNet (classification + segmentation) using masks only as supervision, with full-train-set pseudo masks. A single resnet200d-UNet scored private 0.974 (gold level). Three models: 0.972 / 0.97383 / 0.97599; six models: 0.973 / 0.97412 / 0.97641. Inference ran only the encoder + classification head. They pseudo-labelled the NIH ChestX dataset and retrieved the nearest NIH match for each test image with a metric-learning model. That is effectively exploiting test images' presence in a public dataset and is not transferable. — [2nd](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/discussion/227407)
- **4th (Watercooled).** Single-stage EfficientNet with a UNet segmentation head as a "surrogate loss": segmentation loss weight 50, computed only where annotations exist. No external data. 16 models (B7/B8 at 896–1024 px) were averaged. "Our final submission constitutes our best local CV score, best public LB score, and best private LB score." — [4th](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/discussion/229706)
- **5th (Guanshuo Xu).** Auxiliary segmentation on the second-to-last block plus regression of the ETT tip point; per-line-type "expert" models fine-tuned from a shared warm-up; per-expert NIH pseudo-labelling. CV 0.9712 / public 0.9682 / private 0.9756: 116th public, 5th private. — [5th](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/discussion/230033)

**SIIM-FISABIO-RSNA COVID-19** (mAP; Spearman 0.914; computed from the [leaderboard](https://www.kaggle.com/competitions/siim-covid19-detection/leaderboard)).
- 1st (DungNB, public 1st and private 1st).
  - Pretraining: encoder on CheXpert + ChestX-ray14, then RSNA-pneumonia classification + segmentation, then the competition's 4 classes + segmentation. A model was then retrained with competition train + **pseudo-labelled test set**.
  - **Measured deltas:** EMA (decay 0.997) +0.001–0.002 LB; class-weighted auxiliary BCE+Dice +0.002 CV and LB; **8× TTA +0.003–0.004 CV and public LB; lung-detector crop +0.003 LB vs a fixed 80% centre crop.**
  - Detection was pseudo-labelled on test + PadChest/pneumothorax/VinBig with confidence filters (negative < 0.3, max positive > 0.7, top-2 boxes).
  - BIMCV/RICORD were excluded because of duplicate and leak risk. — [1st](https://www.kaggle.com/competitions/siim-covid19-detection/discussion/263658)
- 2nd: NIH pretraining "improves our CV and LB". Final CV 0.636 / public 0.645 / private 0.634. — [2nd](https://www.kaggle.com/competitions/siim-covid19-detection/discussion/263674)
- 3rd (Yuji & Ian Pan): pretraining on CheXpert and RSNA-2018 "significantly increased performance". Ian: "pseudolabeling did not improve CV". **Nelder-Mead and ridge blending "did not work because they had overfit", so they used simple averaging; merging the two members' models made "significant improvements in both CV/LB".** — [3rd](https://www.kaggle.com/competitions/siim-covid19-detection/discussion/263654)
- 4th (team "RTX 4090", incl. Awsaf and Chris Deotte): knowledge distillation from about 5 best models' soft labels, and pseudo-labels on BIMCV + RICORD + RSNA. "When we looked back we noticed our best CV submission doesn't use any pseudo". Final public 0.654 / private 0.631. — [4th](https://www.kaggle.com/competitions/siim-covid19-detection/discussion/264243)

**VinBigData chest X-ray detection** (mAP@0.4 detection; Spearman 0.696; computed from the [leaderboard](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/leaderboard)). The top solutions relied on WBF ensembling and classifier-based post-processing, not distillation.
- 1st: overweighting the "not validated" ensemble stage, which was checked only on public LB, gave 0.354 public / 0.314 private. Their unselected, less LB-tuned blend scored 0.330 / 0.321. — [1st (team)](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/discussion/231511); [1st (path)](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/discussion/229724)
- 3rd: a multi-label classifier with **soft labels 1 / 0.67 / 0.33 depending on whether 3, 2 or 1 radiologists marked the class**, used to scale detector confidences. Per-class multiplication scored public 0.348 / private 0.305; using only the "no finding" score gave 0.332 / 0.305. The big public gain did not transfer. — [3rd](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/discussion/229636)

### Inferences
- In the knee task, the "privileged information" is the 58 expert-labelled studies plus whatever structured text the reports contain; there are no pixel annotations. RANZCR-style feature distillation from an annotation-rendered teacher therefore does not transfer directly. What transfers is: (i) soft-label self-distillation from an ensemble teacher over the noisy report labels (aneurysm 4th: +0.03 CV / +0.01 private; RANZCR: soft > hard); (ii) dense auxiliary supervision wherever any localisation signal exists.
- Expected size: once a strong ensemble exists, pseudo-labelling or distillation typically adds +0.001–0.01 AUC. The upper end applies only when the unlabelled or weakly labelled pool is large relative to the cleanly labelled set, which is our situation (4,349 report-labelled vs 58 expert-labelled).

### Gaps
- No write-up quantified distillation *into a smaller student for inference speed*. I found no VinBigData top solution that used distillation. The COVID 1st-place slide deck (Google Drive link in the post) might contain more ablations but was not read.

## Q5. How did winners handle very noisy or weak labels, and small or variable validation sets?

### Takeaway
Six patterns recur. (1) Clean the labels with OOF: drop high-loss samples, as in LSDC 2nd (+1% public and private), or flip confident contradictions, as in aneurysm 5th. (2) Fix labels manually where possible: aneurysm 2nd (private +0.032 together with better annotations); LSDC 3rd. (3) Use soft rather than hard targets: ensemble soft pseudo-labels, annotator-agreement weights, and slice labels proportional to visibility. Plain label smoothing was neutral or harmful in LSDC 5th and mammography 1st. (4) Treat CV on noisy labels with suspicion, because test labels may be cleaner (LSDC 8th). (5) For small or variable validation, use repeated or multi-seed CV, fewer and larger folds, and a single trusted metric. (6) Never pick checkpoints or blends on the public LB (aneurysm 5th: −0.046 public→private; VinBig 1st: −0.007 private).

### Cited Findings
- OOF-based removal: samples with |label − OOF ensemble prediction| ≥ 0.8 were removed (class coefficients for minority classes) and models retrained, giving "1% on both public and private". — [LSDC 2nd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution)
- OOF-based relabelling: training negatives with OOF "present" > 0.9 were flipped to positive, combined with pseudo-labelled external data. OOF AUC rose by +0.0007 to +0.008 depending on the model. — [Aneurysm 5th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution)
- Manual correction of side and position label mix-ups, together with improved segmentation annotations, took public 0.871→0.888 and private 0.825→0.857 at fixed TTA. — [Aneurysm 2nd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution)
- "We manually reviewed all annotations and corrected erroneous labels by hand" (coordinates). — [LSDC 3rd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453)
- Without radiology domain knowledge, manual relabelling failed. SSL improved LB but not CV, and the author guessed that test labels are cleaner than train labels. — [LSDC 8th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539548)
- The 6th-place team found that "denoising techniques" did not help them. — [LSDC 6th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/541813)
- Label smoothing "not worked". — [LSDC 5th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539472)
- A soft positive label of 0.8–0.9 gave no clear gain over label smoothing 0.1 in the final ablation, though it helped one fold. — [Mammography 1st](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/392449)
- Soft labels from annotator agreement (1 / 0.67 / 0.33). — [VinBig 3rd](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/discussion/229636)
- Weak study labels turned into soft slice labels via organ visibility. — [Trauma 1st](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447449); [C-spine 3rd](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/362643)
- A vertebra's label was set to 0 when less than 30% of it was visible in the crop. — [C-spine 4th](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/364837)
- Training the slice-level model only on positive exams (images from negative exams excluded from that loss) improved CV by about 0.01–0.02. — [PE 2nd](https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection/discussion/193401)
- Validation with few positives: 5-fold CV was inconsistent, so he switched to 4 folds and tracked PR-AUC, ROC-AUC, best pF1 and threshold rather than the unstable metric alone. — [Mammography 1st](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/392449)
- Mammography 2nd trained 5 folds but then experimented only on fold 0, after finding that its score "exhibit[ed] a strong correlation" with CV and LB. — [Mammography 2nd](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/391676)
- The same model trained 100 times gave CV 0.53–0.62. The winner averaged over many repeats and kept a simple baseline. — [MGMT 1st](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/281347)
- A single 1,000-study holdout was "reliable enough" for a large, clean dataset. — [PE 1st](https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection/discussion/194145)
- Balanced patient sampling per fold "helps to have the correlation between CV and LB". — [ICH 5th](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/discussion/117232)
- A team that evaluated on "only 50 series" and selected the epoch by public LB went from 2nd public to 5th private (0.893 → 0.847). — [Aneurysm 5th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution); [leaderboard](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/leaderboard)
- Hanley & McNeil's standard error of AUC is the basis for the noise calculation in the Inferences below. — [Hanley & McNeil 1982, Radiology](https://pubs.rsna.org/doi/10.1148/radiology.143.1.7063747)

### Inferences
- **Our labels:** 4,349 report-derived training labels (likely errors: negation, unmentioned treated as negative, ambiguous grading) plus 58 expert-labelled studies. Report-derived negatives are probably the noisier side. The 2nd-place LSDC recipe maps directly: train the ensemble, get 5-fold OOF, then down-weight or drop samples with label 0 and OOF very high (and the reverse), per finding with finding-specific thresholds. Our existing ensemble's OOF is the prerequisite.
- **Validation noise (my Hanley–McNeil calculation, illustrative).** For one finding with AUC about 0.95, the SE is about 0.035 with 20 positives and 0.020 with 60 positives. Averaging 12 findings (assumed independent) gives a macro-AUC SE of roughly 0.006–0.010 for a few hundred to ~900 studies. On the 58-study expert set, per-finding SE is about 0.05–0.14. Consequences: (a) public-LB differences under about 0.005 are mostly noise; (b) the 58 expert studies can sanity-check noise handling but should not tune blend weights; (c) the main selection signal should be 5-fold OOF on all 4,407 studies, ideally multi-seed. Paired differences between correlated models have smaller SE than these absolute values.
- If the test set is radiologist-labelled and cleaner than the report-derived training labels (unknown), LSDC 8th suggests that improvements in label cleaning or regularisation may appear on LB but not in noisy-label CV. Checking on the 58 expert studies is the only offline proxy.

### Gaps
- None of the reviewed winning write-ups trained directly on NLP-mined radiology-report labels, so there is no direct Kaggle precedent for report-label noise. How the knee test labels were produced (expert vs report) is not established here.

## Q6. Which blending practices (rank averaging, per-target weights, weights fitted on OOF vs LB) held up on private, and what do the shake-ups teach?

### Takeaway
Simple or constrained blends held up. Simple averages won RANZCR 1st, LSDC 3rd and COVID 3rd. Per-target weights picked on OOF worked for trauma 3rd and RANZCR 3rd. Low-dimensional stackers (an MLP per target, a residual over Nelder-Mead) worked in LSDC 4th and 6th. Free-form stacking or Nelder-Mead often overfit (COVID 3rd, trauma 3rd). Rank-averaging gave tiny AUC gains, worthwhile only per column (RANZCR 1st: +0.0003). Weights tuned on the public LB repeatedly cost private score (VinBig 1st, aneurysm 5th). For AUC competitions with tight score spreads, ranks inside the top 50 shuffle by about 5–15 places even when solutions are sound.

### Cited Findings
**Shake-up summary (all computed from Kaggle leaderboards; "Spearman" = private top-200 rank vs public rank).**

| Competition (metric) | Spearman | Median rank change, private top 50 | Private winner's public rank | Public #1's private rank |
|---|---|---|---|---|
| LSDC 2024 (log loss) | 0.918 | 5 | 5 | 2 |
| Aneurysm 2025 (AUC) | 0.850 | 2 | 3 | 2 |
| Abdominal Trauma 2023 (log loss) | 0.913 | 5 | 2 | 2 |
| C-spine 2022 (log loss) | 0.811 | 3 | 2 | 4 |
| Mammography 2023 (pF1) | 0.705 | 17 | 22 | 6 |
| MGMT 2021 (AUC) | 0.034 | 676 | 1040 | >200 |
| RANZCR 2021 (macro AUC) | 0.725 | 14 | 2 | 4 |
| SIIM COVID 2021 (mAP) | 0.914 | 11 | 1 | 1 |
| VinBig 2021 (mAP) | 0.696 | 16 | 2 | 2 |
| PE 2020 (log loss) | 0.966 | 4 | 1 | 1 |

Sources: [LSDC LB](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/leaderboard), [Aneurysm LB](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/leaderboard), [Trauma LB](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/leaderboard), [C-spine LB](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/leaderboard), [Mammography LB](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/leaderboard), [MGMT LB](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/leaderboard), [RANZCR LB](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/leaderboard), [COVID LB](https://www.kaggle.com/competitions/siim-covid19-detection/leaderboard), [VinBig LB](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/leaderboard), [PE LB](https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection/leaderboard). ICH 2019 is omitted because its two-stage format makes public and private not comparable.

**Blending evidence.**
- Simple averages of soft predictions, "no weighting"; rank-averaging only for the 5 columns where it helped (+0.00032 CV). — [RANZCR 1st](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/discussion/226633)
- Logistic regression fitted 8 ensemble coefficients (one per training setup) on local validation: +0.00013 CV, +0.0005 public, +0.00013 private over an equal-weight 8-setup ensemble. — [RANZCR 3rd](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/discussion/228775)
- Simple average of 16 models; best CV = best public = best private. — [RANZCR 4th](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/discussion/229706)
- Nelder-Mead and ridge overfit, so they used simple averaging. Merging two members' independently developed models gave a "significant" CV and LB gain. — [COVID 3rd](https://www.kaggle.com/competitions/siim-covid19-detection/discussion/263654)
- Per-target weighted average; metric-optimising stacking was "slightly worse… due to overfit". — [Trauma 3rd](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447464)
- The strongest single model beat the ensemble on private. — [Trauma 2nd](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447453)
- Simple average of 30 models: single-model CV 0.3858 → ensemble 0.3643. — [LSDC 3rd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453)
- MLP that learns the residual over a Nelder-Mead blend, plus per-level GBDT stackers on same-type predictions only. — [LSDC 4th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539443)
- MLP stacker with each target independent ("surprisingly… worked best") and weights shared across levels; the team trusted CV over LB. — [LSDC 6th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/541813)
- A stage-3 MLP on per-study mean/min/max of slice predictions, blended 30:70 with max aggregation, gave "2–3 LB points". — [C-spine 5th](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/363232)
- Running stage 2 per stage-1 model, rather than averaging the stage-1 outputs, gave their best private. — [LSDC 5th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539472)
- The heavily public-LB-validated stage cost 0.007 private. — [VinBig 1st](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/discussion/231511)
- Delaying blending until the end avoided chasing public LB. — [Mammography 4th](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/391208)
- A "relatively conservative" ensemble chosen because of shake risk. — [Mammography 3rd](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/391725)
- Two final submissions kept deliberately different (best CV vs best LB); the best-CV one had no pseudo-labels. — [COVID 4th](https://www.kaggle.com/competitions/siim-covid19-detection/discussion/264243)
- Picking the best epoch by public LB with a 50-series holdout preceded the top-6's largest drop. — [Aneurysm 5th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution)
- Six-model ensemble: CV +0.001 over three models translated into private +0.0004. — [RANZCR 2nd](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/discussion/227407)
- Stacking-only model (private 0.043) was better than the selected final (0.045) but was not chosen. — [ICH 3rd](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/discussion/117223)

### Inferences
- For macro AUC in our setting: (1) average logits or probabilities across models per finding, with rank-averaging as a per-finding option chosen on OOF; (2) if weights are fitted, fit them on 5-fold OOF over about 4.4k studies with a constrained method (non-negative, sum to 1, or logistic regression on a few model groups). Never fit them on the 58 expert studies or on public LB; (3) a per-finding stacker can use cross-finding features (e.g. meniscus tear ↔ cartilage loss), as LSDC 5th did with level-wise co-occurrence, but LSDC 4th found cross-target stacking "not effective", so test it on OOF.
- Use the two final-submission slots as a hedge: one best-OOF blend and one LB-informed blend. Given the leaders' 0.961–0.964 public scores, expect top-50 ranks to move by about 5–15 places on private; RANZCR, an AUC competition with tight scores, had a median of 14.

### Gaps
- No write-up compared rank vs probability averaging across a whole ensemble on private, beyond RANZCR 1st's column-wise note.

## Q7. Ranked list: most transferable tricks for the knee setting, with expected gain and cost

### Takeaway
To close the 0.943 → 0.96 gap within 14 days on 1×L4 plus 1×4070S and about 30 Kaggle GPU-hours a week, the highest expected value per GPU-hour is: (1) OOF-driven label-noise handling; (2) multi-series fusion with view dropout and routing series to the findings they show best; (3) merging with or blending an independent pipeline; (4) flip, slice-offset and multi-crop TTA, sized to the 9 h budget; (5) soft-label self-distillation; (6) training-recipe fixes (CutMix/Mixup, EMA, short schedules, BiLSTM + auxiliary attention loss in the MIL head). ROI cropping and dense auxiliary supervision were the largest single wins in CT and MRA, but need localisation labels we probably lack. They are worth it only if about a day of box annotation is feasible. Gains below are my estimates (inferences) anchored on the cited measured deltas.

### Cited Findings
Measured anchors for each trick (details and links in Q1–Q6):
- OOF label cleaning: +1% log loss on both public and private ([LSDC 2nd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution)); +0.0007 to +0.008 OOF AUC ([aneurysm 5th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution)); better labels gave private +0.032 AUC ([aneurysm 2nd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution)).
- Multi-series fusion and view routing: one of the top-3 elements in [LSDC 2nd (Ian)](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539452); missing view = zeros plus 10% view dropout in [LSDC 6th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/541813); three-series attention in [LSDC 3rd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453).
- Merging independent pipelines: "significant" gain ([COVID 3rd](https://www.kaggle.com/competitions/siim-covid19-detection/discussion/263654)); 0.75/0.25 blend of a one-stage model improved the ensemble ([LSDC 7th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539439)).
- TTA: 4× gave +0.026 public / +0.012 private, and 8× with left–right label swap a further +0.010 / +0.005 ([aneurysm 2nd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution)); 8× gave +0.003–0.004 ([COVID 1st](https://www.kaggle.com/competitions/siim-covid19-detection/discussion/263658)); 27-crop aggregation "significant" ([LSDC 2nd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539452)).
- Soft pseudo-distillation: +0.03 CV / +0.01 private ([aneurysm 4th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/4th-place-solution)); soft beats hard ([RANZCR 1st](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/discussion/226633)); "not significant" ([LSDC 3rd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453)).
- Recipe changes:
  - CutMix p=1.0 was the largest AUC gain in [LSDC 5th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539472).
  - CutMix, flips with label swap and strong augmentation gave about +0.01 CV each ([aneurysm 6th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/6th-place-solution)).
  - EMA +0.001–0.002 ([COVID 1st](https://www.kaggle.com/competitions/siim-covid19-detection/discussion/263658)).
  - Attention-MIL then BiLSTM + auxiliary loss: 0.37→0.35→0.33 log loss; small backbones and short schedules won ([LSDC 1st](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/540091)).
  - The two-step AUC-first-then-head-only scheme "significantly" improved results ([LSDC 5th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539472)).
- ROI cropping at higher resolution: about 0.70→0.80 ([aneurysm 4th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/4th-place-solution)); +0.03–0.05 ([aneurysm 5th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution)); 0.77→0.81 from higher-resolution feature maps ([aneurysm 3rd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/3rd-place-solution)); log loss 0.39→0.35 private from crop models ([trauma 2nd](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447453)); +0.003 from a lung crop ([COVID 1st](https://www.kaggle.com/competitions/siim-covid19-detection/discussion/263658)).
- About a day to hand-label ROI slices for 1,000 scans ([PE 2nd](https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection/discussion/193401)); 571 self-annotated boxes ([mammography 1st](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/392449)).
- Dense auxiliary supervision: +0.01–0.03 ([trauma 1st](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447449)); 0.876→0.902 ([aneurysm 1st](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution)).
- Inference engineering:
  - All models resident on the GPU with one pass over the test data fit 67 folds in 9 h ([RANZCR 1st](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/discussion/226633)).
  - Dropping decoders at inference ([RANZCR 2nd](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/discussion/227407); [aneurysm 2nd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution)).
  - uint8 batches normalised on GPU ([C-spine 3rd](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/362643)); dicomsdl and GPU preprocessing ([trauma 2nd](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447453)).
  - Frame subsampling of long stacks ([aneurysm 6th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/6th-place-solution)); OOF-mean fallback on failure ([aneurysm 1st](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution)).

### Inferences
Ranked by expected macro-AUC gain per unit cost. All gain figures are my estimates for this competition.

1. **OOF-driven noise handling for report labels.** For each finding, take 5-fold OOF from the current ensemble. Drop or down-weight label≠prediction outliers (start with |y − p| ≥ 0.8, as in LSDC 2nd), or replace them with soft targets (e.g. 0.5·y + 0.5·p). Retrain. Check on the 58 expert studies and on public LB, not on noisy-label CV alone.
   - Expected gain: +0.003–0.010.
   - Cost: about 1 retrain cycle (1–2 days on L4/4070S), no new inference cost.
   - Risk: removing true hard positives on rare findings; use finding-specific thresholds and cap removals at a few %.
2. **Series-to-finding routing plus study-level multi-series fusion.**
   - Per finding, use the planes and sequences a radiologist would read, e.g. sagittal fluid-sensitive for cruciates and menisci, coronal for collaterals and menisci, axial for patellofemoral.
   - Fuse per-series embeddings with attention over series. A missing series becomes zeros plus a mask, with 10–20% series dropout in training (LSDC 6th).
   - The study head can be retrained on cached per-series embeddings, which is cheap.
   - Expected gain: +0.003–0.010. Cost: low–moderate (a head retrain; maybe 1 extra series model).
3. **Team merge or independent-pipeline blend before the Oct 15 merge deadline** (from project memory; team target "gold ≈0.954+").
   - Average per finding with equal weights or OOF-constrained weights.
   - Expected gain: +0.003–0.01 if the pipelines differ (backbone, sampling, series handling). Compute cost is near zero.
   - Prerequisite: shared folds or OOF for weight fitting.
4. **TTA sized to the budget.** Options are horizontal flip (label-invariant for knee findings: medial and lateral stay anatomically fixed while laterality mirrors), ±1 slice offset, and 2–3 crop scales.
   - The budget is 9 h / about 1,300 studies ≈ 25 s wall-clock per study, or about 50 GPU-seconds across 2×T4 (≈4.5 s per series wall-clock). Each extra TTA view competes with ensemble members.
   - Expected gain: +0.002–0.006. Cost: inference time only.
5. **Soft-label self-distillation.** Mix report labels with ensemble-teacher OOF soft labels (e.g. α=0.3–0.5) and train 1–2 student models. Students can also be smaller, to free inference time.
   - Expected gain: +0.002–0.008 (aneurysm 4th's +0.01 private is the upper anchor). Cost: 1 retrain per student.
6. **Recipe fixes in the 2.5D attention-MIL.**
   - Add a BiLSTM/GRU before attention pooling plus an auxiliary per-slice attention loss (LSDC 1st).
   - CutMix/Mixup (in LSDC 5th and aneurysm 6th it was among the largest gains).
   - EMA; short schedules (LSDC 1st: 7–14 epochs); small or medium backbones rather than large ones.
   - Expected gain: +0.002–0.008 combined. Cost: retrains, about 1–3 days.
7. **ROI / sub-region crops at higher resolution, plus dense auxiliary supervision.** Only if we can hand-box about 200–500 key regions (joint centre, menisci, ACL) in about a day (PE 2nd and mammography 1st precedents), or derive crops from attention maps.
   - Knee MRI FOV is already tight, so the gain is likely smaller than in CT/MRA.
   - Expected gain: +0.002–0.01, highly uncertain. Cost: high (annotation plus a new stage-1 model plus inference time).
8. **External knee MRI data for pretraining or pseudo-labelling.** Mammography got +0.02 F1; LSDC 6th and aneurysm 6th saw no gain.
   - Expected gain: uncertain (0 to +0.005).
   - Cost: high (download, preprocessing, licence and competition-rule checks). Lowest priority given 14 days.
9. **Inference engineering**, as an enabler rather than a gain:
   - load every model once and decode each DICOM series once;
   - uint8 to GPU normalisation; dicomsdl;
   - a per-series try/except falling back to OOF-mean probabilities;
   - cap slices per series (subsample long stacks);
   - drop any decoder or auxiliary heads.
   - This lets items 3–5 fit into 9 h.
10. **Selection discipline.** Choose by multi-seed 5-fold OOF macro AUC on about 4.4k studies plus a sanity check on the 58 expert studies. Keep one "best OOF" and one "best LB" final. Distrust public differences under about 0.005. This protects the score rather than adding to it.

### Gaps
- I could not verify whether the knee competition provides any localisation labels (boxes, slice indices), whether the test labels are expert or report-derived, the public/private split fraction, or the external-data rules. Each strongly changes the ranking of items 1, 7 and 8.
- Gain estimates for items 1–8 are extrapolations from other modalities and metrics (log loss, mAP, pF1, AUC); no knee-MRI Kaggle precedent was found.
