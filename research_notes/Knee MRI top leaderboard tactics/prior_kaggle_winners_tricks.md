# Prior Kaggle medical-imaging winners: the tricks that moved teams from "good" to "top" (knee MRI addendum, 16 Oct 2026)

**Scope.** This is an addendum to `research_notes/Knee MRI prior work and tricks/past_kaggle_winners.md` (8 Oct). That file already covers:
- LSDC 2024, 1st–9th place;
- aneurysm 2025, 1st–6th place;
- trauma 2023, C-spine 2022, ICH 2019 and mammography 2023;
- RANZCR, SIIM-COVID, VinBig, MGMT and PE;
- shake-up statistics and blending evidence.

**New in this file:**
- aneurysm 7th, 8th, 9th, 11th, 15th and 18th place;
- HMS 2024, 1st–4th place (how they used label quality from expert vote counts);
- runtime facts from two 2025 3D-volume contests (CZII and BYU);
- one compact card per competition;
- a cross-competition table of tricks ranked by evidence.

Methods for uncertain report-derived labels (CheXpert U-Ones/U-Zeros, LSR, noisy-label losses) are in `learning_from_reports.md` and are not repeated here.

**Method.**
- Kaggle pages are JavaScript-rendered, so I read them through the r.jina.ai reader proxy. The links below are the original Kaggle URLs.
- Write-up URLs came from the community index [faridrashidi/kaggle-solutions](https://github.com/faridrashidi/kaggle-solutions) (`data/competitions.yml`).
- That index shows no RSNA-style medical-imaging classification contest finished between the aneurysm contest (Oct 2025) and today. Aneurysm 2025 is still the most recent analogue.
- No competition data was downloaded, no code was run and no repository was cloned. GitHub READMEs were read online.

**Tags.**
- "[8 Oct notes]": verified in the earlier file, not re-read today.
- "[second-hand]": taken from a third-party summary, not from the authors.

**Metrics.**
- Higher is better: AUC (aneurysm, RANZCR).
- Lower is better: weighted log loss (LSDC, trauma, C-spine, ICH) and KL divergence (HMS).
- Mammography used probabilistic F1 (pF1).
- "AUC-equivalent" can only be judged directly for the AUC contests. For the others I report the native metric.

## Q1. RSNA 2025 Intracranial Aneurysm Detection: series and modality handling, localisation, 3D vs 2.5D, and what won

### Takeaway
**What won.** The top 3 all ran 3D models on a localised region of interest (ROI) with dense voxel-level supervision: vessel-segmentation pretraining, an auxiliary segmentation loss, or a per-voxel head. 2.5D/2D pipelines took 4th–6th, 8th, 9th, 11th, 15th and 18th. The best of them (4th, private 0.848) finished 0.031 behind 1st (0.879).

**What almost every team shared.** Localise first, then crop. Keep native slice spacing; through-plane resampling hurt every team that tested it. Flip augmentation with left/right label swap. Heavy negative sampling.

**Largest measured single gains:**

| Change | Gain | Team |
|---|---|---|
| Segmentation pretraining of the backbone | +0.108 CV | 1st |
| ROI crop | about +0.10 | 4th |
| ROI crop | +0.03–0.05 | 5th |
| 2.5D neighbouring slices | +0.03 CV | 4th |
| Auxiliary segmentation loss | +0.026 CV | 1st |
| Resolution / finer feature maps | +0.014 to +0.04 | 1st, 3rd |
| Learned sequence or transformer aggregation | +0.012 to +0.021 private | 15th |
| Test-time augmentation (TTA) | +0.012 private | 2nd |

**Two warnings.**
- Runtime was a real constraint. Four write-ups (7th, 8th, 9th, 11th) report submissions or checkpoints lost to timeouts.
- A CV-only gain from self-distillation on OOF logits reversed on both leaderboards (11th).

### Cited Findings
**Context.**
- 1,147 teams competed for nine prize places: 1 tomoon33, 2 BraveCoWCoW, 3 BTYND, 4 Harshit Sheoran, 5 "more CV challenge pls", 6 Ian, Theo & Bartley, 7 MIC-DKFZ, 8 Konni, 9 "Vibes and Genius Trade-Off". — [RSNA 2025 winners](https://www.rsna.org/news/2025/november/2025-ai-challenge-winners)
- The Educational Merit Award (for code clarity, completeness and efficiency) went to Ian, Theo & Bartley and to MIC-DKFZ. — [RSNA 2025 winners](https://www.rsna.org/news/2025/november/2025-ai-challenge-winners)
- Private / public scores:
  - 1st–6th: 0.879/0.883, 0.867/0.900, 0.851/0.867, 0.848/0.872, 0.847/0.893, 0.840/0.845 [8 Oct notes] — [leaderboard](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/leaderboard)
  - 7th: 0.83/0.83 — [7th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution)
  - 9th: 0.8230/0.83082, their selected 2-fold stack — [9th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/9th-place-solution)
  - 11th: 0.82201/0.87689 — [11th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/11th-place-solution)
  - 15th: 0.80592/0.82494 — [15th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/15th-place-solution)
  - 18th: about 0.80/0.81 — [18th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/18th-place-solution)

**Design by rank (dimensionality and supervision).**

| Rank | Localise / crop | Classifier input | Backbone | Slice/volume aggregation | Dense supervision |
|---|---|---|---|---|---|
| 1 | Coarse nnU-Net, then a 140 mm ROI | 3D, 128×256×256 | nnU-Net encoder pretrained on vessel segmentation | Vessel-masked pooling into a location-aware transformer | Yes (auxiliary sphere segmentation + segmentation pretraining) |
| 2 | Tri-axial 2D nnU-Net ROI | 3D, 224³ | Multitask 3D nnU-Net | Classification head on the encoder | Yes (vessel and aneurysm segmentation) |
| 3 | YOLOv8 boxes on mid-sagittal and mid-coronal slices; 90/120 mm ROI | 3D, 196³ | 3D ResNet-18 | Per-voxel head, Top-N mean | No |
| 4 | DINOv3 ViT box regressor | 2.5D (−2, 0, +2), 384 px | CoaT-Lite-Medium (+ MaxViT) | Max over slices | No |
| 5 | Hand-boxed brain detector | 2D/2.5D, 384 px | ViT-L / EVA-L; mit-b4 FPN | Not recorded | Segmentation head in mit-b4 |
| 6 | Skull crop plus an extra fixed-ratio crop | 2.5D, 384 px | coatnet_rmlp_2_rw_384, maxvit_rmlp_base_rw_384 | Sequence model | No (refined localiser labels) |
| 7 | Fixed 200×160×160 mm cube | 3D patches 96×160×128 | nnU-Net ResEnc-M, from scratch | Blob heatmap, max | Yes (blob regression) |
| 8 | None (spot sampling) | 2.5D, 448 crop | ConvNeXt-Base (DINOv3 weights) | Transformer over the whole sequence | No |
| 9 | None | 2.5D, 512 px (YOLO); 2D encoder + 3D head | YOLOv11m, EffNetV2-S | Max, then GBDT stacker | Auxiliary 2D segmentation in one model |
| 11 | TotalSegmentator brain mask, then a 3D U-Net crop around the circle of Willis (CoW) | 2.5D, 384 px | coatnet_rmlp_2_rw_384, EffNetV2-S | RNN and BERT heads | CoW segmentation features + keypoint head |
| 15 | Rule-based crop/pad to 384 | 2.5D Faster R-CNN | ResNet50-FPN v2, ConvNeXtV2-T | DeiT-S transformer over RoI features | Detection boxes |
| 18 | 2.5D vessel segmentation, then per-vessel ROIs | 2.5D, 224 px | EffNet-B0 / V2-S | Top-k mean | Vessel segmentation |

Sources:
- 1st–6th [8 Oct notes]: [1st](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution), [2nd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution), [3rd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/3rd-place-solution), [4th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/4th-place-solution), [5th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution), [6th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/6th-place-solution)
- New today: [7th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution), [8th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/8th-place-solution), [9th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/9th-place-solution), [11th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/11th-place-solution), [15th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/15th-place-solution), [18th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/18th-place-solution)

**Cards for 1st–6th** (compact; full detail in the 8 Oct notes, Q2).

- **1st (tomoon33).**
  - Data cleaning: about 60 series excluded; slices filtered to the majority Rows×Cols/spacing; volumes reoriented and z-scored.
  - Inference: EMA weights, 4 folds, left–right flip TTA. Runtime about 18 s per series on 2×T4 (4.0 + 10.7 + 3.3 s). If the pipeline fails, it falls back to per-class OOF-mean probabilities.
  - Ablation on folds 0–2 at 96×192×192: full model 0.902.

    | Variant | CV |
    |---|---|
    | Full model | 0.902 |
    | Without segmentation pretraining | 0.794 |
    | Without auxiliary segmentation loss | 0.876 |
    | All loss weights set to 1 | 0.884 |
    | Without the location-aware transformer | 0.896 |
    | Full resolution 128×256×256 | 0.916 |

  — [1st](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution) [8 Oct notes]
- **2nd (BraveCoWCoW).**
  - Robustness: a modality head; spacing inferred from slice count when tags were missing; an orientation classifier for T2.
  - Inference runs only the encoder and classification head, about 9 h on 2×T4.
  - Ablation (public/private): baseline 0.844/0.813 → 4× TTA 0.871/0.825 → corrected labels and annotations 0.888/0.857 → left–right augmentation with label swap and 8× TTA 0.898/0.862 → 2-model ensemble 0.900/0.867.
  - Did not work: try/except fallbacks; they deliberately ran without them.

  — [2nd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution) [8 Oct notes]
- **3rd (BTYND).**
  - Smaller stride (feature map 8³ → 25³) with input 128³ → 196³: single-fold LB 0.77 → 0.81.
  - A spacing regressor for missing DICOM tags: private 0.84 → 0.85.
  - 11-model ensemble: 0.86/0.84.
  - Did not work: vessel segmentation; modality/plane auxiliary heads; MIL, LSTM and decoder heads.

  — [3rd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/3rd-place-solution) [8 Oct notes]
- **4th (Harshit Sheoran, solo, 14 days).**
  - CV path (public/private in brackets): ROI crop 0.805 (0.78/0.78) → rotation ±25° 0.83 (0.80/0.81) → 2.5D 0.86 (0.86/0.83) → soft pseudo-labels on about 500k slices 0.89 (0.86/0.84) → + MaxViT 0.896 (0.87/0.84).
  - The ROI crop moved the score "from 0.7 to 0.8".
  - Weighted sampling did not help.

  — [4th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/4th-place-solution) [8 Oct notes]
- **5th.**
  - Brain-crop detector: +0.03–0.05. Horizontal flip with left/right swap: +0.01.
  - Negatives with OOF > 0.9 relabelled as positive, plus pseudo-labelled external MRA: OOF up by +0.0007 to +0.008.
  - Final ensemble: OOF 0.8823, public 0.89.
  - Epochs were picked on the public LB with a 50-series holdout. The team fell from 2nd public to 5th private (−0.046).

  — [5th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution) [8 Oct notes]
- **6th (Ian, Theo & Bartley).**
  - Aggregation: max over slices CV 0.87 → sequence model 0.88 → 3-model ensemble 0.895. Public and private were both 0.84.
  - Each of the following gave about +0.01 CV: an extra fixed-ratio ROI crop; strong geometric and colour augmentation; CutMix/Mixup; horizontal flip with target flip; 2 adjacent frames as channels; spacing-aware neighbour sampling; refined localiser labels.
  - External data did not help.
  - Long stacks were subsampled to save time.

  — [6th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/6th-place-solution) [8 Oct notes]

**7th place (MIC-DKFZ): a single 3D nnU-Net, no ensemble, no TTA.**
- Pipeline and crop: DICOM → NIfTI, reoriented to RAS. A fixed 200×160×160 mm cube over the central-superior head, sized to contain every training aneurysm. — [7th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution)
- Data quality control: mildly misoriented series were flipped; 10 volumes with severe artefacts were dropped. — [7th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution)
- Preprocessing: resampled to the median spacing [0.70, 0.47, 0.47] mm; global z-score. — [7th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution)
- Model: nnU-Net ResEnc (3D full resolution), from scratch. The task was framed as 14-channel blob-heatmap regression: 13 locations plus "present" as their pixelwise max. Blobs were distance-transform spheres; a blob radius of 65 voxels was best on fold 1 (15–95 tested). — [7th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution)
- Loss: BCE on the hardest 20% of voxels (TopK). — [7th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution)
- Training: patch 96×160×128, batch 32, SGD with poly LR. 5-fold CV stratified by modality; most tuning was done on fold 1 only. About 4.5 days on 4×A100-40GB. — [7th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution); [MIC-DKFZ repo](https://github.com/MIC-DKFZ/kaggle-rsna-intracranial-aneurysm-detection-2025-solution)
- Augmentation: no left/right mirroring, because labels encode laterality. — [7th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution)
- Inference: patch-wise prediction, per-channel max, then max over patches. Split across both T4s, about 8 h. Checkpoints trained beyond 1,500 epochs timed out on Kaggle even though only the weights differed. — [7th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution)
- Results: public/private 0.83/0.83. Internal validation was about 0.9 and "did not carry over"; the authors could not explain why. — [7th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution)
- What worked: cropping to the ROI beat whole-volume processing (not quantified). — [7th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution)
- What did not work:
  - nnDetection (Retina U-Net) scored better on public but worse on private and in internal validation;
  - isotropic 1.0 mm resampling "substantially worsened our results";
  - co-training with external aneurysm datasets;
  - larger patches.

  — [7th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution)

**8th place (Konni; the author links Kaggle "khabel" and GitHub "KonradHabel"): 2.5D, no resampling, no 3D, no segmentation.**
- Framing: outlier detection, "we have only a small number of aneurysm spots... but can sample a huge number of negative spots". — [8th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/8th-place-solution)
- Stage 1, a per-image classifier on positive and negative spots:
  - Input: the centre slice plus neighbours as the R and B channels, at two step sizes (1, and 2 to skip a frame).
  - Backbone: ConvNeXt-Base with DINOv3 weights. Images resized to 512, random 448 crop for training, centre crop for validation.

  — [8th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/8th-place-solution)
- Augmentation: ShiftScaleRotate ±5°, RandomRotate90 (p = 1), GridDropout / CoarseDropout / GridDistortion (p = 0.5), and left/right flips with label flip. — [8th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/8th-place-solution)
- Hard negatives: spots where an earlier model's OOF predictions were false positives were sampled more often. — [8th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/8th-place-solution)
- Stage 2: a transformer over the whole sequence of stage-1 features. Step-1 and step-2 sequences were both used as augmentation. — [8th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/8th-place-solution)
- Ensemble and runtime:
  - 4 ConvNeXt-Base models across the two step sizes.
  - Stacks longer than 192 images were subsampled to every n-th image, because of the 12 h limit and variable Kaggle hardware speed.
  - "Nearly half of my submissions are simply timeouts."
  - No CV or LB numbers are given.

  — [8th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/8th-place-solution)

**9th place ("Vibes and Genius Trade-Off"): 2.5D YOLO detectors + a 2D-encoder/3D-head model + GBDT stackers.**
- Inputs:
  - YOLO: 512 px, R/G/B = slices i−1, i, i+1. Negatives: 10 evenly spaced slices per series.
  - "Flayer": EffNetV2-S slice encoder, a Conv3d head, depth 64.

  — [9th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/9th-place-solution)
- 5-fold CV (combined mean of location AUC and presence AUC):

  | Model | CV |
  |---|---|
  | YOLO11m | 0.8056 |
  | EffNetV2-S YOLO | 0.8139 |
  | YOLO ensemble | 0.8322 |
  | Flayer without auxiliary 2D segmentation head | 0.7748 |
  | Flayer with auxiliary head | 0.7829 (+0.008) |

  — [9th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/9th-place-solution)
- Stacking:
  - "Parallel" meta-classifier, 44 features: CV 0.852 averaged, 0.858 with Nelder-Mead.
  - The selected 2-fold result scored public 0.83082 / private 0.8230, a CV-to-private gap of about 0.03.

  — [9th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/9th-place-solution)
- What worked:
  - 2.5D *without* z-resampling: +"0.02+" CV. Z-axis resampling, tested for 1–2 weeks, was "consistently worse".
  - Mixup 0.4 and mosaic 0.4 for YOLO.

  — [9th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/9th-place-solution)
- What did not work:
  - An early YOLO-only ensemble scored about 0.78 LB, with presence AUROC about 0.58, which they attributed to too few negatives.
  - ConvNeXt in half precision gave NaN losses.
  - Filtering YOLO boxes by 3D vessel segmentation cut recall.
  - A keypoint graph model scored well on CV but caused timeouts.

  — [9th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/9th-place-solution)

**11th place (solo): the closest design to ours. 2.5D coatnet_rmlp_2_rw_384 + RNN heads.**
- Stage 1, localisation:
  - TotalSegmentator brain masks (CT and MR modes; the larger mask kept) trained a 3D U-Net at 128³, which crops the ROI around the circle of Willis (CoW).
  - Crop parameters were tuned so that every annotated coordinate stays inside the crop.

  — [11th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/11th-place-solution)
- Stage 2: a 2.5D CoW segmentation model (EffNetV2-B0, 224 px, Dice only 0.56). It was kept for its positional features and to drop slices without vessels. A good 3D CoW model could not be trained. — [11th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/11th-place-solution)
- Stage 3, the aneurysm model:
  - 2.5D, 384 px; classification + keypoint heads.
  - Backbones: coatnet_rmlp_2_rw_384 (5 folds, plus one all-data model) and EffNetV2-S.
  - Three rounds of negative mining: CoW logits first, then OOF hard negatives twice.
  - Vessel-focused crops; focal loss (α = 0.75); EMA; self-distillation on OOF logits.

  — [11th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/11th-place-solution)
- Stage 4: residual LSTM/GRU, BiLSTM, LSTM and BERT heads over slice features, with EMA and heavy mixup (0.5) because the heads "start to overfit quickly". — [11th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/11th-place-solution)
- Results:

  | Submission | CV | Public | Private |
  |---|---|---|---|
  | RNN ensemble | 0.8909 | 0.87689 | 0.82201 |
  | + OOF self-distillation | 0.9035 | 0.86703 | 0.81321 |

  Self-distillation gained +0.0126 CV but lost −0.0099 public and −0.0088 private. — [11th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/11th-place-solution)
- Runtime on 2×T4: 11 aneurysm models + 45 RNN models in under 12 h. MONAI preprocessing on GPU; each volume split in two across the GPUs; RNNs run in two parallel groups. There were "timeout problems in the final days". — [11th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/11th-place-solution)

**15th place: a 2.5D Faster R-CNN feeding a transformer over RoI features.**
- Preprocessing:
  - Resampled to 2.0 mm slices and 0.5 mm in-plane, then cropped/padded to 384×384.
  - Designed to be orientation-agnostic, because some direction cosines were wrong.

  — [15th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/15th-place-solution)
- Multi-plane handling: trained on all planes (axial, coronal, sagittal) but ran inference only on the finest (least-interpolated) plane. — [15th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/15th-place-solution)
- Two models:
  - Model 1: 3-channel input, fasterrcnn_resnet50_fpn_v2, 48 mm boxes.
  - Model 2: 5-channel input, ConvNeXtV2-tiny, 32 mm boxes.

  — [15th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/15th-place-solution)
- Stage 2: a DeiT-Small transformer with 3D sinusoidal positions over RoI features. It was trained on 20× augmented cached features, with label smoothing; a horizontal flip swaps the left/right labels. — [15th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/15th-place-solution)
- Public/private scores:

  | Model | Detector + max only | + transformer | Private gain from transformer |
  |---|---|---|---|
  | Model 1 | 0.78602/0.77517 | 0.80415/0.78740 | +0.012 |
  | Model 2 | 0.78177/0.75816 | 0.80140/0.77955 | +0.021 |

  The 2-model weighted average scored 0.82494/0.80592, +0.019 private over the best single model. — [15th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/15th-place-solution)

**18th place: 2.5D vessel segmentation, per-vessel ROIs, then an ROI classifier.**
- Segmentation: U-Net with EffNet-B0, 3-channel input at 512 px (more accurate than 1 channel), Dice about 0.66. — [18th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/18th-place-solution)
- ROI classifier:
  - 2.5D (3 slices) at 224 px with aspect-preserving resize; ROI-level AUC about 0.94.
  - Series score: the top-k mean over ROIs. Class score: ROI probability × vessel-class probability.
  - Inference used only the last 150 mm along z, capped at 200 slices.

  — [18th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/18th-place-solution)
- What did not work:
  - a dedicated MRI-T2 model;
  - false-positive filtering ("may remove high-probability predictions that help AUROC");
  - their series-level aggregation step.

  — [18th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/18th-place-solution)

### Inferences
- **3D won here for a reason we don't share.** The top-3 3D models were driven by dense voxel labels (vessel masks, aneurysm coordinates). Where only study- or level-level labels exist, as in LSDC, 2.5D won. The parts that transfer to the knee are the ROI step, preprocessing robustness and the aggregation heads, not 3D itself.
- **Our backbone is in the right family.** Our CoAtNet-rmlp-2 384 2.5D design matches the 6th- and 11th-place designs, which used the same backbone; both added a sequence head over slice features. The 6th place reached 0.84/0.84 with only 3 models; its +0.01 CV levers are cheap to try.
- **Keep native slice spacing.** Through-plane resampling hurt independently in 7th and 9th (and 8th avoided it on purpose). Keep native slices and pick neighbours with spacing awareness rather than resampling.
- **Don't build per-series specialists.** A dedicated series model (the 18th place's T2 model) gave nothing, as with LSDC specialists. Share one backbone across our 5 series slots.
- **Treat OOF-derived CV gains as suspect.** In the 11th place, OOF self-distillation raised CV by +0.013 and cost about 0.01 on both leaderboards. My interpretation: OOF teacher predictions for a fold's training samples come from teachers that saw that fold's validation studies, so CV is optimistic. The same risk applies to our distillation labels and OOF-based label weights. Confirm on LB, or use teachers whose folds match the students' folds.
- **Budget a time margin.** Timeouts hit 4 of the 6 lower-ranked write-ups. Keep about 15–20% of the 9 h limit spare, and make heavy late changes (more checkpoints, a new TTA) only after a timed dry run.

### Gaps
- The index has no 10th or 12th–14th write-ups. The 8th-place post reports no CV or LB numbers. The 7th place's CV-to-LB gap (about 0.9 → 0.83) is unexplained.
- I did not verify the aneurysm runtime limit on the rules page. The 8th and 11th authors both refer to 12 h; our knee competition allows 9 h.

## Q2. RSNA 2024 Lumbar Spine (LSDC): two-stage localise-then-classify, crop sizes, series-specific models, and the gain from each stage

### Takeaway
**The shared pipeline.** All top-9 teams:
- localised the level/slice and keypoint;
- classified a few-slice crop around it with small 2D/2.5D CNNs plus attention, MIL or RNN aggregation;
- fused series;
- ensembled small models.

**The gains authors actually measured or emphasised:**
- slice aggregation (1st: attention-MIL, then BiLSTM before attention with an auxiliary attention loss; public log loss 0.37 → 0.35 → 0.33);
- robustness to stage-1 errors through jitter and multi-crop aggregation (1st, 2nd);
- treating levels and sides as extra samples, 5–10× more data (3rd);
- OOF-based removal of noisy labels (2nd: about 1% on public and private).

**What did not help.** Larger backbones, ViTs, 3D CNNs, per-level or per-side specialists, extra series for a condition, and long schedules.

### Cited Findings
Compact cards. Full detail is in the 8 Oct notes, Q1.

| Place | Localise | Crop / slices | Backbone | Aggregation | Measured gains | Did not work |
|---|---|---|---|---|---|---|
| 1st (solo) | 3D ConvNeXt slice classifier + 2D keypoint regression | 5 slices; condition-specific windows on a 512² resize (e.g. sagittal T2 spinal: 96 px left / 32 px right / ±40 px vertical) | ConvNeXt-S, EffNetV2-S | Attention MIL → BiLSTM + auxiliary attention loss | 0.37 → 0.35 → 0.33 public; "crucial": slice shift ±2 (probability = localiser error rate) and ±10 px coordinate jitter | Mamba / self-attention instead of BiLSTM; ConvNeXt-L/B (worse than S); ViT; extra series; long training |
| 2nd (Ian, Yuji, Kevin, Brendan) | YOLOX boxes; CNN-transformer slice localiser | 27 crops per target (3 slice × 9 in-plane offsets), median/mean | ConvNeXt-S, maxvit_tiny_tf_512, coatnet_1_rw_224, NFNet-l0, CSN-R101 | 5-image MIL; attention pooling | OOF noise removal about 1% public and private; multi-crop "significantly" better; sagittal T2 + axial T2 for spinal stenosis | Attention / CLAM-MIL only added diversity |
| 3rd (SonySpine) | CenterNet keypoints; all coordinates manually corrected | 15 sagittal / 10 axial slices, 160–320 px | ResNet18, ConvNeXt-N/T, EffNetV2-S, MaxViT-N | Attention over slices; levels and sides as independent samples (5×/10× data) | Single model CV 0.3858 → 30-model average 0.3643; flip TTA on axial improved LB | One-stage model; multi-level models; specialists; 3D CNN; 2D-CNN + LSTM; focal loss; long training |
| 4th | 2.5D CNN + LSTM for axial levels; UNet keypoints | Multi-view sagittal T1 / sagittal T2 / axial crops | — | Attention pooling *before* a transformer, with auxiliary loss ("key technique") | MLP residual over a Nelder-Mead blend; per-level GBDT stackers | Cross-target stacking features |
| 5th | Gaussian heatmaps (z weight halved per slice away) | — | — | Level-wise RNN ("significantly improved") | Two-step training (AUC first, then head-only on the weighted loss): "significantly"; CutMix p = 1.0 was the best augmentation | ±1 z jitter; left/right label flips; label smoothing |
| 6th (NVSpine) | — | 3-view model: 288×288×9; missing view = zeros; 10% view dropout | CoAtNet + RNN in the sagittal model (about 0.42 private alone); 3-view backbone not recorded | RNN; MLP stacker | More frames helped ("5 was not enough") | Pseudo-labels on external data; denoising; auxiliary localisation decoder; bi-encoder |

Sources [8 Oct notes]: [1st](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/540091), [2nd team](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution), [2nd Ian](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539452), [3rd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453), [4th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539443), [5th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539472), [6th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/541813).

**New details read today.**
- Ian Pan's part scored CV 0.3805 / LB 0.35. He "did not test my retrained models on LB after the noise reduction step described by Yuji, but I imagine it would have been 0.34". — [2nd write-up](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution)
- Ian on why noise reduction was tried: he noticed "a fair number of incorrect labels" while reviewing errors, and expected models to "learn through the noise", but Yuji's test "resulted in a notable improvement in our LB score". — [2nd write-up](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution)
- 2nd-place series weighting for spinal stenosis: sagittal 1, axial 0.5, with "not much difference" from equal weights. — [2nd write-up](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution)
- [second-hand] A Japanese summary of the top solutions credits Brendan Artley's part of the 2nd-place ensemble with:
  - the centre 24 frames per sequence;
  - 9-rotation TTA;
  - training data mixed 50:50 from labelled and pseudo-labelled samples;
  - attention pooling.

  The same summary says ROI extraction effectively multiplied training data "5 to 10 times", and that the 2nd place's noise removal (excluding samples far from their predicted values) "led to a large improvement". — [zenn.dev summary](https://zenn.dev/mkj/articles/ba630a8837ee72)

### Inferences
- **Make the knee crop model robust to anchor errors.** The 1st place's "crucial" trick has a direct knee analogue: during training, jitter the knee anchor (crop centre and scale) with the localiser's measured error distribution.
- **Use multi-crop TTA around the anchor.** Ian Pan's 27-crop aggregation suggests averaging over 3–5 in-plane offsets and/or 2 crop scales around the anchor. This is the cheapest way to use the anchored-crop idea at inference.
- **Upgrade the MIL head before adding anything else.** Every LSDC aggregation upgrade happened inside the head: BiLSTM before attention, attention before a transformer, auxiliary attention losses. For our per-finding attention MIL, this is a head-only change that could be trained on cached slice embeddings.
- **Share weights, don't specialise.** LSDC repeatedly found that sharing weights across levels/sides (more samples) beats specialist models. For the knee, keep one backbone for all findings and series, and put finding specificity in the per-finding attention.

### Gaps
- I could not re-verify the exact 2nd-place noise rule. The 8 Oct notes recorded |label − OOF| ≥ 0.8 with class coefficients for moderate/severe. Today the reader proxy returned only Ian's section of the team write-up, and neither the index nor search surfaced a separate post by Yuji. The coefficients remain unknown.
- Most LSDC gains are described only as "significant"; numeric per-stage deltas exist only for the 1st, 2nd and 3rd places.

## Q3. Other contests (trauma 2023, C-spine 2022, mammography 2023, ICH 2019, HMS 2024, CZII and BYU 2025): compact cards

### Takeaway
**CT volumes (trauma, C-spine, ICH).** The winning formula repeats: organ or vertebra localisation and crop; 2.5D CNNs on adjacent slices; an RNN or attention over slices; soft slice labels built from study labels × visibility; auxiliary segmentation. Measured gains came mainly from crops (trauma 2nd: private log loss 0.39 → 0.35), auxiliary segmentation (trauma 1st: 0.01–0.03), sequence heads, and cheap stage-3 stackers (C-spine 5th: "2–3 LB points").

**Mammography.** The measured gains came from external data (about +0.02 F1). ROI detectors were used but never ablated, and the 4th place found that neither an ROI extractor nor higher resolution helped.

**HMS 2024.** It is EEG, not imaging, but it is the clearest Kaggle precedent for labels whose *quality varies per sample* (the number of expert votes). All of the top 4 trained on everything first, or with quality-weighted loss, and then fine-tuned on or validated with high-agreement samples only.

**CZII and BYU 2025.** These 3D contests add two runtime facts: TorchScript/TensorRT export gave a "200% speedup increase" (2× or 3× depending on how it is read), and matching patch shapes gave 4× faster inference.

### Cited Findings
**RSNA 2023 Abdominal Trauma** [8 Oct notes]
- **1st:**
  - Pipeline: 3D segmentation crop → 96 slices as (32, 3, H, W) → 2D CNN + GRU, max over slices.
  - Labels: soft slice labels = study label × organ visibility.
  - Auxiliary Dice loss: +0.01 to +0.03 log loss.
  - Backbones: CoaT-lite, EffNetV2-s.

  — [1st](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447449)
- **2nd (Theo Viel):**
  - 2D + RNN: 0.41/0.39 → + organ-crop models: 0.37/0.35 (his best private) → full ensemble: 0.35/0.35.
  - "My best sub is my strongest single model."
  - dicomsdl decoding and GPU preprocessing kept the run under 4 h.

  — [2nd](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447453)
- **3rd (Yuji Ariyasu):** organ cubes into 2.5D + 3D models; per-target weighted average; metric-optimising stacking "slightly worse… due to overfit". — [3rd](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447464)

**RSNA 2022 Cervical Spine** [8 Oct notes]
- **1st:**
  - Segmentation: 3D, 128³, trained from 87 masks.
  - Input: per-vertebra crops, 15 slices × 5 channels + the mask as a 6th channel.
  - Model: 2.5D CNN + LSTM. 3D CNN was "not satisfactory".
  - Inference took 7.5 h.

  — [1st](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/362607)
- **3rd:**
  - Soft slice labels: vertebra-volume ratio × study label.
  - Attention over the RNN output "helped a lot".
  - Final stage: frozen CNN plus a new RNN trained on study labels.
  - uint8 batches normalised on GPU; 4.5 h.

  — [3rd](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/362643)
- **4th:** 3D only worked when video-pretrained (ir-CSN-152). — [4th](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/364837)
- **5th:** a stage-3 MLP on per-study mean/min/max of slice predictions, blended 30:70 with max aggregation: "2–3 LB points". — [5th](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/363232)

**RSNA 2019 Intracranial Hemorrhage** [8 Oct notes]
- 1st: three 2D pipelines (windows / adjacent slices / both) + two sequence models; "Sequence model wins… there is no shakeup". — [1st](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/discussion/117210)
- 2nd: the LSTM also received deltas to neighbouring slices' embeddings. — [2nd](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/discussion/117228)
- 3rd: neighbouring-slice concatenation was the biggest gain. — [3rd](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/discussion/117223)

**RSNA 2023 Screening Mammography**
- **1st (dangnh0611, 22nd on public):**
  - ROI detector: YOLOX-nano at 416 px with LINEAR interpolation. Validation AP 96.26/94.21, against 94.09/91.60 with AREA interpolation and 82.09 on the second validation set at 768 px.
  - Classifier: 4 × ConvNeXt-small at 2048×1024.
  - External data: OOF F1 0.4921 → 0.5161 (about +0.02).
  - Soft positive label 0.8–0.9 was no clearer than label smoothing 0.1.
  - Checkpoint grid search over 525 combinations: best OOF pF1 0.5187 vs worst 0.4951.
  - "Transfer the computation task to GPU as much as possible."

  — [1st](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/392449)
- **2nd:** external pretraining at 1280 px, then fine-tuning at 1536; external data took public 0.57 → 0.58 and private 0.51 → 0.53; one high-resolution model rather than an ensemble [8 Oct notes]. — [2nd](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/391676)
- **4th:** neither an ROI extractor nor higher resolution helped [8 Oct notes]. — [4th](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/391208)

**HMS 2024 Harmful Brain Activity** (KL divergence, lower is better; labels are vote distributions from varying numbers of experts)
- **1st (Team Sony):**
  - Staged training by vote count: stage 1 on samples with more than 1 vote (one member) or on all data for 5 epochs (another); stage 2 on samples with ≥10 votes for 15 epochs. "Labels of vote: 1 appeared to be unreliable, so excluded from training." — [HMS 1st](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492560)
  - Label smoothing: "Add offset of 0.02 to each vote before normalization", which regularises low-vote labels more strongly. — [HMS 1st](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492560)
  - Validation used only vote ≥ 10, one sample per EEG, GroupKFold(5). — [HMS 1st](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492560)
  - Blending: non-negative linear regression per target. — [HMS 1st](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492560)
  - Entmax output (α ≈ 1.03) "improved both public and private LB score around 0.004" with no CV gain. — [HMS 1st](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492560)
  - The post gives no separate number for the staged training. — [HMS 1st](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492560)
- **2nd:**
  - Stage 1: loss weight = voters/20, 15 epochs, LR 1e-3. Stage 2: only samples with ≥6 votes, weight 1, 5 epochs, LR 1e-4. Validation only on ≥6 votes. — [HMS 2nd](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492254)
  - The 6-model ensemble moved private only 0.28 → 0.27. — [HMS 2nd](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492254)
- **3rd (credits @darraghdog):**
  - De-duplication: the 106k rows were near-duplicates; reverse-engineering them left 6,350 "true" samples, which gave "quite good cv/lb correlation". — [HMS 3rd](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492471)
  - Pretrained 2D models were trained only on high-quality data; samples with <8 votes, and pseudo-labels, gave no benefit. — [HMS 3rd](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492471)
  - 1D models trained from scratch did use low-vote data, with decaying weights:
    - samples with 3–8 votes, with loss weight decayed from 0.8 to about 0.2;
    - samples with 1–2 votes as a separate loss decayed from 0.5 to 0.

    Including low-vote samples is listed as −0.01 (KL; lower is better). — [HMS 3rd](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492471)
  - Blending: a learned linear blend over 23 base models plus a per-class bias. The bias moved them from 9th to 4th on public. — [HMS 3rd](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492471)
  - Scores (CV/public/private): best single 2D 0.232/0.24/0.29; final 0.207/0.22/0.27. — [HMS 3rd](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492471)
- **4th (members credited as Yuji, tattaka, bilzard, yu4u):**
  - Members' staged recipes: pretrain on all data, then fine-tune on vote ≥ 10 (Yuji); vote > 7 only (tattaka); all data for 32 epochs, then vote > 7 for 32 epochs (yu4u). — [HMS 4th](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492240)
  - Ensembling: single model CV 0.2318 → 4-model 2D ensemble 0.2135. — [HMS 4th](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492240)
  - Blending method: an nn.Linear(48, 6) blender scored public 0.21 / private 0.27, against Nelder-Mead's 0.22 / 0.28. — [HMS 4th](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492240)

**CZII CryoET 2025 and BYU Flagellar Motors 2025** (3D detection in tomograms; Kaggle code competitions)
- **CZII 1st:**
  - Models were "exported via jit or TensorRT, which gave 200% speedup increase and enabled us to have a slightly larger ensemble". The post does not say which export was used in the final run.
  - Segmentation outputs were rank-matched to the detector's score distribution, per class, before blending.

  — [CZII 1st](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/561510)
- **BYU 1st:**
  - An 8-seed ensemble; making patch height equal to width made inference "4x faster" at about the same score; inference took about 10 h.
  - The data was "quite noisy", so the team manually added missing motors in Napari and added external tomograms. No gain is quantified.

  — [BYU 1st](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/discussion/583143)

### Inferences
- **Soft slice labels.** Study label × visibility (trauma 1st, C-spine 3rd) is the standard way to get slice-level supervision from study labels. For the knee, our per-finding attention weights or anatomical priors could act as "visibility", but that is a 6-day stretch.
- **HMS maps directly onto our label situation.** "Number of expert votes" ≈ "agreement across our 31 LLM label tables". The HMS evidence, in order of strength:
  1. validate only on high-agreement labels;
  2. a short final stage on high-agreement labels;
  3. low-agreement samples help only with down-weighting, and helped from-scratch models more than pretrained ones.

  Our CoAtNet is pretrained, which matches the 3rd place's 2D models, where low-quality data gave no benefit.
- **Rank-match before averaging.** CZII's rank matching is relevant when blending heterogeneous models (our CoAtNet with public components). AUC depends only on ranks, so rank-normalising each finding before a weighted average stops one badly calibrated member from dominating.

### Gaps
- I did not review UBC-OCEAN, ISIC 2024 or SIIM write-ups. They are not structural analogues (WSI/tabular/2D), and the index showed no report-label precedent in them.
- The HMS staged-training gain is not quantified separately by any of the top 4.
- The CZII post does not say whether TorchScript or TensorRT was used, or on which GPU.

## Q4. Which tricks repeatedly gave ≥ +0.005 AUC-equivalent? A cross-competition ranking by evidence

### Takeaway
Ranked by consistency of *measured* gains across competitions:
1. Localise, then crop. This includes adding a second crop scale.
2. Adjacent-slice context at native spacing.
3. Flip augmentation/TTA.
4. Learned sequence aggregation over slices.
5. A small ensemble of *different* pipelines.

Each has at least 3 independent measured gains of ≥0.005 AUC (or a clear equivalent in log loss) and few counterexamples.

Dense auxiliary supervision is the single largest lever where masks exist (aneurysm 1st: +0.026, and +0.108 from segmentation pretraining), but the knee has no masks.

**Mixed evidence.** Label cleaning and quality-staged training, soft pseudo-labels / self-distillation, higher resolution and learned blending. Each has both wins and LB reversals.

**Neutral.** Label smoothing and EMA (small but consistent).

### Cited Findings

**Grades.** A = measured ≥ +0.005 AUC-equivalent in at least 3 teams across at least 2 competitions, with few counterexamples. B = measured in 1–2 teams, or mixed. C = mostly neutral or negative.

**1. Localise, then crop (ROI), including a second crop scale.** Grade A.
- Gains:
  - Aneurysm 4th: about 0.7 → 0.8 ([4th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/4th-place-solution)).
  - Aneurysm 5th: +0.03–0.05 ([5th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution)).
  - Aneurysm 6th: extra fixed-ratio crop +0.01 CV ([6th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/6th-place-solution)).
  - Aneurysm 7th: "improved" ([7th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution)).
  - Trauma 2nd: crop models, private log loss 0.39 → 0.35 ([trauma 2nd](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447453)).
  - COVID 1st: lung crop +0.003 over a fixed centre crop ([COVID 1st](https://www.kaggle.com/competitions/siim-covid19-detection/discussion/263658)).
  - Every LSDC top team used crops.
- Counter-evidence: mammography 4th, where the ROI extractor didn't help ([mammo 4th](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/391208)).

**2. Adjacent slices as channels at native spacing.** Grade A.
- Gains:
  - Aneurysm 4th: 2.5D +0.03 CV, +0.02 private ([4th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/4th-place-solution)).
  - Aneurysm 6th: 2 neighbours +0.01; spacing-aware neighbours +0.01 ([6th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/6th-place-solution)).
  - Aneurysm 9th: no z-resampling +0.02 CV ([9th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/9th-place-solution)).
  - ICH 3rd: biggest gain ([ICH 3rd](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/discussion/117223)).
  - Aneurysm 7th: isotropic resampling "substantially worsened" ([7th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution)).
- Counter-evidence: none found.

**3. Flip augmentation and TTA (with label swap where labels are lateral).** Grade A.
- Gains:
  - Aneurysm 2nd: 4× TTA +0.012 private; left–right augmentation + 8× TTA a further +0.005 private ([2nd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution)).
  - Aneurysm 5th: +0.01 ([5th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution)).
  - Aneurysm 6th: +0.01 CV ([6th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/6th-place-solution)).
  - COVID 1st: 8× TTA +0.003–0.004 ([COVID 1st](https://www.kaggle.com/competitions/siim-covid19-detection/discussion/263658)).
  - LSDC 3rd: axial flip TTA improved LB ([LSDC 3rd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453)).
- Counter-evidence: LSDC 5th, where left/right label flips gave no gain ([LSDC 5th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539472)).

**4. Learned sequence aggregation over slices (RNN or transformer instead of max/mean).** Grade A−.
- Gains:
  - LSDC 1st: 0.37 → 0.35 → 0.33 log loss ([LSDC 1st](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/540091)).
  - Aneurysm 6th: +0.01 CV ([6th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/6th-place-solution)).
  - Aneurysm 15th: +0.012 / +0.021 private ([15th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/15th-place-solution)).
  - C-spine 3rd: "helped a lot" ([C-spine 3rd](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/362643)).
  - ICH 1st ([ICH 1st](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/discussion/117210)).
- Counter-evidence:
  - Aneurysm 3rd: MIL/LSTM did not help a 3D model ([3rd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/3rd-place-solution)).
  - LSDC 3rd: 2D-CNN + LSTM failed ([LSDC 3rd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453)).
  - Aneurysm 1st: its transformer added only +0.006 ([1st](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution)).

**5. An ensemble of 2–3 different pipelines.** Grade A−.
- Gains:
  - Aneurysm 2nd: +0.005 private ([2nd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution)).
  - Aneurysm 15th: +0.019 private ([15th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/15th-place-solution)).
  - Aneurysm 6th: +0.015 CV ([6th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/6th-place-solution)).
  - HMS 3rd: private 0.29 → 0.27 ([HMS 3rd](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492471)).
  - LSDC 3rd: CV 0.3858 → 0.3643 ([LSDC 3rd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453)).
- Counter-evidence:
  - Trauma 2nd: best private was a single model ([trauma 2nd](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447453)).
  - Aneurysm 4th: adding MaxViT gave +0 private ([4th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/4th-place-solution)).
  - Aneurysm 7th: a single model was enough for 7th ([7th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution)).

**6. Dense or auxiliary supervision, and segmentation pretraining.** Grade A where masks exist.
- Gains:
  - Aneurysm 1st: auxiliary loss +0.026; segmentation pretraining +0.108 ([1st](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution)).
  - Aneurysm 9th: auxiliary head +0.008 CV ([9th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/9th-place-solution)).
  - Trauma 1st: 0.01–0.03 log loss ([trauma 1st](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447449)).
- Counter-evidence:
  - Aneurysm 3rd: modality/plane auxiliary heads did not help ([3rd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/3rd-place-solution)).
  - LSDC 6th: auxiliary localisation decoder did not help ([LSDC 6th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/541813)).

**7. Strong geometric augmentation and Mixup/CutMix.** Grade B+.
- Gains:
  - Aneurysm 4th: rotation ±25° +0.025 CV, +0.03 private ([4th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/4th-place-solution)).
  - Aneurysm 6th: +0.01 for strong augmentation and +0.01 for CutMix/Mixup ([6th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/6th-place-solution)).
  - LSDC 5th: CutMix was the best augmentation ([LSDC 5th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539472)).
  - HMS 3rd: augmentations −0.03 KL ([HMS 3rd](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492471)).
- Counter-evidence: none found.

**8. Higher resolution or finer feature maps.** Grade B.
- Gains:
  - Aneurysm 1st: +0.014 ([1st](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution)).
  - Aneurysm 3rd: +0.04 LB on a single fold ([3rd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/3rd-place-solution)).
- Counter-evidence:
  - LSDC: small 160–320 px models won ([LSDC 3rd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453)).
  - Mammography 4th: higher resolution didn't help ([mammo 4th](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/391208)).

**9. Label cleaning, correction and quality-staged training.** Grade B.
- Gains:
  - LSDC 2nd: "1%" on public and private ([LSDC 2nd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution)).
  - Aneurysm 2nd: corrected labels and annotations +0.032 private ([2nd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution)).
  - Aneurysm 5th: OOF relabelling +0.0007–0.008 OOF ([5th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution)).
  - HMS 3rd: low-vote samples with decaying weights −0.01 KL ([HMS 3rd](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492471)).
- Counter-evidence:
  - LSDC 6th: denoising didn't help ([LSDC 6th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/541813)).
  - HMS 3rd: low-vote data gave no benefit for pretrained 2D models ([HMS 3rd](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492471)).

**10. Learned per-target blending weights.** Grade B−.
- Gains:
  - HMS 4th: linear blender private 0.27 vs Nelder-Mead 0.28 ([HMS 4th](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492240)).
  - HMS 1st: non-negative least squares per target ([HMS 1st](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492560)).
  - Trauma 3rd: per-target weights ([trauma 3rd](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447464)).
- Counter-evidence:
  - COVID 3rd: Nelder-Mead and ridge overfit ([COVID 3rd](https://www.kaggle.com/competitions/siim-covid19-detection/discussion/263654)).
  - Trauma 3rd: stacking overfit ([trauma 3rd](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447464)).
  - Aneurysm 9th: Nelder-Mead stack CV 0.858 vs private 0.823 ([9th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/9th-place-solution)).

**11. Soft pseudo-labels or self-distillation.** Grade C+ (mixed).
- Gains:
  - Aneurysm 4th: +0.03 CV, +0.01 private ([4th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/4th-place-solution)).
  - RANZCR 1st: soft beats hard, +0.0005 CV ([RANZCR 1st](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/discussion/226633)).
- Counter-evidence:
  - Aneurysm 11th: +0.013 CV but −0.009 private ([11th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/11th-place-solution)).
  - HMS 3rd: pseudo-labels didn't help ([HMS 3rd](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492471)).
  - LSDC 3rd: "not significant" ([LSDC 3rd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453)).
  - COVID 4th: best CV used no pseudo-labels ([COVID 4th](https://www.kaggle.com/competitions/siim-covid19-detection/discussion/264243)).

**12. EMA.** Grade B− (small).
- Gains: COVID 1st +0.001–0.002 ([COVID 1st](https://www.kaggle.com/competitions/siim-covid19-detection/discussion/263658)); used by aneurysm [1st](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution) and [11th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/11th-place-solution).
- Counter-evidence: none found.

**13. Label smoothing.** Grade C (neutral).
- Mammography 1st: on par with soft labels ([mammo 1st](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/392449)).
- HMS 1st: used a 0.02 vote offset as smoothing, not ablated ([HMS 1st](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492560)).
- LSDC 5th: no gain ([LSDC 5th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539472)).

**14. Robust series handling (orientation, spacing, fallbacks).** Grade B (protects score).
- Gains: aneurysm 3rd spacing regressor +0.01 private ([3rd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/3rd-place-solution)); aneurysm 1st OOF-mean fail-safe ([1st](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution)).
- Counter-evidence: aneurysm 2nd ran without fallbacks ([2nd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution)).

### Inferences
**Current position.** Our CoAtNet scores 0.948 alone and 0.956 when blended with public models; the target is 0.969 and the top of the LB is 0.968.

**Plan for the 6 days before 22 Oct, ordered by evidence × cost.** The gain figures are my estimates for the blend, not measurements.
1. **Train our own CoAtNet on knee-anchored crops.** Reuse the public localiser's anchors. Jitter the anchor during training by its measured error (LSDC 1st). Train at 100 mm, plus optionally a second scale such as 120 mm, for diversity (aneurysm 6th's extra crop was worth +0.01 CV).
   - This combines the most consistent trick (localise, then crop) with our proven backbone and MIL head.
   - Estimate: +0.005–0.015. A100/H100 time should make 5 folds feasible within the window.
2. **Flip TTA plus 2–3 in-plane offset/scale TTA, on the strongest members.**
   - For the knee, a left–right flip is label-preserving: medial and lateral are anatomical and survive mirroring. Flip coronal and axial images left–right, and reverse the sagittal slice order, so the series stay consistent.
   - Estimate: +0.002–0.006. Costs inference time only.
3. **Upgrade the aggregation head.**
   - Add a BiLSTM/GRU before the per-finding attention pooling, with an auxiliary loss on the attention logits.
   - Train it on cached slice embeddings. That is cheap: the 11th place ran 45 RNN heads over 11 backbones, and C-spine 3rd trained a new RNN over a frozen CNN.
   - Several heads per backbone add diversity at almost no inference cost.
   - Estimate: +0.002–0.008.
4. **Quality-staged fine-tuning using agreement across the 31 LLM tables (HMS recipe).** Details in Q5.
   - Estimate: +0.002–0.008, but confirm on LB: CV on noisy labels can't confirm it.
5. **Blend conservatively.**
   - Rank-normalise each finding.
   - Use equal weights or constrained per-finding weights fitted on OOF only where every member has OOF on the same folds.
   - Spend both final-submission slots as a hedge (best-OOF vs best-LB).
6. **Skip in this window:** 3D models, auxiliary segmentation, external data, distillation for speed. Their evidence is either conditional on dense labels or mixed, and they cost more than 6 days.

### Gaps
- No competition measured these tricks on report-derived multi-label knee MRI. All gains above transfer across modality and metric, and the knee-specific numbers are estimates.
- Log-loss and KL gains cannot be converted into AUC; I only rank them by direction and consistency.

## Q5. How did winners handle noisy or weak labels, and what transfers to labels mined from radiology reports?

### Takeaway
No Kaggle winner trained on labels mined from free-text reports. The closest precedents are HMS (per-sample label quality = number of expert votes) and LSDC/aneurysm (labels with errors found through OOF disagreement).

**What worked, from strongest evidence down:**
1. Fix or remove labels that strongly disagree with OOF predictions, and confirm on LB (LSDC 2nd: about 1% on public and private; aneurysm 2nd: corrected labels +0.032 private).
2. Validate only on high-quality labels, so CV tracks LB (HMS 1st, 2nd and 3rd).
3. Train on everything, with quality-weighted loss or decaying weights, then fine-tune on the high-quality subset (HMS 1st, 2nd and 4th).
4. Manual relabelling where domain knowledge exists (LSDC 3rd, aneurysm 2nd, BYU 1st).

**What did not work reliably:** plain label smoothing, and self-distillation on OOF logits (it inflated CV and lowered LB in aneurysm 11th).

### Cited Findings
**Removing or flipping OOF outliers.**
- LSDC 2nd: removed training samples whose label disagreed strongly with the team's OOF ensemble, then retrained. It gave "1% on both public and private" [8 Oct notes]. Ian estimated his part would have moved from LB 0.35 to 0.34. — [LSDC 2nd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution)
- Aneurysm 5th: negatives with OOF > 0.9 were relabelled positive. Together with external pseudo-labels, OOF rose by +0.0007 to +0.008. — [5th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution)

**Manual correction.**
- Aneurysm 2nd: corrections plus better annotations gave private 0.825 → 0.857. — [2nd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution)
- LSDC 3rd corrected every coordinate label by hand. — [LSDC 3rd](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453)
- BYU 1st added missing motors by hand because the data was "quite noisy". — [BYU 1st](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/discussion/583143)

**Quality-aware training (HMS).**
- 1st:
  - stage 1 on all labels (or labels with >1 vote), stage 2 on ≥10 votes;
  - single-vote labels dropped as unreliable;
  - +0.02 offset per vote as label smoothing;
  - validation on ≥10 votes only.

  — [HMS 1st](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492560)
- 2nd: loss weight = voters/20 in stage 1; stage 2 on ≥6 votes with a 10× lower LR; validation on ≥6 votes. — [HMS 2nd](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492254)
- 3rd:
  - pretrained 2D models trained on high-quality labels only; samples with <8 votes and pseudo-labels did not help them;
  - from-scratch 1D models used 3–8-vote samples (weight decayed 0.8 → about 0.2) and 1–2-vote samples (0.5 → 0), worth −0.01 KL;
  - de-duplicating near-copies gave "quite good cv/lb correlation".

  — [HMS 3rd](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492471)
- 4th: pretrain on all, then fine-tune on vote ≥ 10; or train on vote > 7 only. — [HMS 4th](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492240)

**Soft labels and smoothing.**
- Mammography 1st: a soft positive label (0.8/0.9) was no clearer than label smoothing 0.1. — [mammo 1st](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/392449)
- LSDC 5th: label smoothing gave no gain [8 Oct notes]. — [LSDC 5th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539472)
- Aneurysm 15th used label smoothing on both heads (not ablated). — [15th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/15th-place-solution)

**Self-distillation as a noise fix.**
- Aneurysm 11th used self-distillation on OOF logits "to handle label noise": CV 0.8909 → 0.9035, public 0.87689 → 0.86703, private 0.82201 → 0.81321. — [11th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/11th-place-solution)
- Aneurysm 4th's soft pseudo-labels on unlabelled slices helped instead: +0.03 CV, +0.01 private. — [4th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/4th-place-solution)

**Test labels may be cleaner than training labels.** In LSDC 8th, a self-supervised loss improved LB but not CV. The author guessed that "the labels of test dataset are cleaner than train dataset" [8 Oct notes]. — [LSDC 8th](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539548)

### Inferences
**Mapping HMS onto our labels.** Per finding and study, the fraction of our 31 LLM tables that call a finding positive plays the role of the vote distribution.
- **(a) Model selection.** Report OOF macro-AUC on high-agreement cells too (e.g. mean < 0.1 or > 0.9), next to the full-label OOF. HMS teams found high-quality-only validation correlated better with LB.
- **(b) Training order.** The repo already has a consensus mask (88% of cells kept) and OOF-based label weights. If the mask is applied from epoch 0, the HMS variant to try is the staged order: train on all cells with soft targets (the mean of the 31 tables), then run a short final stage at a lower LR on high-agreement cells only.
- **(c) Low-agreement cells.** HMS 3rd's from-scratch models gained from decaying weights, while pretrained 2D models gained nothing from low-quality data. For our pretrained CoAtNet, expect the clean-subset fine-tune to matter more than the weighting.

**A leakage trap in OOF-based fixes (my inference from aneurysm 11th).** OOF-based relabelling, removal and distillation can all inflate CV.
- Two mechanisms: validation samples get cleaned with the same OOF predictions used for evaluation, or teacher folds differ from student folds.
- Defences: evaluate every variant against the *original* labels on untouched folds, prefer fold-matched teachers, and accept a change only if it also holds on LB or on the expert-labelled studies.

**Cap the removals.** LSDC 2nd removed only high-disagreement samples. For rare findings, removing "confident disagreements" can delete the true hard positives. Cap removals per finding and prefer down-weighting to deletion.

### Gaps
- No Kaggle precedent uses report-mined (or LLM-mined) labels. The test-label process for the knee competition is not established here.
- The exact LSDC 2nd-place thresholds and class coefficients could not be re-verified (see Q2 Gaps). No HMS team quantified its staged training on its own.

## Q6. How did winners fit big ensembles into the time limit (FP16, TensorRT, cached decoding, smaller inputs)?

### Takeaway
Winners bought time in eight ways:
1. Crop first, so heavy models see small inputs (most top solutions).
2. Run only the encoder and classification head at inference; drop decoders and auxiliary heads (aneurysm 2nd, RANZCR 2nd).
3. Decode each series once and preprocess on GPU (dicomsdl, MONAI, uint8 → GPU normalisation).
4. Split volumes and models across both T4s (aneurysm 7th, 11th).
5. Subsample long stacks (aneurysm 6th, 8th, 18th).
6. Run many cheap second-stage heads over cached features rather than many backbones (aneurysm 11th: 45 RNN heads over 11 backbones in under 12 h).
7. Export to TorchScript or TensorRT (CZII 1st: "200% speedup increase", a larger ensemble).
8. Add fail-safes (aneurysm 1st: OOF-mean fallback).

Timeouts were the most common failure mode in the 2025 aneurysm contest. No RSNA top write-up I read documents TensorRT, and none documents distilling into smaller students for speed.

### Cited Findings
**Runtime per series or per run.**
- Aneurysm 1st: about 18 s per series on 2×T4 (preprocessing 4.0 s + segmentation 10.7 s + classification 3.3 s) [8 Oct notes]. — [1st](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution)
- Aneurysm 2nd: encoder + classification head only, about 9 h on 2×T4 [8 Oct notes]. — [2nd](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution)
- Aneurysm 7th: one 3D nnU-Net, patch inference split across 2×T4, about 8 h. Checkpoints trained beyond 1,500 epochs timed out although only the weights differed. — [7th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution)

**Subsampling long stacks.**
- Aneurysm 8th kept only every n-th image for stacks over 192 images: "Nearly half of my submissions are simply timeouts." — [8th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/8th-place-solution)
- Aneurysm 6th subsampled frames by ½ for stacks over 64 frames and by ¼ over 128 [8 Oct notes]. — [6th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/6th-place-solution)
- Aneurysm 18th capped inference at 200 slices within the last 150 mm along z. — [18th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/18th-place-solution)

**Cheap heads over shared backbones.** Aneurysm 11th ran 11 backbone models + 45 RNN models in under 12 h on 2×T4: MONAI GPU preprocessing, each volume split in two across the GPUs, RNNs run in two parallel groups. They still had timeouts in the final days. — [11th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/11th-place-solution)

**Models dropped for time.** Aneurysm 9th dropped a strong-CV keypoint graph model because it caused timeouts. — [9th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/9th-place-solution)

**Export and patch shape.**
- CZII 1st: "exported models via jit or TensorRT, which gave 200% speedup increase and enabled us to have a slightly larger ensemble". — [CZII 1st](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/561510)
- BYU 1st: matching patch height and width was "4x faster" at about the same score; the 8-seed ensemble still took about 10 h. — [BYU 1st](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/discussion/583143)

**Decoding and preprocessing on GPU.**
- Trauma 2nd: dicomsdl and GPU preprocessing kept the run under 4 h [8 Oct notes]. — [trauma 2nd](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447453)
- C-spine 3rd: uint8 batches normalised on GPU, about 4.5 h [8 Oct notes]. — [C-spine 3rd](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/362643)
- C-spine 1st: inference took 7.5 h [8 Oct notes]. — [C-spine 1st](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/362607)
- Mammography 1st: "transfer the computation task to GPU as much as possible". — [mammo 1st](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/392449)

**Keeping every model resident on GPU.** RANZCR 1st held all models on the GPU and made one pass over the test data, fitting 67 folds into 9 h [8 Oct notes]. — [RANZCR 1st](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/discussion/226633)

**FP16 caveat.** Aneurysm 9th saw ConvNeXt produce NaN losses in half precision. This was in training, not inference. — [9th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/9th-place-solution)

### Inferences
**Budget arithmetic.** About 1,300 studies in 9 h leaves about 25 s of wall-clock per study across 2×T4 (my arithmetic). Since the backbone pass over 96–192 slices dominates, the cheapest diversity is:
- several MIL/RNN heads per backbone over the same cached slice embeddings;
- two crop scales sharing one decode and resample pass;
- flip TTA only on the 1–2 strongest backbones.

**Engineering checklist:**
- decode each series once and share it across all models;
- uint8 to GPU, normalise on GPU;
- FP16 autocast at inference, with a numerical check against FP32 on a few studies;
- put one model family per GPU, or split studies across the GPUs;
- cap slices per series;
- a per-study try/except fallback that returns the per-finding training prevalence or the OOF mean;
- a timed dry run on a public-test-sized subset, keeping at least 15–20% of the time spare.

**TorchScript/TensorRT export is optional.** It is worth trying only if the ensemble is time-bound after the steps above. CZII's "200% speedup increase" (2–3×) is the only documented Kaggle data point I found, and export adds failure risk in the last days.

### Gaps
- No RSNA top-place write-up I read reports TensorRT or ONNX use, measured T4 throughput per backbone, or knowledge distillation into smaller students for runtime. I cannot quantify those options for CoAtNet-rmlp-2 on a T4.
- The mammography 1st-place repo README is a stub ("Code is under refactor"), and its write-up does not describe DICOM decoding or export details. — [repo README](https://github.com/dangnh0611/kaggle_rsna_breast_cancer)
