# In-domain self-supervised (continued) pretraining for knee-MRI classification: evidence and a 1-2 day A100/H100 recipe

Evidence grades: **[A]** primary source read in full (paper text, table, code or config) this session; **[B]** primary source, but only the abstract, model card or README was checked; **[C]** preprint, single dataset, author's own claim, or a secondary summary of a primary source; **[D]** search snippet, anecdote, or attribution not verified. Research date: 16 Oct 2026. Our setting: 4,349 labelled train studies, about 650k 2D slices (about 20k series), 12 noisy report-derived binary labels, macro AUC, 2.5D (3 slices as RGB) attention-MIL on CoAtNet-rmlp-2 (ImageNet-12k, 384 px) and ConvNeXt.

## 1. How much does in-domain SSL continued pretraining improve over ImageNet init with ~1k-10k noisy labels?

### Takeaway
Starting from a strong ImageNet init, in-domain SSL gives a small but fairly reliable gain. The closest analogue to us is CheXpert: multi-label AUC, report-derived noisy labels, and SSL on the same training images. There the gain was about +1.0 AUC point over a supervised-ImageNet baseline, and the in-domain step on its own added only about 0.06 points. Large gains (+5 to +15 points) show up in three situations: unlabelled data far outnumber labelled data, the baseline is weak (random init or a ViT without a good init), or the metric is a linear probe or kNN instead of full fine-tuning. For a fully fine-tuned CoAtNet, a realistic hope is +0.002 to +0.008 macro AUC for a single model, which is not enough on its own to close 0.956 to 0.969. Its main value is probably as a diverse ensemble member (inference).

### Cited Findings
- **Azizi et al., ICCV 2021 (MICLe) [A].** With ResNet-152 (2x), dermatology top-1 was 63.36 ± 0.12 for supervised ImageNet, 69.43 ± 0.43 for SimCLR ImageNet→Derm, and 70.02 ± 0.22 for MICLe ImageNet→Derm (+6.7 points). CheXpert mean AUC went from 0.7625 (supervised ImageNet) to 0.7729 (SimCLR ImageNet→CheXpert), about +1.0 point; the paper's text says "1.1%" — [ar5iv 2101.05224](https://ar5iv.labs.arxiv.org/html/2101.05224); [ICCV page](https://openaccess.thecvf.com/content/ICCV2021/html/Azizi_Big_Self-Supervised_Models_Advance_Medical_Image_Classification_ICCV_2021_paper.html); [Google blog](https://research.google/blog/self-supervised-learning-advances-medical-image-classification/)
- **Azizi: the in-domain step itself [A].** ImageNet→Derm SSL beat ImageNet-only SSL by 1.92 ± 0.16% top-1. Derm-only versus ImageNet→Derm was 66.93 vs 67.63 (ResNet-50 4x) and 66.43 vs 68.30 (ResNet-152 2x). On CheXpert, CheXpert-only versus ImageNet→CheXpert was 0.7683 vs 0.7689 (ResNet-152 2x, ablation table). Fine-tuning from random init reached only 52.48 top-1 — [ar5iv 2101.05224](https://ar5iv.labs.arxiv.org/html/2101.05224)
- **Azizi: scale of unlabelled data [A].** Derm SSL used 454,295 unlabelled images against 15,340 labelled training cases (about 30x more). CheXpert SSL used only the training split, i.e. the same images as fine-tuning. MICLe matched the supervised baseline using 20% (ResNet-50 4x) or 30% (ResNet-152 2x) of labels — [ar5iv 2101.05224](https://ar5iv.labs.arxiv.org/html/2101.05224)
- **Survey of SSL on radiological images [B].** SSL pretraining "generally improves downstream task performance compared to full supervision", "most prominently when unlabelled examples greatly outnumber labelled examples" — [arXiv 2309.02555](https://arxiv.org/abs/2309.02555)
- **Huang et al. 2023, npj Digital Medicine systematic review [B/C].** The review covers 79 studies. A secondary summary says SSL outperforms supervised learning, that it is unclear which strategy is best, and that combinations of methods work best — [Springer](https://link.springer.com/10.1038/s41746-023-00811-0); [Stanford HAI summary](https://hai.stanford.edu/news/could-self-supervised-learning-be-game-changer-medical-image-classification)
- **Ma et al. 2022, "Benchmarking and Boosting Transformers…" [C].** Three findings: good initialization matters more for transformers than for CNNs; masked image modelling (MIM) SSL generalizes better than supervised pretraining; and domain-adaptive SimMIM (continued from ImageNet on 926,028 chest X-rays from 13 datasets) was state of the art on ChestX-ray14 and best on all five targets. The domain-adaptive numbers come from a patent summary and were not verified. Their ResNet-50 ChestX-ray14 AUCs: scratch 80.40, supervised ImageNet 81.70, DINO ImageNet 81.41 — [PMC9646404](https://pmc.ncbi.nlm.nih.gov/articles/PMC9646404); [patent US12394186](https://patents.google.com/patent/US12394186); [code](https://github.com/JLiangLab/BenchmarkTransformers)
- **Zhou et al. (ISBI 2023), MAE "self pre-training" on the target training set only [C].** arXiv v1 reports +9.4% mAUC on ChestX-ray14 lung-disease classification. Segmentation Dice went 77.4→78.9 (brain tumour) and 78.8→83.5 (multi-organ, N=30). The baseline definition and whether 9.4% is absolute or relative were not verified — [arXiv 2203.05573](https://arxiv.org/abs/2203.05573); [v1 PDF](https://arxiv.org/pdf/2203.05573v1)
- **RAD-DINO (Nature Machine Intelligence 2025) [B].** An encoder "continually pre-trained with medical scans by adopting the DINOv2" approach. The model card lists it as fine-tuned from `dinov2-base` on 882,775 public chest X-rays — [arXiv 2401.10815](https://arxiv.org/html/2401.10815v2); [model card](https://huggingface.co/microsoft/rad-dino)
- **"Data Scaling Laws for Radiology Foundation Models" (Sep 2025) [B].** Continued pretraining of RAD-DINO and MedImageInsight on up to 3.5M chest X-rays at constant compute. For some tasks "as few as 30k in-domain samples are sufficient to surpass open-weights foundation models" — [arXiv 2509.12818](https://arxiv.org/abs/2509.12818)
- **DIET-CP (NeurIPS 2025) [C].** Label-free continued pretraining gives "a significant performance boost for state-of-the-art models such as DINOv3 using only 1000 images". A secondary summary reports average linear-probe gains of +4.81 F1 (DINOv2) and +4.43 F1 (DINOv3), and kNN gains of +17.77 and +12.44 — [arXiv 2509.06990](https://arxiv.org/abs/2509.06990); [NeurIPS](https://neurips.cc/virtual/2025/127235)
- **LeJEPA (Nov 2025) [C].** On Galaxy10, in-domain LeJEPA pretraining of small models beat transfer from DINOv3 ViT-S/16. Full fine-tuning: 82.72 (ConvNeXt-V2 Nano) and 83.28 (ResNet-34) vs 81.60. Frozen backbone at full supervision: 78.17 and 76.52 vs 71.38. This is one dataset and not replicated — [arXiv 2511.08544](https://arxiv.org/pdf/2511.08544)
- **Krones (May 2026) [C].** Across chest X-ray, chest CT and retina OCT, "only pre-training on data closely matching the target modality significantly improves downstream performance"; SSL "can outperform supervised methods" depending on context — [arXiv 2605.08819](https://arxiv.org/abs/2605.08819)
- **Truong et al., ML4H 2021 [B].** DINO ImageNet initialization "consistently outperforms all the other initializations" on medical tasks, and SSL ImageNet features beat supervised ImageNet features — [PMLR](https://proceedings.mlr.press/v158/truong21a.html); [arXiv 2108.10048](https://arxiv.org/abs/2108.10048)

### Inferences
- The relevant comparison for us is a strong supervised ImageNet-12k CoAtNet versus the same model plus in-domain SSL continuation. That is closer to Azizi's "in-domain step" (+1.9 top-1 with 30x extra unlabelled derm images; +0.06 AUC points on CheXpert using the same training images) than to the headline +6.7.
- Our unlabelled pool is the same 4,349 studies we already train on, so this is self pre-training, not extra data. Any gain must come from the objective (slice-level, dense or invariance features that weak study-level MIL labels do not teach), not from new patients. That argues for modest single-model gains.
- Most large reported gains are for linear probes or kNN (DIET-CP, MedAug), for ViTs (Zhou, Ma), or for weak baselines. None of these describe our fully fine-tuned hybrid CNN at 384 px.
- Realistic expectation (inference, not measured): +0.002 to +0.008 macro AUC per model, plus some decorrelation for the ensemble. Treat in-domain SSL as one component of the top-5 push, not the main lever.

### Gaps
- I found no study that measures continued SSL from a supervised ImageNet-12k hybrid (CoAtNet or MaxViT) on medical data.
- I could not get exact RAD-DINO versus DINOv2-init classification deltas (full paper tables not retrieved).
- The Ma 2022 domain-adaptive table values were not verified; only the patent summary was seen.
- I found no published knee-MRI study with about 4k labelled studies that compares ImageNet init against ImageNet plus in-domain SSL.

## 2. Which SSL method works best for MRI slices/volumes, and for ConvNet vs ViT/hybrid backbones?

### Takeaway
No method wins everywhere. Joint-embedding, view-invariance methods (DINO/iBOT, SimCLR/MoCo, BYOL) are favoured when the signal is spatially localized, which describes most knee lesions, and when labels are noisy. Masked image modelling (MAE/SimMIM for ViTs; SparK or FCMAE for CNNs) is more robust when the labelled set is small and is the cheapest to run. JEPA-style objectives suit global anatomy. For ConvNets, use SparK (any CNN, including ConvNeXt V1) or FCMAE (ConvNeXt V2 only, because V1 collapses). For CoAtNet, a DINO-style objective is the lowest-risk option because it needs no change to the architecture.

### Cited Findings
- **"Pretext Matters" (Mar 2026) [C].** For spatially localized signal (histopathology), "JEAs are more effective due to their view-invariance objective". For globally structured anatomy (liver ultrasound), "JEPAs are optimal" — [arXiv 2603.22649](https://arxiv.org/abs/2603.22649)
- **Wolf et al., Scientific Reports 2023 [B].** CNNs were pretrained on a large unlabelled CT set and fine-tuned on several CT classification tasks. "The SparK pre-training method is more robust to the training dataset size than the contrastive methods"; the authors recommend it for small annotated datasets — [PMC10662445](https://pmc.ncbi.nlm.nih.gov/articles/PMC10662445/); [arXiv 2308.06534](https://arxiv.org/abs/2308.06534)
- **Khanal et al. 2024 [B].** Among 8 SSL methods (rotation, jigsaw, jigmag, SimCLR, MoCo v2, Barlow Twins, VAE, BigBiGAN) on ResNet-18, contrastive pretraining "performs the best in all datasets". Generative methods underperformed — [arXiv 2401.07990](https://arxiv.org/html/2401.07990)
- **Ergün et al. (Jun 2026), 3D brain MRI [C].** MAE with a spectral-domain reconstruction loss "consistently achieves superior downstream performance" across five disease-detection tasks. JEPA with variance-covariance regularization helps when information "spans multiple decorrelated feature dimensions" — [arXiv 2606.13315](https://arxiv.org/abs/2606.13315)
- **ConvNeXt V2 / FCMAE [B].** Masked pretraining of ConvNeXt V1 shows feature collapse in the MLP layers. GRN fixes this and is most effective together with FCMAE. A secondary source gives V2+FCMAE 84.6% vs V1+FCMAE 83.7% on ImageNet-1k for the Base model — [ar5iv 2301.00808](https://ar5iv.labs.arxiv.org/html/2301.00808); [CVPR slides](https://cvpr.thecvf.com/media/cvpr-2023/Slides/22892_lw8881R.pdf)
- **SparK architecture support [A].** The official code supports ResNets (timm), ConvNeXt (official ConvNeXt code) and custom CNNs (implement `get_downsample_ratio`, `get_feature_map_channels`, `forward`). The mask patch size equals the CNN's downsample ratio — [SparK pretrain README](https://github.com/keyu-tian/SparK/tree/main/pretrain)
- **DINO for CNNs [A].** "This code also works for training DINO on convolutional networks, like ResNet-50", with adapted optimization arguments — [DINO README](https://github.com/facebookresearch/dino)
- **MRI-CORE (Jun 2025) [B].** DINOv2 objective on 6.9M MRI slices from 116,806 volumes (more than 9 sequence types), initialized from SAM and trained for a limited number of epochs. Evaluated on 22 tasks (segmentation-focused); beats SAM and MedSAM — [arXiv 2506.12186](https://arxiv.org/html/2506.12186v2)
- **DINO on OAI knee MRI (ISMRM 2025, abstract #4014) [C].** DINO pretraining on 3D knee MRI "significantly outperforms training from scratch" and "improves classification performance over the supervised pretrained model". No numbers are given — [ISMRM archive](https://archive.ismrm.org/2025/4014.html)
- **MedDINOv3 (Sep 2025) [C].** A multi-stage DINOv3 recipe for domain-adaptive pretraining on CT-3M (3.87M axial CT slices). Gram anchoring was "optional"; segmentation is reported competitive with or better than state of the art — [arXiv 2509.02379](https://arxiv.org/html/2509.02379v1)
- **"Does DINOv3 set a new medical vision standard?" (Sep 2025) [C].** Off-the-shelf DINOv3 is strong on NIH-14 chest X-ray and CT-RATE but weak on whole-slide images, EM and PET. Larger models or higher resolution "does not reliably" help, and NIH-14 AUC peaks at 512 px — [arXiv 2509.06467](https://arxiv.org/html/2509.06467v2)
- **Transformers versus CNNs [C].** Ma et al. 2022: "good initialization is more crucial for transformer-based models than for CNNs" — [PMC9646404](https://pmc.ncbi.nlm.nih.gov/articles/PMC9646404)
- **BrainFound [D].** Extends DINOv2 to 3D brain MRI and reportedly outperforms other SSL strategies and supervised baselines, "particularly in label-scarce and multi-contrast settings". This comes from a search summary and the arXiv attribution (2510.23415) was not opened — [arXiv 2510.23415](https://arxiv.org/abs/2510.23415v2)

### Inferences
- Most of the 12 knee findings are local: meniscal and ligament tears, cartilage defects, bone-marrow oedema, effusion. That points to a JEA (DINO-style self-distillation, ideally with an iBOT-style masked-patch term for ViTs) as the first choice, with MIM as a cheaper second option for an additional model.
- **CoAtNet-rmlp-2** has MBConv conv stages followed by relative-position attention stages. SparK's sparse-conv masking is designed for CNNs and has not been validated through attention stages, and FCMAE needs GRN. DINO or BYOL-style objectives are architecture-agnostic, so use DINO for CoAtNet. SimMIM-style input masking with a mask token (which works for Swin) is a fallback with no evidence for CoAtNet.
- **ConvNeXt V1 (ImageNet or CLIP-LAION):** do not run FCMAE on it (collapse risk). Use DINO or SparK. Alternatively, switch to ConvNeXt V2 and run FCMAE from the official `convnextv2_*.fcmae` SSL weights.
- **ViT path:** continued DINOv2/DINOv3 (the RAD-DINO, Roth, MedDINOv3 and MRI-CORE pattern) has the most evidence. It would be a new ensemble member, not an upgrade of the current encoders.

### Gaps
- I found no head-to-head of DINO vs SparK vs MAE on knee MRI (or any musculoskeletal MRI) classification.
- No evidence on SSL for CoAtNet or MaxViT hybrids in medical imaging.
- The MRI-CORE classification numbers and the ISMRM knee abstract's numbers were not available.

## 3. MRI-specific SSL ideas that fit multi-series knee data, and knee-MRI SSL evidence

### Takeaway
Using multiple images from the same patient or study as positives (MICLe, MedAug) is the best-supported "medical-specific" upgrade. Same-study positives were best in MedAug, and MICLe added about +1.2 top-1 over SimCLR. For knee MRI the cleanest analogue is cross-sequence, same-plane positives: fluid-sensitive vs non-fluid sagittal slices at the same physical position. Cross-plane positives (sagittal vs coronal) only make sense at study level, after MIL pooling. Slice-position and slice-order pretext tasks have weak or only segmentation evidence. Published knee-MRI SSL (MRNet-scale, trained from scratch) does not beat ImageNet-supervised baselines. Video SSL (VideoMAE, V-JEPA) is data-efficient but would need a new 3D ViT and has no MRI classification evidence.

### Cited Findings
- **MICLe [A].** Positive pairs come from multiple images of the same pathology in the same patient case. The average gain over SimCLR was 1.18 ± 0.09% top-1 (ResNet-50 4x: 67.63 → 68.81; ResNet-152 2x: 68.30 → 68.43). It was applied to dermatology only — [ar5iv 2101.05224](https://ar5iv.labs.arxiv.org/html/2101.05224)
- **MedAug (MLHC 2021) [B].** The best positive-pair strategy used images from the same patient and the same study, across all lateralities. It gave +14.4% mean AUC over the ImageNet baseline for pleural effusion with a linear layer trained on 1% of labels. Choosing hard negatives from metadata did not beat baselines — [arXiv 2102.10663](https://arxiv.org/abs/2102.10663)
- **Contrasts as views (Chalcroft et al., 2025) [B].** Multiple MRI contrasts simulated from one quantitative MRI scan were used as views of the same anatomy, with gains over baseline SSL "especially in low-data settings". Evaluation was on segmentation and denoising, not classification — [arXiv 2501.12057](https://arxiv.org/abs/2501.12057); [UCL Discovery](https://discovery.ucl.ac.uk/id/eprint/10216760)
- **Positional Contrastive Learning (Zeng et al., 2021) [B].** Each slice gets a normalized z-position in [0,1]. Slices with close positions are positives, which avoids false negatives from partition-based global contrast. It substantially improved CT/MRI segmentation in semi-supervised and transfer settings. A 2025 follow-up criticizes PCL's hard threshold, which must be tuned per dataset — [arXiv 2106.09157](https://arxiv.org/abs/2106.09157)
- **Knee MRI, jigsaw SSL from scratch (Manna et al. 2020) [A].** Uses MRNet (1,130 train exams, 208 ACL-positive) and a custom CNN trained from scratch on one RTX 2080 Ti. ACL AUC was 0.848 (accuracy 76.62%), against 0.963 (86.63%) for the supervised MRNet baseline — [arXiv 2007.07761](https://arxiv.org/abs/2007.07761)
- **SB-SSL (2022) [C].** Slice-based self-supervised transformers using fewer than 1,000 cases and no external pretraining data reached ACL accuracy 89.17% and AUC 0.954 — [arXiv 2208.13923](https://arxiv.org/abs/2208.13923)
- **SKID (2021) [D].** Pretext learning on knee "MR video" frames for imbalanced multi-label classification. Search snippets give ACL AUC 0.893 and meniscus AUC 0.810, with weaker PIRL, SimCLR and MoCo-v2 baselines; not verified in the full text — [arXiv 2104.10481](https://arxiv.org/abs/2104.10481)
- **OAI knee DINO (ISMRM 2025) [C].** DINO on 3D knee MRI improved classification over supervised pretraining (qualitative only) — [ISMRM](https://archive.ismrm.org/2025/4014.html)
- **VideoMAE (NeurIPS 2022) [B].** Works on "very small datasets (i.e., around 3k-4k videos) without using any extra data". "Data quality is more important than data quantity", and domain shift between pretraining and target "is an important issue". Extremely high masking (90-95%) still works. Results: UCF101 91.3%, HMDB51 62.6% with a vanilla ViT — [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/416f9cb3276121c42eebb86352a4354a-Paper-Conference.pdf); [arXiv 2203.12602](https://arxiv.org/abs/2203.12602)
- **Slices as video frames [C].** A targeted search found no paper applying VideoMAE to MRI slices-as-frames for classification. Adjacent work treats volumes as 3D patches, for example the 3D brain MRI MAE/JEPA study — [arXiv 2606.13315](https://arxiv.org/abs/2606.13315)
- **I-JEPA compute [B].** ViT-H/14 pretrained on ImageNet in under 72 h on 16 A100 GPUs (effective batch 2048) — [ijepa repo](https://github.com/facebookresearch/ijepa); [HF docs](https://huggingface.co/docs/transformers/v4.51.3/model_doc/ijepa)

### Inferences
- **Cross-sequence positive (recommended add-on to DINO; low engineering cost).** With probability 0.3-0.5, take the second global crop from the other series in the same plane of the same study (for example, sagittal fluid-sensitive and sagittal non-fluid) at the nearest physical slice position, matched by DICOM ImagePositionPatient along the slice normal. This teaches invariance to contrast while keeping anatomy and pathology identical. It is the MRI version of MedAug's "same study" positives and Chalcroft's "contrasts as views".
- **Cross-plane positives (sagittal vs coronal) at slice level are anatomically mismatched.** A random sagittal slice and a random coronal slice show different structures, and the lesion may be in one view only. Slice-level positives would teach the model to ignore lesion content. Use cross-plane contrast only at study level: embed each series through encoder plus attention-MIL pooling and run InfoNCE between the pooled series embeddings of the same study. Risk: shortcut learning via site, scanner or patient signatures. Mitigation: sample negatives from the same site and apply strong intensity augmentation. This is higher engineering effort; do it only if time allows.
- **Slice-position (PCL) and slice-order pretext tasks.** Knee FOV and coverage vary, and sagittal slice order runs medial→lateral differently for left and right knees. Any position-based pairing needs laterality-aware normalization from patient coordinates. Evidence is limited to segmentation, so this is low priority.
- **VideoMAE / V-JEPA 2.** Our scale (about 20k series) is in VideoMAE's data-efficient range, but it needs a new ViT video backbone and a new 2.5D/3D inference path within the T4 9 h budget. With 6 days left before 22 Oct, skip it.
- **2.5D consistency.** Pretrain on exactly the same 3-slice channel stacking (i-1, i, i+1) and intensity normalization as fine-tuning, so the SSL-adapted stem matches the fine-tuning input. Randomly reversing slice order (through-plane flip) is a valid augmentation only if fine-tuning also uses it.

### Gaps
- I found no knee or MSK paper with sagittal/coronal or PD-FS/T1 positive pairs for classification.
- No published comparison of slice-level vs study-level contrastive learning for weakly labelled MRI MIL.
- The SKID numbers are unverified.

## 4. Exact recipes, epochs from ImageNet weights vs scratch, and compute budgets

### Takeaway
Continued SSL from a pretrained checkpoint needs far fewer steps than training from scratch: HPT converges in about 5k MoCo-v2 steps (up to 80x faster) and overfits after that on small datasets. Successful continued runs fall between about 1.3M samples seen (HPT), about 90M (RAD-DINO), and 2 GPU-hours to 3 A100-days (Roth et al., histopathology). For our 650k slices, plan 20-50 epochs (13-33M samples) at 224 px with checkpoints every few epochs. Estimated cost is about 30-100 A100-hours per backbone for DINO-style training and about 10-30 for SparK/FCMAE-style. That fits in 1-2 days on 8×H100 for 2-3 backbones. Budget twice the naive throughput estimates.

### Cited Findings
- **HPT (Reed et al., WACV 2022) [A].** Uses MoCo-v2 on ResNet-50 starting from an 800-epoch MoCo-v2 ImageNet base. HPT "typically converges by 5k steps of pretraining regardless of the target dataset size". On CheXpert, HPT at 5k steps matched target-only SSL at 200k steps; on Chest-X-ray-kids it matched 100k steps. On 15 of 16 datasets HPT or HPT-BN matched or beat Base or Target at 400k steps (the "up to 80x"). HPT-BN (training only BatchNorm γ/β, about 0.2% of parameters) is recommended for datasets under about 10k images and was consistently best with fewer than 5k samples — [ar5iv 2103.12718](https://ar5iv.labs.arxiv.org/html/2103.12718); [WACV](https://openaccess.thecvf.com/content/WACV2022/html/Reed_Self-Supervised_Pretraining_Improves_Self-Supervised_Pretraining_WACV_2022_paper.html)
- **Azizi recipe [A].** SimCLR with LARS. Derm: batch 512, LR 0.3, up to 150k steps. CheXpert: best was batch 1024, LR 0.5, up to 100k steps (temperature searched over {0.1, 0.5, 1.0}). X-ray augmentations: random crop, rotation up to 45°, horizontal flip, colour jitter at strength 0.5; Gaussian blur dropped. MICLe: initialized from SimCLR, batch 128, LR 0.1, 100k steps (final model batch 1024, 1000 epochs). Fine-tuning: 30k steps, batch 256, SGD momentum 0.9, grid over LR and weight decay. Pretraining at 224 px, fine-tuning at 448 (derm) or 224 (CheXpert). Compute: about 12 h on 64 TPU cores for ResNet-50 (1x), batch 512, 100 epochs — [ar5iv 2101.05224](https://ar5iv.labs.arxiv.org/html/2101.05224)
- **RAD-DINO released checkpoint [B].** Initialized from `dinov2-base` (ViT-B/14). Shorter image side 518 px, 882,775 images. The released checkpoint is iteration 35k, chosen by linear probe from a 100k-iteration run. 40 images per GPU on 64 A100-80GB GPUs, 2,560 GPU-hours, fp16 with FSDP. The paper's model used private data and fewer GPUs — [model card](https://huggingface.co/microsoft/rad-dino)
- **Roth et al., ISBI 2024 [B].** Continued DINOv2 ViT-S pretraining on histopathology took "two hours or three days" on a single A100 (0.08-0.6% of CTransPath's compute) and matched or beat CTransPath/RetCCL. The authors note a limited study (2 training sets, 3 test sets) — [arXiv 2401.04720](https://arxiv.org/abs/2401.04720); [code](https://github.com/beneroth13/dinov2)
- **ExPLoRA (ICML 2025) [C].** Initialize from DINOv2/MAE, unfreeze 1-2 blocks (plus norm layers), apply LoRA to the rest, and continue SSL on the new domain. Reported +7-8% linear probe on satellite images. An ablation snippet shows 77.48% with 100 GPU-hours versus 54.29% for full pretraining from scratch — [arXiv 2406.10973](https://arxiv.org/abs/2406.10973)
- **SparK defaults (`pretrain/utils/arg_util.py`) [A].** mask=0.6, input 224, batch 4096, base_lr 2e-4 (lr = base_lr × bs/256), weight decay 0.04 rising to 0.2, 1600 epochs, 40 warmup epochs, gradient clip 5, LAMB optimizer. The 384-px ConvNeXt-L variant uses mask 0.75, batch 2048, base LR 4e-4 — [arg_util.py](https://github.com/keyu-tian/SparK/blob/main/pretrain/utils/arg_util.py); [README](https://github.com/keyu-tian/SparK/tree/main/pretrain)
- **ConvNeXt V2 FCMAE defaults (`main_pretrain.py`) [A].** mask_ratio 0.6, norm_pix_loss on, decoder depth 1 with dim 512, weight decay 0.05, blr 1.5e-4, default 800 epochs (TRAINING.md uses 1600 for Base, effective batch 64×64=4096), 40 warmup epochs, RandomResizedCrop scale (0.2, 1.0) at 224 — [main_pretrain.py](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/main_pretrain.py); [TRAINING.md](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/TRAINING.md)
- **timm hosts SSL-only ConvNeXt V2 weights [A].** `timm/convnextv2_tiny.fcmae`, `convnextv2_base.fcmae` and `convnextv2_large.fcmae` resolve on Hugging Face (HTTP 200) — [HF](https://huggingface.co/timm/convnextv2_base.fcmae)
- **DINOv3 provides SSL-distilled ConvNets [A].** `dinov3_convnext_tiny/small/base/large` (29M/50M/89M/198M parameters) and ViT-S/B/L. Weights are gated under the "DINOv3 License"; the repo ships training configs (pretrain, gram anchoring, high-res adaptation) — [dinov3 repo](https://github.com/facebookresearch/dinov3)
- **DINO defaults and timings [A].** `main_dino.py` defaults: out_dim 65536, momentum_teacher 0.996, teacher_temp 0.04, weight decay 0.04→0.4, clip_grad 3.0, 64 images per GPU, 100 epochs, 10 warmup epochs, AdamW, global crops scale (0.4, 1), 8 local crops at (0.05, 0.4). ResNet-50 recipe: `--optimizer sgd --lr 0.03 --weight_decay 1e-4 --weight_decay_end 1e-4 --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14`. ViT-S/16 for 100 epochs on 8 GPUs took 1.75 days; 300 epochs on 16 GPUs took 2.6 days — [DINO README](https://github.com/facebookresearch/dino); [main_dino.py](https://github.com/facebookresearch/dino/blob/main/main_dino.py)
- **MAE compute [B].** 1600 epochs of ViT-L took 31 h on 128 TPU-v3 cores, versus 36 h for 300 epochs of MoCo v3 — [arXiv 2111.06377](https://arxiv.org/pdf/2111.06377)
- **Throughput anchors: timm RTX 3090 AMP NHWC training, samples/s [A].** resnet50 1218; vit_small_patch16_224 1058; convnext_tiny 819; convnext_small 487; vit_base_patch16_224 393; convnext_base 339; convnext_tiny at 384: 282; convnext_small at 384: 166 — [timm benchmark CSV](https://github.com/huggingface/pytorch-image-models/blob/main/results/benchmark-train-amp-nhwc-pt112-cu113-rtx3090.csv)
- **CoAtNet-rmlp-2 at 384 [A].** 73.9M parameters, 47.7 GMACs, 209.4M activations; ImageNet-12k pretrained, fine-tuned on ImageNet-1k — [HF card](https://huggingface.co/timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k)
- **Augmentation sensitivity [A].** In HPT, CheXpert accuracy dropped as the augmentation policy changed, and an HPT run from a supervised base was more augmentation-sensitive on BDD and CheXpert — [ar5iv 2103.12718](https://ar5iv.labs.arxiv.org/html/2103.12718)

### Inferences

**Compute estimates.** These are not measured. Assumptions: A100 ≈ 2× RTX 3090 in AMP training; H100 ≈ 1.5-2× A100; one epoch = 650k 2.5D samples. DINO with 2×224 global and 6×96 local crops plus the teacher forward pass costs about 3.8× one 224 forward+backward. Dense-masked MIM (SparK/FCMAE) costs about 1.1×. CoAtNet-2 at 224 is about 16 GMACs, roughly ConvNeXt-B.

| Backbone @ SSL res | DINO A100-h per epoch | DINO 50 ep (naive) | SparK/FCMAE 50 ep (naive) |
|---|---|---|---|
| ConvNeXt-T @224 | ~0.42 | ~21 | ~6 |
| ConvNeXt-S @224 | ~0.71 | ~35 | ~10 |
| ConvNeXt-B / CoAtNet-rmlp-2 @224 | ~1.0 | ~50 | ~15 |
| ConvNeXt-S @384 | ~2.0 | ~100 | ~30 |
| CoAtNet-rmlp-2 @384 | ~3.8 | ~190 | ~55 |
| ViT-S/16 @224 | ~0.32 | ~16 | ~2-4 (MAE) |
| ViT-B/16 @224 | ~0.87 | ~44 | ~5-10 (MAE) |

- **Cross-check.** DINO's README run (ViT-S, 100 ImageNet epochs, 8 GPUs, 1.75 days) implies roughly 2× the naive estimate. Plan for 2×: CoAtNet-2 DINO at 224 for 50 epochs ≈ 100 A100-h ≈ 55-65 H100-h ≈ 7-8 h on 8×H100.
- **Do SSL at 224 (or 256) and fine-tune at 384, as Azizi did (224 → 448).** DINO at 384 on CoAtNet-2 costs about 4× more.
- **The data loader will be the bottleneck.** Ten crops per sample at about 3-4k samples/s means 30-40k crops/s. Pre-decode series to uint8 arrays (about 320 px, per-series memmap), build triplets on the fly, and run augmentations on the GPU (kornia or torchvision v2 on CUDA). Otherwise a 64-core CPU will starve 8 H100s.

**Proposed 1-2 day recipe, in priority order.**

- **R1 (primary): DINO-style continued self-distillation of the current backbones.**
  - Init: CoAtNet-rmlp-2 IN-12k and/or the ConvNeXt we already fine-tune, at 224.
  - Phase 0, about 1 epoch: freeze the backbone and train only the DINO head, so the randomly initialized head cannot distort the pretrained features (LP-then-FT reasoning; Kumar et al. ICLR 2022 not re-verified here).
  - Then unfreeze with layer-wise LR decay 0.7-0.8 and a backbone peak LR about 0.1× the from-scratch value. Use AdamW, weight decay 0.04→0.2, global batch 512-1024, out_dim 16k-65k, teacher momentum 0.996→1, teacher temperature 0.04→0.07.
  - Crops: global RandomResizedCrop(224, scale 0.25-1), 6 local crops (96, scale 0.05-0.25).
  - MRI augmentations: brightness/contrast ±0.3, gamma 0.7-1.5, blur, Gaussian/Rician noise, smooth multiplicative bias field, slice-gap jitter (i±1 or i±2), horizontal flip only if fine-tuning uses it. No hue or saturation.
  - Cross-sequence same-plane positive for the second global view with p = 0.3-0.5 (see Q3).
  - Length: 20-50 epochs, saving checkpoints at epochs 5, 10, 20, 30, 50.
  - Norms: CoAtNet's conv stages probably use BatchNorm (check the timm config), so use SyncBN or keep BN in train mode with ≥64 images per GPU. ConvNeXt uses LayerNorm.
- **R2 (cheap, SSL-base path, adds a diverse member): ConvNeXt V2 FCMAE continued from `convnextv2_base.fcmae`.**
  - Settings: mask 0.6, norm_pix_loss, decoder depth 1 with dim 512, blr 1.5e-4, 5 warmup epochs, 50-100 epochs at 224.
  - Fine-tune in the 2.5D MIL at 384.
  - This matches the best-supported pattern (SSL base → in-domain SSL: HPT, Azizi, Ma, RAD-DINO).
  - Alternative: SparK on our ConvNeXt V1 (mask 0.6, LAMB, base_lr 2e-4·bs/256), which is experimental from a supervised or CLIP base.
- **R3 (optional ViT member).** Continue DINOv2 ViT-B/14 or DINOv3 ViT-B/16 using the official code on knee slices, at 224 then a short high-res stage, for about 40-90 A100-h; or use ExPLoRA (LoRA plus 1-2 unfrozen blocks) to limit forgetting. Check the DINOv3 licence and the competition's external-model rules first.
- **R4 (experimental, only if R1 works and time remains).** Study-level cross-plane InfoNCE between MIL-pooled sagittal and coronal series embeddings of the same study.

**Checkpoint selection and acceptance.**
1. Cheap proxy: frozen-feature study-level kNN or linear probe on OOF labels at each checkpoint (minutes). DIET-CP shows kNN and linear probes can move a lot without guaranteeing fine-tuning gains, so treat this only as a sanity check.
2. Fine-tune 2 folds with identical recipe and seeds against the baseline. Promote to 5 folds only if mean macro AUC improves by at least +0.003, or if the run adds clear ensemble decorrelation.

**Timeline to 22 Oct (6 days).**
- Day 1: preprocess, then launch R1 (CoAtNet) and R2 in parallel, 4 H100s each.
- Day 2: 2-fold checks.
- Days 3-4: 5-fold fine-tunes of the winners.
- Day 5: ensemble weighting.
- Day 6: buffer.

**Things not to do.**
- Do not pretrain at 384.
- Do not run FCMAE on ConvNeXt V1.
- Do not exceed about 50 epochs without a fine-tune check (HPT showed overfitting after about 5k steps on small sets).

### Gaps
- I did not find measured A100/H100 SSL throughput for CoAtNet or ConvNeXt; the table is extrapolated from RTX 3090 timm numbers and FLOP ratios.
- I found no published LR or epoch schedule specifically for continued DINO from a supervised checkpoint.
- The default DINO base LR was not re-verified.
- The exact MAE 75% mask default and the VideoMAE epoch counts were not re-verified this session.

## 5. Does SSL pretraining help fine-tuning under noisy labels?

### Takeaway
Yes, consistently on synthetic (symmetric) noise: rotation self-supervision cut CIFAR-10 error by 4-6 points, and C2D added more than 27 points at 90% noise. In medical data the effect is smaller and depends on the dataset. Contrastive SSL helps most, ImageNet init is usually as good or better than in-domain SSL from scratch, and the benefit shrinks with class-dependent noise and disappears above the flipping threshold. Report-derived knee labels are class-dependent and systematic (missed findings), so expect only a small robustness gain.

### Cited Findings
- **Hendrycks et al., NeurIPS 2019 [A].** Average test error over 11 corruption strengths, without vs with an auxiliary rotation-prediction task:

  | Setting | CIFAR-10 | CIFAR-100 |
  |---|---|---|
  | No correction | 27.4 → 21.8 | 52.6 → 47.4 |
  | GLC (5% trusted) | 14.6 → 10.5 | 48.3 → 43.2 |
  | GLC (10% trusted) | 11.6 → 9.6 | 39.1 → 36.8 |

  — [ar5iv 1906.12340](https://ar5iv.labs.arxiv.org/html/1906.12340)
- **Contrast to Divide (C2D), WACV 2022 [B].** Self-supervised pre-training before noisy-label methods gave more than 27% improvement over the previous state of the art on CIFAR-100 at 90% noise, and +3% top-1 on mini-WebVision (WebVision and ImageNet validation). It reduces "the warm-up stage's susceptibility to noise level" — [WACV PDF](https://openaccess.thecvf.com/content/WACV2022/papers/Zheltonozhskii_Contrast_To_Divide_Self-Supervised_Pre-Training_for_Learning_With_Noisy_Labels_WACV_2022_paper.pdf); [arXiv 2103.13646](https://arxiv.org/abs/2103.13646)
- **Khanal et al. 2024, five medical datasets (Fetal US, DermNet, COVID-QU-Ex, MURA, NCT-CRC-HE-100K) [B].** Setup: ResNet-18, symmetric noise 0.5-0.8, class-dependent noise 0.3-0.7. Contrastive pretraining (MoCo v2, Barlow Twins, SimCLR) was best. ImageNet init "performed relatively better" overall and beat in-domain SSL on DermNet; in-domain contrastive beat ImageNet only on Fetal US, which has the largest domain gap. MoCo v2 initialized from ImageNet improved robustness further (DermNet only). Pretraining "no longer offers an advantage beyond the label noise rate flipping threshold", and the margin is "relatively small" with class-dependent noise — [arXiv 2401.07990](https://arxiv.org/html/2401.07990)
- **Khanal et al. 2023 [B].** SSL-initialized models "can effectively learn better features and improve robustness against noisy labels" on NCT-CRC-HE-100K and COVID-QU-Ex with induced noise — [arXiv 2308.04551](https://arxiv.org/abs/2308.04551v1)
- **CheXpert as a noisy-label analogue [A].** CheXpert labels are report-derived. In-domain SSL there gave about +1.0 AUC point over supervised ImageNet and about +0.06 points for the in-domain step (see Q1) — [ar5iv 2101.05224](https://ar5iv.labs.arxiv.org/html/2101.05224)

### Inferences
- Our noise is asymmetric and class-dependent (radiology reports omit findings, and LLM label tables disagree), which is the regime where Khanal saw the smallest SSL benefit.
- SSL init will complement our existing label work (OOF-mixed labels, consensus mask), not replace it. Any robustness effect will show up as better fine-tuning stability across seeds and folds, so measure seed variance as well as the mean.
- The best-supported combination is an ImageNet-initialized model plus in-domain contrastive or DINO continuation (Azizi, HPT, and Khanal's MoCo-from-ImageNet result), followed by our noisy-label fine-tuning.

### Gaps
- No study I found measures SSL init against report-derived multi-label noise in MRI.
- Khanal's numeric gains are only in figures; no tables were available.

## 6. Has in-domain SSL helped top Kaggle solutions?

### Takeaway
I found no verified Kaggle medical-imaging solution (RSNA, SIIM, HuBMAP, ISIC) in which image SSL on the competition images gave a quantified gain. The nearest examples are an NLP analogue (pretraining on the competition's unlabelled text helped NBME 2nd place), tabular SSL in ISIC 2024 2nd place, and a non-Kaggle copy-detection challenge where SSL was central to every winner. This is weak evidence that, for Kaggle-style fine-tuning on a few thousand labelled cases, ImageNet-pretrained CNNs usually remain hard to beat with in-domain image SSL.

### Cited Findings
- **NBME (text) 2nd place [D].** Found pretraining on the competition's unlabelled data helped more than off-the-shelf checkpoints (per search summary) — [Kaggle writeup](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/writeups/boring-comment-2-solution)
- **ISIC 2024 2nd place [D].** Self-supervised pre-training with tabular data, then image-model fine-tuning; patient-grouped folds — [Kaggle discussion](https://www.kaggle.com/c/isic-2024-challenge/discussion/532704)
- **Understanding Clouds 6th place [D].** Pretraining on a derived brightness-threshold pseudo-task — [Kaggle writeup](https://www.kaggle.com/competitions/understanding_cloud_organization/writeups/limerobot-6th-simple-solution-pre-training-single-)
- **Image Similarity Challenge (DrivenData, Meta) [B].** SSL was "a key component across all of the winning solutions" for copy detection, which is retrieval, not classification — [DrivenData blog](https://blog.drivendata.org/blog/image-similarity-winners)

### Inferences
- In-domain image SSL is rare in Kaggle medical top-10 write-ups. Teams more often use external supervised pretraining, pseudo-labels or distillation.
- If this works here, it is a differentiator rather than an established trick. Keep the go/no-go fold check strict.
- In a Kaggle code competition the hidden test images are not available offline. SSL sees only train images (plus any public test samples), so it cannot be transductive on test, and including validation folds in SSL slightly inflates OOF CV relative to LB (small, but nonzero).

### Gaps
- I did not search each RSNA competition's top-10 write-ups individually (2019 ICH, 2022 cervical spine, 2023 abdominal trauma, 2024 lumbar spine, 2025 aneurysm). A per-competition scan might find counter-examples.

## 7. Failure modes: when does in-domain SSL hurt or fail?

### Takeaway
The main risks for us are as follows.
1. Continuing SSL from a supervised checkpoint, which HPT found "considerably worse" than from an SSL base. Our CoAtNet is supervised IN-12k.
2. Over-training on a small dataset (HPT overfits after about 5k steps).
3. SSL from scratch losing to ImageNet init.
4. Fine-grained features degrading: DIET-CP saw linear-probe drops on fine-grained sets.
5. Architecture/objective mismatch, such as FCMAE on ConvNeXt V1.
6. Augmentation sensitivity on grayscale medical images.
7. Shortcut learning in metadata-based positives.

### Cited Findings
- **Supervised base [A].** HPT starting from a supervised base was "considerably worse" on DomainNet and more sensitive to augmentation changes on BDD and CheXpert — [ar5iv 2103.12718](https://ar5iv.labs.arxiv.org/html/2103.12718)
- **Overfitting and no-gain cases [A].** HPT overfit after about 5k steps, most visibly on UC-Merced, Flowers, Chest-X-ray-kids and Pascal. It performed about the same as Base on COCO, Pascal and UC-Merced, and lost to target-only SSL on Quickdraw. Chest-X-ray-kids dropped when the base was trained for only 200 epochs — [ar5iv 2103.12718](https://ar5iv.labs.arxiv.org/html/2103.12718)
- **From scratch loses to ImageNet [A/B].** Azizi: Derm-only SSL 66.43 vs ImageNet→Derm 68.30 — [ar5iv 2101.05224](https://ar5iv.labs.arxiv.org/html/2101.05224). Khanal: ImageNet init relatively better than in-domain SSL except on the most out-of-domain dataset — [arXiv 2401.07990](https://arxiv.org/html/2401.07990). Knee jigsaw SSL from scratch reached 0.848 ACL AUC vs 0.963 for the supervised ImageNet-init MRNet — [arXiv 2007.07761](https://arxiv.org/abs/2007.07761)
- **Domain-specific supervised pretraining is not automatically better [B].** ImageNet ResNet-50 "tend[s] to outperform" RadImageNet across 7 medical tasks — [arXiv 2302.08272](https://arxiv.org/abs/2302.08272)
- **Continued SSL can reduce linear-probe accuracy on fine-grained sets [C].** DIET-CP reported drops for DINOv3 on FGVC-Aircraft and Food-101 — [arXiv 2509.06990](https://arxiv.org/abs/2509.06990)
- **Feature collapse [B].** FCMAE on ConvNeXt V1 collapses in the MLP layers without GRN — [ar5iv 2301.00808](https://ar5iv.labs.arxiv.org/html/2301.00808)
- **Label-set size [B].** Contrastive methods are less robust than SparK as the labelled set shrinks (CT) — [PMC10662445](https://pmc.ncbi.nlm.nih.gov/articles/PMC10662445/)
- **Domain shift [B].** Domain shift between pretraining and target data matters, and data quality beats quantity (VideoMAE) — [arXiv 2203.12602](https://arxiv.org/abs/2203.12602)
- **Scale and resolution do not reliably help [C].** Off-the-shelf DINOv3 on medical tasks is not uniformly strong, and larger models or resolution do not reliably help — [arXiv 2509.06467](https://arxiv.org/html/2509.06467v2)
- **Metadata hard negatives give no gain [B].** MedAug — [arXiv 2102.10663](https://arxiv.org/abs/2102.10663)
- **High label noise [B].** The benefit vanishes above the label-noise flipping threshold — [arXiv 2401.07990](https://arxiv.org/html/2401.07990)

### Inferences
- **Mitigations for our case:**
  - Freeze the backbone for about 1 epoch (head warm-up), use a low backbone LR with layer decay, and stop early (20-50 epochs) with fold-checked checkpoints.
  - Consider BN-only or partial unfreezing (HPT-BN, ExPLoRA style) as a low-risk variant.
  - Prefer SSL-base checkpoints (ConvNeXt V2 `.fcmae`, DINOv2/v3) for new ensemble members.
- **Contrastive shortcuts.** Same-study positives can be solved by matching site, vendor or patient signatures: intensity histograms, FOV, noise texture, coil shading. Use strong per-view intensity augmentation and same-site negatives, and watch for collapse in kNN-by-site.
- **Resolution and preprocessing.** Pretraining at 224 then fine-tuning at 384 can shift low-level statistics (a small risk, given Azizi's 224→448). Keep intensity normalization identical between SSL and fine-tuning.
- **Rules and data handling.** Copying competition data to a rented GPU host is internal compute, not redistribution, but use private storage, delete afterwards, and never push images or derived datasets to public hubs. Check the competition rules on external compute and on gated pretrained weights (DINOv3 licence) before relying on R3.

### Gaps
- I found no quantified evidence on how much fine-grained ImageNet-12k features degrade during continued SSL on MRI, or on how to detect it early beyond fold fine-tunes.
- No knee-specific evidence on site/vendor shortcuts in contrastive pretraining.
