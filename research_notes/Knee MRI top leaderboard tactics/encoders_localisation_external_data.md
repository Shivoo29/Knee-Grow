# Image encoders, knee localisation tools and external knee MRI data (releases up to Oct 2026): licences, access, evidence and plug-in feasibility

Research date: 2026-10-16. This note builds on our 8 Oct notes (`Knee MRI prior work and tricks/external_data_and_backbones.md` and `knee_mri_literature.md`) and does not repeat them. It focuses on releases from 2025-2026, licence and access status, and how each candidate would plug into our pipeline.

Pipeline context: CoAtNet-rmlp-2 (ImageNet-12k) at 384 px, 2.5D 3-slice windows, attention MIL, 140 mm centre crop. The strongest public component uses an intensity-heuristic knee anchor plus 100 mm close-ups. Weakest findings: lateral OA, patellofemoral OA, synovitis, lateral meniscus. Inference budget: 9 h on Kaggle 2xT4 for about 1,300 studies.

Rule applied, as stated in the brief: external data and models must be freely and publicly available, free of charge, and need no escalated (institutional) review or sign-off. OAI-derived data and weights are conditional and should be kept out of the clean final. KneeCoT is banned. fastMRI+ is unresolved.

Legend:
- **ungated** = direct download, no approval.
- **auto-gated** = click-through terms on Hugging Face, approved instantly.
- **manual-gated** = the publisher approves each request.
- **key** = a licence key requested through a form.
- UNVERIFIED = not confirmed against a primary source in this session.
- [S] = search-snippet or secondary summary only.

How the speed numbers were made:
- Hugging Face licence, gating, upload date and parameter count come from the Hugging Face API for each repo. Each is cited as the model page.
- Speed is timm's own inference benchmark (RTX 4090, AMP, PyTorch 2.9.1). It is expressed as throughput relative to our backbone, `coatnet_rmlp_2_rw_384` (407.6 img/s at 384 px), on the same GPU — [timm benchmark CSV](https://github.com/huggingface/pytorch-image-models/blob/main/results/benchmark-infer-amp-nchw-pt291-cu128-4090.csv).
- No T4 numbers were found. Absolute T4 time should be scaled from our own CoAtNet T4 timing, and the 4090 ratio will not transfer exactly.

---

## Q1. Which general vision backbones (DINOv3, EVA02, SigLIP2, ConvNeXt-V2, MaxViT, TinyViT, EfficientNetV2, CAFormer, newer 2026 families) are worth trying against our CoAtNet, and what is the evidence on MRI or RSNA Kaggle MRI competitions?

### Takeaway
No source shows any newer general backbone beating an ImageNet-12k/21k CoAtNet on knee MRI. The top RSNA 2024 lumbar-spine MRI solutions all used small or medium ImageNet-family timm backbones plus an explicit localisation stage. Pick a second backbone for **licence cleanliness, architectural diversity and T4 cost**, not leaderboard folklore:
- ConvNeXt-B (CLIP-LAION-2B → IN-12k, Apache-2.0, 1.76× CoAtNet throughput) is the cleanest fast diverse option.
- EVA02-B (MIT) and SigLIP2-B (Apache-2.0) are licence-clean ViT alternatives.
- DINOv3 and the new 2026 EUPE family are freely downloadable, but their licences are custom (DINOv3) or non-commercial (EUPE).

### Cited Findings

**Table Q1. General backbones (all in timm)**

| Model (timm / HF id) | HF upload | Pretraining | Params | Throughput vs CoAtNet-2-384 (4090 AMP) | GMACs @ res | IN-1k top-1 | Licence / access |
|---|---|---|---|---|---|---|---|
| `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` (ours) | 2023-01-20 | IN-12k → IN-1k | 73.9M | 1.00× (407.6 img/s @384) | 47.7 @384 | 87.39 | Apache-2.0, ungated — [HF](https://huggingface.co/timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k) |
| `convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384` | 2023-03-31 | LAION-2B CLIP → IN-12k → IN-1k | 88.6M | 1.76× (716.4 @384) | 45.2 @384 | 87.14 | Apache-2.0, ungated — [HF](https://huggingface.co/timm/convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384) |
| `convnext_base.dinov3_lvd1689m` | 2025-09-11 | DINOv3 SSL, LVD-1689M | 87.6M | 1.76× (same arch as ConvNeXt-B) | 45.2 @384 | n/a (no head) | "dinov3-license" (custom). Ungated in timm; `facebook/dinov3-*` repos are **manual-gated** — [timm HF](https://huggingface.co/timm/convnext_base.dinov3_lvd1689m); [facebook HF](https://huggingface.co/facebook/dinov3-convnext-base-pretrain-lvd1689m) |
| `convnext_small.dinov3_lvd1689m` | 2025-09-11 | same | 49.5M | 2.50× (1018 @384) | 25.6 @384 | n/a | dinov3-license, ungated — [HF](https://huggingface.co/timm/convnext_small.dinov3_lvd1689m) |
| `vit_base_patch16_dinov3.lvd1689m` | 2025-09-17 | same | 85.6M | 3.29× **at 256** (1340); 384 not benchmarked | 23.6 @256 | n/a | dinov3-license, ungated — [HF](https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m) |
| `vit_large_patch16_dinov3.lvd1689m` | 2025-09-17 | same | 303M | 1.12× at 256 (457) | 82.4 @256 | n/a | dinov3-license, ungated — [HF](https://huggingface.co/timm/vit_large_patch16_dinov3.lvd1689m) |
| `eva02_base_patch14_448.mim_in22k_ft_in22k_in1k` | 2023-03-31 | MIM IN-22k → IN-22k → IN-1k | 87.1M | 0.77× (313 @448) | 107.1 @448 | **88.68** | MIT, ungated — [HF](https://huggingface.co/timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k) |
| `vit_base_patch16_siglip_384.v2_webli` (SigLIP 2) | 2025-02-21 | WebLI image-text | 93.2M | 2.07× (845 @384) | 56.1 @384 | n/a | Apache-2.0, ungated — [HF](https://huggingface.co/timm/vit_base_patch16_siglip_384.v2_webli) |
| `convnextv2_base.fcmae_ft_in22k_in1k_384` | 2023-01-05 | FCMAE → IN-22k → IN-1k | 88.7M | not in bench | — | 87.64 | **CC-BY-NC-4.0**, ungated — [HF](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k_384) |
| `maxvit_rmlp_base_rw_384.sw_in12k_ft_in1k` | 2023-01-20 | IN-12k | 116.1M | 0.68× (278 @384) | 71.0 | 87.81 | Apache-2.0 — [HF](https://huggingface.co/timm/maxvit_rmlp_base_rw_384.sw_in12k_ft_in1k) |
| `maxvit_base_tf_384.in21k_ft_in1k` | 2022-12-02 | IN-21k | 119.7M | 0.60× (244) | 73.8 | 87.93 | Apache-2.0 — [HF](https://huggingface.co/timm/maxvit_base_tf_384.in21k_ft_in1k) |
| `tiny_vit_21m_384.dist_in22k_ft_in1k` | 2023-09-01 | IN-22k distillation | 21.2M | 2.98× (1215) | 13.7 | 86.09 | Apache-2.0 — [HF](https://huggingface.co/timm/tiny_vit_21m_384.dist_in22k_ft_in1k) |
| `tf_efficientnetv2_m.in21k_ft_in1k` | 2022-12-13 | IN-21k | 54.1M | 2.19× @384 / 1.39× @480 | 15.9 @384 | 84.77 @384 / 86.01 @480 | Apache-2.0 — [HF](https://huggingface.co/timm/tf_efficientnetv2_m.in21k_ft_in1k) |
| `caformer_b36.sail_in22k_ft_in1k_384` | 2023-05-05 | IN-22k | 98.8M | 0.44× (181) | 72.3 | 88.06 | Apache-2.0 — [HF](https://huggingface.co/timm/caformer_b36.sail_in22k_ft_in1k_384) |
| `hgnetv2_b5.ssld_stage2_ft_in1k` | 2024-02-12 | SSLD distillation, IN-1k | 39.6M | 5.37× @288 (2190) | 10.8 @288 | (not extracted) | Apache-2.0 — [HF](https://huggingface.co/timm/hgnetv2_b5.ssld_stage2_ft_in1k) |
| `vit_base_patch14_reg4_dinov2.lvd142m` | 2023-10-30 | DINOv2, LVD-142M | 86.6M | 0.85× @518 (347) | 152.3 @518 | n/a | Apache-2.0 — [HF](https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m) |
| **EUPE** (Meta, new): `facebook/EUPE-ViT-{T,S,B}`, `EUPE-ConvNeXt-{T,S,B}`; timm `*.eupe_lvd1689m` | facebook 2026-03-23/27; timm 2026-05-27 | Distilled from several domain-expert foundation encoders through one large proxy teacher — [card](https://huggingface.co/facebook/EUPE-ViT-B); paper [arXiv 2603.22387](https://arxiv.org/abs/2603.22387) | ViT-B 85.7M; ConvNeXt-B 87.6M | ≈ ViT-B / ConvNeXt-B | — | n/a | **FAIR Noncommercial Research License**, ungated — [HF](https://huggingface.co/timm/vit_base_patch16_dinov3_qkvb.eupe_lvd1689m); [HF](https://huggingface.co/timm/convnext_base.eupe_lvd1689m) |
| NVIDIA C-RADIOv3-B / C-RADIOv4-SO400M | 2025-04-24 / 2026-01-26 | Agglomerative distillation | 98.3M / 431M | not benchmarked | — | — | NVIDIA Open Model License, ungated — [HF](https://huggingface.co/nvidia/C-RADIOv3-B); [HF](https://huggingface.co/nvidia/C-RADIOv4-SO400M) |
| Meta Perception Encoder `vit_pe_core_base_patch16_224.fb` | 2025-06-20 (timm) | Image-text contrastive | 93.7M | not benchmarked | — | — | Apache-2.0 — [HF](https://huggingface.co/timm/vit_pe_core_base_patch16_224.fb); [facebook](https://huggingface.co/facebook/PE-Core-B16-224) |
| Apple AIMv2-L | 2024-10-29 | Autoregressive multimodal | 309M | — | — | — | "apple-amlr" licence — [HF](https://huggingface.co/apple/aimv2-large-patch14-224) |

Throughput and GMACs come from the [timm 4090 benchmark](https://github.com/huggingface/pytorch-image-models/blob/main/results/benchmark-infer-amp-nchw-pt291-cu128-4090.csv). Top-1 comes from [timm results-imagenet.csv](https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv). Ratios are my arithmetic.

**Evidence from RSNA Kaggle MRI competitions (RSNA 2024 Lumbar Spine Degenerative Classification, sagittal and axial MRI)**
- Final standings: 1st Avengers, 2nd IanPan-Kevin-Yuji-Bartley, 3rd SonySpine s & tkmn & Moyashii — [RSNA news](https://www.rsna.org/news/2024/november/2024-ai-challenge-winners).
- **2nd place** (Artley's part) is a 2-stage pipeline:
  - Stage 1 passes "the eight middle frames of each image sequence into a 2D backbone with an attention head to predict ten coordinates". Predictions fell "within 5 pixels of the ground truth ~99.1% of the time". The image is then cropped around each disc.
  - Stage 2 runs the crop sequence through an encoder → LSTM → attention pooling, with pseudo-labels and confident-learning label cleaning — [README](https://github.com/brendanartley/RSNA-2024-Competition).
  - Its config backbone is `hgnetv2_b5.ssld_stage2_ft_in1k` — [repo configs](https://github.com/brendanartley/RSNA-2024-Competition).
- **3rd place** configs use, alongside a keypoint stage 1 — [repo](https://github.com/Moyasii/Kaggle-2024-RSNA-Pub):
  - `tf_efficientnet_b6/b5/b3.ns_jft_in1k`
  - `convnext_tiny.in12k_ft_in1k_384`, `convnext_nano`
  - `tf_efficientnetv2_s.in21k_ft_in1k`, `efficientnetv2_rw_t`
  - `resnet18.fb_ssl_yfcc100m_ft_in1k`
  - `maxvit_(rmlp_)nano_rw_256`, `coatnet_nano_rw_224`
  - `convnextv2_atto`
- **4th place** training scripts are named for `resnetrs50`, `swinv2_tiny`, `convnext` and `caformer_s18`, plus a `keypoint` stage — [repo](https://github.com/tattaka/rsna-2024-lumbar-spine-degenerative-classification-public).
- None of these top solutions used a medical foundation model (from the configs above).

**Evidence on DINOv3 in medical imaging**
- "Does DINOv3 Set a New Medical Vision Standard?" uses frozen DINOv3, slice by slice, on Medical Segmentation Decathlon segmentation. Average Dice: DINOv3-B 71.3 and DINOv3-L 71.0, vs nnU-Net 81.4. MRI tasks: brain 66.8 vs 78.9, hippocampus 75.3 vs 84.1, prostate 79.8 vs 91.3 (B vs nnU-Net). Scaling "does not consistently follow" (B ≥ L). **No MRI classification task** was evaluated — [arXiv 2509.06467v2](https://arxiv.org/html/2509.06467v2).
- OrthoFoundation (knee X-ray + MRI) ran a transferability estimation (SFDA, PED, LogME, PARC, ITM) on the knee ACL-injury task and chose **DINOv3-L** as the best general initialisation among its candidates — [README](https://github.com/ytrsk/OrthoFoundation).
- Licence precedent (from our 8 Oct notes): the DINOv3 License allows commercial use, but redistribution must include the Agreement and "Built with DINOv3". DrivenData ruled DINOv3-based models not prize-eligible in one competition — [Meta licence](https://ai.meta.com/resources/models-and-libraries/dinov3-license); [DrivenData forum](https://community.drivendata.org/t/is-dinov3-allowed-to-use/11488).

**Older context (verified in the 8 Oct notes, not re-checked here)**
- Brain-tumour MRI linear probe: DINOv2 ViT-L 0.974 AUROC vs IN-21k ViT-L 0.975 — [Baharoon et al.](https://arxiv.org/html/2312.02366v4).
- Medical Slice Transformer (DINOv2 2D encoder plus a transformer over slices) vs 3D ResNet on knee meniscus tear: 0.85 vs 0.69 — [arXiv 2411.15802](https://arxiv.org/abs/2411.15802).

### Inferences
- **Expect diversity, not a single-model jump.** No paper or Kaggle write-up shows DINOv3, SigLIP2 or EVA02 beating a fine-tuned IN-12k CoAtNet on knee MRI. A second backbone family mainly buys ensemble diversity.
- ConvNeXt is the most different inductive bias from CoAtNet (a hybrid conv-attention net). MaxViT-rmlp is a sibling of CoAtNet-rmlp (same `rw` family), so it adds less diversity and is 0.68× the speed.
- **Clean, fast, diverse pick:** `convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384` (Apache-2.0). On the same GPU it needs about 0.57× CoAtNet's time per image (1/1.76).
- **Clean ViT picks:**
  - EVA02-B (MIT, best IN-1k top-1, but 0.77× throughput at 448; it can run at 384 with interpolated position embeddings).
  - SigLIP2-B/16-384 (Apache-2.0, 2.07×).
  - SigLIP2's image-text pretraining may give weaker local features for small lesions than DINO-style or MIM pretraining. This is a hypothesis, not tested.
- **DINOv3 ConvNeXt-B** costs the same as ConvNeXt-B and is ungated through timm, so it passes the "freely available" test. Its custom licence is the open question for the final. EUPE (2026) is non-commercial research-only and should be kept out of the final for the same reason. ConvNeXt-V2 FCMAE weights are CC-BY-NC.
- Practical notes (general knowledge, not sourced here):
  - T4 (Turing) has no bf16, so test fp16 stability of ViTs (EVA02, DINOv3) before committing.
  - Use `dynamic_img_size=True` (timm ViTs) or interpolated position embeddings for 384-px 3-slice inputs.
  - Use layer-wise LR decay for ViTs.
- Kaggle 2xT4 lets two different backbones run in parallel, one per GPU. A second family costs wall-clock time only if one GPU is already saturated.

### Gaps
- No T4 throughput was found for any candidate. The 4090 ratios are indicative only.
- No head-to-head of DINOv3, SigLIP2, EVA02 or EUPE vs CoAtNet or ConvNeXt on knee or any MRI classification task was found.
- The RSNA 2024 lumbar 1st-place (Avengers) backbone was not found; the write-up was not reachable through search.
- DINOv3 ViT-B/16 throughput at 384 is not in the timm benchmark (only 256).

---

## Q2. Which medical / MRI foundation encoders exist, which include knee MRI (and OAI / fastMRI / SKM-TEA / MRNet) in pretraining, and what are their licences, access terms and knee benchmarks?

### Takeaway
Only three released checkpoints have substantial knee-MRI pretraining:
- **OrthoFoundation** (2D DINOv3-L ViT, MIT repo, weights public via GitHub LFS). Its pretraining **includes OAI and fastMRI**, so it is conditional under our rule.
- **PRISM / "MARS"** (3D Swin; over half of its volumes are knee MRI and 26 of its 34 datasets are private; fastMRI possibly included; weights licence unstated).
- **OrthoDiffusion** (3D diffusion, private data, MIT, heavy).

None reports a comparison against a fine-tuned ImageNet 2D backbone. Everything else is the wrong modality or has a restrictive licence:
- MedDINOv3 is CT; RAD-DINO is chest X-ray; Triad covers brain, breast and prostate.
- MedSigLIP (HAI-DEF), Curia (RAIL-M research-only), BiomedParse (CC-BY-NC-SA) carry restrictive licences.

### Cited Findings

**Table Q2. Medical / MRI encoders**

| Model | Release | Pretraining data (knee? OAI / fastMRI / SKM-TEA / MRNet?) | Modality fit | Size / speed | Licence and access | Knee / MSK evidence |
|---|---|---|---|---|---|---|
| **OrthoFoundation-L** | Paper v1 26 Jan 2026 ([arXiv](https://arxiv.org/abs/2601.18250)); repo pushed 25 Aug 2026 | DINOv3-L init plus DINO-style continued SSL on **1,251,655 knee images**: 357,670 radiographs and 893,985 MRI slices. Public part: **OAI** (~100k X-ray + ~200k MRI slices) and **fastMRI** (~300k MRI slices). Private part: 4 centres of Peking University Third Hospital, 130,567 patients — [README](https://github.com/ytrsk/OrthoFoundation) | 2D slices, ViT-L/16 | `OrthoFoudation-L.pth` is **1.21 GB** (Git LFS pointer `size 1213056638`) — [repo](https://github.com/ytrsk/OrthoFoundation). Plain ViT-L/16 runs at 2.34× CoAtNet throughput at 224 and 0.77× at 384 (61.6 / 191.2 GMACs). timm's DINOv3 ViT-L/16 runs at 1.12× at 256 (82.4 GMACs) — [timm bench](https://github.com/huggingface/pytorch-image-models/blob/main/results/benchmark-infer-amp-nchw-pt291-cu128-4090.csv) | Repo **MIT** ([LICENSE](https://github.com/ytrsk/OrthoFoundation/blob/main/LICENSE)), weights ungated via GitHub LFS. Paper **CC BY-NC-ND 4.0** ([arXiv](https://arxiv.org/abs/2601.18250)). DINOv3-derived, so DINOv3 License pass-through is likely (UNVERIFIED) | After full fine-tuning: ACL AUROC 98.49, PCL 84.31, MCL 93.34, cartilage lesion 93.38, meniscus injury accuracy 84.74. ACL AUROC 94.39 with 12.5% of labels. **No baselines in README** — [README](https://github.com/ytrsk/OrthoFoundation). Abstract: "state-of-the-art (SOTA) performance across 14 downstream tasks" and matches supervised baselines with 50% of labels — [arXiv](https://arxiv.org/abs/2601.18250) |
| **OrthoFoundation-MSK-5M** | Repo pushed 2026-07-21 | Expanded corpus of ~5M MSK images (~4M knee, ~1M ankle, hip and shoulder). Includes the OAI/fastMRI subset; a list of 2,000 OAI cases is provided — [README](https://github.com/ytrsk/OrthoFoundation-MSK-5M) | same | 1.31 GB LFS checkpoint — [repo](https://github.com/ytrsk/OrthoFoundation-MSK-5M) | No LICENSE detected by the GitHub API (licence field empty) — [repo](https://github.com/ytrsk/OrthoFoundation-MSK-5M) | Same reported table as above (paper results are for the L model) |
| **PRISM** (the published name of **"MARS"**) | arXiv Aug 2025 ([2508.07165](https://arxiv.org/html/2508.07165v2)); Nature BME Jul 2026 ([HKUST post](https://smartx.cse.ust.hk/2026/07/15/mars/)) | 336,476 3D MRI volumes from 8 public and 26 private datasets. "Over half of the volumes originate from knee MRI scans". Largest private part: "Private-Knee_int, 10,000 cases and 50,000 volumes". Acknowledgements name ADNI, **fastMRI** and PPMI, but whether fastMRI was in pretraining is UNVERIFIED. OAI, MRNet and SKM-TEA appear only downstream — [arXiv](https://arxiv.org/html/2508.07165v2) | **3D** Swin-B/L/H, 1 mm isotropic, 96³ crops — [arXiv](https://arxiv.org/html/2508.07165v2) | Weights on Google Drive (size not checked) — [GitHub](https://github.com/zqiuak/MARS) | Code **Apache-2.0** ([GitHub](https://github.com/zqiuak/MARS)); **no weights licence stated** | MRNet abnormality accuracy 80.05% vs Swin-B from scratch ~77.78%. OAI KL grading 64.20% vs 63.20% (Swin-B) and 64.08% (MAE-3D). OAI-ZIB segmentation Dice 93.20%. **No ImageNet baseline** — [arXiv](https://arxiv.org/html/2508.07165v2) |
| **OrthoDiffusion** | arXiv Feb 2026; HF 2026-05-29 | 15,948 unlabelled knee MRIs, private ("MRI data is **not publicly released**") — [README](https://github.com/lt-0123/OrthoDiffusion) | **3D**, three orientation-specific diffusion backbones, input 16×256×256 PDW per plane — [arXiv](https://arxiv.org/html/2602.20752v1) | 3 × 553 MB (`axial_model.pt`, `coronal_model.pt`, `sagittal_model.pt`) — [HF](https://huggingface.co/lanstat0123/orthodiffusion) | **MIT**, ungated — [HF](https://huggingface.co/lanstat0123/orthodiffusion); [GitHub](https://github.com/lt-0123/OrthoDiffusion) | Private test macro-AUC over 8 findings (ACL, PCL, medial meniscus, lateral meniscus, MCL, LCL, patellar dislocation, effusion): FT 90.79 vs 3D ResNet-18 83.15 and 3D U-Net 82.63. Lowest per-class: lateral meniscus 85.02. **No ImageNet-2D baseline** — [arXiv](https://arxiv.org/html/2602.20752v1) |
| **KneePreM** | arXiv 25 Sep 2026 | 3D U-Net MAE on **19,011 OAI series** (4,791 participants), so OAI-derived — [arXiv 2609.31461](https://arxiv.org/abs/2609.31461) | 3D | — | **No code or weights stated** | fastMRI+ 2-label AUC 0.722 with 50% of data (beats full-data random/SuPreM). Compared only with random init and SuPreM — [arXiv](https://arxiv.org/abs/2609.31461) |
| **MRI-CORE** (Duke) | arXiv 2404.09957 (2024); repo 2025 | Duke-110K: 116,806 volumes, 18 body locations, internal (8 Oct notes) | 2D SAM ViT-B at **1024×1024** — [README](https://github.com/mazurowski-lab/mri_foundation) | ViT-B at 1024 px, heavy | README: "released under the **Apache 2.0 License**". The GitHub API detects no LICENSE file. Weights on Google Drive — [README](https://github.com/mazurowski-lab/mri_foundation) | KneeMRI linear probe exists but has no ImageNet baseline (8 Oct notes) |
| **Triad** | arXiv 2502.14064 (Feb 2025) | **Triad-131K: 131,170 3D MRI volumes of brain, breast, prostate. No knee** — [ilex card](https://huggingface.co/ilex-hub/triad.swinb-mae.1) | 3D | — | MIT per the ilex re-host card, which cites the upstream LICENSE. The GitHub API finds no licence file in [wangshansong1/Triad](https://github.com/wangshansong1/Triad) | None on knee |
| **MedDINOv3** | 2025-09-03 | CT-3M: 3,868,833 axial **CT** slices from 16 public datasets. "Direct use for MRI … not recommended" — [card](https://huggingface.co/ricklisz123/MedDINOv3-ViTB-16-CT-3M) | 2D ViT-B/16 | — | Apache-2.0, ungated | None (CT only) |
| **RadImageNet** | 2022 | 1.35M CT/MRI/US images, **includes knee MRI** (8 Oct notes) — [Mei 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/) | 2D CNNs (ResNet50, DenseNet121 …) | small | Repo **MIT** ([GitHub](https://github.com/BMEII-AI/RadImageNet)). Weights on Google Drive; HF third-party copies tagged MIT ([HF](https://huggingface.co/Lab-Rasool/RadImageNet)) | See Q5. Own paper +4.8 / +4.5 AUC points on knee ACL / meniscus; independent replications did not reproduce this |
| **BiomedCLIP** | 2023 | PMC-15M figure-caption pairs | 2D ViT-B/16, 224 | — | MIT, ungated — [HF](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | None found |
| **RAD-DINO** | 2024 (licence changed to MIT 2025) | Chest X-ray only | 2D ViT-B/14, 518 | 86.6M | MIT, ungated — [HF](https://huggingface.co/microsoft/rad-dino) | Wrong modality |
| **MedImageInsight** (Microsoft) | Oct 2024 | Multi-domain, including MRI — [arXiv 2410.06542](https://arxiv.org/pdf/2410.06542) | 2D | — | Open-weight licence **UNVERIFIED**. The Foundry listing says "provided as-is for research and model development exploration" — [Foundry](https://ai.azure.com/catalog/models/MedImageInsight-onnx) [S]. A third-party HF copy is tagged MIT — [HF](https://huggingface.co/lion-ai/MedImageInsights) | None found |
| **MedSigLIP-448** (Google) | 2025-07-09 | Not read (card is gated) | 2D SigLIP, 448 | 878M incl. text tower | **Health AI Developer Foundations** terms, **auto-gated** — [HF](https://huggingface.co/google/medsiglip-448) | Not checked |
| **Curia / Curia-2 / Curia-MAE** (Raidium) | 2025-09-01 / 2026-03-02 / 2026-08-06 | Curia: 150k CT/MRI exams (8 Oct notes) | 2D ViT-B (~86M) | — | **research-only RAIL-M**, auto-gated — [Curia](https://huggingface.co/raidium/curia); [Curia-2](https://huggingface.co/raidium/curia-2); [Curia-MAE](https://huggingface.co/raidium/Curia-MAE) | Curia has a "kneeMRI" head; no metrics (8 Oct notes) |
| **MedicalNet** | 2019 (HF 2022) | 3D segmentation corpora (details UNVERIFIED) | 3D ResNet | — | MIT — [HF](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet50) | None |
| **SAM-Med3D-turbo** | 2024-2025 | SA-Med3D-140K | 3D, 128³ | 100.5M (ilex copy) | Code Apache-2.0 ([GitHub](https://github.com/uni-medical/SAM-Med3D)); ilex re-host Apache-2.0 ([HF](https://huggingface.co/ilex-hub/sam_med3d.turbo.1)) | Not evaluated for classification |

### Inferences
- **OrthoFoundation is the only released 2D encoder with large-scale knee-MRI pretraining**, and it is a ViT-L that fits our 2.5D slice design. It is also the only candidate here plausibly targeted at our weak findings: cartilage lesion AUROC 93.38 is a lateral/PF OA proxy and MCL is 93.34. Two problems:
  - Its pretraining **includes OAI (~200k MRI slices + ~100k radiographs) and fastMRI (~300k slices)**. Under our rule it is OAI-derived and therefore conditional, and the fastMRI part is unresolved. It belongs in experiments only, unless the host clears OAI-derived weights.
  - It is DINOv3-initialised, so the DINOv3 License may apply to redistributed derivatives despite the repo's MIT file. A lawyer-free reading: assume the DINOv3 obligations carry over.
- PRISM/MARS and OrthoDiffusion are 3D models that expect isotropic or 16-slice volumes. They would need a separate 3D branch and were only compared against from-scratch 3D baselines. Not feasible to integrate and validate in 6 days.
- Licence-clean knee-inclusive options for the clean final are thin: RadImageNet (MIT, older 2D CNNs, contradictory evidence) and OrthoDiffusion (MIT, private data, heavy).
- For a top-5 push, the encoder lever is weaker than localisation (Q3) and ensembling (Q1).

### Gaps
- OrthoFoundation's comparison tables (vs ImageNet, plain DINOv3 or RadImageNet) could not be read: the arXiv HTML returned 404 and the README lists no baselines. Its knee-MRI test sets (public or private) are not stated in the README.
- PRISM/MARS's list of 8 public pretraining datasets was not visible in the fetched text, so whether fastMRI or OAI images are in pretraining is UNVERIFIED.
- The MedSigLIP training-data composition and the MedImageInsight official licence were not verified.
- No released checkpoint was found for KneePreM.

---

## Q3. Which knee anatomy localisation / segmentation models can anchor close-up crops, which structures do they cover, what are their licences, and can they run within the T4 budget?

### Takeaway
**TotalSegmentator `total_mr` (Apache-2.0, ungated) segments the femur (left and right) but not tibia, patella or fibula.** Those are in `appendicular_bones_mr`, which needs a licence key (free for non-commercial use, via a form). On clinical knee MRI, TotalSegmentator reached Dice 0.897 for femur, 0.887 tibia and 0.810 patella, which is ample for crop anchoring.

The licence-clean knee-specific segmenters are either OAI-derived (nnU-Net `aagatti/nnunet_knee`, `wq2012` OAI variants) or slow CPU random forests trained on SKI10 (`wq2012/kneeseg`). No public knee-MRI landmark detector with weights was found.

Best plan: run `total_mr` offline on the training set, derive femur-based anchors, and **distil them into a tiny in-kernel localiser**. The top RSNA lumbar-MRI solutions used the same pattern.

### Cited Findings

**Table Q3. Localisation / segmentation tools**

| Tool | Knee structures | Training data / OAI-derived? | Input fit | Runtime | Licence / access | Knee evidence |
|---|---|---|---|---|---|---|
| **TotalSegmentator `total_mr`** | `femur_left`, `femur_right` (plus hip, sacrum, gluteal muscles etc.). **No tibia, patella or fibula** in the 50-class map — [README class map](https://github.com/wasserth/TotalSegmentator) | TotalSegmentator MRI data | "If the taskname ends with `_mr` it works for MR images" — [README](https://github.com/wasserth/TotalSegmentator) | Fastest settings: "~30s on GPU and ~70s on CPU" for a **large CT** image. Options: `--fast` (3 mm model), `--roi_subset`, `--statistics_extra` (writes per-class `centroid_vox` and `bbox_vox`) — [README](https://github.com/wasserth/TotalSegmentator). Knee-MRI time on T4 UNVERIFIED | **"Openly available for any usage (Apache-2.0 license)"**, ungated. Offline use is documented: install and download weights on a machine with internet, then copy — [README](https://github.com/wasserth/TotalSegmentator) | Femur Dice **0.8970 ± 0.0351** on clinical knee MR, after super-resolving the three anisotropic series to an isotropic 256³ volume — [arXiv 2508.06076](https://arxiv.org/html/2508.06076) |
| **TotalSegmentator `appendicular_bones_mr`** | patella, tibia, fibula, tarsal, metatarsal, phalanges_feet, ulna, radius — [README](https://github.com/wasserth/TotalSegmentator) | — | MR | same CLI | "Available with a license (free licenses available for non-commercial usage …; for a commercial license contact …)", set with `totalseg_set_license -l aca_…` — [README](https://github.com/wasserth/TotalSegmentator). Key issuance is via a web form (Streamlit page could not be read). Slicer forum users report licensed tasks not appearing, licence validation hanging, or `set license` failing — [forum 1](https://discourse.slicer.org/t/totalsegmentator-licensed-tasks-not-appearing-in-3d-slicer-interface/43523); [forum 2](https://discourse.slicer.org/t/failed-to-set-totalsegmentator-license/44362) [S] | Tibia **0.8869**, patella **0.8104**, fibula **0.7232** Dice. Mean of four bones 0.8294; binary bone 0.8807 — [arXiv 2508.06076](https://arxiv.org/html/2508.06076) |
| **`aagatti/nnunet_knee`** | 9 classes: patellar, femoral, medial and lateral tibial cartilage; medial and lateral meniscus; femur, tibia, patella bone — [card](https://huggingface.co/aagatti/nnunet_knee) | Dataset tag `oai-zib`, so **OAI-derived** (conditional). OAI-ZIB has no patella or menisci, so the card's provenance is internally inconsistent (UNVERIFIED) | "T2-weighted knee MRI", 3D nnU-Net cascade (3d_lowres → 3d_cascade_fullres) | "~80 seconds per volume" (hardware not stated) — [card](https://huggingface.co/aagatti/nnunet_knee) | MIT, ungated (HF upload 2025-08-24) | Card: mean DSC 0.906; bones >0.95; menisci >0.84 |
| **`wq2012` kneeseg, SKI10 subchondral-atlas model** | Femur, femoral cartilage, tibia, tibial cartilage — [card](https://huggingface.co/wq2012/knee_3d_mri_segmentation_SKI10_subchondral_atlas) | SKI10 (MICCAI 2010), 100 training cases; **not OAI** | 3D volumes; behaviour on 2D thick-slice clinical PD-FS UNVERIFIED | **CPU only**: "~45–90 seconds per volume on an 8-core CPU", 4.0–5.5 GB peak RAM, 1.56 GB checkpoints — [card](https://huggingface.co/wq2012/knee_3d_mri_segmentation_SKI10_subchondral_atlas) | MIT, ungated (HF 2026-01 to 2026-10); library on [GitHub](https://github.com/wq2012/kneeseg) (MIT) | DSC femur 0.9689, tibia 0.9742, femoral cartilage 0.8026, tibial cartilage 0.8001 on a strict 80/20 SKI10 split — [baseline card](https://huggingface.co/wq2012/knee_3d_mri_segmentation_SKI10) |
| `wq2012` OAI variants (OAI-downsampled 6-structure incl. patella; unified SKI10+OAI) | + patella, patellar cartilage | **OAI-derived** — [card](https://huggingface.co/wq2012/knee_3d_mri_segmentation_SKI10_subchondral_atlas) | 3D | CPU, as above | MIT | — |
| **SegmentAnyBone** (Duke) | Binary bone, 17 body parts incl. knee — [README](https://github.com/mazurowski-lab/SegmentAnyBone) | — | 2D slice-wise SAM / MobileSAM | UNVERIFIED | **CC BY-NC 4.0** ([LICENSE](https://github.com/mazurowski-lab/SegmentAnyBone)); weights on Google Drive | Binary knee-bone Dice **0.7392** vs TotalSegmentator 0.8807 — [arXiv 2508.06076](https://arxiv.org/html/2508.06076) |
| **MedSAM2** | Promptable 3D / video; needs box or point prompts | 455,000+ 3D image-mask pairs and 76,000+ video frames — [card](https://huggingface.co/wanglab/MedSAM2) | 3D propagation from a prompted slice | UNVERIFIED | Weights **CC-BY-SA-4.0** ([HF](https://huggingface.co/wanglab/MedSAM2)); code Apache-2.0 ([GitHub](https://github.com/bowang-lab/MedSAM2)) | No knee evidence found |
| **SAM 2.1** (hiera-base-plus) | Promptable | SA-V | Needs prompts | 80.8M params | Apache-2.0, ungated — [HF](https://huggingface.co/facebook/sam2.1-hiera-base-plus) | — |
| **SAM 3** | Text-promptable | — | — | 860M | Licence "other", **manual-gated** — [HF](https://huggingface.co/facebook/sam3) | — |
| **nnInteractive** (DKFZ) | Interactive 3D; an annotation helper | — | — | — | Weights **CC-BY-NC-SA-4.0** ([HF](https://huggingface.co/MIC-DKFZ/nnInteractive)); code Apache-2.0 ([GitHub](https://github.com/MIC-DKFZ/nnInteractive)) | — |
| **BiomedParse** | Text-prompted segmentation | — | 2D | — | Weights **CC-BY-NC-SA-4.0**, auto-gated ([HF](https://huggingface.co/microsoft/BiomedParse)); code Apache-2.0 | Knee classes UNVERIFIED |
| **AutoMedLabel** (npj Digit Med 2026) | SAM / SAM2 / MedSAM fine-tuned on MSK MRI, with detection used to automate prompting — [arXiv 2501.13376](https://arxiv.org/abs/2501.13376) | Datasets not named on the abstract page | — | — | Code Apache-2.0 ([GitHub](https://github.com/gabbieHoyer/AutoMedLabel)); weights not stated | "High concordance to expert annotations", no numbers on the abstract page |
| **Roboflow Universe knee projects** | `kneemriv2` is **classification** (classes normal, abnormal, acl, meniscus, both), CC BY 4.0 — [Roboflow](https://universe.roboflow.com/kneemri/kneemriv2/model/1) [S]. A `meniscus` detection project is CC BY 4.0 with no published description — [Roboflow](https://universe.roboflow.com/isimg-waqpn/meniscus) [S] | Provenance unknown | — | — | CC BY 4.0, free account | None |

**Relevant design evidence**
- The RSNA 2024 lumbar 2nd-place pipeline localised with a 2D backbone plus attention head on 8 middle frames, landed within 5 px about 99.1% of the time, then cropped — [README](https://github.com/brendanartley/RSNA-2024-Competition). The 3rd and 4th places also had keypoint stages — [3rd](https://github.com/Moyasii/Kaggle-2024-RSNA-Pub); [4th](https://github.com/tattaka/rsna-2024-lumbar-spine-degenerative-classification-public).
- Tiny localiser cost in the same timm benchmark — [timm bench](https://github.com/huggingface/pytorch-image-models/blob/main/results/benchmark-infer-amp-nchw-pt291-cu128-4090.csv):
  - `convnext_nano` at 224: 6,224 img/s, 15.3× CoAtNet-2-384
  - `tf_efficientnetv2_s` at 384: 1,459 img/s, 3.58×
- The 2025 knee study that used TotalSegmentator excluded scans with dislocated or subluxated patellae and intense swelling. It notes that the patella is the highest-variance structure — [arXiv 2508.06076](https://arxiv.org/html/2508.06076).

### Inferences
- **Femur-only anchoring is enough for our close-ups, so the licensed task can be skipped.** All of these can be derived from the distal femur alone (proposed geometry, untested):
  - joint line: inferior edge of the condyles on sagittal and coronal
  - medial vs lateral compartment: split at the intercondylar notch, the narrowest point of the femur mask on coronal or axial
  - patellofemoral crop: centred anterior to the trochlear groove on sagittal and axial
  - posterior crop (Baker's cyst, posterior horns): behind the posterior condyles
  - suprapatellar recess crop (effusion, synovitis): anterior-superior to the trochlea

  This keeps the anchor pipeline on Apache-2.0 only. Take laterality from DICOM; `femur_left` vs `femur_right` may be unreliable on single-knee fields of view.
- **Do not run TotalSegmentator or nnU-Net inside the Kaggle kernel.**
  - At an assumed 10–30 s per series and 2–3 series per study, 1,300 studies would take roughly 7–33 GPU-hours (my arithmetic, not measured). Even split over two T4s, the upper half of that range breaks the 9 h budget.
  - `aagatti` (~80 s per volume) and kneeseg (45–90 s on CPU) are worse.
  - Instead, run `total_mr --fast --roi_subset femur_left femur_right --statistics_extra` offline on rented A100/H100 for each training series. Then train a small heatmap or box regressor (for example `convnext_nano` or `tf_efficientnetv2_s`, Apache-2.0) on those pseudo-labels and ship only that model.
- **Fat suppression risk.** On fat-suppressed sequences, marrow signal is suppressed, so `total_mr` may segment bone less reliably on PD-FS/T2-FS than on non-fat-saturated PD/T1. Prefer the non-FS series of each plane for pseudo-labelling, then transfer the anchor to other series through DICOM patient coordinates. This is a hypothesis to check on about 50 studies by eye first.
- MedSAM2 or SAM 2.1 could refine a tibia or patella mask from femur-derived box prompts. That adds complexity for little anchor gain, so it is not a 6-day item.
- `appendicular_bones_mr`, SegmentAnyBone, nnInteractive and BiomedParse carry non-commercial licences, and the first needs a key. If any is used even only to make training pseudo-labels, it is part of the solution, so keep them out of the clean final.

### Gaps
- No study was found reporting TotalSegmentator accuracy on **raw** anisotropic 2D clinical knee series without super-resolution, or per sequence (PD-FS vs PD vs T1).
- The TotalSegmentator academic-licence issuance process could not be confirmed (the form page is a JavaScript app), so it is unknown whether a key is instant or manually reviewed.
- No T4 runtimes were found for any segmentation tool on knee MRI.
- No public knee-MRI landmark or keypoint detector with downloadable weights was found.

---

## Q4. Which public knee MRI datasets are usable under the rule, and would any help our 12 findings, especially the weak ones (lateral OA, patellofemoral OA, synovitis, lateral meniscus)? (2026 updates only)

### Takeaway
No newly released (2025-2026) labelled knee MRI dataset was found. The 2026 knee foundation-model papers keep their data private or rely on OAI.

Among accessible sources:
- MRNet and KneeMRI only add ACL and meniscus labels with no side split.
- SKM-TEA has 155 cases.
- fastMRI+ is the only clean-ish source touching our weak findings (meniscus-tear and cartilage boxes without side or compartment labels), and it is unresolved.
- OAI/MOAKS has all four weak findings but is excluded.

Recommendation: spend none of the 6 days on external data.

### Cited Findings
- **MRNet.** The Stanford AIMI page links a Redivis download and asks for citation by DOI 10.71718/rcbp-8c35. Its "Terms & Conditions" and "License" sections render empty. Data: 1,370 exams; 1,104 abnormal, 319 ACL tears, 508 meniscal tears; "labels were obtained through manual extraction from clinical reports" — [AIMI MRNet](https://aimi.stanford.edu/datasets/mrnet-knee-mris).
  - AIMI datasets are available "without charge for non-commercial research purposes" — [AIMI shared datasets](https://aimi.stanford.edu/shared-datasets) [S].
  - The RUA prohibits commercial use and redistribution (8 Oct notes) — [AIMI commercial use](https://aimi.stanford.edu/datasets-commercial-use).
  - Access is an individual login plus a click-through; no institutional sign-off was described. This is UNVERIFIED for Redivis specifically.
- **fastMRI+ label coverage relevant to weak findings** (8 Oct notes) — [Sci Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC8983757/):
  - meniscus tear 5,658 boxes / 663 subjects, **no side label**
  - cartilage partial-thickness 2,985 / 588 and full-thickness 615 / 122, **no compartment label**
  - effusion 1,311 / 142
  - **no synovitis class**
  - Images need the fastMRI DSA, which is "internal research or educational purposes only" and has a "derived data files" clause — [fastMRI](https://fastmri.med.nyu.edu/).
- **Unofficial re-uploads exist and should not be used** (redistribution conflicts with the fastMRI DSA and the AIMI RUA described in the 8 Oct notes):
  - `AUMLProject/fastmri-knee-singlecoil-rss` (2026-04, ungated) — [HF](https://huggingface.co/datasets/AUMLProject/fastmri-knee-singlecoil-rss)
  - `osbm/fastmri-knee` (manual-gated) — [HF](https://huggingface.co/datasets/osbm/fastmri-knee)
  - `AUMLProject/mrnet-acl` (2026-04) — [HF](https://huggingface.co/datasets/AUMLProject/mrnet-acl)
  - The Roboflow `kneemriv2` classes (normal/abnormal/acl/meniscus/both) mirror MRNet's label set, so it may be derived from MRNet. This is my inference, not confirmed — [Roboflow](https://universe.roboflow.com/kneemri/kneemriv2/model/1).
- **New 2026 knee models whose data are not usable:**
  - OrthoDiffusion: private, "not publicly released" — [README](https://github.com/lt-0123/OrthoDiffusion).
  - OrthoFoundation: OAI + fastMRI + private cohort under "restricted access … subject to institutional review" — [README](https://github.com/ytrsk/OrthoFoundation).
  - KneePreM: OAI only — [arXiv](https://arxiv.org/abs/2609.31461).
- **OAI-derived artefacts (conditional or excluded under our rule):**
  - 3DReasonKnee labels and OAIZIB-CM (8 Oct notes)
  - KneePreM
  - OrthoFoundation weights
  - `aagatti/nnunet_knee`
  - `wq2012` OAI variants
  - `PhenoOAlab/Enthesophyte_OAI_Knee` (manual-gated) — [HF](https://huggingface.co/datasets/PhenoOAlab/Enthesophyte_OAI_Knee)
- **Unchanged from the 8 Oct notes:**
  - KneeMRI (Zenodo, CC BY-NC-ND 4.0): ACL only
  - SKM-TEA (AIMI RUA): 155 qDESS cases with 16 pathology box classes
  - K2S: not distributed after the challenge
  - KneeCoT: banned
  - MOST: institutional signature

### Inferences
Per weak finding, among sources that pass the rule:
- **Lateral meniscus:** fastMRI+ meniscus boxes (if the side can be derived from box x-position and laterality) or MRNet meniscus labels (no side). Both are weak, and fastMRI+ is unresolved.
- **Lateral OA and PF OA:** only indirect fastMRI+ cartilage boxes (no compartment label). No clean source has compartment OA grades.
- **Synovitis:** none (MOAKS effusion-synovitis and Hoffa-synovitis are OAI-only).
- With 6 days left, external data cannot plausibly beat the anchor and ensemble levers on these findings. Skip it unless the host clears fastMRI+ for free within a day.

### Gaps
- No search turned up a 2025-2026 public knee MRI dataset with compartment-level OA, synovitis or side-specific meniscus labels. It is possible one exists outside HF, Zenodo and arXiv.
- The current Redivis access flow for MRNet (instant vs reviewed) is not confirmed.

---

## Q5. What evidence (2023-2026) shows domain-specific pretraining beating ImageNet for knee MRI classification, and how large are the effects?

### Takeaway
Domain-specific pretraining shows clear gains only against **from-scratch** baselines or in **low-label** settings: +7.6 macro-AUC (OrthoDiffusion vs 3D ResNet-18), +2.3 accuracy points (PRISM vs Swin-B from scratch), and label-efficiency claims. Against well-fine-tuned ImageNet 2D backbones the evidence is null or contradictory (RadImageNet), or simply absent (all 2025-2026 knee foundation models). At about 4,300 labelled studies, expect at most a small single-model gain. The main value of a domain encoder would be ensemble diversity.

### Cited Findings

| Study (year) | Comparison | Effect | Caveat |
|---|---|---|---|
| RadImageNet, Mei et al. 2022 | RadImageNet vs ImageNet, 2D knee slices (1,021 ACL-tear and 4,201 meniscal-tear images) | ACL AUC 0.97 vs 0.91, reported as +4.8%; meniscus 0.96 vs 0.92, reported as +4.5% (figures as quoted in our 8 Oct notes) — [PMC9530758](https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/) | Own paper, small 2D sets |
| Juodelyte et al. 2023 | RadImageNet vs ImageNet ResNet50, 7 tasks | "ImageNet tend[s] to outperform" — [arXiv 2302.08272](https://arxiv.org/abs/2302.08272) | Not knee-specific |
| Frees et al. 2025 | RadImageNet vs ImageNet, ACL tear and breast lesion | "We do not find evidence confirming RadImageNet pre-training" is superior. Best ACL AUC 0.9969 (ResNet50, 1D-conv classifier, partial unfreezing) — [arXiv 2508.17567](https://arxiv.org/abs/2508.17567) | Dataset not named on the abstract page |
| OrthoFoundation 2026 | DINOv3-L + knee SSL vs "supervised baselines" | Matches supervised baselines with 50% of labels; ACL AUROC 94.39 at 12.5% labels vs 98.49 with full labels — [arXiv](https://arxiv.org/abs/2601.18250); [README](https://github.com/ytrsk/OrthoFoundation) | No ImageNet numbers available to us |
| OrthoDiffusion 2026 | Knee diffusion SSL vs supervised 3D CNNs from scratch | Macro-AUC 90.79 vs 83.15 (3D ResNet-18), **+7.6**; also stronger at 1.5T (91.89 vs 84.43) and 3T (90.53 vs 82.92) — [arXiv 2602.20752](https://arxiv.org/html/2602.20752v1) | Baselines are from scratch; private data |
| PRISM / MARS 2025-2026 | MRI SSL vs Swin-B from scratch | MRNet abnormality accuracy 80.05% vs ~77.78% (**+2.3**). OAI KL 64.20% vs 63.20% (+1.0) and vs MAE-3D 64.08% (+0.1) — [arXiv 2508.07165](https://arxiv.org/html/2508.07165v2) | No ImageNet baseline |
| KneePreM 2026 | OAI MAE vs random init and SuPreM | fastMRI+ AUC 0.722 with 50% of data beats full-data baselines; baseline values not given — [arXiv 2609.31461](https://arxiv.org/abs/2609.31461) | No ImageNet baseline; no weights |
| MedNet-FS (Diagnostics 2025) [S] | Knee-MRI-specific vs generic or other medical weights, few-shot | Knee-specific "significantly outperformed"; ACL AUC 0.76 on MRNet with 40 support samples — [MDPI](https://www.mdpi.com/2075-4418/15/14/1808/htm) (page returned 403; claim via search summary only) | Low-label only; UNVERIFIED |
| DINOv3 medical benchmark 2025 | Frozen DINOv3 vs supervised | Lags nnU-Net by about 10 Dice on MSD; no MRI classification — [arXiv 2509.06467](https://arxiv.org/html/2509.06467v2) | Frozen features only |
| Kaggle RSNA 2024 lumbar MRI, top 4 | Practice | All used ImageNet-family timm backbones (EfficientNet, ConvNeXt, HGNetV2, MaxViT, CAFormer, SwinV2, ResNet) — [2nd](https://github.com/brendanartley/RSNA-2024-Competition); [3rd](https://github.com/Moyasii/Kaggle-2024-RSNA-Pub); [4th](https://github.com/tattaka/rsna-2024-lumbar-spine-degenerative-classification-public) | Lumbar spine, not knee |
| Azcona 2020 (8 Oct notes) | ImageNet ResNet18 vs scratch, MRNet | +0.036 average AUC from ImageNet init — [arXiv 2010.01947](https://arxiv.org/pdf/2010.01947) | Shows that pretraining matters, not which kind |

### Inferences
- The consistent pattern is that domain SSL beats from scratch and beats ImageNet in few-shot regimes. Beating ImageNet-21k/12k at full fine-tune is unproven.
- With 4,349 labelled studies and report-derived labels, the realistic single-model gain from swapping CoAtNet-IN-12k for a domain encoder is likely ≤ 0.005 macro-AUC. This is an estimate, and gold-58 cannot resolve it (SE about ±0.02).
- An OrthoFoundation experiment is still worth one run. If it helps, it should help most on the subtle findings: its cartilage-lesion AUROC of 93.38 is its closest analogue to our lateral and PF OA. Measure it on the out-of-fold set for the four weak findings, not only on gold-58.

### Gaps
- No 2023-2026 study was found that compares knee-MRI domain pretraining against fine-tuned ImageNet-21k/12k 2D backbones at a few-thousand-study scale.
- OrthoFoundation's baseline tables were not accessible.

---

## Q6. Which 2-3 candidates should we try within 6 days, and exactly how do they plug into the pipeline?

### Takeaway
Ranked:
1. **Replace the intensity-heuristic knee anchor with femur-based anchors** from TotalSegmentator `total_mr` (Apache-2.0), distilled into a tiny in-kernel localiser. This targets lateral OA, PF OA, lateral meniscus and synovitis through better-placed close-ups.
2. **Add a licence-clean second backbone family**, ConvNeXt-B CLIP-LAION-2B→IN-12k at 384 (Apache-2.0, about 0.57× CoAtNet cost), trained with the identical 2.5D MIL recipe for ensemble diversity.
3. **Experiment only:** OrthoFoundation-L as a 224-px close-up encoder, to measure the knee-domain gain. It is OAI- and fastMRI-derived, so keep it out of the clean final unless cleared.

### Cited Findings
- TotalSegmentator anchors:
  - `total_mr` includes `femur_left` and `femur_right` and is "Openly available for any usage (Apache-2.0 license)" — [README](https://github.com/wasserth/TotalSegmentator).
  - Femur Dice on clinical knee MRI: 0.897 — [arXiv 2508.06076](https://arxiv.org/html/2508.06076).
  - `--statistics_extra` outputs per-class `centroid_vox` and `bbox_vox`. `--fast` uses a 3 mm model. `--roi_subset` predicts only the listed classes; `--robust_crop` avoids cut-off crops — [README](https://github.com/wasserth/TotalSegmentator).
- The localise-then-crop design is used by the top RSNA 2024 lumbar MRI solutions. The 2nd place's 2D backbone + attention-head localiser reached ~99.1% within 5 px — [README](https://github.com/brendanartley/RSNA-2024-Competition).
- Our 8 Oct notes found ROI gains largest on the weakest subregions: Tack 2021 medial meniscus anterior horn AUC 0.74 → 0.94 with a box loss — [PMC8675251](https://europepmc.org/article/PMC/PMC8675251).
- ConvNeXt-B CLIP-LAION-2B→IN-12k-384: Apache-2.0, 88.6M params, 716 img/s vs CoAtNet-2-384 at 408 img/s on the same GPU, IN-1k 87.14 — [HF](https://huggingface.co/timm/convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384); [bench](https://github.com/huggingface/pytorch-image-models/blob/main/results/benchmark-infer-amp-nchw-pt291-cu128-4090.csv); [results](https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv).
- OrthoFoundation-L:
  - 1.21 GB ViT-L checkpoint, public via GitHub LFS, repo MIT. Pretraining includes OAI and fastMRI — [README](https://github.com/ytrsk/OrthoFoundation).
  - ViT-L/16 at 224: 954 img/s, 2.34× CoAtNet-2-384 — [bench](https://github.com/huggingface/pytorch-image-models/blob/main/results/benchmark-infer-amp-nchw-pt291-cu128-4090.csv).

### Inferences (6-day plan; all effort and gain figures are my estimates)

**1. Femur-based anchor and localiser (highest expected value for the weak findings; low licence risk)**
- **Day 1.** Convert one non-fat-suppressed sagittal, coronal and axial series per study to NIfTI, keeping the DICOM affine (dcm2niix or SimpleITK). On an A100, run:
  `TotalSegmentator -i s.nii.gz -o out --task total_mr --roi_subset femur_left femur_right --fast --ml --statistics_extra` (add `--robust_crop` if masks look cut off).
  QA 50 studies by eye. Also measure the distance between the new anchor and the current intensity-heuristic anchor, and flag any above 15 mm.
- **Day 2.** Run on all 4,349 training studies; estimated a few GPU-hours. Derive anchors in patient coordinates:
  - inferior condyle line (joint line)
  - intercondylar notch (medial/lateral split)
  - trochlear groove (PF centre)
  - posterior condyle line
  - suprapatellar point

  Train a small localiser (`convnext_nano` or `tf_efficientnetv2_s` at 224-256, Apache-2.0) to regress these points per series from the existing slices. It is roughly 15× cheaper per image than CoAtNet-2-384, so inference cost is negligible.
- **Days 3-4.** Re-generate the 100 mm close-ups from the new anchors:
  - patella / PF on axial and sagittal
  - posterior
  - medial and lateral compartments split at the notch on coronal
  - a suprapatellar/Hoffa crop for synovitis and effusion

  Retrain the close-up model and compare out-of-fold AUC per finding for lateral OA, PF OA, lateral meniscus and synovitis against the heuristic-anchor version.
- **Clean-final status:** Apache-2.0 tools only. Do not use `appendicular_bones_mr`: it needs a key and is non-commercial.

**2. Second backbone, ConvNeXt-B CLIP→IN-12k-384 (Apache-2.0)**
- Code: `timm.create_model('convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384', pretrained=True, num_classes=0, global_pool='')` feeds the same attention-MIL head with the same 3-slice 2.5D input.
- Use the model's own `pretrained_cfg` mean and std (CLIP statistics, not ImageNet).
- Train 1-2 seeds on rented H100s in parallel with item 1, and blend at rank level.
- If the DINOv3 licence is acceptable for the final, `convnext_base.dinov3_lvd1689m` is a same-cost drop-in to try.
- EVA02-B (MIT) is the stronger-on-ImageNet alternative, at about 1.3× CoAtNet cost at 448.

**3. OrthoFoundation-L probe (experiment only; conditional licence)**
- Load into a ViT-L/16 (DINOv3-style) backbone. Its `nets/` code defines the architecture, and the key mapping to timm's `vit_large_patch16_dinov3` is UNVERIFIED.
- Fine-tune on the 100 mm close-ups at 224 px only, where it runs at 2.34× CoAtNet-384 throughput.
- If it beats the item-1 close-up model by ≥ 0.01 out-of-fold AUC on the weak findings, raise "OAI- and fastMRI-derived pretrained weights" with the host before using it. Otherwise drop it.

**Not worth the 6 days:**
- PRISM/MARS and OrthoDiffusion: 3D, separate branch, only from-scratch baselines.
- Triad (no knee), MedDINOv3 (CT), RAD-DINO (chest X-ray).
- BiomedCLIP, MedImageInsight, MedSigLIP and Curia: no knee evidence and/or restrictive licences.
- MRI-CORE: needs 1024 px.
- kneeseg random forests: CPU-slow.
- `aagatti/nnunet_knee`: OAI-derived.
- All external datasets (Q4).

### Gaps
- Real T4 timings for the ConvNeXt-B second model and the localiser must be measured in a Kaggle dry run. The 4090 ratios are only a guide.
- `total_mr` femur robustness on fat-suppressed, thick-slice knee series is untested. This is the main risk for item 1, which is why the plan prefers non-FS series and QA first.
- Whether OrthoFoundation weights load cleanly into timm's DINOv3 ViT-L, and whether the DINOv3 License attaches to them, were not verified.
