# Image-grounded refinement of noisy report-derived labels (RSNA Knee 2026): local VLM annotators, multi-view co-training, self-training without confirmation bias, clean-anchored relabelling

Round-2 research note, written 2026-10-16. It covers methods that use the MRI images themselves to refine labels that were averaged from 31 LLM report-extraction tables. It does not repeat what the sibling notes already hold:
- `noisy_labels_auc_ensembling.md`: VisualCheXbert background, ELR/NVUM/DivideMix on chest X-ray (CXR) report noise, Gold Loss Correction (GLC), Dawid–Skene/Snorkel over the 31 tables, and the theory that class-conditional noise cannot change asymptotic AUC.
- `prior_kaggle_winners_tricks.md`: HMS vote-staged training and the RSNA aneurysm out-of-fold (OOF) reversals.

**Evidence grades**
- **A**: peer-reviewed, with a controlled ablation of the exact mechanism, at scale.
- **B**: peer-reviewed or strong preprint; the mechanism is relevant but the domain or noise type differs.
- **C**: a single small study, a preprint without ablation, or a Kaggle write-up.
- **D**: forum anecdote, secondary summary, or numbers I could not check.

**Tags**
- [PR] peer-reviewed; [PP] preprint; [K] Kaggle write-up or forum.
- [team brief]: from the assignment text or our repo.
- [computed]: my arithmetic. [estimate]: my estimate, not benchmarked.
- [via sibling note]: taken from a round-2 sibling note and not re-fetched.
- **[RAW-TEXT]**: needs the raw report text. Team rule: we only use the 31 extracted tables, so this is not allowed.
- **[API]**: needs an external API. Forbidden.
- **[LICENCE]**: licence caveat.

---

## Q1. Local vision-language models (VLMs) as image-aware annotators for knee MRI

### Takeaway
Published zero-shot evidence says local VLMs are weak knee-MRI graders.
- MedGemma 1.5 4B, the best open model in Google's own benchmark, scores 64.7% balanced accuracy averaged over brain, knee and abdominal MRI conditions. Qwen3-VL-4B is at chance (49.6%).
- On knee MOAKS grading, Qwen2.5-VL-7B scores 0.47 zero-shot. Fine-tuned 3B-class models reach about 0.60.
- No local model has a published per-finding AUC for any of our 12 findings.

So a local VLM has two safe uses:
- **(a)** One extra input, validated on gold, for uncertain or silent cells of findings that show as bright fluid: effusion, synovitis, Baker's cyst and contusion. It should never replace a label outright.
- **(b)** A teacher fine-tuned with LoRA (low-rank adaptation), whose medical pretraining makes its errors differ from ours.

A 4B model can label all 4.3k studies in about 1 h (16 key slices per study) to about 5 h (85-slice volumes) on one A100 [estimate].

### Cited Findings

#### Published knee-MRI results (all are image-only unless stated)
- **MedGemma 1.5 4B** [PP, grade B for the numbers, C for knee relevance]:
  - Input: 3D CT/MR volumes become sequences of 2D slices. Each slice is rescaled to 896×896, with at most 85 slices per query (21,760 vision tokens, i.e. 256 per slice). Larger volumes are sampled equidistantly along z.
  - MRI preprocessing: each volume is min-max normalised and the grey value is copied into R, G and B (no windowing).
  - The benchmark is "MRI Dataset 1": brain (acute infarct, haemorrhage, multiple sclerosis), **knee (meniscal tears, fractures)** and abdomen (liver disease, pancreatic lesions). Positives and negatives were sub-sampled to balance. ACL tear and effusion are not evaluated.
  - Accuracy: MedGemma 1.5 4B **64.7**; MedGemma 1 4B 51.3; MedGemma 1 27B 57.4; **Qwen3-VL-4B 49.6**; Gemma 3 4B 51.1; Gemma 3 27B 50.5; Gemini 3.0 Flash 60.3; Gemini 3.0 Pro 55.5.
  - MRI Dataset 1 was also used in MedGemma's pretraining, distillation and RL, so the comparison favours MedGemma.
  - The abstract says MRI improved "11%", while the table gap is 13.4 points.
  - There is no per-body-part breakdown and no AUC.
  - Sources: [MedGemma 1.5 technical report, arXiv 2604.05081 (HTML v2)](https://arxiv.org/html/2604.05081v2); [abstract](https://arxiv.org/abs/2604.05081); [Google Research blog](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/)
- **3DReasonKnee** [PP, grade B]:
  - Data: 7,970 OAI DESS knee volumes (160 slices each) and 494k question/box/reasoning/severity quintuples on MOAKS. The MOAKS items cover bone marrow lesions, cartilage, osteophytes, meniscal damage, ligaments/tendons, Hoffa-synovitis, effusion-synovitis and periarticular features, graded 0–3.
  - Grading accuracy, zero-shot: Qwen2.5-VL-3B 0.158; Qwen2.5-VL-7B 0.470; o1 0.477.
  - With an instruction schema: 0.146 / 0.504 / 0.568. Qwen2.5-VL-7B given the ground-truth region: 0.556.
  - After supervised fine-tuning (SFT): Qwen2.5-VL-3B 0.613 (0.588 with chain-of-thought); Med3DVLM 0.596.
  - Authors: "even state-of-the-art models struggle with MOAKS grading in zero-shot settings". Meniscus and osteophyte reasoning "remains weak even with the correct region provided". Med3DVLM often produced invalid outputs zero-shot.
  - The abstract says five VLMs were tested; the setup section names four.
  - Source: [3DReasonKnee, arXiv 2510.20967](https://arxiv.org/html/2510.20967v1)
- **GPT-5, zero-shot meniscal tear on MRNet** **[API]** [PR, small pilot, grade C]:
  - Design: 100 exams (50/50) from the MRNet validation set. Sagittal T2 and coronal T1 series (the paper is internally inconsistent about which series), sent as 256×256 PNGs (20–44 slices per series). Prompt asked for "yes"/"no" only.
  - Results: accuracy 0.76 (95% CI 0.668–0.833), sensitivity 0.84, specificity 0.68, F1 0.778. No AUC.
  - Limitations the authors list: the binary prompt allowed no uncertainty, and training-data overlap with MRNet cannot be excluded.
  - It shows what a frontier model can do. It is not usable for us (external API).
  - Source: [PMC12811926](https://pmc.ncbi.nlm.nih.gov/articles/PMC12811926/)
- **MeniOmni** (ICME 2026) [PP/conference, grade D for the numbers]:
  - Data: 746 multi-centre studies, tri-planar input, top-16 slices per view, Stoller severity grading.
  - Zero-shot: Qwen2-VL-72B 55.4% accuracy (quadratic weighted kappa, QWK, 0.492; severe-error rate 17.8%); InternVL2 51.2%; Video-LLaVA 38.5%; GPT-4o 62.5% (58.4% without clinical priors).
  - Fine-tuned Video Swin-B: 63.8% (QWK 0.605).
  - Not evaluated: MedGemma, Qwen2.5/3-VL, Lingshu, Hulu-Med.
  - The specialist rows rise suspiciously smoothly; treat the numbers as unverified.
  - Source: [MeniOmni, arXiv 2605.28161](https://arxiv.org/html/2605.28161)
- **CLIP-style zero-shot** (contrastive, not generative) [PP, grade D]:
  - Knee "ACL" and "Meniscus" sets, accuracy %: UniMed-CLIP 85.82 / 85.27; MedMoE 77.92 / 74.87; BiomedCLIP 47.89 / 40.09; PMC-CLIP 67.09 / 65.03.
  - The datasets are not named and class balance is not given. UniMed-CLIP's training data may include them.
  - Source: [MedMoE, arXiv 2506.08356, Table 1](https://arxiv.org/html/2506.08356)
- **Competition forum** [K, grade D] [via sibling note]:
  - Kirderf's vision-LLM benchmark (images rendered as a mosaic, scored on gold) had incomplete runs. The author asked readers to disregard the early numbers.
  - hengck23 suggested SGMRI-VQA and Hulu-Med-14B (which takes 3D NIfTI) for VLM labelling.
  - Source: [discussion 747658](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747658)
- **Round-1 hint** [team brief, grade D] **[RAW-TEXT]**:
  - An LLM given the report plus the MRI images raised gold-58 label AUC for effusion from 0.835 to ~0.92 and for synovitis from 0.80 to ~0.85.
  - It needs the raw report and, very likely, a frontier API model.
- **Team research (ChatGPT/Fable)** [team brief, chat.md]: "VLM labelling not viable (zero-shot ~50% on knee MOAKS)". This matches the 3DReasonKnee zero-shot numbers above.

#### Candidate local models: size, MRI handling, licence
- **MedGemma** [grade B]:
  - MedGemma 1.5 is listed as a 4B multimodal model. The 27B text-only and 27B multimodal variants belong to MedGemma 1.
  - Use is governed by the Health AI Developer Foundations (HAI-DEF) terms, and the Hugging Face download is gated.
  - Google: "not yet clinical-grade… will likely need further fine-tuning".
  - Sources: [MedGemma collection page](https://developers.google.com/health-ai-developer-foundations/medgemma); [MedGemma 1 model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card-v1)
- **HAI-DEF terms** **[LICENCE]**, as read from the page (not legal advice):
  - "Google will not claim ownership over any original Outputs you generate using HAI-DEF."
  - A "Model Derivative" includes models trained with "methods based on the generation of synthetic data Outputs by HAI-DEF for training that model".
  - Distribution of HAI-DEF or Model Derivatives must carry the Section 3.2 use restrictions as an enforceable provision, plus a copy of the Agreement and notices.
  - Section 3.2 bans uses that could make Google a "manufacturer" of a medical device, and uses in the Prohibited Use Policy.
  - Source: [HAI-DEF terms](https://developers.google.com/health-ai-developer-foundations/terms)
- **Qwen3-VL** [grade D, secondary sources]:
  - Dense 2B/4B/8B/32B models and mixture-of-experts 30B-A3B and 235B-A22B; Apache-2.0 weights per secondary sources; 256K context.
  - The 235B weights are about 471 GB.
  - Sources: [HF collection](https://huggingface.co/collections/Qwen/qwen3-vl); [Simon Willison](https://simonwillison.net/2025/Sep/23/qwen3-vl); [Kaitchup](https://kaitchup.substack.com/p/qwen3-vl-fine-tuning-on-your-computer)
  - Qwen3-VL-4B scored 49.6% on MedGemma's MRI benchmark (above).
- **Hulu-Med** (Zhejiang University, Oct 2025) [PP, grade C]:
  - Sizes: 7B/14B/32B.
  - Training data: 16.7M public or synthetic samples across 14 modalities, including MRI.
  - 3D input: volumes are sliced into a multi-image sequence, with token reduction of up to 55% for 3D/video.
  - Authors claim it beats open models on 27 of 30 benchmarks. No knee-MRI evaluation was found.
  - Code and weights are public. The licence was not checked.
  - Sources: [arXiv 2510.08668](https://arxiv.org/pdf/2510.08668); [GitHub](https://github.com/ZJUI-AI4H/Hulu-Med)
- **Lingshu** [PP, grade C]: evaluated on general medical VQA (MMMU-Med, VQA-RAD, SLAKE, PathVQA, PMC-VQA, OmniMedVQA, MedXpertQA). No knee-specific evaluation was found. — [arXiv 2506.07044](https://arxiv.org/pdf/2506.07044)
- **Decipher-MR** [PP, grade C]:
  - A 3D MRI vision-language foundation model trained on more than 200,000 MRI series from over 22,000 studies, spanning brain, spine, **knee** and other regions, with reports for most studies.
  - Supports zero-shot cross-modal retrieval.
  - Weight release was not verified.
  - Source: [arXiv 2509.21249](https://arxiv.org/html/2509.21249v1)
- **VLMs as noisy-label detectors** (natural images) [PR, grade B for mechanism, D for the number]:
  - DeFT learns a positive and a negative text prompt per class. The negative prompt acts as a learnable threshold that "sieves out noisy labels"; the visual encoder gets parameter-efficient fine-tuning.
  - A secondary summary reports CIFAR-100 with 40% instance-dependent noise: 87.58% vs 75.64% for plain cross-entropy.
  - Sources: [NeurIPS 2024 abstract](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6af08ba9468f0daca4b8dd388cb95824-Abstract.html); [summary](https://liner.com/en/review/visionlanguage-models-are-strong-noisy-label-detectors)

#### Compute
- I found no published images-per-second benchmark for a VLM in vLLM offline batch mode on an A100.
- Forum figures (for example ~764 prompt tokens/s for Qwen2.5-VL-7B) come from online serving and are not comparable. The advice there: tune max-batched-tokens, cap `max_pixels`, and limit images per prompt. — [vLLM forum](https://discuss.vllm.ai/t/speeding-up-vllm-inference-for-qwen2-5-vl/615)
- **[computed / estimate]** MedGemma 1.5 4B, 16 key slices per study (256 tokens each, per the report):
  - Visual tokens: 4,349 × 16 × 256 ≈ 17.8M.
  - Decoder prefill ≈ 2 × 4e9 × 17.8e6 ≈ 1.4e17 FLOPs, about 16–24 min at 100–150 effective TFLOP/s on an A100.
  - The vision encoder at 896 px (~4,096 patches per slice), assuming a ~0.4B SigLIP-class encoder (my assumption, not verified here) ≈ 2.3e17 FLOPs, about 25–40 min.
  - Decoding ~100 tokens per study is minor when batched.
  - Total ≈ 1 h on an A100, roughly 0.4 h on an H100.
- **[estimate]** At 85 slices per study (≈95M tokens, plus attention at ~22k context): about 4–6 h on an A100 for the 4B model.
  - A 27B model is about 6–7× the decoder cost, so roughly 12–15 h on one A100, or 2–4 GPUs.
- Labels are generated offline, so none of this counts against the 9-hour Kaggle inference limit.

### Inferences
- **What the local-VLM prior is worth.** Image-only and zero-shot, a 4B local model starts near MedGemma-1.5 level (~65% balanced accuracy, in-distribution for MedGemma). That is far below the ~0.89–0.90 AUC our report-mean labels already reach on gold. The round-1 effusion jump (0.835→0.92) almost certainly relied on the report text and a frontier model, so it is **[RAW-TEXT]** and **[API]** for us.
- **Where a VLM is most likely to add signal.** Findings shown as large bright-fluid signal on fluid-sensitive sagittal/axial images, where reports omit or undergrade:
  - effusion;
  - Baker's cyst;
  - synovitis (via effusion-synovitis);
  - bone contusion.

  Tears (ACL, menisci), fracture lines and compartment OA are fine-grained. 3DReasonKnee says meniscus reasoning stays weak even when the region is given, and the GPT-5 pilot had 68% specificity on tears.
- **Adjudication protocol** that respects the team rules (my design):
  1. **Candidate cells.** Only cells where the 31-table mean is in [0.3, 0.7] (the 12% our consensus mask drops, per commit 6e26c39), plus "not addressed" cells, for the fluid-visible findings above.
  2. **Prompt.**
     - Give the finding definition at the gold severity threshold (for example "moderate or larger effusion distending the suprapatellar recess").
     - Give the plane and sequence name.
     - Give ~16 slices from the sagittal fluid-sensitive series, plus axial for Baker's cyst.
     - Optionally add the table consensus as a number ("report-derived probability 0.45"). This uses only the extracted tables, not raw text. Test the prompt with and without this prior, because it may anchor the model.
  3. **Score.** Read P("yes") from the token log-probabilities rather than a hard yes/no; the GPT-5 pilot names the binary prompt as a limitation. Average 3–5 prompt or slice-subset variants and record their variance.
  4. **Gate on gold first.** Score all 58 gold studies. Keep the VLM for a finding only if, as a *label*, it beats the report mean on gold by more than the paired-bootstrap noise (see Q4).
  5. **Merge.** Use the VLM score only as one input to the clean-anchored gate (Q4), never as a direct label override. Leave confident report cells untouched.
- **Fine-tuning route.** Fine-tune MedGemma 1.5 4B or Qwen3-VL-8B with LoRA on confident cells, then predict the uncertain cells.
  - This turns the VLM into another image model trained on our labels, but with a different, medical pretraining prior. That is the "different teacher" property the failed OOF mix lacked (Q3).
  - 3DReasonKnee shows SFT roughly quadruples zero-shot grading for a 3B model (0.158→0.613). MeniOmni suggests fine-tuned specialists still edge out zero-shot MLLMs.
- **Licence [LICENCE].**
  - MedGemma outputs are ours.
  - A student trained on MedGemma-generated labels plausibly becomes a HAI-DEF "Model Derivative". Releasing it under an open licence for a Kaggle prize may then conflict with the Section 3.2 pass-through requirement.
  - Qwen3-VL (Apache-2.0, per secondary sources) avoids this. Check the competition's external-model and winner-licence rules before using MedGemma for training labels.

### Gaps
- I found no published zero-shot or few-shot AUC for any local VLM on ACL, MCL, either meniscus, compartment OA, effusion, synovitis, Baker's cyst, contusion or fracture as separate findings. MedGemma 1.5 reports only an average across body parts.
- I found no knee-MRI evaluations for RadFM, M3D-LaMed, BiomedGPT, LLaVA-Med or HuatuoGPT-Vision.
- SGMRI-VQA, suggested on the forum, could not be identified.
- It is unclear whether MedGemma 1.5's "axial slicing" means the acquisition plane for knee volumes. Sagittal or coronal input may be off-distribution.
- VLM throughput is not benchmarked; the estimates above are FLOP arithmetic.
- The licences of Lingshu, Hulu-Med and Decipher-MR were not verified, and neither was the Qwen3-VL licence on the official model cards.
- I found no evidence on in-context few-shot (example studies in the prompt) for MRI.

---

## Q2. Multi-view co-training and cross-view pseudo-labelling

### Takeaway
Co-training's guarantee needs views that are each sufficient and conditionally independent given the label. Deep versions build views with different networks, adversarial view-difference or 3D plane permutation, and report real semi-supervised gains:
- CIFAR-10 with 4k labels: 9.03% error with 2 views vs 12.31% for Mean Teacher.
- Multi-organ CT: more than 4% over fully supervised.

None of this has been tested as relabelling of report labels in knee MRI. Here, plane-disjoint teachers decorrelate *model* error, which the failed OOF mix lacked. They still share the report's *systematic* biases, so co-training should fill masked or silent cells, not overrule confident report cells. Avoid co-teaching-style small-loss selection: it discards hard positives.

### Cited Findings
- **Co-training assumptions** (Blum & Mitchell 1998) [PR, grade A for theory]: each instance has two views, each view is sufficient to learn the task, and the views are conditionally independent given the label. Under these conditions co-training has PAC-style guarantees. — [restated in Deep Co-Training, arXiv 1803.05984](https://arxiv.org/abs/1803.05984); [and in UMCT](https://arxiv.org/abs/1811.12506)
- **Deep Co-Training (Qiao et al., ECCV 2018)** [PR, grade B]:
  - Recipe: two or more networks on the same images. A "view difference" constraint uses adversarial examples so the networks stay different, and a cross-view consistency loss runs on unlabelled data.
  - Error rates: CIFAR-10 (4,000 labels) 9.03 ± 0.18 (2 views), 8.54 (4), 8.35 (8). SVHN (1,000 labels) 3.61, 3.38, 3.29.
  - Baselines: Mean Teacher 12.31 / 3.95; VAT 10.55 / 3.86.
  - Source: [arXiv 1803.05984](https://arxiv.org/abs/1803.05984)
- **Deep Multi-Planar Co-Training (DMPCT; Zhou et al., WACV 2019)** [PR, grade B]:
  - Recipe: a teacher is trained on labelled CT. Its axial, coronal and sagittal predictions are fused into pseudo-labels for unlabelled volumes; a student trains on both; the loop repeats.
  - Data: 210 labelled + 100 unlabelled CTs, 16 structures. Result: more than 4% better than fully supervised, especially with few annotations.
  - Authors: an error in the initial pseudo-label "will be reinforced by the network during the following iterations". That is why they fuse across planes.
  - Source: [arXiv 1804.02586](https://arxiv.org/abs/1804.02586)
- **Uncertainty-aware Multi-view Co-Training (UMCT; Xia et al., WACV 2020 / MedIA)** [PR, grade B]:
  - Views are made by rotating or permuting the 3D axes. Each view's pseudo-label is a fusion of the other views' predictions, weighted by Bayesian (MC-dropout) uncertainty.
  - Evaluated on NIH pancreas and LiTS liver tumour segmentation, and extended to domain adaptation.
  - Sources: [arXiv 1811.12506](https://arxiv.org/abs/1811.12506); [MedIA version, arXiv 2006.16806](https://arxiv.org/abs/2006.16806)
- **Co-teaching+ vs JoCoR** [PR, grade B]:
  - Co-teaching (two nets swap small-loss samples) tends to make the two nets converge.
  - Co-teaching+ keeps "prediction disagreement data only" and runs small-loss exchange within that set ("update by disagreement").
  - JoCoR argues the opposite: train both nets on a joint loss with co-regularisation (agreement), then select small-loss samples by that joint loss.
  - Sources: [Co-teaching+, arXiv 1901.04215](https://arxiv.org/abs/1901.04215); [JoCoR, arXiv 2003.02752](https://arxiv.org/abs/2003.02752)
- **Small-loss selection on real report noise** [via sibling note `noisy_labels_auc_ensembling.md`]: on real CXR report noise, DivideMix fell 12.9 AUC points; co-teaching/JoCoR-style selection drops subtle positives.
- **Co-Correcting (Liu et al., IEEE TMI 2021)** [PR, grade C for us]:
  - Recipe: dual-network mutual learning, label-probability estimation and curriculum-based label correction.
  - Tested only on ISIC-Archive and PatchCamelyon with *synthetic* noise of 5–40%, against six noisy-label methods.
  - Source: [arXiv 2109.05159](https://arxiv.org/abs/2109.05159)
- **Mammography** [PR/PP, grade C]:
  - No co-training paper on CC/MLO views was found.
  - The closest multi-view work: BRAIxMVCCL (CC/MLO global consistency and local co-occurrence) and BRAIxDet (multi-view classifier, then a student–teacher detector on mixed full and weak annotations).
  - Sources: [BRAIxMVCCL](https://ar5iv.labs.arxiv.org/html/2209.10478); [BRAIxDet](https://arxiv.org/html/2301.13418v4)
- **Report–image co-training (patent)** [grade D]: an image classifier and a report classifier are trained together. Images whose own pseudo-label is low-confidence, but whose report pseudo-label agrees with high confidence, are added to training. No public numbers. **[RAW-TEXT]** in its original form. — [USPTO 12512211](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12512211)
- **Why diverse models carry "dark knowledge" (Allen-Zhu & Li, ICLR 2023)** [PR, grade B, theory]:
  - With "multi-view" data, networks that differ only in random seed learn different features. Their ensemble provably improves accuracy, and the gain can be distilled into one model trained to match the ensemble.
  - Self-distillation works as implicit ensemble plus distillation.
  - Source: [arXiv 2012.09816](https://arxiv.org/abs/2012.09816)
- **Competition evidence for lineage and view diversity** [K] [via sibling note `forum_social_github.md`]:
  - The only GitHub-documented gains above the public 0.94x stack come from "label lineage diversity" and "using every series" (darcywjoseph).
  - heliosli's R6 knee-anchored compartment crops carry 80% of the blend weight for lateral OA.
  - Source: [darcywjoseph SOLUTION.md](https://github.com/darcywjoseph/Knee-Abnormality-Detection/blob/HEAD/SOLUTION.md)

### Inferences
- **Two kinds of error.** Plane-disjoint teachers (sagittal-only, coronal-only, axial-only) fix *variance-type* error. Each sees different pixels, so its idiosyncratic mistakes correlate less with the student's.
  - That is the property our 0.6/0.4 OOF mix lacked: its teacher was the same architecture on the same series with the same labels.
  - They do **not** fix *bias-type* error, because all views were trained on the same report labels. Reports omit effusion next to an ACL tear, and they call mild disease positive; every view teacher learns both. Only a teacher anchored to the gold threshold (Q4), an external severity-graded source, or a VLM prompted with severity definitions (Q1) attacks bias.
- **Plane sufficiency is per finding** (radiology convention; my inference, not verified here):
  - ACL: sagittal. MCL: coronal.
  - Menisci: sagittal + coronal.
  - Medial/lateral OA: coronal + sagittal. Patellofemoral (PF) OA: axial + sagittal.
  - Effusion and synovitis: sagittal + axial fluid-sensitive. Baker's cyst: axial + sagittal.
  - Contusion and fracture: any fluid-sensitive plane.

  So "use the other two views" only works for findings that at least two planes can see. Measure each view teacher's gold AUC per finding before trusting its fills.
- **Cross-view pseudo-labelling for our setup** (UMCT-style):
  - Fill a masked or silent cell with the uncertainty-weighted mean of the *sufficient* view teachers' OOF probabilities.
  - Weight each view by 1/variance across test-time augmentation (TTA) or MC-dropout passes.
  - Then train the full multi-plane student. The student differs from every teacher (all planes, different backbone), which satisfies the "student ≠ teacher" condition from Q3.
- **Agreement vs disagreement.** JoCoR versus Co-teaching+ matters less here, because we are not selecting samples by loss. For *relabelling*, use agreement: change a confident report cell only when at least two sufficient views agree against it (Q6 design).

### Gaps
- I found no published plane-disjoint co-training or cross-view pseudo-labelling for knee MRI *classification*. DMPCT and UMCT are CT segmentation.
- I have no measurement of how correlated our plane-specific models' errors actually are. It must be measured on our OOF.
- No mammography co-training paper with numbers was found.

---

## Q3. Self-training that transfers vs fails, and why our OOF relabelling failed

### Takeaway
A student beats its teacher only when it gets something the teacher lacked:
- more (unlabelled) data, plus input noise and capacity (Noisy Student);
- feedback from clean labels (Meta Pseudo Labels);
- a separate head that keeps pseudo-label errors out of the classifier (Debiased Self-Training);
- a stronger, differently trained annotator (ReLabel).

Remove these and the student copies the teacher. Noisy Student without noise scored 83.2% against 83.3% for the supervised baseline. ReLabel with a weak annotator lost 1.4 points.

Our three failures fit that pattern:
- same architecture, same inputs, same noisy labels;
- a high 40% mixing weight;
- judgement on 58 gold studies whose macro-AUC standard error (≈0.013–0.027) is as large as the effects measured.

### Cited Findings
- **Noisy Student (Xie et al., CVPR 2020)** [PR, grade A]:
  - **Recipe:**
    - An unnoised EfficientNet teacher labels 300M JFT images with soft labels.
    - Filter to confidence > 0.3 and balance to at most 130k images per class.
    - The student is equal-or-larger and noised with RandAugment, dropout and stochastic depth.
    - Train jointly on labelled + pseudo-labelled data with a 14:1 then 28:1 unlabelled:labelled batch ratio; iterate.
  - **Results:** ImageNet top-1 88.4%; iterations 87.6 → 88.1 → 88.4. ImageNet-A 61.0→83.7; ImageNet-C mean corruption error 45.7→28.3.
  - **Noise ablation** (EfficientNet-B5 teacher, Table 6; first number with 1.3M unlabelled images, second with 130M):
    - supervised B5: 83.3 / 84.0;
    - Noisy Student: 83.9 / 85.1;
    - student without augmentation: 83.6 / 84.6;
    - student without augmentation, stochastic depth or dropout: **83.2 / 84.3**;
    - noised teacher: 83.7 / 84.4.
  - Authors: "when the student is trained to be exactly the same as the teacher model, the cross entropy loss on unlabeled data would be zero and the training signal would vanish… noising the student is needed so that it does not merely learn the teacher's knowledge."
  - **Findings list:**
    - a large unlabelled set is necessary;
    - soft labels are better for out-of-domain data, where hard labels can hurt;
    - a large student matters;
    - joint training beats pretrain-then-finetune;
    - training the student from scratch is sometimes better than initialising it from the teacher.
  - Sources: [arXiv 1911.04252](https://arxiv.org/abs/1911.04252); [PDF](https://arxiv.org/pdf/1911.04252)
- **Meta Pseudo Labels (Pham et al., CVPR 2021)** [PR, grade A]:
  - The teacher is "constantly adapted by the feedback" of the student's performance on the labelled set, so it produces better pseudo-labels over time.
  - ImageNet 90.2% top-1, 1.6 points above the prior state of the art, using JFT-300M unlabelled.
  - Teacher and student are trained together.
  - Source: [arXiv 2003.10580](https://arxiv.org/abs/2003.10580)
- **Debiased Self-Training (DST; Chen et al., NeurIPS 2022)** [PR, grade A for semi-supervised learning]:
  - Bias types: "data bias" and "training bias", the latter from fitting one's own wrong pseudo-labels.
  - **Recipe:**
    - The main head trains only on clean labels and generates the pseudo-labels.
    - A separate nonlinear pseudo head (projection 2048) trains only on pseudo-labels, drives the backbone, and is discarded at inference.
    - A worst-case head fits labelled data while disagreeing maximally with the pseudo-labels; the backbone is trained to stay correct even under it (minimax, λ = 1).
    - Confidence threshold 0.7.
  - **Results:** average FixMatch 75.4 → DST 78.3 (from scratch, 4 labels per class). CIFAR-100 with a pretrained ResNet-50: 53.1 → 70.4. Abstract: +6.3% average over the state of the art.
  - **Cost:** ~7% more training time (104 → 111 h); nothing extra at inference.
  - **Failure modes it documents:**
    - FixMatch pseudo-labels exceed 70% coverage while their accuracy falls below 60%;
    - a "Matthew effect" (the worst 20 classes fall to 1.0%);
    - with pretrained models, declines often do not recover.
  - Source: [arXiv 2202.07136](https://ar5iv.labs.arxiv.org/html/2202.07136)
- **Pseudo-labelling and confirmation bias (Arazo et al., IJCNN 2020)** [PR, grade A]:
  - "Naive pseudo-labeling overfits to incorrect pseudo-labels due to the so-called confirmation bias."
  - Fixes: soft pseudo-labels, mixup, and a minimum of k labelled samples per mini-batch (k = 16 works for CIFAR-10/100).
  - CIFAR-10/100 with 4,000 labels: error 11.40 / 48.54 naive → 7.16 / 41.80 with mixup. With 500 labels, mixup alone still gives 32.10, and adding k cuts it sharply.
  - Sources: [arXiv 1908.02983](https://arxiv.org/abs/1908.02983); [PDF](https://arxiv.org/pdf/1908.02983)
- **Early-Learning Regularisation (ELR; Liu et al., NeurIPS 2020)** [PR, grade B]: networks first fit clean labels in an "early learning" phase, then memorise false ones. ELR builds target probabilities from the model's own temporally ensembled outputs and adds a regulariser toward them, so memorisation is avoided implicitly. — [arXiv 2007.00151](https://arxiv.org/abs/2007.00151). On real CXR report noise it gained ~+1 AUC point [via sibling note].
- **SELF (Nguyen et al., ICLR 2020)** [PR, grade B]: running averages of predictions over the whole training set (self-ensembles) progressively filter out wrong labels. Filtered samples leave the supervised loss and are used as unlabelled data. — [arXiv 1910.01842](https://arxiv.org/abs/1910.01842)
- **Self-distillation theory** [PR, grade B]:
  - Mobahi et al. (NeurIPS 2020): each self-distillation round strengthens regularisation by limiting the basis functions. "A few rounds… may reduce over-fitting, further rounds may lead to under-fitting." — [arXiv 2002.05715](https://arxiv.org/abs/2002.05715)
  - Das & Sanghavi (ICML 2023): student loss = ξ·ℓ(teacher, student) + (1−ξ)·ℓ(labels, student). At high *random* label noise the optimal ξ exceeds 1, and ξ > 1 empirically beats ξ ≤ 1 at 30–50% corrupted labels. They characterise the corruption range where the student beats the teacher (logistic regression). — [arXiv 2301.13304](https://arxiv.org/abs/2301.13304)
- **Bootstrapping (Reed et al. 2014)** [PP, widely used, grade B]: target = β·label + (1−β)·prediction, with β = 0.95 (soft) or 0.8 (hard), i.e. a 5–20% model weight. — [arXiv 1412.6596](https://arxiv.org/abs/1412.6596)
- **Confident learning (Northcutt et al., JAIR 2021)** [PR, grade B]: out-of-sample (cross-validated) predicted probabilities and per-class thresholds estimate the noisy-vs-true joint distribution, then likely label errors are pruned. — [arXiv 1911.00068](https://arxiv.org/abs/1911.00068). In RSNA 2024 lumbar, 2nd place applied it on the last day with no reported gain [via sibling note].
- **Medical self-training** [PP, grade C]:
  - Rajan et al.: Noisy Student with mixup and confidence tempering on CXR matched large-data performance "using 85% lesser labeled data". — [arXiv 2005.02231](https://arxiv.org/abs/2005.02231) [snippet]
  - CheXFusion used Noisy Student with soft pseudo-labels for CheXpert's uncertain and unmentioned labels in the long-tailed CXR task. — [arXiv 2308.03968](https://arxiv.org/abs/2308.03968) [snippet]
- **Kaggle reversals and small gains** [K] [via sibling note `prior_kaggle_winners_tricks.md`]:
  - RSNA aneurysm 11th: OOF self-distillation **+0.0126 CV, −0.0099 public, −0.0088 private**. — [11th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/11th-place-solution)
  - Aneurysm 5th: relabelling negatives with OOF > 0.9 as positive, plus pseudo-labelled external MRA: OOF +0.0007 to +0.008. — [5th](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/5th-place-solution)
  - HMS 3rd: samples with fewer than 8 votes, and pseudo-labels, "gave no benefit". — [HMS 3rd](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492471)
- **Knee forum** [K] [via sibling note `forum_social_github.md`]:
  - A participant's OOF-mix denoising showed a large gold gain, but darcywjoseph found the same kind of loop did not transfer (LB 0.948 with or without it). — [SOLUTION.md](https://github.com/darcywjoseph/Knee-Abnormality-Detection/blob/HEAD/SOLUTION.md)
  - An agreement filter (train only on cells with table mean > 0.9 or < 0.1, dropping ~25%) "improved my results on CV and public LB". — [745861](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745861)
- **Ours** [team brief, chat.md]:
  - 0.6 report / 0.4 own 4-fold OOF: gold 0.93 → 0.918.
  - A participant's pseudo-label loop: +0.026 on gold offline, nothing on the LB.
  - In-sample distillation:
    - teacher = 0.7 Raptor + 0.3 compact, trained on all train data (Raptor also on +2,399 OAI knees), so there is no clean OOF;
    - target = 0.6 teacher + 0.4 label mean;
    - students reached gold 0.928–0.930 against teacher blend 0.931, with rank correlation 0.94–0.96 to the teacher, and added nothing.
- **[computed]** Hanley–McNeil standard error (SE) of one finding's AUC on 58 studies at AUC 0.90:
  - 9 positives: 0.071; 15: 0.056; 20: 0.049; 29: 0.042.
  - Macro over 12 findings with per-finding SE 0.045: 0.013 if findings are independent, 0.027 if their correlation is 0.3.
  - Paired difference between two models (per-finding AUC correlation r = 0.8–0.9): 0.020–0.029 per finding, about 0.006–0.017 at the macro level.
  - Method: [Hanley & McNeil 1982](https://pubs.rsna.org/doi/10.1148/radiology.143.1.7063747)

### Inferences
- **Why the 0.6/0.4 OOF mix could not help** (mechanism):
  - **What the image model learns.** It estimates P(report label | image).
    - If report noise is independent of the image given severity, that estimate is already monotone in true severity, which is the AUC-optimal ranking for the gold labels too (sibling theory note).
    - If the noise depends on the image (omission next to a dominant finding, "mild = positive"), the model *learns* that bias.
  - **What its OOF predictions add.** Either way, OOF predictions from the same family bring no new information about gold. They add (i) label-variance reduction, a weak label-smoothing effect, and (ii) the model's own estimation errors, which is confirmation bias.
  - **Mixing weight.** At 40%, term (ii) dominates. The literature's safe range is 5–20% (Reed bootstrapping).
  - **Weaker annotator.** Four-fold OOF models each saw 75% of the data, so they are weaker annotators than the final model. ReLabel shows a student "follows the performance of the machine annotator" and loses accuracy when the annotator is weak.
  - **Theory does not transfer.** Das & Sanghavi's benefit (ξ ≥ 1) needs *random* noise and a smoother teacher. Our dominant noise is systematic.
- **Why in-sample distillation added nothing.**
  - The teacher's predictions on its own training studies are partly memorised labels, and the student was not noised, larger or given extra data. Noisy Student Table 6 shows exactly this outcome (student ≈ baseline).
  - A student that copies a teacher already in the blend cannot add blend diversity. Rank correlation 0.94–0.96 is the symptom (Allen-Zhu & Li: distillation transfers the teacher's features rather than adding new ones).
- **Why +0.026 on gold did not show on the LB.**
  - With 58 gold studies, a +0.026 macro gain is about 1.5–4 paired SEs, depending on the inter-finding correlation [computed]. That is before correcting for selecting the best loop variant and for possible leakage.
  - Leakage path: OOF teachers for a fold's training studies were trained on that fold's validation studies (11th-place aneurysm pattern), or gold studies were used to tune the loop.
  - Our own −0.012 (0.93 → 0.918) is also within about 1–2 SEs, so "hurt" is not established either. The LB is the better arbiter.
- **Minimal changes that turn a failing loop into one with a chance**, in order of evidence strength:
  1. **New information in the teacher.** It must see different inputs (a plane-disjoint view, Q2), a different pretraining prior (VLM fine-tune, Q1; external MOAKS data), or be anchored by clean labels (Q4). The same model family re-run on the same series does not count.
  2. **Relabel only where reports are weak.** Masked uncertain cells (12%) and silent "not addressed" cells. Teachers should be trained *only* on confident cells, so they do not learn the omission bias (CheXpert U-Ignore → U-SelfTrained; WRENCH "uncovered data").
  3. **Small weights on confident cells.** At most 10–20% teacher weight, and only where at least two independent views agree against the report (bootstrapping range; DMPCT/UMCT fusion).
  4. **A different, noised student.**
     - Equal-or-larger student, with strong augmentation, stochastic depth and dropout (Noisy Student).
     - Train it from scratch rather than from teacher weights.
     - Soft targets; mixup; at least half of each batch from confident cells (Arazo's k).
     - A DST-style separate head for filled cells, dropped at inference.
  5. **One round.** Do not iterate (Mobahi; DMPCT's error-reinforcement warning).
  6. **Honest evaluation.**
     - Gold-58 and the public LB only, or nested cross-fitting (fills for fold k regenerated by teachers that never saw fold k).
     - Judge the student as a *blend member*, not a stand-alone model.
- **Doing nothing.** Mixing 20% of a *different*, stronger teacher into confident cells is low-risk. Mixing 40% of a same-family OOF is the setting the evidence predicts will not help.

### Gaps
- I did not fetch exact numbers for ELR and SELF beyond their mechanisms. The real-report-noise numbers for ELR, NVUM and DivideMix are in the sibling note.
- I found no study of OOF self-relabelling on multi-label medical *ranking* (AUC) with *systematic* report noise, so the diagnosis above is my inference from mechanism.
- The participant's +0.026 loop details (hyperparameter selection on gold or not) are unknown.

---

## Q4. Clean-anchored relabelling with the 58 gold studies

### Takeaway
With 58 gold studies, the only safe clean-anchored relabeller is a tiny model:
- a pooled, penalised logistic "gate" with 2–4 shared slopes (report mean, cross-view image score, optional VLM score, table disagreement) and per-finding intercepts;
- fit by leave-one-study-out;
- used to decide, per finding group, whether image evidence may change labels at all.

Veit et al.'s label-cleaning network shows the idea works when the clean set is 0.4% of the data (40k of 9M). The gains were modest: +0.56 MAP, +3.9 AP over all annotations.

### Cited Findings
- **Label-cleaning network (Veit et al., CVPR 2017)** [PR, grade B]:
  - **Recipe:**
    - The cleaning network takes the noisy label vector (as a bag of words) plus visual features from a shared Inception-V3, concatenated.
    - It is residual, so "it only needs to learn the difference between the noisy and clean labels".
    - It is supervised by human-verified labels.
    - The classifier is trained on cleaned labels for the noisy set and on verified labels where available.
  - **Data:** ~9M Open Images with noisy labels and ~40k verified (≈0.4%).
  - **Results:**
    - baseline: 83.82 AP over all annotations (AP_all), 61.82 mean AP (MAP);
    - fine-tuning on mixed labels: 84.80 / 61.90;
    - fine-tuning on clean only: 85.88 / 61.53;
    - label cleaning, trained jointly: **87.67 / 62.38**.
  - Most effective with 20–80% false-positive annotations.
  - Source: [arXiv 1701.01619](https://arxiv.org/abs/1701.01619)
- **Meta Pseudo Labels** uses feedback from the labelled set to adapt the teacher (Q3). Its labelled set was all of ImageNet, not a 1% clean set. — [arXiv 2003.10580](https://arxiv.org/abs/2003.10580)
- **Small-clean-set reweighting and correction** (GLC, Learning-to-Reweight, Meta-Weight-Net): per the sibling note, these do not change AUC asymptotically under class-conditional noise, and they are high-variance with 58 studies. GLC gave small gains on CIFAR at ~1% trusted data; 58/4,349 ≈ 1.3%. [via sibling note `noisy_labels_auc_ensembling.md`]
- **Privileged information (PI)**, i.e. features available only at training time (annotator IDs, agreement statistics), can "explain away" label noise [PR, grade B]:
  - TRAM shares knowledge learned with PI and approximately marginalises it at test time.
  - Pi-DUAL uses a gate steered by PI to separate clean from noisy labels.
  - A follow-up studies when PI helps on CIFAR-N/H and a large-scale dataset.
  - Sources: [TRAM, arXiv 2202.09244](https://arxiv.org/abs/2202.09244); [Pi-DUAL, arXiv 2310.06600](https://arxiv.org/abs/2310.06600); [When does PI explain away label noise?, arXiv 2303.01806](https://arxiv.org/abs/2303.01806)
- **Selecting a subset of weak labels** [PR, grade C]: keeping a high-quality subset of weakly labelled data (β < 1), chosen by a "cut statistic", gave more accurate training labels and beat COSINE's confidence-based selection. — [arXiv 2206.02914](https://arxiv.org/abs/2206.02914) [snippet]
- **Competition evidence on overfitting gold-58** [K] [via sibling note `forum_social_github.md`]:
  - Eight public label sets vs gold: for 11 of 12 findings no set is clearly better (bootstrap 95% CIs). MCL, with 9 positives, cannot be ranked.
  - JEV showed per-finding blending tuned on gold overfits.
  - Source: [747540](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747540)
- **[computed]** Per-finding AUC SE on 58 studies is 0.04–0.07 at AUC 0.90 (Q3). A 12-parameter per-finding weight vector tuned on gold is therefore mostly fitting noise.

### Inferences
- **Gate specification** (my design; parameter budget ≤ ~16):
  - For gold cell (study i, finding f): logit P(gold = 1) = a_f + b·z_rep + c·z_img + d·z_dis (+ e·z_vlm).
    - z_rep: logit of the 31-table mean.
    - z_img: logit of the cross-view, cross-fitted image score (Q2).
    - z_dis: the table standard deviation.
    - z_vlm: optional (Q1).
  - b, c, d, e are shared across findings, or across three groups:
    - report-reliable: ACL, MCL, fracture;
    - omission-prone, fluid-visible: effusion, synovitis, Baker's cyst, contusion;
    - threshold-prone: medial/lateral/PF OA and the menisci.
  - a_f are per-finding intercepts, which do not affect per-finding AUC.
  - Fit with ridge on all 58 × 12 = 696 cells. Choose the penalty by leave-one-study-out.
- **Decision rule.** Allow image-driven relabelling for a group only if both hold:
  - the study-level bootstrap (2,000 resamples) 90% interval for c excludes 0;
  - leave-one-study-out, the gate's output *as a label* beats the report mean on that group's gold AUC.

  Otherwise keep report labels for that group.
- **Producing refined labels.**
  - For training cells in allowed groups, use the gate's probability as the soft label, but only on masked or silent cells.
  - For confident cells, move at most 10–20% toward it, and only where at least two views agree.
  - This is Veit's "learn the noisy → clean mapping conditioned on images", shrunk to the size 58 studies can support.
- **Leakage controls.**
  - Gold studies must never train any teacher.
  - z_img for gold studies must come from the *same kind* of single fold model as for training studies: assign gold to a pseudo-fold. A 5-model average would be smoother than single-fold OOF and would miscalibrate the gate.
  - Do not report the gate's own training fit as a gold result; report its leave-one-out score.
- **PI alternative** (no relabelling at all):
  - Feed the 31-table vector as privileged information into a training-only branch whose logit is added to the image logit. At test time, use the image branch alone (TRAM / Pi-DUAL style).
  - This lets the network attribute disagreement-driven noise to the PI branch instead of memorising it.
  - It is cheap to add to our MIL head, but untested on systematic report noise.

### Gaps
- No paper tests clean-anchored relabelling with a clean set as small as 58 studies (1.3%) on multi-label medical AUC.
- Exact positive counts per finding on gold-58 were not available here. MCL is 9 per the forum; the computed SEs assume 9–29.
- PI methods were tested on CIFAR-N/H and ImageNet-PI-type data, not on medical report noise.

---

## Q5. Evidence that image-grounded relabelling of report labels helped (CXR, natural images, Kaggle)

### Takeaway
The strongest evidence that image-grounding improves report labels is VisualCheXbert: +0.14 F1 agreement with radiologists reading the *images*. But it maps raw report text to image-model outputs, so it is **[RAW-TEXT]** for us.

Image-only relabelling gives small gains:
- CheXpert's self-trained relabelling of uncertain labels: mean AUC 0.894 vs 0.889 for ignoring them, inside the CIs.
- ReLabel: +1.4 top-1, but only with a much stronger, externally trained annotator. A weak annotator costs −1.4.
- Kaggle OOF relabelling: +0.0007 to +0.008 OOF, sometimes reversing on the LB.

### Cited Findings
- **VisualCheXbert (Jain et al., ACM CHIL 2021)** [PR, grade A for the mechanism] **[RAW-TEXT]**:
  - A biomedical BERT maps the report directly to *image* labels. Training is "supervised by a computer vision model trained to detect medical conditions from chest X-ray images".
  - Agreement with radiologists labelling the images: +0.14 F1 over the existing report labeller (95% CI 0.12–0.17).
  - It agrees with image-reading radiologists better than radiologists reading the reports do, by 0.12 (0.09–0.15) to 0.21 (0.18–0.24) F1.
  - Source: [arXiv 2102.11467](https://arxiv.org/abs/2102.11467)
- **CheXpert uncertainty-label approaches (Irvin et al., AAAI 2019)** [PR, grade A for the comparison]:
  - U-SelfTrained recipe: train with uncertain labels ignored until convergence, then "re-label each of the uncertainty labels with the probability prediction outputted by the model". Never replace 0/1 labels.
  - Validation AUC (atelectasis, cardiomegaly, consolidation, edema, pleural effusion):

    | Approach | Atel. | Cardio. | Consol. | Edema | Effusion | Mean [computed] |
    |---|---|---|---|---|---|---|
    | U-Ignore | 0.818 | 0.828 | 0.938 | 0.934 | 0.928 | 0.889 |
    | U-Zeros | 0.811 | 0.840 | 0.932 | 0.929 | 0.931 | 0.889 |
    | U-Ones | 0.858 | 0.832 | 0.899 | 0.941 | 0.934 | 0.893 |
    | U-SelfTrained | 0.833 | 0.831 | 0.939 | 0.935 | 0.932 | 0.894 |
    | U-MultiClass | 0.821 | 0.854 | 0.937 | 0.928 | 0.936 | 0.895 |

  - The best approach differs by pathology (U-Ones for atelectasis and edema, U-MultiClass for cardiomegaly). Per-pathology CIs are about ±0.03–0.06.
  - Source: [arXiv 1901.07031](https://arxiv.org/abs/1901.07031)
- **ReLabel ImageNet (Yun et al., CVPR 2021)** [PR, grade A]:
  - Recipe: pixel-wise multi-label maps from EfficientNet-L2 trained on JFT-300M, computed once; random crops are labelled by pooling the map.
  - Cost: +0.5% training time, against more than 4× for on-the-fly distillation from a B7 teacher.
  - ResNet-50: 77.5 → **78.9** (+1.4).
  - "The performance of the target model overall follows the performance of the machine annotator. When the machine supervision is not sufficiently strong (e.g., EfficientNet-B1), the trained model shows a severe performance drop (76.1%)."
  - Source: [arXiv 2101.05022](https://arxiv.org/abs/2101.05022)
- **Veit et al.** (Open Images, image-conditioned label cleaning): +0.56 MAP and +3.9 AP_all (Q4). — [arXiv 1701.01619](https://arxiv.org/abs/1701.01619)
- **WRENCH weak-supervision benchmark (NeurIPS 2021 Datasets)** [PR, grade B, text domain]:
  - COSINE (self-training plus contrastive regularisation on weak labels) "could successfully improve the performance of fine-tuned language models".
  - "The superiority of COSINE suggests that those uncovered data should also be used in training an end model."
  - Soft labels act like label smoothing against noisy weak labels.
  - Source: [arXiv 2109.11377](https://arxiv.org/abs/2109.11377)
- **Kaggle** [K] [via sibling notes]:
  - Aneurysm 5th: OOF relabel +0.0007 to +0.008 OOF.
  - Aneurysm 11th: OOF self-distillation +0.0126 CV but −0.0099 public.
  - HMS (top 4): train on everything, then fine-tune or validate on high-agreement samples. The gain is not quantified separately. — [HMS 1st](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492560)
  - Knee: a participant blending LLM report labels with image-model pseudo-labels at a fixed ratio reached 0.960 [team brief]. A label swap alone moved a DINOv2 from 0.931 to 0.942 LB. — [745214](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745214)

### Inferences
- **A VisualCheXbert analogue that needs no raw text.** Fit, cross-fitted, a gradient-boosted model g(T) → cross-view image score.
  - T is the full 31-table × 12-finding vector, so cross-finding cues count (effusion present → synovitis more likely).
  - Use g(T) as an "image-calibrated report label" for silent or uncertain cells.
  - Because g sees only the table view, the image model's study-specific errors average out. Only the report-pattern → image mapping survives. That is the co-training idea with views = (tables, images).
  - It takes CPU minutes. It does **not** read report text.
  - Evidence grade C: a mechanism analogue, untested in our setting.
- **Realistic size of the gain.** The CXR and Kaggle analogues suggest image-only relabelling gains are small, about +0.000 to +0.005 AUC, and fragile. Bigger moves have come from genuinely *new* information: a stronger external annotator (ReLabel) or report text read with image-grounded targets (VisualCheXbert). That is consistent with the #3 team's "labels… AI, not by hand. And don't rely on reports alone".

### Gaps
- I found no paper or Kaggle write-up doing image-only relabelling of report-derived labels on knee MRI or any other musculoskeletal MRI.
- The participant's fixed-ratio 0.960 recipe (ratio, teacher lineage) is unknown.

---

## Q6. Synthesis: evidence-graded catalogue and a concrete design that differs from what failed

### Takeaway
Do not re-run self-relabelling. Run one experiment that changes every ingredient the evidence links to failure:
- **Teachers:** trained only on confident cells, on different planes or backbones, cross-fitted.
- **Scope:** fill only masked or silent cells; move confident cells by at most 20%, and only on multi-view agreement.
- **Gold:** a gate of at most ~16 parameters decides per finding group whether image evidence may change labels at all.
- **Student:** noised, equal-or-larger, from scratch, with a separate head for filled cells.
- **Rounds:** one.
- **Judgement:** gold-58 with paired bootstrap, plus the LB as a blend member.
- **VLM:** an optional, gold-validated input for fluid-visible findings only.

Expected gain is small, about +0.000 to +0.004 public LB [estimate]. It is larger only if a new information source (VLM or external severity-graded teacher) beats the report labels on gold for effusion, synovitis or OA.

### Cited Findings

#### Catalogue (sources in the cited sections above)

| # | Method (paper) | Core idea / recipe | Reported gain | Compute | Fails when | Grade | Raw text? |
|---|---|---|---|---|---|---|---|
| 1 | Noisy Student ([1911.04252](https://arxiv.org/abs/1911.04252)) | Unnoised teacher, soft PL on large unlabelled pool; equal/larger noised student (RandAugment, dropout, stochastic depth); joint training; iterate | ImageNet 88.4%; +0.6 to +1.1 over B5 baseline in ablation | 1 extra training per round, larger student | No noise (student 83.2 vs baseline 83.3); small unlabelled pool; hard labels out-of-domain | A | No |
| 2 | Meta Pseudo Labels ([2003.10580](https://arxiv.org/abs/2003.10580)) | Teacher updated by the student's labelled-set loss | ImageNet 90.2% (+1.6) | Teacher + student jointly (≈2× memory) | Needs a large clean labelled set; 58 gold would overfit | A (not for us) | No |
| 3 | Debiased Self-Training ([2202.07136](https://ar5iv.labs.arxiv.org/html/2202.07136)) | Separate pseudo head (dropped at test) + worst-case head | FixMatch 75.4→78.3 avg; CIFAR-100 pretrained 53.1→70.4 | +7% train time | Not tested on noisy-label relabelling; minimax tuning | A (semi-supervised) / C (us) | No |
| 4 | Arazo pseudo-labelling ([1908.02983](https://arxiv.org/abs/1908.02983)) | Soft PL + mixup + ≥k (=16) labelled per batch | CIFAR-10 4k: 11.40→7.16 err | ≈0 | Few labelled samples without k; hard PL | A | No |
| 5 | Bootstrapping ([1412.6596](https://arxiv.org/abs/1412.6596)) | Target = β·label + (1−β)·pred, β = 0.8–0.95 | Robustness on synthetic noise | ≈0 | Weight too high (our 0.4) or same-family teacher | B | No |
| 6 | ELR ([2007.00151](https://arxiv.org/abs/2007.00151)) / SELF ([1910.01842](https://arxiv.org/abs/1910.01842)) | Regularise toward (ELR) or filter by (SELF) temporal ensembles | ELR ≈ +1 AUC on real CXR report noise (sibling) | ≈0 / small | Systematic noise the model learns early | B | No |
| 7 | Confident learning ([1911.00068](https://arxiv.org/abs/1911.00068)) | Cross-validated probs + per-class thresholds → prune | No reported gain in lumbar 2nd (sibling) | K-fold OOF | Prunes hard true positives; same-family OOF | B / C | No |
| 8 | Deep Co-Training ([1803.05984](https://arxiv.org/abs/1803.05984)) | ≥2 nets, view-difference via adversarial examples | CIFAR-10 4k: 9.03 (2 views) vs 12.31 Mean Teacher | ×views | Views collapse to the same function | B | No |
| 9 | DMPCT ([1804.02586](https://arxiv.org/abs/1804.02586)) / UMCT ([1811.12506](https://arxiv.org/abs/1811.12506)) | Plane-specific models; fused, uncertainty-weighted pseudo-labels | >4% over supervised (CT, few labels) | ×3 planes | Initial pseudo-label errors reinforced over iterations | B | No |
| 10 | Co-teaching+ / JoCoR ([1901.04215](https://arxiv.org/abs/1901.04215), [2003.02752](https://arxiv.org/abs/2003.02752)) | Two nets exchange small-loss samples (disagreement vs agreement) | Synthetic-noise benchmarks | 2× | Drops hard positives; DivideMix −12.9 AUC on real CXR report noise (sibling) | B / D for us | No |
| 11 | Co-Correcting ([2109.05159](https://arxiv.org/abs/2109.05159)) | Dual nets + curriculum label correction | Best of 7 methods, synthetic 5–40% noise | 2× | Only synthetic noise tested | C | No |
| 12 | Label-cleaning net ([1701.01619](https://arxiv.org/abs/1701.01619)) | Residual map (noisy labels + image features) → clean, trained on a verified subset | +0.56 MAP, +3.9 AP_all | Small head | Clean set too small for a network (ours is 58) | B | No |
| 13 | ReLabel ([2101.05022](https://arxiv.org/abs/2101.05022)) | One-time label maps from a much stronger external annotator | ResNet-50 +1.4 top-1 | One-time pass; +0.5% train | Weak annotator: −1.4 | A | No |
| 14 | CheXpert U-SelfTrained ([1901.07031](https://arxiv.org/abs/1901.07031)) | Ignore uncertain → train → fill uncertain with probs; keep 0/1 | Mean AUC 0.894 vs 0.889 U-Ignore (n.s.) | 2 trainings | Gain within CIs | A (as a null-ish result) | No |
| 15 | VisualCheXbert ([2102.11467](https://arxiv.org/abs/2102.11467)) | Text model trained on image-model outputs | +0.14 F1 vs report labeller | BERT fine-tune | Needs raw text | A | **Yes** |
| 16 | WRENCH / COSINE ([2109.11377](https://arxiv.org/abs/2109.11377)) | Self-training over weak labels, including uncovered data | Improves end models (text) | ≈2× | Confidence-based selection is beaten by cut-statistic selection | B | No |
| 17 | PI: TRAM / Pi-DUAL ([2202.09244](https://arxiv.org/abs/2202.09244), [2310.06600](https://arxiv.org/abs/2310.06600)) | Training-only branch on privileged features explains away noise | Gains on CIFAR-N/H and similar | Small head | Untested on systematic report noise | B / C | No |
| 18 | VLM noisy-label detector, DeFT ([NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6af08ba9468f0daca4b8dd388cb95824-Abstract.html)) | Positive/negative prompts as a learnable threshold; PEFT | CIFAR-100 40% IDN 87.58 vs 75.64 (secondary) | PEFT | No medical/MRI evidence | B / D | No |
| 19 | Local VLM zero-shot (MedGemma 1.5 [2604.05081](https://arxiv.org/html/2604.05081v2); 3DReasonKnee [2510.20967](https://arxiv.org/html/2510.20967v1)) | Slices + prompt → grade/probability | MRI 64.7% balanced acc; knee MOAKS 0.47 zero-shot, 0.61 SFT | ~1–6 h A100 per pass (4B) [estimate] | Fine-grained tears and OA; anchoring on the prior; licence | C | No |
| 20 | Self-distillation theory ([2002.05715](https://arxiv.org/abs/2002.05715), [2301.13304](https://arxiv.org/abs/2301.13304)) | Rounds amplify regularisation; ξ > 1 optimal under high random noise | Theory plus small experiments | — | Many rounds underfit; systematic noise | B | No |

### Inferences

#### Design: "masked teachers, cross-view fill, clean-anchored gate" (MCG)
Built for 6 days and A100/H100 rental. Each step names what it changes from our failures.

0. **Hygiene** (½ day, CPU).
   - Fix the 5 folds, grouped by patient.
   - Hold gold-58 out of every teacher.
   - Store the producing fold for every OOF prediction, and assign gold studies to a pseudo-fold so their scores come from a single fold model, like the OOF scores.
   - *Changes:* the leakage and scale mismatch behind optimistic gold numbers (Q3, 11th-place aneurysm).
1. **Masked, view-specific teachers** (~1–1.5× one full 5-fold run [estimate]).
   - Train three view groups per fold, each on confident cells only (consensus mask plus "not addressed" cells for omission-prone findings):
     - sagittal: PD-FS + PD/T1;
     - coronal: fluid + non-fluid;
     - axial.
   - Use a different backbone family for at least one view (ConvNeXt vs CoAtNet).
   - *Changes:* teachers no longer share inputs and architecture with the student (co-training; Allen-Zhu & Li), and they do not learn the omission bias (CheXpert U-Ignore; WRENCH).
2. **Measure before use** (hours, CPU).
   - For each finding and view: gold-58 AUC.
   - On OOF: Spearman correlation between view teachers' residuals (teacher − report).
   - Keep, per finding, only views that are sufficient (gold AUC within ~0.03 of the full model) and decorrelated.
   - *Kill switch:* if no finding has two sufficient views with residual correlation clearly below that of two re-seeded copies of one model, stop. Co-training has nothing to add.
3. **Cross-view fill** (CPU).
   - Fill score per cell = mean of the sufficient views' OOF probabilities, weighted by inverse TTA variance (UMCT).
   - Optional variant: g(T), the table-pattern → image-score regression from Q5 (no raw text), blended with the direct score.
4. **Clean-anchored gate** (CPU, minutes).
   - Pooled ridge-logistic on gold cells, as specified in Q4.
   - Relabelling is allowed per finding group only if c > 0 survives the study-level bootstrap and the leave-one-out label AUC beats the report mean.
   - Refined targets:
     - masked or silent cells in allowed groups → gate probability;
     - confident cells → at most 10–20% toward the fill, only where at least two sufficient views agree at |fill − report| > 0.6, capped at 5% of cells per finding;
     - everything else → the report mean.
   - *Changes:* replaces a hand-set 0.6/0.4 global mix with gold-anchored, scope-limited, small-weight changes (Veit; Reed; Q4 SE arithmetic).
5. **Optional VLM input** (1–6 A100-hours for 4B [estimate]).
   - MedGemma 1.5 4B, or Qwen3-VL-8B to avoid the HAI-DEF derivative question.
   - Zero-shot with the gold-threshold definitions; P("yes") from log-probabilities; 3–5 slice-subset variants.
   - Only for effusion, synovitis, Baker's cyst and contusion; added as z_vlm to the gate and kept only if its coefficient survives the bootstrap.
   - **[LICENCE]** check before any MedGemma-derived label is used to train a submission model.
6. **Student** (one full 5-fold run, +7% for the extra head).
   - Multi-plane MIL CoAtNet/ConvNeXt, equal-or-larger than the teachers, from ImageNet initialisation (not teacher weights).
   - Strong 3D augmentation, stochastic depth ~0.2–0.3, dropout; soft targets; mixup.
   - At least 50% of each batch from unmodified confident cells.
   - A DST-style pseudo head trained only on modified cells, dropped at inference.
   - One round only.
   - *Changes:* student ≠ teacher and pseudo-label errors are kept out of the main head (Noisy Student; DST; Arazo; Mobahi).
7. **Judge correctly.**
   - Gold-58 macro and per-finding AUC with a paired study-level bootstrap against the same recipe on the report mean.
   - The student's marginal value as a *blend member* (gold and public LB).
   - Do not trust the OOF CV of a student trained on fills unless the fills were regenerated per outer fold (nested cross-fitting, roughly 4× teacher cost).
   - *Stop rule:* if the paired gold gain is below 1 SE (~0.006–0.017 macro [computed]) and the public LB moves less than +0.002 as a blend member, drop the pipeline.

#### Minimal 2-day version
- Skip the VLM.
- Train only one extra teacher set: a different backbone on axial + coronal series, 4 folds.
- Use it with our existing 4-fold CoAtNet OOF as the two "views".
- Fill only the 12% masked cells through the gate; leave confident cells untouched.
- Train one noised student with a DST head.
- This isolates the one change the literature most supports, namely a different teacher on report-weak cells only, at about 1.5 GPU-days [estimate].

#### What this design deliberately does not do
- No small-loss sample selection (co-teaching, JoCoR, DivideMix): it drops subtle positives, and DivideMix collapsed on real report noise.
- No per-finding weights fitted freely on gold: 12+ free parameters on 58 studies.
- No iterated loops.
- No in-sample teacher targets.
- No raw report text and no external API.

### Gaps
- None of the 20 catalogue methods has been tested on knee-MRI report labels. The design's expected gain (+0.000 to +0.004 LB) is my extrapolation from:
  - CheXpert U-SelfTrained (+0.005 mean AUC, not significant);
  - Kaggle OOF relabels (+0.0007 to +0.008 OOF, with LB reversals);
  - ReLabel (+1.4 top-1 only with a much stronger annotator).
- How decorrelated plane-specific models are on our data is unknown until step 2 runs.
- Whether gold-58 has enough positives per finding group for a bootstrap-stable gate coefficient is unknown; the counts were not checked here.
- The HAI-DEF "Model Derivative" status of a model trained on MedGemma-generated labels, and the competition's winner-licence terms, need a direct reading of both documents.
- VLM throughput and VLM label quality on gold must be measured. No published knee-MRI AUC exists for any local VLM.
