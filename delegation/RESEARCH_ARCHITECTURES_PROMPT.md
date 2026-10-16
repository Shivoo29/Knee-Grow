# Research prompt: novel architectures for multi-plane knee MRI (re-run of the agent that died 16 Oct)

Paste everything below into a deep-research model. Ask for a Markdown report with links and evidence grades.

---

Research NOVEL architectures and inductive biases from the 2021-2026 literature for multi-plane knee MRI (and closely related multi-series MRI/CT) study-level multi-label classification that could add >= +0.005 macro ROC-AUC to a strong 2.5D CNN attention-MIL baseline and be implemented in 2-3 days.

## Context
Kaggle "RSNA Knee Abnormality Detection" 2026. 12 binary findings per knee MRI study: ACL, MCL, medial/lateral meniscus tear, medial/lateral/patellofemoral OA, effusion, synovitis, Baker's cyst, bone contusion, acute fracture. Metric: macro ROC-AUC. Each study has sagittal fluid-sensitive and non-fluid, coronal fluid and non-fluid, and axial series. ~4,349 train studies. Labels are noisy, extracted from reports; gold definitions use severity thresholds (OA = high-grade cartilage loss; effusion = moderate/large).
Baseline: CoAtNet-rmlp-2 384 px encoder over 2.5D windows (3 adjacent slices as RGB), 96-192 slices from 5 series slots, per-finding attention-MIL (12 learned class queries), optional slot embeddings and bi-GRU head. A public model with knee-anchored 100 mm close-ups (patellar, posterior, medial, lateral crops) is stronger.
Weakest findings: lateral OA, PF OA, synovitis, lateral meniscus. Inference: offline Kaggle 2xT4 <= 9 h for ~1,300 studies. Public LB: ours 0.956, top 0.968; goal 0.969 by 22 Oct 2026. A100/H100 rentable.
Already known (do not re-research): top Kaggle teams localise -> crop -> 2.5D CNN -> attention/GRU/transformer -> ensemble; 3D models only won with dense voxel labels; through-plane resampling hurt; the key need is a model whose errors are decorrelated from the public blend (within-class rho <= ~0.8).

## Questions
1. Knee-MRI architectures with measured gains over simple 2.5D CNNs (MRNet and others): ELNet, MRPyrNet, cross-plane fusion transformers, anatomy-guided attention (region tokens, bone-mask guidance), graph networks over anatomical regions, multi-scale crop fusion.
2. Cross-plane fusion: early vs late, cross-attention between sagittal and coronal token sets, plane-specific heads; which findings benefit.
3. Label structure: Query2Label, ML-Decoder, C-Tran, ML-GCN, conditional heads (effusion->synovitis, contusion<->ACL, OA<->Baker's); effect on macro AUC.
4. Ordinal / latent-severity heads (CORAL, CORN) trained on binary report labels: do they rank better under a stricter severity threshold? Theory or experiments.
5. Slice-sequence models: transformer, Mamba/S4, positional/anatomical encodings, attention supervision vs plain attention-MIL.
6. MIL variants with ~100-200 instances and noisy bag labels: TransMIL, DSMIL, DTFD-MIL, ACMIL (ECCV 2024), additive MIL, instance dropout.
7. Multi-scale input (global 140 mm + local 100 mm in one model) vs separate models.

## Output
A catalogue for each idea: paper, venue/year, dataset, gain vs baseline (numbers), compute, implementation sketch for our pipeline, and evidence grade (peer-reviewed / preprint / anecdote). Then the top 3 ideas ranked by expected gain per engineering day. Mark anything unverified.
