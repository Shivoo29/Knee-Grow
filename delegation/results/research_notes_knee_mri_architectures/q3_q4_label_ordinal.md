# Label-Structure and Ordinal Heads for Multi-Label Medical Image Classification

## Q1: Do transformer-decoder label heads (Query2Label / ML-Decoder / C-Tran) beat independent per-class queries on multi-label tasks? Numbers?

### Takeaway
Yes on natural-image multi-label benchmarks: query-based transformer-decoder heads consistently add +0.8 to ~+3 mAP over GAP/linear baselines on MS-COCO, with Query2Label reaching 91.3% mAP and ML-Decoder 91.1–91.4% mAP; no direct medical-imaging head-to-head vs independent per-class queries was found (gap).

### Cited Findings
- Query2Label (Liu et al., arXiv 2021) uses learnable label embeddings as queries in a standard Transformer decoder; cross-attention pools class-specific features from the backbone feature map, followed by per-class binary classification — [Source](https://arxiv.org/abs/2107.10834)
- Query2Label establishes 91.3% mAP on MS-COCO and "consistently outperform[s] all previous works on five multi-label datasets including MS-COCO, PASCAL VOC, NUS-WIDE, and Visual Genome" — [Source](https://arxiv.org/abs/2107.10834)
- Query2Label's gain is largest where labels need localized evidence: per-size breakdown shows improvement over the TResNetL baseline on small/medium/large objects (e.g. reported 39.5 vs 37.8 on small, 77.5 vs 74.2 on medium, 86.1 vs 84.2 on large in the paper's Table 7) — [Source](https://ar5iv.labs.arxiv.org/html/2107.10834)
- ML-Decoder (Ridnik et al., WACV 2023) is a Transformer-decoder classification head that removes the self-attention block (cutting the decoder's quadratic dependence on number of queries to linear) and adds a group-decoding scheme with a fixed number of group queries expanded via a group fully-connected layer — [Source](https://arxiv.org/abs/2111.12933)
- ML-Decoder reaches 91.1% mAP on MS-COCO multi-label (WACV 2023 proceedings version), 91.4% mAP in the arXiv version, 31.1% ZSL mAP on NUS-WIDE, and 80.7% top-1 on ImageNet with vanilla ResNet50 — [Source](https://openaccess.thecvf.com/content/WACV2023/html/Ridnik_ML-Decoder_Scalable_and_Versatile_Classification_Head_WACV_2023_paper.html); [Source](https://arxiv.org/abs/2111.12933)
- ML-Decoder with the same 80 queries matches Transformer-decoder accuracy, and on Open Images (9600 classes) full Transformer-decoder training is out-of-memory even at batch size 1 while ML-Decoder group-decoding adds only ~10–20% FLOPs and improves mAP over GAP; group ratio N/K=48 gives the full score benefit, N/K=96 loses only ~0.1% — [Source](https://arxiv.org/html/2111.12933v2)
- C-Tran (Lanchantin et al., CVPR 2021) is a Transformer *encoder* over concatenated image features + label embeddings, trained with Label Mask Training (randomly mask 25–100% of labels as "unknown" with ternary positive/negative/unknown state embeddings) so it can condition on partial/extra labels at inference — [Source](https://arxiv.org/abs/2011.14027)
- C-Tran absolute mAP gains of 0.8 on COCO-80 and 1.0 on VG-500 over prior SOTA were described as notable, achieved without predefined label-relation inputs such as word embeddings — [Source](https://ar5iv.labs.arxiv.org/html/2011.14027)
- C-Tran needs only 2–4 Transformer layers (mAP 85.0/85.1/84.3 for 2/3/4 layers on COCO-80) but is memory-intensive in labels: upper bound ~2000 labels on 4× Titan X GPUs — [Source](https://openaccess.thecvf.com/content/CVPR2021/papers/Lanchantin_General_Multi-Label_Image_Classification_With_Transformers_CVPR_2021_paper.pdf)
- Official open implementations exist for all three: Query2Label — [Source](https://github.com/SlongLiu/query2labels); ML-Decoder — [Source](https://github.com/Alibaba-MIIL/ML_Decoder); C-Tran — [Source](https://github.com/QData/C-Tran)

### Inferences
- For a 12-finding knee-MRI attention-MIL head, the 12-class regime is far below the scale where group-decoding matters; a plain per-finding-query cross-attention head (Query2Label-style, 1 decoder layer) is the right complexity point, and ML-Decoder's fixed/non-learnable-query result (same accuracy with fixed queries) suggests random-query initialization is safe.
- Compute cost estimate (UNVERIFIED, by analogy): one cross-attention decoder layer over ~12 queries × window features adds low-single-digit % FLOPs over the backbone — consistent with ML-Decoder's 10–20% overhead at 9600 classes scaling down sharply at 12 classes; verify by profiling.
- The small-object localization gain in Q2L is the most transferable argument to knee MRI: small focal findings (e.g. meniscal tears, cartilage defects) benefit most from per-label spatial pooling vs global average pooling.

### Gaps
- No published head-to-head of transformer-decoder label heads vs independent per-class queries on a medical multi-label task (CheXpert/ChestX-ray14/knee MRI) was found; all numeric gains above are natural-image benchmarks.
- Query2Label venue: cited here as arXiv 2021 (UNVERIFIED whether it appeared at a peer-reviewed venue).

## Q2: Does modelling label co-occurrence (ML-GCN, conditional chains, CRF/graph layers) improve macro AUC? Any medical evidence?

### Takeaway
On natural images ML-GCN-style co-occurrence modelling gives roughly +2 mAP over co-occurrence-agnostic baselines; the single strongest medical datapoint is CheXGCN, a label-co-occurrence GCN for chest X-ray that beat SOTA baselines on both ChestX-ray14 and CheXpert — though exact mean-AUC deltas were not retrievable and are recorded as a gap.

### Cited Findings
- ML-GCN (Chen et al., CVPR 2019) builds a directed graph over labels from word embeddings, learns inter-dependent classifiers via stacked GCNs with a re-weighted co-occurrence correlation matrix (thresholded conditional probabilities P(Lj|Li)) to control over-smoothing/overfitting, applied to CNN image features end-to-end — [Source](https://arxiv.org/abs/1904.03582)
- ADD-GCN (ECCV 2020) reports 85.2% mAP on MS-COCO, outperforming ML-GCN by 2.2% and SSGRL by 1.4%, and beating its own baseline by 5.5% — implying ML-GCN ≈ 83.0% mAP on COCO by subtraction — [Source](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660647.pdf)
- A-GCN follow-up argues ML-GCN's hand-crafted count-and-threshold correlation matrix is inflexible/sub-optimal and learns the label graph adaptively end-to-end with a sparsity constraint — [Source](https://arxiv.org/html/1909.13005)
- Classifier Chains (Read et al.) transform multi-label prediction into a chain of binary classifiers where classifier j sees all previous labels as extra features, explicitly modelling conditional dependence P(lj | x, l1..lj−1) — [Source](https://users.ics.aalto.fi/jesse/papers/chains.pdf)
- Probabilistic Classifier Chains (Dembczynski et al., ICML 2010) formalize chains as estimating the joint label distribution via the product rule, and warn that "label correlation" is often invoked intuitively without specifying when modelling it actually helps — [Source](https://icml.cc/Conferences/2010/papers/589.pdf)
- A 2024 analysis proposes a conditional-dependency score detecting when interdependency modelling beats binary relevance, reporting Spearman rank correlation 0.79 between the score and the chain-vs-independent performance gap — [Source](https://arxiv.org/html/2411.02638v1)
- CheXGCN (Chen et al., IEEE JBHI 2020) is a label co-occurrence GCN framework for multi-label chest X-ray classification; "extensive experiments on the ChestX-Ray14 and CheXpert datasets have demonstrated the effectiveness of CheXGCN as compared with the state-of-the-art baselines" — [Source](https://www.semanticscholar.org/paper/Label-Co-Occurrence-Learning-With-Graph-Networks-Chen-Li/1a601743b98788bfd3195bf26a8f3af8fd6294f6)
- Follow-up CXR work confirms the pattern: a 2023 label-semantic GCN paper reports its AUC scores "better than those of comparative models such as CheXGCN" (i.e. CheXGCN is the baseline to beat), and a 2026 graph-guided cross-attention CXR method reports mean AUC 0.849 on ChestX-ray14 and 0.815 on CheXpert — [Source](https://ieeexplore.ieee.org/document/10505562/); [Source](https://www.nature.com/articles/s41598-026-53115-0)
- Clinically-inspired hierarchical CXR classification (2025) structures labels by clinical hierarchy rather than flat co-occurrence — [Source](https://arxiv.org/html/2502.03591v1)

### Inferences
- Medical evidence directionally supports co-occurrence modelling (CheXGCN wins on both major CXR benchmarks), but the ranking metric in CXR literature is mean AUC over pathologies, which maps directly onto the macro-AUC objective for knee findings.
- For knee MRI, unconditional co-occurrence counts (ML-GCN style) risk baking in dataset artefacts (e.g. OA–Baker's cyst correlation driven by age); conditional/chain structure with the image features always present is safer than pure label-prior graphs. The Dembczynski caution applies: only impose structure on pairs with suspected *conditional* dependence (effusion|synovitis given the image), per the 0.79-Spearman diagnostic idea.
- Chain-order sensitivity is a real risk (label order affects chain performance; ensembles of chains mitigate it) — [Source](https://scikit-learn.org/stable/auto_examples/multioutput/plot_classifier_chain_yeast.html); a fixed clinically-motivated order (primary lesion → secondary sign) avoids random-order pathology.

### Gaps
- Exact CheXGCN mean-AUC numbers and deltas vs baselines could not be retrieved (PubMed page blocked; IEEE paywalled). Record as UNVERIFIED gap: need the IEEE JBHI 2020 paper Table values.
- No published conditional-chain or GCN experiment on knee-MRI findings (effusion–synovitis, ACL–contusion, OA–Baker's) was found; Q4 recommendations below are theory, marked as such.
- No macro-AUC (vs mAP) ablation isolating the co-occurrence component on any multi-label benchmark was found.

## Q3: CORAL/CORN trained on binary noisy labels — any evidence they rank better when the gold threshold is stricter (e.g. high-grade OA only)? Theory or experiments?

### Takeaway
No direct experiment found (binary-trained ordinal head evaluated against a stricter gold threshold). The mechanism story is theoretically coherent — ordinal heads learn a graded latent severity score, so raising the decision threshold at eval should isolate severe cases — and CORAL→CORN has measured MAE gains on age estimation, plus knee-OA precedent exists for ordinal losses on KL grades; but the binary-labels-to-strict-threshold claim is THEORY, unverified.

### Cited Findings
- CORAL (Cao et al., arXiv Jan 2019; Pattern Recognition Letters 2020) reformulates K-rank ordinal regression as K−1 binary tasks sharing one weight vector with ordered biases, with a theorem guaranteeing rank-monotonicity (non-increasing biases ⇒ decreasing confidence scores) without example-specific cost weighting — [Source](https://ar5iv.labs.arxiv.org/html/1901.07884)
- CORAL-CNN (ResNet-34) MAE on MORPH-2: 2.64 ± 0.02 vs OR-CNN 2.83 ± 0.03 vs cross-entropy 3.34 ± 0.07; AFAD: 3.47 vs 3.51 vs 3.60; CACD: 5.25 vs 5.38 vs 5.65 (mean ± SD over 3 seeds) — [Source](https://ar5iv.labs.arxiv.org/html/1901.07884)
- CORAL guarantees zero rank inconsistencies while OR-CNN averages ~0.8–2.3 inconsistencies per prediction, and inconsistencies concentrate on wrong predictions — presented as evidence inconsistency harms performance — [Source](https://ar5iv.labs.arxiv.org/html/1901.07884)
- CORN (Shi et al., arXiv Nov 2021; Pattern Analysis and Applications 2023) removes CORAL's weight-sharing constraint by training on conditional probabilities P(y > rk | y > rk−1) and recovering unconditional ranks via the chain rule; authors report it "improves the performance substantially compared to the CORAL reference approach" — [Source](https://arxiv.org/abs/2111.08851)
- Reference implementations for both: — [Source](https://github.com/Raschka-research-group/coral-pytorch)
- Knee-OA ordinal precedent: a contrastive domain-adaptation framework for KL grading explicitly "selected CORN as a representative ordinal regression" method — [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC13603584/)
- Directly relevant evaluation paradigm: "Leveraging deep learning to infer continuous predictions from ordinal labels" evaluates severity prediction on retinopathy of prematurity and knee osteoarthritis among three medical datasets — [Source](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0001248)
- Original CORAL venue/year: Pattern Recognition Letters vol. 140 (2020), pp. 325–331 (arXiv preprint Jan 2019, so "AAAI 2019" attribution is incorrect) — [Source](https://www.sciencedirect.com/science/article/pii/S016786552030413X); CORN venue: Pattern Analysis and Applications 2023 — [Source](https://arxiv.org/abs/2111.08851)

### Inferences (THEORY — explicitly unverified, no experiment found)
- Why a CORAL/CORN-style head could beat a binary head under a stricter gold threshold even when trained on binary noisy labels: (1) the K−1-threshold decomposition forces the shared representation to order cases by severity rather than merely separating "mentioned vs not mentioned" in the report; (2) the latent score s(x) = Σk fk(x) is monotone in severity by construction, so sweeping the operating threshold at eval traces a severity ranking, and a stricter gold standard (high-grade OA) corresponds to a higher cut on the same axis; (3) CORAL's rank-consistency removes contradictory threshold crossings that flatten the top of the ranking. None of this has been tested with report-derived binary labels — the failure mode is that with only K=2 observed, the ordinal head collapses to a single logistic unit and all of the above is vacuous unless severity supervision comes from elsewhere (report text grades, multi-rater disagreement, or pseudo-grades).
- Practical consequence: to get any ordinal signal from binary labels you must manufacture rank structure — e.g. derive 3–4 pseudo-grades from report language ("trace/small" < "moderate" < "large/severe") or from mention-hedge cues, then train CORAL/CORN on those pseudo-ranks and evaluate ranking (AUROC/average precision) against the strict gold threshold. This is a hypothesis to test, not an established result.
- Suggested decisive experiment (2–3 days, if pseudo-grades extractable): train (a) binary logistic head vs (b) CORAL head on pseudo-graded labels on the same backbone; compare AUROC on the full binary gold labels AND on the strict-severity-only gold subset; success = (b) wins by a larger margin on the strict subset. If pseudo-grades are unavailable, skip ordinal heads — a well-calibrated binary ranker is the correct baseline.

### Gaps
- No paper found training CORAL/CORN on binary labels and evaluating ranking under a stricter severity threshold. The exact CORN-vs-CORAL numeric deltas were also not retrieved (abstract claims "substantial" improvement without numbers in the abstract page).
- The "AAAI 2019" venue guess in the brief is wrong for CORAL (PRL 2020); CORN is PAA 2023, not a 2021 conference paper.

## Q4: Application to correlated finding pairs (effusion–synovitis, ACL–contusion, OA–Baker's) — what structure to impose? + 2–3-day implementation sketch

### Takeaway
No knee-specific structure-learning experiment exists; the recommended structure is: shared per-finding query cross-attention base (Q1) + small directed conditional edges only on the three clinically-motivated pairs + optional pseudo-grade ordinal output on gradable findings (Q3-theory). Everything below is THEORY grounded in the Q1–Q3 findings.

### Cited Findings
- C-Tran's label-mask training (mask 25–100% of labels per sample, predict masked from unmasked + image) is the closest validated mechanism for learning conditional structure P(y_masked | y_known, x) and yields the +0.8/+1.0 mAP gains cited in Q1 — [Source](https://ar5iv.labs.arxiv.org/html/2011.14027)
- CheXGCN validates co-occurrence GCNs on chest X-ray multi-label tasks (Q2), while A-GCN warns that fixed count-based graphs can be sub-optimal and should be learned/sparsified — [Source](https://arxiv.org/html/1909.13005); [Source](https://www.semanticscholar.org/paper/Label-Co-Occurrence-Learning-With-Graph-Networks-Chen-Li/1a601743b98788bfd3195bf26a8f3af8fd6294f6)
- Classifier-chain theory says order matters and ensembles mitigate it; clinically fixed orders (primary → secondary) are the principled choice over random orders — [Source](https://users.ics.aalto.fi/jesse/papers/chains.pdf); [Source](https://scikit-learn.org/stable/auto_examples/multioutput/plot_classifier_chain_yeast.html)

### Inferences (THEORY — proposed structure, no direct evidence)
- Pair-by-pair structure proposal:
  1. Effusion → synovitis (asymmetric conditional): synovitis query cross-attends to window features AND to the effusion logit/embedding (gated residual: s_syn = f(x) + g · h(effusion)). Direction fixed by clinical plausibility (fluid is the visible cue conditioning synovial assessment), avoiding chain-order search.
  2. ACL ↔ bone contusion (bidirectional correlation, single mechanism): one shared "pivot-shift mechanism" latent queried once, feeding both heads (a 2-node group query in ML-Decoder terms), rather than two directed edges — fewer params, matches the single-trauma-mechanism story.
  3. OA ↔ Baker's cyst (confounded co-occurrence): do NOT wire a direct edge (age-mediated correlation risks a spurious prior); instead share the GroupFC-style decoder context or condition both on an age/degeneration covariate if available. This follows the Dembczynski/A-GCN caution against baking unconditional correlations into the head.
- General rule: at most 3 sparse directed edges on 12 findings; everything else stays conditionally independent given the image (binary relevance), so a mispredicted parent cannot cascade. Consider C-Tran-style label-mask training (randomly reveal 0–3 gold/pseudo labels at train time) as the mechanism that teaches the conditional edges without test-time label leakage.
- 2–3-day implementation sketch for the 12-finding attention-MIL head:
  - Day 1 (base head, ~1 day): freeze the window-feature MIL backbone; implement 12 learnable finding queries × 1-layer cross-attention decoder (hidden dim = backbone dim, 4 heads) + per-finding linear + sigmoid, BCE with asymmetric/pos-weighted loss. Baseline metric: per-finding AUROC + macro AUC on val. (Mirrors Query2Label single-decoder-layer design — [Source](https://arxiv.org/abs/2107.10834).)
  - Day 2 (structure, ~1 day): add (a) the effusion→synovitis gated edge, (b) the shared ACL–contusion mechanism query, (c) label-mask auxiliary training (mask 0–50% of findings, predict masked). Ablate each addition vs Day-1 macro AUC; drop anything < +0.005 val macro-AUC. Keep total head params < ~5% of backbone (UNVERIFIED estimate — profile to confirm).
  - Day 3 (ordinal probe, ~0.5–1 day, only if pseudo-grades extractable from reports): replace the linear+sigmoid on gradable findings (effusion size, OA grade, cartilage) with a CORAL head over 3–4 text-derived pseudo-grades; eval AUROC on full vs strict-severity gold subsets per the Q3 experiment. If pseudo-grades are unavailable, spend Day 3 on calibration/threshold selection instead.
  - Guardrails: edges read *detached* parent logits during training (no gradient cascade); at inference all parents come from the image (no gold labels), so the C-Tran partial-label path is train-only regularisation.

### Gaps
- Zero direct evidence on which of the three pair structures (directed edge vs shared latent vs covariate conditioning) helps knee-MRI macro AUC; the sketch's ablation plan is the de-risking mechanism.
- Whether report text supports pseudo-grade extraction for CORAL training is a data question, not a literature question — needs a same-day report-language audit before committing Day 3.
- Compute-cost numbers for the head itself are UNVERIFIED estimates by analogy to ML-Decoder's 10–20% overhead at 9600 classes; actual profiling required.
