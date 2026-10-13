# T02 - Fact sheet on other strong PUBLIC notebooks (for a rank-blend with our 0.954 base) (~60 min)

Goal: decide which other public notebooks could be rank-blended with the 0.954 base (heliosli/rsna-knee-blend-gold-dataset-restored, already forked in kaggle_base954/).
Use the local authenticated Kaggle CLI to PULL CODE ONLY into delegation/results/pub/ (never run anything, never download competition data):
  kaggle kernels pull <owner/slug> -p delegation/results/pub/<slug> -m
Candidates (verify each slug exists; if not, `kaggle kernels list --competition rsna-knee-abnormality-detection --sort-by voteCount --page-size 40` and pick others with claimed LB >= 0.945):
  matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed, romantamrazov/rsna-knee-dinosaur-v5, jiweiliu/rsna-knee-fast-2xt4-inference,
  pjmathematician/rsna-knee-d4-blend (already known: DINOv2 + RadImageNet + CoAtNet readers, LB 0.943), mattiaangeli/bend-the-knee-to-the-dinosaurs.
For EACH write a row: slug | claimed LB (from its title/markdown, quote the text) | model families | dataset_sources (list) | does it use OAI-derived weights? (look for 'oai' in dataset names/markdown; else "unknown")
| estimated GPU runtime on 2xT4 for ~1,300 studies (quote any runtime numbers printed in its markdown/log; else "unknown") | how it finds test data (RSNA_COMP_ROOT? hard-coded path?) | output file name.
Then answer: (1) which 1-2 are most likely to be DIFFERENT enough from base954 (Raptor CoAtNet + ConvNeXt-small compact) to add diversity; (2) can their weights be attached from a Kaggle dataset without internet; (3) total runtime if chained after base954 (~3.5 h): fits in 9 h?
Output: delegation/results/T02.md.
