# T06 (GLM, thinking task): Plan B blend strategy from public notebooks

Read `delegation/PREAMBLE.md` first and obey it. You only READ and write one results file. Time-box: 3 hours.

## Situation (facts, Oct 14)
- Metric: macro ROC-AUC over 12 findings, rank-based per column. Kaggle notebook: T4 x2, no internet, 9 h hard limit
  on the hidden test run. Deadline Oct 22. Goal: top 10 (public LB ~0.963; #1 0.966).
- Our leaderboard: base954 fork = 0.954 (`kaggle_base954/base954.ipynb`: Raptor CoAtNet 70% + Heliosli ConvNeXt
  "compact" 30%, percentile-rank blend). base954 + our distilled CoAtNet students 0.15 = 0.954 (no gain).
  Students alone = 0.948. base954 + our label-trained CoAtNets 0.15 = 0.953.
  Our students were distilled FROM base954, so they are near-copies of it (Spearman 0.959).
- New public best (not yet verified by us): Raptor + Heliosli "R7" ConvNeXt-small 50/50 rank blend ~0.957
  (our unchanged fork `kaggle_r7_957/r7_957.ipynb`, commit run in progress).
- Open rule risk: Raptor weights are trained with OAI data (`..._oai_...` in the weight file name); the host has not
  ruled on OAI. R7 is said to use fastMRI auxiliary training. One of our two final picks must avoid disputed
  external data.

## Inputs (all local, read-only)
- `delegation/inputs/public_nb/*/` six public notebooks + `kernel-metadata.json` each (Versia 7, Heliosli 0.957,
  Jenil 0.956, "one unused checkpoint 0.956", lavinwins 0.950 dual reader, degnonguidi).
- `kaggle_base954/base954.ipynb`, `kaggle_r7_957/r7_957.ipynb`, `kaggle_base954_fleet/fleet_v6.py` (our CoAtNet
  inference), `kaggle_students/build_nb.py` (how we add our models as a rank branch).

## Questions to answer (with file:cell or file:line evidence for every claim)
1. Component table: for each distinct MODEL used across these notebooks (Raptor, compact, R7, orthofoundation,
   goodpjw2008 ConvNeXt reader, any other): weights dataset, architecture, input pipeline (slices, crop, res),
   external data used in training as stated in the notebook text (OAI, fastMRI, none, unknown), and any runtime
   evidence (timeouts, deadlines, timing prints). Say "unknown" when the notebook does not say.
2. How do the 0.956 variants differ from the 0.957 one (exact diffs: weights, per-finding weights, extra checkpoint)?
   What does "one unused checkpoint" add?
3. Propose exactly 3 blend notebooks we can build by Oct 20, each with: components, fixed rank-blend weights
   (per finding if justified), expected runtime vs 9 h, why its errors should differ from R7+Raptor, and the risk.
   One of the 3 must be the "safe" pick without OAI-trained or fastMRI-trained components (say which components
   qualify and what evidence says so).
4. Rank-blend math: with rank averaging, when does adding a third model that is weaker alone (e.g. 0.940) help a
   0.957 pair? Give the condition in terms of correlation and individual AUC, and say which of our own models
   (label-trained CoAtNet version 5/6 at LB ~0.936-0.941, distilled students 0.948) could plausibly help, and at
   what weight to try first. Our evidence: weight 0.15 gave -0.001 and 0.000 on base954.
5. Risks in the R7 notebook: silent fallbacks, time guards, anything that could make the hidden run fail or
   silently degrade. Quote lines.

## Output
`delegation/results/T06_blend_strategy.md`. First 15 lines = SUMMARY (the 3 proposed notebooks in one line each +
top 3 risks). No LB predictions as facts; mark estimates "ESTIMATE". Do not edit any other file.
