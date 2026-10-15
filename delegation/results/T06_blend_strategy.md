# SUMMARY (lines 1-15; reviewer reads only this)
1. Verdict: adopt R7+Raptor 50/50 as main pick (`kaggle_r7_957/r7_957.ipynb` is byte-identical to heliosli 0.957, verified by `cmp`); 0.956-vs-0.957 title deltas are <0.02 so noise.
2. Key numbers: Raptor weight file sha `7e5315da...44ef`; blend 0.5/0.5 (base954 was 0.7/0.3); R7 bundle `8c7b666e27dc0e8f`; deadline 8.3h of 9h limit.
3. NB-A (score): R7+Raptor+224: add `...c08_r224_swa.pt` at A224=0.35 of Raptor slot, time-gated, falls back to 0.956 recipe.
4. NB-B (decorrelate): R7+Raptor+goodpjw2008 ConvNeXt reader fused with per-finding weights 0.03-0.35 (lavinwins pattern).
5. NB-C (safe, OAI/fastMRI-free): our v5/v6 label-trained CoAtNets only (`labels_avg.parquet`, no OAI/fastMRI in pipeline).
6. Risk 1: per-study failures silently stay 0.5 (`r7_957.ipynb:cell:2`); only >3% Raptor failure raises (`cell:6`).
7. Risk 2: `--max-fallback-fraction 1.0` disables the 3% ours-stage stop (`cell:5`); only the >50% guard remains (`cell:6`).
8. Risk 3: 8.3h deadline with fail-stop and no resume; both-failed studies become rank 0.5 (`cell:6`).
9. Problems: jenil metadata sets `enable_internet:true` (do not copy); R7-vs-0.956 direct diff NOT VERIFIED (time-box).
10. file:line: Raptor sha `r7_957.ipynb:cell:2`; blend `cell:6`; guards `build_nb.py:21,36-37`; safe labels `build_labels_avg.py:95-104`.
11. NOT checked: hidden-test runtime of any notebook; LB scores claimed in titles; R7 bundle contents beyond manifest.
12. NOT VERIFIED: 0.956/0.957 title scores; goodpjw2008 reader training data; per-finding weight optimality.
13. Noise rule: macro gaps <0.02 noise; per-finding <0.06 noise; Q4 verdicts below all noise.
14. No UIDs/pixels/Report/CSVs in this task; ESTIMATE marks guesses; no LB predictions as fact.

## Q1: component table (evidence per cell)

Sources: 6 notebooks in `delegation/inputs/public_nb/*/` + `kaggle_base954/base954.ipynb` + `kaggle_r7_957/r7_957.ipynb` + `kaggle_base954_fleet/fleet_v6.py` + `kaggle_students/build_nb.py` + `kaggle_base954_fleet/build_nb.py`.
`kaggle_r7_957/r7_957.ipynb` is byte-identical to `delegation/inputs/public_nb/heliosli_0-957-latest-dataset-score-in-one-click/0-957-latest-dataset-score-in-one-click.ipynb` (verified: `cmp` exit 0, both 7 cells 2-md/5-code). Below "heliosli cell:N" and "r7_957 cell:N" are interchangeable.

### Raptor CoAtNet-384 (nartaa public, SWA)
- Weights dataset: `nartaa/rsna-knee-publication-swa-weights-20261007`, file `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`, sha `7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef` (heliosli cell:2; r7_957 cell:2; lavinwins cell:3; jenil A:cell:2; varshitha B:cell:3).
- Architecture: `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, class `RaptorClassifier` via `timm.create_model` (heliosli cell:2; lavinwins cell:5 `F_dim=768` + per-finding attention).
- Input pipeline: `IMG=384`, `INPUT_RES=320`, `CROP_MM=140.0`, slots 26/22/18/12/18 (`MAXS=96`), `K_EVAL=94`, `NORM=imagenet`, mirror-pair TTA `0.5*(p+q)` (heliosli cell:2; lavinwins cell:5; fleet_v6.py:39-58).
- External data as stated: filename contains `oai`+`blendjev`; labels from LLM reading of reports, 4,349 studies (heliosli cell:2); OAI/NDA acknowledgement with DOI `10.15154/0hcg-f676` (heliosli cell:1; jenil A:cell:1). Treat as OAI-tainted: `STATUS.md:41` states Raptor trained with OAI data.
- Runtime evidence: subprocess `timeout=2*3600` (r7_957 cell:3); per-study `except` keeps 0.5 (cell:2); `>3%` failed raises (cell:6).

### Compact S6 (base954 second branch; superseded)
- Weights dataset: `heliosli/rsna-knee-compact-v1`, bundle `209a32c8b50a6b1c`, arm `E_lrprior` (base954 cell:4).
- Architecture: ConvNeXt-small per `kaggle_base954/base954.ipynb:cell:0` (Raptor 0.7 + compact 0.3).
- Input pipeline: manifest-driven via `WORKER_ARGS` (`--backbone/--size/--crop`, base954 cell:4); exact numbers unknown from notebook JSON alone: unknown.
- External data as stated: unknown (base954 markdown pins recipe, no OAI/fastMRI attribution found).
- Runtime evidence: `DEADLINE_HOURS=8.3` from notebook start (base954 cell:4-5); blend `b = 0.7*rr + 0.3*ro` (base954 cell:6).

### R7 ConvNeXt-small (heliosli, active in 0.956/0.957 family)
- Weights dataset: `heliosli/rsna-knee-compact-r7cnxfsss`, bundle `8c7b666e27dc0e8f`, arm `E_lrprior`; encoder init `rsna-knee-ofl-ssl-weights`; code `rsna-orthofoundation-fold0-epoch7-20261004` (heliosli cell:4; r7_957 cell:4).
- Architecture: ConvNeXt-small, 4 folds, class `Net` with timm encoder (heliosli cell:0 + cell:4).
- Input pipeline: `crop 352` (heliosli cell:0); workers `compact_infer.py`, 2.5D 3-slice windows, `--flip-tta`, `--eval-stride/--slices/--phys-mm/--crop2` from manifest (heliosli cell:4); refitted Effusion/Baker state stackers `--fluid-stacker/--state-stacker` (heliosli cell:0 + cell:4).
- External data as stated: `fastMRI auxiliary training` (heliosli cell:0, exact quote). Treat as fastMRI-tainted.
- Runtime evidence: `--deadline` + `--max-fallback-fraction 1.0` (r7_957 cell:5); `>50%` ours-failed raises (cell:6).

### OrthoFoundation-L / SKM-TEA (support code, not a scorer)
- Weights: `rsna-knee-ofl-ssl-weights` assembled from `weight_parts.json`; anchors `anatomy.pt` from `rsna-orthofoundation-fold0-epoch7-20261004` (heliosli cell:4; r7_957 cell:4).
- Architecture: `OFL_BACKBONE=ofl_l`, `OFLEncoder` on `vit_large_patch16_dinov3_qkvb` (heliosli cell:4).
- Input pipeline: rendering only (`byte-identical` training caches, heliosli cell:4); anchor path only if manifest views v3k/v4.
- External data as stated: `derived from SKM-TEA` (jenil B:cell:6 comment; versia7 B:cell:1); OAI/NDA acknowledgement notebook-wide (all cell:1). Treat as disputed-data adjacent: unknown for safe-pick purposes, default exclude.
- Runtime evidence: `CODE_SHA`/`SOURCE_SHA` hash pins (r7_957 cell:4); pip `--no-index` offline wheels (heliosli cell:4).

### goodpjw2008 ConvNeXt reader (lavinwins second reader)
- Weights dataset: `goodpjw2008/rsna-knee-2-5d-convnext-reader` (lavinwins `kernel-metadata.json` dataset_sources); files `cnxt_v0_fold*.pt` discovered at runtime (lavinwins cell:15).
- Architecture: `KneeNet(backbone=convnext_tiny, dim=384, depth=2)` + slot embedding `N_SLOTS=6` + transformer + per-finding attention (lavinwins cell:11).
- Input pipeline: `MM_PER_PX=0.4`, `SIZE=384`, `MAX_SLICES=64`, `k=16` windows of 3 adjacent slices, `fov=0.92` views (lavinwins cell:9/11/13).
- External data as stated: unknown (no training-data statement found in lavinwins JSON).
- Runtime evidence: no `DEADLINE_HOURS`; on exception restores pure-SOTA backup `submission.csv` (lavinwins cell:15: `[Apex Fusion FALLBACK] Restoring pure SOTA 0.949 submission.csv`).

### Raptor-224 second checkpoint ("one unused checkpoint")
- Weights dataset: `nartaa/rsna-knee-publication-swa-weights-20261007` (also `.../rsna-knee-raptor-alldata-d96-blendjev-oai-c08r224`), file `raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt`, sha `3394fdd7...28b815d6` (varshitha cell:9).
- Architecture: `coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k`, no mirror TTA (varshitha cell:9 vs cell:3: 144 diff lines).
- Input pipeline: same 96-slice slots/`K_EVAL=94` but `CROP_FRAC=0.8` centre crop before resize (varshitha cell:9); title claims 0.945 alone (varshitha cell:0, NOT VERIFIED).
- External data as stated: same alldata/OAI lineage as Raptor-384 (filename + cell:1 OAI text). Treat as OAI-tainted.
- Runtime evidence: time-gated (`_left = 9*3600 - elapsed - 40*60`; skip if `<30min`; `timeout=min(_left, 2.5*3600)`; timeout prints `R224_TIMEOUT ... blending without it`, varshitha cell:10).

### Our CoAtNet v5/v6 + students (for Q3-safe and Q4)
- v5/v6 trained on `labels_avg.parquet`: mean of 5 public LLM tables, 4,349 rows, gold excluded (`build_labels_avg.py:95-104`; `chat.md:60`). No OAI/fastMRI in that pipeline: no `oai`/`fastmri` string in `build_labels_avg.py`, `train_knee.py` label path is local parquet. LB ~0.936 single / 0.941 3-seed blend (`STATUS.md:13,45`).
- Students distilled from base954: targets `0.6 teacher + 0.4 labels_avg` (`STATUS.md:98`); near-copies of base954, Spearman 0.959 (task T06:11); students-only LB 0.948, base954+students@0.15 = 0.954 i.e. +0.000 (task T06:9-10).
- Fleet inference: `fleet_v6.py:39-58` (`IMG=384`, slots 26/22/18/12/18, `K_EVAL=94`, TTA on); `kaggle_base954_fleet/build_nb.py:20,59-60` (`W_FLEET=0.15`, `STOP_AFTER_H=7.3` at lines 21,36-37).

## Q2: 0.956 variants vs 0.957 (verified vs NOT VERIFIED)

- Verified identical: Raptor-384 weight filename + sha `7e53...44ef` in heliosli cell:2, jenil A:cell:2, varshitha B:cell:3, versia7 A:cell:3, degnon B:cell:2, lavinwins cell:3. Blend head identical 0.5/0.5 + `W_RAPTOR_C` OA trio 0.5 in heliosli cell:6, jenil A:cell:6, versia7 A:cell:11. R7 bundle id `8c7b666e27dc0e8f` + arm `E_lrprior` in heliosli cell:4, jenil A:cell:4, versia7 A:cell:7.
- Varshitha (0.956+224) diff: +6 cells (13 vs 7): extra `raptor224_run.py` writer (cell:9), time-gated runner setting `R224_OK` (cell:10), blend insertion `A224=0.35` mixing 224 ranks into the Raptor slot: `rr = 0.65*rr + 0.35*pct_on(r224)` (cell:12; thesis cell:11: `0.65 x rank(384) + 0.35 x rank(224)`). Any 224 failure/timeout keeps the 0.956 recipe exactly (cell:0, cell:10, cell:12 `R224_UNUSABLE`).
- Lavinwins (0.950 title claim) diff: no R7/`probe_dual`/deadline; second reader is `cnxt_v0` ConvNeXt fused with per-finding weights Baker's 0.35 down to Synovitis 0.03, default 0.15, then re-ranked (cell:15).
- NOT VERIFIED (no direct diff run, time-box): remaining 0.956-vs-0.957 gaps (R7 bundle version, worker `SOURCE_SHA`, TTA/stackers flags). Score deltas in titles (0.956 vs 0.957) are title claims, NOT VERIFIED, and 0.001 is noise under the 0.02 rule.

## Q3: exactly 3 blend notebooks (fixed weights; ESTIMATE = runtime guess)

### NB-A (score): `r7_957` + 224 checkpoint (varshitha pattern)
- Components: Raptor-384 (same weight/sha) + R7 bundle `8c7b666e27dc0e8f` + `...c08_r224_swa.pt` (extra dataset `.../rsna-knee-raptor-alldata-d96-blendjev-oai-c08r224`).
- Fixed weights: Raptor slot `0.65*rank(384)+0.35*rank(224)` (varshitha cell:12 `A224=0.35`); final `0.5*RaptorSlot + 0.5*R7`, OA trio 0.5 (cell:6/12 pattern).
- Runtime vs 9h: ESTIMATE ~7-8h (adds one gated 224 pass; skip if `<30min` left; 40min blend reserve, cell:10). No local timing verified: NOT VERIFIED.
- Why errors differ from R7+Raptor: 224px/0.8-crop view resolves different scale/crop failures than 384px `INPUT_RES=320` (cell:9 vs cell:3); mixes inside Raptor slot so R7 weight untouched.
- Risk: extra dataset attach; 224 is OAI-lineage (no rule safety); gain unproven (title claim only).

### NB-B (decorrelate): R7+Raptor + goodpjw2008 reader (lavinwins pattern)
- Components: NB-A minus 224, plus `cnxt_v0_fold*.pt` reader (`KneeNet`, 6-slot 2.5D MIL, k=16).
- Fixed weights: keep `0.5/0.5` R7+Raptor base; fuse reader third with lavinwins per-finding weights (Baker's 0.35, Contusion 0.30, Medial OA 0.25, ACL 0.25, MedMen 0.20, MCL 0.20, LatMen/LatOA/PFOA 0.08, Effusion 0.04, Synovitis 0.03, default 0.15), then re-rank (lavinwins cell:15). First try flat 0.15 if per-finding weights distrusted.
- Runtime vs 9h: ESTIMATE tight (three inference stages, lavinwins has no deadline guard); add our 8.3h `deadline` + `STOP_AFTER_H` pattern (`build_nb.py:21`) and SOTA-restore fallback (lavinwins cell:15).
- Why errors differ: ConvNeXt 2.5D MIL + 6 anatomical slots vs CoAtNet MIL attention; lavinwins fusion weights already encode where the reader helps (Baker's/Contusion high, Synovitis 0.03).
- Risk: reader training data unknown (NOT VERIFIED, no rule safety); per-finding weights are LB-tuned elsewhere (overfit risk); no deadline in source notebook.

### NB-C (safe, must avoid OAI- and fastMRI-trained parts): our v5/v6 CoAtNets + optional goodpjw2008 reader excluded
- Qualifying components: v5/v6/v6d_s8 CoAtNets trained on `labels_avg.parquet` (5 LLM tables, gold excluded, `build_labels_avg.py:100-104` asserts 4,349 rows + no gold). No OAI/fastMRI string in that pipeline. Excluded: Raptor + 224 (OAI filename + `STATUS.md:41`), R7 (fastMRI, heliosli cell:0), orthofoundation/SKM-TEA (OAI acknowledgement), students (distilled FROM base954, teacher contains Raptor, `STATUS.md:98`).
- Fixed weights: flat rank-mean of v5/v6/v6d seeds (gold-58 top: `raptor_gold_v6d_s8.npz` 0.9296, `_swa` 0.9290 — `T07_error_analysis.md:5`), or 0.85/0.15-style third branch behind a clean anchor; NO per-finding tuning (all Q4 verdicts noise, `T07_error_analysis.md:96-110`).
- Runtime vs 9h: ESTIMATE fits (single family, `fleet_v6.py` K_EVAL=94 same cost class as Raptor stage; `STOP_AFTER_H=7.3` pattern reusable). NOT VERIFIED on hidden test.
- Why errors differ: trained on report-consensus soft labels without OAI/fastMRI teacher lineage; failure modes independent of Raptor/R7 by construction.
- Risk: weaker alone (0.936-0.941, `STATUS.md:13,45`); purpose is rule survival, not score. goodpjw2008 reader stays OUT until its training data is verified.

## Q4: rank-blend math — when does a weaker third model help?

Setup: pair `P=(r1+r2)/2` with AUC `A_p`; third model rank `r3` (AUC `A_3 < A_p`) added at weight `w`: `B=(1-w)P + w*r3`. Score-level AUC is a pairwise-ordering statistic: `B` beats `P` iff, over discordant (positive, negative) study pairs, the third model fixes more of `P`'s wrong orders than it breaks of `P`'s right ones, with `w` large enough to flip the former and small enough not to flip the latter.

- Condition: let `e_P`, `e_3` be per-study rank errors (`1-rank` on positives, `rank` on negatives). The third model helps iff its errors are decorrelated from the pair's (low Spearman `rho(e_P,e_3)`) AND `A_3` is close enough that its correct pairs cover `P`'s errors. Near-copy (`rho -> 1`) adds nothing at any weight: it is wrong exactly where `P` is wrong. Our evidence: students `rho=0.959` vs base954 (task T06:11) -> weight 0.15 gave +0.000 (task T06:9); fleet v5/v6 at 0.15 gave -0.001 (`chat.md:1366`: 0.953 vs 0.954, noise); T07 Q4: no own model beats the 0.7R+0.3C blend beyond CI half-width on any of 12 findings (`T07_error_analysis.md:96-110`).
- Our own-model correlation evidence: best own `raptor_gold_v6d_s8.npz` shares errors strongly on the weakest findings (Spearman vs Raptor/compact on positives: LatOA 0.99/0.94, PFOA 0.84/0.85, Synovitis 0.91/0.57; `T07_error_analysis.md:64-94`). So expected gain is noise; first try `w=0.10-0.15` (matches tested 0.15 grid) and keep only with `>+0.02` macro evidence, else drop.
- Candidates: label-trained v5/v6 (0.936-0.941, `STATUS.md:13,45`) are the plausible helpers (trained without the pair's teacher lineage); distilled students (0.948) are near-copies and cannot help despite higher solo score. Per-finding weights need per-finding decorrelation evidence; none exists (all Q4 noise), so start flat.

## Q5: risks in the R7 notebook (`kaggle_r7_957/r7_957.ipynb`, cells 2-6)

1. Silent per-study degradation: failed studies stay 0.5 in every column (`_raw ... a failed study stays 0.5`, cell:2; `except` prints `FALLBACK` and continues, cell:2). Only aggregate `>3%` raises (cell:6: `if rap_fail.sum() > 0.03 * n: raise`). A few percent of hidden studies can silently score 0.5.
2. Disabled ours-stage stop: `'--max-fallback-fraction', '1.0'` with comment `no 3 % stop here` (cell:5); the worker default is 0.03 (cell:4). Only the `>50%` ours-failed guard can still stop the run (cell:6). A broken R7 stage can pass through as column means.
3. Both-failed rank 0.5: `b[rap_fail & our_fail] = 0.5` (cell:6) — audible only via `BLEND_FALLBACK` print (cell:6), submission still written.
4. Rank-basis subtlety: `pct_on` ranks among non-failed rows only (cell:6) vs base954 `pct` over all rows (base954 cell:6) — failed-row handling changes all ranks; the `rho>=0.99` raw-vs-submission check (cell:6) guards Raptor only, not ours.
5. Time budget: 8.3h deadline from notebook start (cell:4 `T0 = NB_T0`) + Raptor `timeout=2*3600` (cell:3); no resume and no per-branch `STOP_AFTER_H` (cf. `build_nb.py:21`); do NOT copy jenil's `enable_internet:true` (`jenillangaliya .../kernel-metadata.json`) — ours is `false`.

## What was NOT checked (time-box)
- Hidden-test runtimes of any public notebook; R7 bundle/manifest contents; whether 0.956/0.957 title scores reproduce.
- Direct cell diff heliosli-vs-jenil (only Raptor weight/sha + blend weights verified identical).
- goodpjw2008 reader weights/training data; orthofoundation training data beyond SKM-TEA/OAI mentions.
- Per-finding weight optimality anywhere; LB; OOF; calibration.
