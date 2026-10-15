# SUMMARY (lines 1-15; reviewer reads only this)
1. Verdict: weakest blend findings: Lateral OA AUC 0.7950 CI [0.5833,0.9656] n_pos 11; Synovitis AUC 0.8399 CI [0.7167,0.9307] n_pos 27; PF OA AUC 0.8764 CI [0.7525,0.9704] n_pos 21.
2. Shared errors: mean Raptor-compact Spearman over positives = 0.82; see lines 16+ for pairs.
3. Key numbers: blend macro 0.9313; Raptor macro 0.9229 (possibly in-sample); compact macro 0.9265.
4. Best ours by macro: raptor_gold_v6d_s8.npz 0.9296; raptor_gold_v6d_s8_swa.npz 0.9290; raptor_gold_v6d_s8_epens.npz 0.9290.
5. Problems: none blocking; file filter keeps _epens despite `_ep` substring (see Methods).
6. file:line: rank def score_blend_npz.py:63; labels score_blend_npz.py:9-11; inputs teacher/c0_*.csv:1.
7. NOT checked: LB; OOF; training provenance; per-series causes; calibration; thresholds.
8. NOT VERIFIED: teacher chunk-0 coverage beyond 58/58 id match; clinical meaning of misses.
9. Raptor numbers are possibly in-sample (trained on alldata); treat gaps vs Raptor as noise-prone.
10. Macro gaps <0.02 noise; per-finding gaps <0.06 noise (PREAMBLE).
11. No UIDs / pixels / Report text in this file; misses listed as study index 0-57.
12. Bootstrap: 1000 resamples of 58, seed 0; degenerate resamples skipped (n_valid in table).
13. Rank error e=(1-rank)|pos else rank; Spearman on e (positives + all-58).
14. Q4: ours beats blend only if diff > half-width else noise; outputs .md+.py; CPU only.

## Methods (inputs, definitions, provenance)
- Root: project root; script: delegation/results/T07_error_analysis.py; output: delegation/results/T07_error_analysis.md.
- Gold npz kept: 34 files (final/swa/epens); dropped epoch checkpoints: 18 files.
- Dropped files (contain `_ep` without `_epens`): raptor_gold_coat224_v1_ep13.npz, raptor_gold_coat224_v1_ep15.npz, raptor_gold_convnext_s1_ep10.npz, raptor_gold_convnext_s1_ep11.npz, raptor_gold_convnext_s2_ep11.npz, raptor_gold_convnext_s2_ep13.npz, raptor_gold_r50_v1_ep13.npz, raptor_gold_r50_v1_ep15.npz, raptor_gold_v5s1_ep10.npz, raptor_gold_v5s1_ep8.npz, raptor_gold_v6_ep11.npz, raptor_gold_v6_ep12.npz, raptor_gold_v6_s6_ep12.npz, raptor_gold_v6_s6_ep13.npz, raptor_gold_v6d_s10_ep11.npz, raptor_gold_v6d_s10_ep13.npz, raptor_gold_v6d_s8_ep13.npz, raptor_gold_v6d_s8_ep15.npz.
- Kept files: raptor_gold_coat224_v1.npz, raptor_gold_coat224_v1_epens.npz, raptor_gold_coat224_v1_swa.npz, raptor_gold_convnext_s1.npz, raptor_gold_convnext_s1_epens.npz, raptor_gold_convnext_s1_swa.npz, raptor_gold_convnext_s2.npz, raptor_gold_convnext_s2_epens.npz, raptor_gold_convnext_s2_swa.npz, raptor_gold_dinov2_s11.npz, raptor_gold_effnetv2_s1.npz, raptor_gold_r50_v1.npz, raptor_gold_r50_v1_epens.npz, raptor_gold_r50_v1_swa.npz, raptor_gold_v4base.npz, raptor_gold_v5.npz, raptor_gold_v5_swa.npz, raptor_gold_v5s1.npz, raptor_gold_v5s1_epens.npz, raptor_gold_v5s1_swa.npz, raptor_gold_v5s2_swa.npz, raptor_gold_v6.npz, raptor_gold_v6_epens.npz, raptor_gold_v6_s3_swa.npz, raptor_gold_v6_s6.npz, raptor_gold_v6_s6_epens.npz, raptor_gold_v6_s6_swa.npz, raptor_gold_v6_swa.npz, raptor_gold_v6d_s10.npz, raptor_gold_v6d_s10_epens.npz, raptor_gold_v6d_s10_swa.npz, raptor_gold_v6d_s8.npz, raptor_gold_v6d_s8_epens.npz, raptor_gold_v6d_s8_swa.npz.
- Gold shapes: pred (58,12), truth (58,12), ids (58,); truth shared across kept files: True; ids order shared: True.
- Positives per finding: ACL=24, MCL=9, Medial Meniscus=26, Lateral Meniscus=23, Medial OA=15, Lateral OA=11, PF OA=21, Effusion=35, Synovitis=27, Baker's=12, Contusion=19, Fracture=18; flag AUC unreliable if n_pos<5: none flagged.
- Teacher: teacher/c0_raptor_raw.csv (1145 rows) and teacher/c0_compact.csv (1145 rows); columns StudyInstanceUID + 12 LAB in order; loaded with usecols excluding Report; aligned to gold ids order; 58/58 gold ids matched.
- Raptor caveat: trained on alldata, may have seen gold studies; mark Raptor numbers possibly in-sample.
- Rank: per-finding percentile rankdata/58 average-method (score_blend_npz.py:63); blend = 0.7*rank(Raptor)+0.3*rank(compact); AUC via sklearn roc_auc_score per finding, macro = mean of 12.
- Bootstrap: 1000 resamples with replacement of 58 studies, seed 0; per-finding AUC of fixed blend scores; degenerate (single-class) resamples skipped; n_valid reported; CI = 2.5/97.5 percentiles.
- Rank error per study per finding: e = 1-rank if truth=1 else rank (own-model percentile ranks); worst misses = positives with largest blend e; Spearman = scipy spearmanr on e vectors (pairs among Raptor/compact/best), over positives and over all 58.
- Q4: per finding, best-ours-AUC minus blend-AUC vs CI half-width (hi-lo)/2; exceeds => report model+diff else noise. Also note 0.06 per-finding noise rule.

## Table 1: per-finding AUC (Raptor possibly in-sample)
| Finding | n_pos | Raptor | Compact | Blend 0.7R+0.3C | raptor_gold_v6d_s8.npz | raptor_gold_v6d_s8_swa.npz | raptor_gold_v6d_s8_epens.npz |
|---|---|---|---|---|---|---|---|
| ACL | 24 | 0.9645 | 0.9841 | 0.9743 | 0.9620 | 0.9620 | 0.9620 |
| MCL | 9 | 0.9433 | 0.9637 | 0.9592 | 0.9501 | 0.9478 | 0.9478 |
| Medial Meniscus | 26 | 0.9820 | 0.9645 | 0.9820 | 0.9760 | 0.9784 | 0.9784 |
| Lateral Meniscus | 23 | 0.8957 | 0.9155 | 0.9155 | 0.8907 | 0.8894 | 0.8870 |
| Medial OA | 15 | 0.9814 | 0.9721 | 0.9845 | 0.9829 | 0.9829 | 0.9829 |
| Lateral OA | 11 | 0.7756 | 0.8288 | 0.7950 | 0.8317 | 0.8317 | 0.8317 |
| PF OA | 21 | 0.8597 | 0.8842 | 0.8764 | 0.8739 | 0.8726 | 0.8726 |
| Effusion | 35 | 0.9727 | 0.9590 | 0.9814 | 0.9888 | 0.9901 | 0.9901 |
| Synovitis | 27 | 0.8292 | 0.8262 | 0.8399 | 0.8399 | 0.8363 | 0.8375 |
| Baker's | 12 | 0.9764 | 0.9801 | 0.9783 | 0.9783 | 0.9746 | 0.9764 |
| Contusion | 19 | 0.9366 | 0.9568 | 0.9433 | 0.9379 | 0.9393 | 0.9379 |
| Fracture | 18 | 0.9583 | 0.8826 | 0.9458 | 0.9431 | 0.9431 | 0.9431 |
| macro | 58 | 0.9229 | 0.9265 | 0.9313 | 0.9296 | 0.9290 | 0.9290 |
- Macro gaps vs blend: best-ours -0.0017, Raptor -0.0084, compact -0.0048; all |gap|<0.02 => noise.

## Table 2: blend bootstrap 95% CI (1000 resamples, seed 0)
| Finding | n_pos | Blend AUC | CI lo | CI hi | half-width | n_valid |
|---|---|---|---|---|---|---|
| ACL | 24 | 0.9743 | 0.9280 | 1.0000 | 0.0360 | 1000 |
| MCL | 9 | 0.9592 | 0.9038 | 0.9968 | 0.0465 | 1000 |
| Medial Meniscus | 26 | 0.9820 | 0.9534 | 1.0000 | 0.0233 | 1000 |
| Lateral Meniscus | 23 | 0.9155 | 0.8278 | 0.9798 | 0.0760 | 1000 |
| Medial OA | 15 | 0.9845 | 0.9536 | 1.0000 | 0.0232 | 1000 |
| Lateral OA | 11 | 0.7950 | 0.5833 | 0.9656 | 0.1911 | 1000 |
| PF OA | 21 | 0.8764 | 0.7525 | 0.9704 | 0.1089 | 1000 |
| Effusion | 35 | 0.9814 | 0.9541 | 1.0000 | 0.0229 | 1000 |
| Synovitis | 27 | 0.8399 | 0.7167 | 0.9307 | 0.1070 | 1000 |
| Baker's | 12 | 0.9783 | 0.9325 | 1.0000 | 0.0338 | 1000 |
| Contusion | 19 | 0.9433 | 0.8855 | 0.9880 | 0.0513 | 1000 |
| Fracture | 18 | 0.9458 | 0.8743 | 0.9915 | 0.0586 | 1000 |

## Q3: 3 weakest blend findings (worst misses as study index 0-57; best model = raptor_gold_v6d_s8.npz)
### Lateral OA: blend AUC 0.7950 CI [0.5833,0.9656] n_pos 11
- Worst 5 misses (index, rank_pos 1=lowest, negatives_above, blend_err):
  - idx 50: rank_pos 3/58, negatives_above 45, blend_err 0.9483
  - idx 39: rank_pos 20/58, negatives_above 29, blend_err 0.6552
  - idx 6: rank_pos 37/58, negatives_above 13, blend_err 0.3621
  - idx 43: rank_pos 41/58, negatives_above 10, blend_err 0.2931
  - idx 18: rank_pos 46/58, negatives_above 6, blend_err 0.2069
- Spearman of rank errors over positives (11): Raptor-compact 0.954; Raptor-best 0.991; compact-best 0.941.
- Spearman of rank errors over all 58: Raptor-compact 0.810; Raptor-best 0.930; compact-best 0.860.
- Reading: |rho|<0.3 weak sharing, 0.3-0.6 partial, >0.6 shared misses (descriptive bands, not tests).
### Synovitis: blend AUC 0.8399 CI [0.7167,0.9307] n_pos 27
- Worst 5 misses (index, rank_pos 1=lowest, negatives_above, blend_err):
  - idx 10: rank_pos 9/58, negatives_above 23, blend_err 0.8448
  - idx 38: rank_pos 15/58, negatives_above 18, blend_err 0.7414
  - idx 39: rank_pos 17/58, negatives_above 17, blend_err 0.7069
  - idx 20: rank_pos 20/58, negatives_above 15, blend_err 0.6552
  - idx 2: rank_pos 23/58, negatives_above 13, blend_err 0.6034
- Spearman of rank errors over positives (27): Raptor-compact 0.633; Raptor-best 0.905; compact-best 0.574.
- Spearman of rank errors over all 58: Raptor-compact 0.739; Raptor-best 0.909; compact-best 0.717.
- Reading: |rho|<0.3 weak sharing, 0.3-0.6 partial, >0.6 shared misses (descriptive bands, not tests).
### PF OA: blend AUC 0.8764 CI [0.7525,0.9704] n_pos 21
- Worst 5 misses (index, rank_pos 1=lowest, negatives_above, blend_err):
  - idx 49: rank_pos 3/58, negatives_above 35, blend_err 0.9483
  - idx 0: rank_pos 21/58, negatives_above 18, blend_err 0.6379
  - idx 28: rank_pos 23/58, negatives_above 17, blend_err 0.6034
  - idx 47: rank_pos 28/58, negatives_above 13, blend_err 0.5172
  - idx 10: rank_pos 34/58, negatives_above 8, blend_err 0.4138
- Spearman of rank errors over positives (21): Raptor-compact 0.885; Raptor-best 0.844; compact-best 0.848.
- Spearman of rank errors over all 58: Raptor-compact 0.782; Raptor-best 0.919; compact-best 0.790.
- Reading: |rho|<0.3 weak sharing, 0.3-0.6 partial, >0.6 shared misses (descriptive bands, not tests).

## Q4: does any ours beat blend by more than CI half-width? (else noise)
| Finding | Blend | CI half-width | Best ours (file) | Best AUC | diff | verdict |
|---|---|---|---|---|---|---|
| ACL | 0.9743 | 0.0360 | raptor_gold_v6_s6_swa.npz | 0.9902 | +0.0159 | noise |
| MCL | 0.9592 | 0.0465 | raptor_gold_v5.npz | 0.9864 | +0.0272 | noise |
| Medial Meniscus | 0.9820 | 0.0233 | raptor_gold_v6d_s10_swa.npz | 0.9880 | +0.0060 | noise |
| Lateral Meniscus | 0.9155 | 0.0760 | raptor_gold_v5s2_swa.npz | 0.9093 | -0.0062 | noise |
| Medial OA | 0.9845 | 0.0232 | raptor_gold_v6_s6.npz | 0.9938 | +0.0093 | noise |
| Lateral OA | 0.7950 | 0.1911 | raptor_gold_dinov2_s11.npz | 0.8356 | +0.0406 | noise |
| PF OA | 0.8764 | 0.1089 | raptor_gold_v6d_s8.npz | 0.8739 | -0.0026 | noise |
| Effusion | 0.9814 | 0.0229 | raptor_gold_v6d_s8_swa.npz | 0.9901 | +0.0087 | noise |
| Synovitis | 0.8399 | 0.1070 | raptor_gold_v6_s3_swa.npz | 0.8602 | +0.0203 | noise |
| Baker's | 0.9783 | 0.0338 | raptor_gold_v4base.npz | 0.9946 | +0.0163 | noise |
| Contusion | 0.9433 | 0.0513 | raptor_gold_v5.npz | 0.9555 | +0.0121 | noise |
| Fracture | 0.9458 | 0.0586 | raptor_gold_v6d_s10.npz | 0.9597 | +0.0139 | noise |

## All our models macro AUC (kept 34, desc)
- raptor_gold_v6d_s8.npz: 0.9296
- raptor_gold_v6d_s8_swa.npz: 0.9290
- raptor_gold_v6d_s8_epens.npz: 0.9290
- raptor_gold_v6d_s10.npz: 0.9271
- raptor_gold_v6d_s10_swa.npz: 0.9254
- raptor_gold_v6d_s10_epens.npz: 0.9251
- raptor_gold_v6_s3_swa.npz: 0.9249
- raptor_gold_v6_s6_epens.npz: 0.9219
- raptor_gold_v6_s6_swa.npz: 0.9218
- raptor_gold_v5s2_swa.npz: 0.9215
- raptor_gold_v6_s6.npz: 0.9210
- raptor_gold_v5_swa.npz: 0.9156
- raptor_gold_v6.npz: 0.9153
- raptor_gold_v5.npz: 0.9152
- raptor_gold_dinov2_s11.npz: 0.9148
- raptor_gold_v6_epens.npz: 0.9139
- raptor_gold_v6_swa.npz: 0.9137
- raptor_gold_v5s1_epens.npz: 0.9129
- raptor_gold_v5s1_swa.npz: 0.9120
- raptor_gold_v5s1.npz: 0.9094
- raptor_gold_v4base.npz: 0.8925
- raptor_gold_convnext_s1_epens.npz: 0.8838
- raptor_gold_convnext_s1.npz: 0.8837
- raptor_gold_convnext_s1_swa.npz: 0.8837
- raptor_gold_coat224_v1.npz: 0.8823
- raptor_gold_coat224_v1_epens.npz: 0.8821
- raptor_gold_coat224_v1_swa.npz: 0.8809
- raptor_gold_convnext_s2_swa.npz: 0.8710
- raptor_gold_convnext_s2.npz: 0.8709
- raptor_gold_convnext_s2_epens.npz: 0.8703
- raptor_gold_effnetv2_s1.npz: 0.8510
- raptor_gold_r50_v1.npz: 0.8445
- raptor_gold_r50_v1_epens.npz: 0.8438
- raptor_gold_r50_v1_swa.npz: 0.8430

## Checks not done / NOT VERIFIED
- NOT VERIFIED: clinical cause of misses; image-view cause; calibration; thresholds; OOF; LB.
- Did not open data/train.csv; did not print UIDs/pixels/Report text.
- CPU only; inputs read-only; outputs only in delegation/results/.
