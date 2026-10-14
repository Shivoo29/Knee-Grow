# RSNA Knee: where we are (Oct 12, 2026)

Short version for humans. `chat.md` is the long log for AI handoff.
Every number here was measured by us or read from source code/LB, unless marked "not verified".

## Scoreboard

| | Score |
|---|---|
| Our best LB | **0.954** (base954, public notebook fork) |
| Top 10 cutoff | 0.963 |
| #1 | 0.966 |
| Our own models alone | 0.941 (CoAtNet version 5, 44-slice, seeds 0+1+2 averaged) |

Deadlines: merge Oct 15 (not merging), final **Oct 22**. 5 submissions/day, quota resets 5:30 AM IST.

## What went wrong before Oct 12

- Everything was blended with an old public notebook, `pjmathematician/rsna-knee-d4-blend` (LB 0.943). Whatever we added, the score stayed 0.943.
  It is a big multi-stage ensemble (from its own notebook text and attached datasets):
  1. **DINOv2-small** (frozen) + 20 small per-member heads.
  2. **"A5" attention pooling** over slice features, 5 folds.
  3. **ResNet-50 pretrained on RadImageNet** with three head layouts (E10/E13/E11).
  4. **Raptor CoAtNet** by dreaddevelopment, run as "four Raptor views" (notebook's words); weights come from 3 datasets: raptor-knee-maxspan, -native384, -native384dense.
  5. **Four extra CoAtNet readers** by mattiaangeli:
     - residual-gated
     - D4 depth-zone
     - Global96
     - Repair-v1

     Averaged 25% each, then blended 40% into Raptor.
- Many "breakthroughs" were noise on the 58 gold studies (±0.02 macro, ±0.06 per finding).
- Real bugs:
  - Submission V3 fed CoAtNet version 5 seed 2 the wrong input.
  - One submission timed out at 9 h.
  - A fluid-flag parsing bug made inference pick different series than training did.

## What we did today (Oct 12)

1. **Found a better base.** The public notebook `heliosli/rsna-knee-blend-gold-dataset-restored` is a 70/30 rank blend of two models:
   - Raptor CoAtNet by nartaa (0.949, trained with OAI data)
   - ConvNeXt-small compact model by Heliosli (4 folds)

   Forked it unchanged and submitted: **0.954 confirmed.**
2. **Audited all 19 submissions.** Our best own result is 0.941, a 3-seed blend. A single model never beat 0.936 on the LB.
3. **Label research:** scored every public label table on gold-58.
   - Best: JEV hybrid at 0.904.
   - Bigger LLMs do not help: Qwen-32B 0.869, GPT-5.6 0.849, Opus-4.5 0.892.
   - Report-based labels top out at about 0.90, so we dropped the "build a better LLM labeller" plan.
4. **Found a forgotten model.** CoAtNet version 6 (96-slice) seed 3 (gold 0.925) was never uploaded; uploaded now. Trained new seeds on node1:
   - CoAtNet version 6 seed 4 = **0.927**, our best single model.
   - CoAtNet version 6 seed 5 is still training.
5. **Fixed a real bug.** In `rsna_preproc.py`, `x or -1` turned Fluid_Sensitive = 0 into -1. Inference then picked different series than training for about 5% of studies. GLM-5.3 (zcode) found it; we fixed it.
6. **Read the Kaggle forum** via `kaggle competitions topics`.
   - Using LLM APIs on the reports is allowed.
   - KneeCoT is banned.
   - **OAI has no ruling yet.** Raptor uses OAI data, so we must keep one OAI-free final pick.
7. **Tested eval tricks.**
   - Mirror TTA: no gain.
   - Lower resolution (320): impossible with our CoAtNet.
8. **Built the fleet notebook** `shivoo29/rsna-knee-base954-fleet`:
   - Formula: 0.85 × base954 + 0.15 × our fleet (CoAtNet version 5 seeds 0+1+2; CoAtNet version 6 seeds 0, 3, 4).
   - Uses only scripts whose LB scores are proven.
   - Time guard so it cannot hit the 9 h limit.
   - Test run in progress.

## Next steps

| When | What | Why |
|---|---|---|
| Tonight | Submit fleet notebook when its test run passes | Does our fleet add anything on top of 0.954? |
| Oct 13 | If it helps, try fleet weight 0.25; add CoAtNet version 6 seed 5 | GitHub JunhaoLiXD/RSNA_Knee_Abnormality_Detection reports 0.950 after adding its own model to a public notebook (their README; not verified by us) |
| Oct 13–15 | **Distillation:** run base954 on the train studies, mix its predictions 50/50 with the labels, train a new student on node1 | Labels top out at ~0.90; the base954 ensemble is better than that. Our biggest remaining lever |
| Oct 16–20 | Add the student to the fleet if its gold-58 score beats 0.927 | |
| Oct 21–22 | Pick 2 finals: (1) best LB, (2) an **OAI-free** safe one | In case OAI is banned |

## Honest expectation

- Fleet: about 0.954 → 0.955–0.957.
- Top 10 (0.963) needs distillation to work. Not guaranteed.

## Model names

- **CoAtNet version 5:** 44 slices per study, labels = mean of 5 public label tables.
- **CoAtNet version 6:** 96 slices per study (sagittal 48, coronal 30, axial 18), same labels.
- File names are short: `v6_s4` = CoAtNet version 6 seed 4.

## Rules we follow

- Never paste report text or images anywhere.
- Gold-58 differences under ~0.02 are noise.
- Every result is compared against a baseline.
- No co-author lines in commits. The user pushes to GitHub.
- node1: full permission (since Oct 12). After the comp, push everything to GitHub and clear `~/projects/knee`.

## Update Oct 13 evening
- base954 + old fleet scored 0.953 (base954 alone 0.954): our label-trained models add nothing.
- Teacher = base954 run on all train studies (4 Kaggle chunks), distilled labels = 0.6 teacher + 0.4 report labels (0.3/0.7 for Lateral OA, PF OA).
- Distilled students (CoAtNet version 6, 96 slices): seed 7 best gold-58 0.9279, seed 8 best 0.9296 (previous best seed 0.9273). Gold-58 cannot separate these from noise; leaderboard decides.
- Submitted: students only, and 0.85 base954 + 0.15 students (Kaggle descriptions are swapped, notebook names are correct). Scores pending.
- Running: 24-epoch student (node1), student seed 10 (PC). Plan B if students do not help: rank-blend base954 with other strong public notebooks; final picks = best LB + an OAI-free safe one.

## Update Oct 14 evening: pivot to OOF relabelling
- LB: R7 + Raptor fork 0.956 (best), R7 + label-trained CoAtNets 0.952 (OAI-free), R7 alone 0.949.
- Forum (hengck23, varshitha, datnt): gold labels are radiologists reading IMAGES, not reports (report labels cap ~0.90);
  public LB = ~390 studies, noise ~0.003; single models reach 0.960-0.961 with good labels.
- Gold-58 measurement: report labels 0.899, image teacher 0.932, **0.6 image + 0.4 report labels 0.941**;
  our own label-trained CoAtNet 0.921 -> 0.939 when mixed the same way. Image + report information is complementary.
- Problem: teacher predictions on TRAIN studies are in-sample, so distillation targets were mostly report labels again.
- Plan: 4-fold OOF image predictions on all train studies (report labels only, T05 recipe) -> targets =
  mix(OOF image, report labels) -> train final models on them -> blend with R7 + Raptor. PC runs folds 0,1 now;
  node1 folds 2,3 (`pc_code/run_node1_oof.sh 2 3`).
