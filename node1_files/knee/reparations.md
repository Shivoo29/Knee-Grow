I'd look at the **data first, specifically the labels**, then at **how the MRI is fed in** (laterality and plane coverage), then use the biology to explain the weak classes. The model comes last. This is based on your experiment ledger, logs and audit notes on `knee`.

## Why it isn't the model

- **Every model lands in the same place.** CoAtNet, ConvNeXt, EffNetV2, MaxViT and BiLSTM, on 44, 96 and 192 slices, all score between 0.92 and 0.94. Different models with different inputs ending up at the same score points to something they all share: the labels or the preprocessing.
- **Ensembles never helped.** EXP-006, 008 and 009 are all marked NO_GAIN. Blending only helps when models make different mistakes, so yours are making the same ones.
- **192 slices didn't beat 96.** Slice count isn't the bottleneck.
- **The training labels aren't real labels.** Only 58 studies in `train.csv` have expert labels. The other ~4,350 were scored from free-text reports in Spanish, Dutch, German and other languages, using LLM extraction plus pseudo-labels. A model can't learn to be more accurate than the labels it's trained on.

## Where the score is lost

From your latest run (`train_coat_v6d24_s9.log`):

| Group | Classes | AUC |
|---|---|---|
| Already solved | ACL, MCL, Medial Meniscus, Medial OA, Effusion, Baker's, Contusion, Fracture | 0.94–0.98 |
| **Holding the average down** | **Lateral OA 0.79, Synovitis 0.85, PF OA 0.86, Lateral Meniscus 0.90** | |

Raising just those four to 0.95 puts the macro AUC around **0.963**. The other eight classes have almost nothing left to give, so any work that doesn't target these four is wasted.

## What I'd check next, in order

1. **How good are the training labels themselves? (CPU only, minutes)** *Corrected:* the label files you train on (`labels_distil_v1`, `labels_avg`, `labels_v4`, `labels_llm_soft`) contain none of the 58 expert-labelled studies, so they can't be scored directly. Score the process that produced them instead, run on those studies. For the report-derived `HYBRID` labels that's already done: 0.921 macro AUC on 74 hand-checked studies, with Synovitis at 0.80, Lateral OA at 0.87 and PF OA at 0.93 (see the follow-up section below). So the Synovitis labels are weak, but the Lateral OA and PF OA labels are not the ceiling. The labels you actually train on, `labels_distil_v1`, still need measuring. Where labels are weak, the likely cause is that radiologists leave findings out rather than writing them as negative. Lateral OA often goes unmentioned when medial OA dominates, and PF OA appears under different names in each language (e.g. "condromalacia rotuliana" in Spanish). The fix is to treat "not mentioned" as unknown and exclude it from the loss, instead of counting it as a negative.

2. **Is left/right orientation handled correctly? (CPU only)** This is my top suspect. Both lateral classes are weak and both medial classes are strong. If some left knees are mirrored wrongly, lateral and medial get swapped. Medial findings are common enough to hide the damage, lateral ones aren't. Test it by splitting the AUC by left vs right knee, and by checking whether swapping the medial and lateral predictions improves any studies. You already have `anatomical_mirror.py` and `inspect_laterality_coverage.py`, and EXP-011 had to switch off coronal flips, which also suggests flipping has caused trouble.

3. **Does each finding get the view it needs?** Your 192-slice build filled only 87% of coronal slots. Coronal is the main view for lateral-compartment OA. PF OA is best seen on axial slices, which got the smallest share. EXP-011's lateral-compartment crop already lifted Lateral OA from 0.79 to 0.85 on fold 0, which shows that targeted input works where more model capacity didn't.

4. **Can you trust the measurement?** With only 58 expert-labelled studies, your bootstrap gives ±0.015 standard deviation and a 95% range of roughly 0.88 to 0.96. Lateral OA has just 11 positives. That means you can't tell 0.928 from 0.94, and some ledger decisions may rest on noise. I'd add confidence intervals to the ledger and grow the expert set, even by 100–200 hand-checked studies focused on the four weak classes.

5. **Only then, change the model.** That means per-class crops (coronal for lateral, axial for PF), and a separate approach for Synovitis, whose positive cases look different from each other partly because the term covers several different findings.

Steps 1 and 2 cost no GPU time, so your 20h training / 10h submission budget is untouched. I can run both on `knee` now and come back with per-class numbers.

---

## Follow-up: how to check how good the training labels are

Your local project already holds a partial answer, which changes the earlier advice.

### Correction to step 1

None of the label files you train on (`labels_distil_v1`, `labels_avg`, `labels_v4`, `labels_llm_soft`) contain any of the 58 expert-labelled studies. That's deliberate, to keep the test set clean, but it means you can't score those files directly. You can only score the process that produced them, run on the expert-labelled studies.

That has partly been done already. `Knee-Grow/data/labels/README.md` scores the report-derived labels (`HYBRID_labels_scores.csv`) at **0.921 macro AUC** on a 74-study hand-checked set (the 58 expert studies plus 16 checked by reading reports).

### The key comparison: label quality vs model quality, per class

Each label score is a separate per-class AUC (`metrics.json`); only the 0.921 is a macro average. Put them next to your model's scores:

| Class | Labels (74 studies) | Model (58 studies) | Gap | What it means |
|---|---|---|---|---|
| **Lateral OA** | 0.873 | 0.793 | **−0.080** | The model gets far less out of the images than the labels contain. **Not a label problem.** |
| **PF OA** | 0.931 | 0.856 | **−0.075** | Same. |
| Lateral Meniscus | 0.905 | 0.903 | 0.000 | The model matches its labels. The labels are probably the ceiling here. |
| Synovitis | 0.804 | 0.854 | +0.050 | The model already beats its labels. Only better labels will help. |
| Effusion | 0.893 | 0.979 | +0.086 | The model overcomes the label noise. |
| Contusion | 0.890 | 0.949 | +0.059 | Same. |

This narrows things down. **For Lateral OA and PF OA, fixing the labels comes second.** The labels already carry information the model isn't using, which points back to how the images are fed in: left/right orientation, coronal and axial coverage. That's steps 2 and 3 above, and those two classes account for most of the 0.033 you're missing.

Three caveats:

- The two columns use different test sets, 74 vs 58 studies.
- The 58 expert studies were used to pick the best label source for each class, so the label scores are a bit optimistic.
- You train on `labels_distil_v1`, not `HYBRID`, and nobody has measured its quality.

### How to check the labels properly (all CPU)

1. **Score the labels you actually train on.** `labels_distil_v1` is 35% teacher model predictions plus 65% averaged pseudo-labels. Rebuild that mix for the 58 expert studies from its parts: the teacher predictions on those studies are already saved in `raptor_gold_*.npz`. Then score it class by class. Mixing in the teacher's own predictions can amplify the model's mistakes, which would show up here.

2. **Find out what kind of errors the labels make.** For each weak class, take the expert-labelled studies where label and truth disagree most, read the report, and sort each case into one of four bins:
   - **Report silent.** The finding isn't mentioned. Fix: ignore "not mentioned" in the loss instead of counting it as negative.
   - **Extraction miss.** The report says it but the extractor missed it. Fix: better prompts or a re-run.
   - **Convention mismatch.** Mismatches like small effusion vs moderate, or "suspected" findings. Fix: rules.
   - **Image-only truth.** The expert label came from the images, so no text extractor can get it. This is the true limit of text labels.

   Which bin dominates tells you the fix. That's about 50 reports, and I can translate them.

3. **Estimate the noise rate on all 4,349 training studies, not just 58.** Take the cases where the model's held-out prediction confidently contradicts the label (`purify_labels_oof.py` does this), sample about 30 per weak class, and read the reports. The share that turn out to be real label errors is your noise rate.

4. **Measure agreement between the three label sources, class by class.** The sources are stevenleehans, pilkwang and JEV. This needs no expert labels. Classes where they disagree are the noisy ones. `FINAL2_labels_with_confidence.csv` already rates each label's confidence, so check whether the model does better on high-confidence studies than low-confidence ones.

5. **Compare positive rates in the labels with the expert set.** If the labels show far fewer positives than the expert set for a class, reports are leaving that finding out, which is the "report silent" bin from check 2.

Run 1 and 2 first. They're the cheapest, and together they settle whether the labels limit Synovitis and Lateral Meniscus.
