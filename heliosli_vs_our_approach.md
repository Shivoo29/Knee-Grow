# How heliosli's model looks at a knee, and why it beats ours on the weak findings

Written 2026-10-16. Based on the R6 model's manifest (`manifest.json` in `heliosli/rsna-knee-r6-frozen-fe055406-w1`) and the training and preprocessing code embedded in the "Blend mine in" notebook (`px_cache.py`, `compact_train.py`). R6 is the ConvNeXt model in your current submission.

Where the code supports a feature but the manifest doesn't record whether R6 used it, the document says so.

---

## 1. The short story

Both approaches use the same basic idea: take three neighbouring MRI slices as one image (2.5D), run an image network over many such windows, and let an attention layer decide which windows matter for each of the 12 findings.

The difference is **what the network gets to look at, and how it's taught.**

- **Our team's models** (the Raptor recipe) look at one fixed 140 mm square cut from the **middle of each image**, at modest resolution, from a fixed number of slices per scan type. They are trained with one simple loss on one table of labels, checked only on the 58 gold studies.
- **heliosli's R6** first **finds the knee** in each scan with a model. It then cuts several **100 mm close-ups aimed at specific parts of the knee**: the kneecap, the back of the knee, the inner and outer halves. It shows these alongside the full image. It is trained with **4-fold cross-validation**, more windows per study, and extra losses aimed at ranking.

The first difference, the crops, is the main reason R6 is better on **Lateral OA, PF OA and Baker's cyst**. That's why heliosli's newest blend gives R6 80 % of the weight for Lateral OA.

---

## 2. How our models see a knee

Every study goes through the same steps (`train_knee.py`, `build_corpus96_stream.py`):

1. **Fixed slots.** Pick one series per slot and take a fixed number of slices from it. 96-slice version: Sagittal fat-sat 26, Sagittal 22, Coronal fat-sat 18, Coronal 12, Axial 18. Extra series are dropped; T1 is never used.
2. **One fixed crop.** Cut a **140 mm square around the centre of the image**, resized to 384 px (≈ 0.36 mm per pixel). The model then uses the centre 320 px, so ≈ 117 mm.
3. **Train** on 12 random windows per study, with soft labels from `labels_avg.parquet`, one weighted BCE loss, only a ±10 % brightness change as augmentation, and no flips.
4. **Choose the model** by its score on Gold-58. No folds, no out-of-fold predictions.

The crop doesn't know where the knee is. If the knee sits off-centre, part of it falls outside. Even when centred, the kneecap sits near the front edge and the compartments get only a share of the 384 pixels.

---

## 3. How heliosli's model sees a knee

### 3.1 Step 1: a model finds the knee

For every series, a segmentation model trained on the SKM-TEA knee dataset (`knee_anchor_bank_v2.py`, weights pinned by `anchor_weights_sha256` in the manifest) finds:
- the **joint centre**, the point between thigh bone and shin bone, and
- the **kneecap (patella) centre**, when it can see it.

This runs on the test scans too. Your run log shows `"anchors": {"skm_tea_joint": 1}`. If the joint can't be found, the code falls back to the image centre.

### 3.2 Step 2: close-ups are placed relative to the knee

From those points, fixed rules place **100 mm square close-ups**. Distances are in **millimetres**, using each scan's pixel spacing, so they mean the same physical distance on every scanner.

| Close-up | Where its centre goes | Used on |
|---|---|---|
| **Joint centre** | On the detected joint centre | Sagittal, axial |
| **Kneecap** | On the detected patella. If not found: **40 mm in front** of the joint, plus 25 mm up on sagittal | Sagittal, axial |
| **Back of knee** | **35 mm behind** the joint (sagittal also 5 mm down) | Sagittal, axial |
| **Inner side (medial)** | **30 mm to one side** of the joint | Coronal |
| **Outer side (lateral)** | **30 mm to the other side** | Coronal |
| **Full image** | The whole field of view, so nothing is ever out of sight | All planes |

The distances were set by hand. The author checked image montages of the 58 gold studies; a code comment on the kneecap rule says 30 mm forward still cut the patella, so they used 40.

### 3.3 The kneecap crop, step by step

A sagittal slice is a side view of the knee. The kneecap sits at the front.

**Our crop (centred on the image):**

```
  ┌────────────────────────────────┐
  │              femur             │
  │      ┌──────────────────┐      │
  │      │                  │  patella   ← near the edge, squeezed or cut
  │      │   image centre   │      │
  │      │                  │      │
  │      └──────────────────┘      │
  │              tibia             │
  └────────────────────────────────┘
```

**heliosli's kneecap crop (centred on the patella):**

```
  ┌────────────────────────────────┐
  │              femur   ┌───────────────┐
  │                      │    patella    │  ← in the middle, full size
  │       joint ●        │   trochlea    │
  │                      └───────────────┘
  │              tibia             │
  └────────────────────────────────┘
        ● = joint centre found by the model
```

1. The model finds the joint centre (●) and, if it can, the patella.
2. The 100 mm square is centred on the patella, or 40 mm forward and 25 mm up from ●.
3. That square is resized to 384 px. **100 mm over 384 px ≈ 0.26 mm per pixel**, about 40 % sharper than our ≈ 0.36 mm.
4. The model sees the kneecap and the groove it slides in (the trochlea) whole and in the middle, which is exactly where PF OA (cartilage wear between the two) happens.

On axial slices (top-down) the same thing happens with 40 mm forward. Coronal slices (front view) get no kneecap crop, because the kneecap only appears in a few front slices there. They get the inner and outer side crops instead.

### 3.4 Which crop helps which finding

| Finding | Crop that helps | Why |
|---|---|---|
| **PF OA** | Kneecap (sagittal + axial) | Patellar and trochlear cartilage seen whole, at higher resolution |
| **Lateral OA, Medial OA** | Inner / outer side (coronal) | Each half of the joint gets its own sharp image; thin cartilage is easier to judge |
| **Lateral / Medial meniscus** | Inner / outer side (coronal) + joint centre (sagittal) | Same reason; the menisci sit between the bones on each side |
| **Baker's cyst** | Back of knee (sagittal + axial) | The cyst sits behind the knee |
| **ACL, PCL** | Joint centre (sagittal); back of knee for PCL | The cruciate ligaments run through the middle of the joint |
| **Effusion, Synovitis** | Full image + kneecap crop | Fluid collects above and around the kneecap (suprapatellar pouch) |

---

## 4. Everything else heliosli does differently

| Area | Our models | heliosli's R6 | Recorded in R6's manifest? |
|---|---|---|---|
| **What the model sees** | One 140 mm image-centred crop | Full image + 100 mm knee-anchored close-ups (view set `v4`) | ✅ `views: v4` |
| **Which series** | Fixed slots per scan type; extras dropped | Every series, after removing exact duplicates | From code |
| **Thin scans** | Treated like any other | Slices thinner than 1.75 mm averaged into 3.5 mm slabs, so every series looks alike | ✅ `phys_mm: 3.5` |
| **Backbone** | CoAtNet-2 (ImageNet), fully trained | ConvNeXt-small (ImageNet-22k), **first 16 blocks frozen** | ✅ |
| **Windows per study in training** | 12 | **176** | ✅ `train_windows: 176` |
| **Image size** | 384, centre 320 at inference | 384 with a **random 352 crop** in training (a small shift) | ✅ `crop: 352` |
| **Validation** | Gold-58 only | **4-fold cross-validation**, out-of-fold predictions for every training study, Gold-58 never trained on or read by the trainer | ✅ `folds: 4`, `gold_in_training: 0` |
| **Labels** | `labels_avg.parquet`: average of 5 public tables, yes/no-style soft labels | Their own report-derived table (`probe_targets_opusv11b…`) that includes **graded categories** per finding, e.g. Baker's: none / small / moderate / large / ruptured | Target file name ✅; categories from code and `state_spec.json` |
| **Extra model output** | 12 logits | 12 logits + a **"state" head** predicting those graded categories | From code |
| **Ranking loss** | None | **Matched-pair ranking**: a positive case trained against a hard negative with the same other findings, plus grade pairs (moderate/large over none/small) | Supported in code; whether R6 had it on isn't recorded |
| **Augmentation** | ±10 % brightness | **Anatomical mirror with label swap** (coronal/axial flipped left-right, sagittal slice order reversed, inner and outer crops swapped), position jitter, gamma/contrast | Supported in code (v9.4–9.5); not recorded |
| **External data** | None (but the CoAtNets you blend with used OAI) | Code supports a **fastMRI+ lesion-box side task**; R6's run name `cnx_r6fm` suggests it was on | Not recorded. **Licence risk**, see section 6 |
| **Extra small models** | None | Fluid-volume model for Effusion; graded-state model for Baker's; left/right knee classifier | ✅ |
| **Prediction-time flip** | Mirror TTA in some kernels | Mirror TTA with medial/lateral outputs swapped back (MCL from the plain pass) | In the notebook (`FLIP_TTA = True`) |

---

## 5. Why this makes R6 better on the weak findings

**1. More pixels where the finding is.** OA is judged on cartilage only 2–3 mm thick. Our crop gives ≈ 0.36 mm per pixel over the whole knee. R6's close-ups give ≈ 0.26 mm per pixel over just the region that matters, and they always contain it. For PF OA, our crop often puts the kneecap at the edge, as the author noticed. R6's kneecap crop puts it in the middle.

**2. The knee is found, not assumed.** Our crop trusts that the knee is in the middle of the image. R6 finds it first, so off-centre knees don't lose a compartment.

**3. A fair comparison of the two sides.** Coronal close-ups of the inner and outer halves give Medial and Lateral OA (and the two menisci) separate, equally sharp views. Mirror augmentation swaps them along with their labels, which teaches the model what "lateral" means rather than where it usually appears in the image.

**4. Training that matches the metric and the labels.**
- The score is AUC, which only cares about ranking. A matched-pair ranking loss trains exactly that.
- Graded categories ("small" vs "large" Baker's) carry more information than a single yes/no.
- 4 folds give honest out-of-fold scores, so model choices aren't based on 58 studies alone.
- 176 windows per study in training cover much more of each scan than 12.

**The evidence it works:**
- heliosli's per-finding blend gives R6 **80 % of Lateral OA** and **65 % of Baker's**, the findings the crops target.
- Our CoAtNets score only **0.76–0.79** on Lateral OA on Gold-58, their worst finding, in every run.
- Adding R6 next to nartaa's CoAtNets raised the public score from **0.949 to 0.956**, the biggest single jump on the public board.

**Where R6 is *not* better:** Synovitis. The blend gives the CoAtNets 80 % there, likely because they were trained with OAI synovitis labels and see the whole joint. Close-up crops don't help a finding that is spread around the joint.

---

## 6. Caveats

- **Corrections to earlier notes.**
  - I said R6 uses a self-supervised MRI backbone ("OFL"). Its manifest says **ConvNeXt-small initialised from ImageNet-22k**. The OrthoFoundation-L (OFL) self-supervised encoder is an *option* in the trainer, and its weights are attached to the notebook, but R6 doesn't use it as its backbone.
  - Earlier docs described the kneecap crop as fixed image fractions. That was the older `v2/v3` rule; R6 uses the knee-anchored `v4` version described here.
- **fastMRI.** The trainer can add a fastMRI+ lesion-box task, and R6's run name suggests it was used. fastMRI needs an application and data-sharing agreement, which the hosts' rule ("institutional research agreement … clearly do not" qualify) likely excludes. Using R6 as public weights is covered by the hosts' Oct 14 ruling, but **if you train your own model, don't copy the fastMRI part.**
- **Knee locator.** The anchor model is trained on SKM-TEA, and the attached dataset also contains a KneeXNet localizer (OAI-derived, no clear licence). For your own training, the simpler `knee_anchor()` function in `px_cache.py` finds the knee from image brightness alone, needs no external model, and avoids both questions. The author notes it was less accurate.
- Not everything in the trainer was necessarily switched on for R6. Rows marked "not recorded" in section 4 are possible, not confirmed.

---

## 7. What we could copy, in order of value

1. **Knee-anchored close-ups** (`v4` views) on top of our existing 96-slice corpus. Use `knee_anchor()` for the joint centre, then the same mm offsets: kneecap +40 mm forward / 25 mm up, back +35 mm, sides ±30 mm. Biggest expected gain for Lateral OA and PF OA.
2. **Anatomical mirror augmentation with label swap** during training. Cheap, and it fits the medial/lateral findings.
3. **4-fold training with out-of-fold predictions**, so choices stop depending on 58 studies, and label cleaning (first-principles file, Tier 1) becomes possible.
4. **More windows per study** in training (from 12 towards 100+), memory permitting.
5. **A ranking loss** next to BCE, and graded sub-labels from the reports for findings with grades (OA, Effusion, Baker's).

heliosli's full pipeline (`native_cache.py` → `px_cache.py` → `compact_train.py`) is public inside the notebook and the OrthoFoundation dataset. It has hard-coded Windows paths and is built for their machines, so adapting the ideas to your team's code is likely faster than running theirs as-is.
