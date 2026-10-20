# Final Rulebook — Knee MRI Report Labeling (12-Label Schema)

**Version:** Final (consolidates deepseek_rulebook, rulebook, rulebook_gemini, master_clinical_rulebook_calibration_log, gpt_rulebook, Knee_MRI_Annotation_Rulebook_v4, and the effusion–synovitis agreement analysis)
**Purpose:** Teach a model *how* the target dataset turns report text into labels, so it can label reports it has never seen.
**Scope:** This is a reverse-engineered, dataset-specific labeling standard, not a clinical diagnostic guideline.

---

## 0. How to Read This Rulebook

This document contains **rules, not examples**. Each rule describes a mechanism: what kind of textual evidence moves a label, in which direction, and how strongly. Apply the mechanism to whatever words the report uses.

Three ideas run through every rule:

1. **The dataset rewards named diagnoses, not inferred ones.** A label turns on when the report *names* the condition, or when a finding reaches a defined severity threshold. It does not turn on because a reader could infer the condition from descriptive signs.
2. **The final summary is where the radiologist decides what matters.** The Impression/Conclusion is the strongest evidence. A finding left out of a present Impression is usually treated as not label-worthy, with fluid findings as the main exception.
3. **A soft score is a confidence that the dataset label is 1, not a severity.** A mild but explicitly named finding can score high. A severe-sounding but unnamed finding can score low.

When two rules seem to conflict, use the precedence order in Section 2.3.

---

## 1. The 12 Labels

Return exactly these labels in this order:

| # | Label | Meaning in this schema |
|---|---|---|
| 1 | `ACL` | Anterior cruciate ligament tear (grade 2 or higher) |
| 2 | `MCL` | Medial collateral ligament tear (grade 2 or higher) |
| 3 | `Medial Meniscus` | Definite surfacing tear of the medial meniscus |
| 4 | `Lateral Meniscus` | Definite surfacing tear of the lateral meniscus |
| 5 | `Medial OA` | Osteoarthritis / high-grade cartilage disease, medial tibiofemoral compartment |
| 6 | `Lateral OA` | Same, lateral tibiofemoral compartment |
| 7 | `PF OA` | Same, patellofemoral compartment (patella and/or trochlea) |
| 8 | `Effusion` | Joint effusion / excess intra-articular fluid |
| 9 | `Synovitis` | Inflammation or thickening of the synovium |
| 10 | `Baker's` | Baker's (popliteal) cyst |
| 11 | `Contusion` | Bone bruise / named trauma contusion |
| 12 | `Fracture` | Structural or insufficiency fracture, or bony avulsion |

Everything else is **out of schema**: PCL, LCL/FCL, posterolateral corner, popliteus tendon, MPFL, retinacula, posteromedial oblique ligament, tendinosis, bursitis, IT-band friction, plica, loose bodies, ganglion cysts. Out-of-schema findings never get mapped onto the nearest available label. They may only serve as *context* for a target label when a rule below says so.

---

## 2. Core Principles

### 2.1 P0 — The Explicit Diagnostic Statement Principle

This is the supreme rule. A descriptive finding is **never** upgraded to a positive label unless one of these is true:
- the report names the diagnosis (tear, osteoarthritis, synovitis, contusion, fracture, cyst, effusion), **or**
- the finding reaches a full-thickness-equivalent threshold defined in the label rules (for example, full-thickness cartilage loss for OA).

Descriptive findings include signal change, edema, thickening, contour irregularity, cartilage thinning, osteophytes, and fluid quantity words. These can *support* a named diagnosis, but they cannot *create* one.

When any lower rule seems to license an inference, P0 blocks it.

### 2.2 P1 — Impression Supremacy

- The Impression/Conclusion overrides the Findings when they disagree.
- An Impression-level **definite** statement beats a Findings-level hedge or negation.
- An Impression-level **negation** beats a Findings-level positive.
- A finding in the Findings that is **omitted from a present Impression** is suppressed to 0 for ligament, meniscus, and OA labels (see Rule R1.2).
- Fluid labels (Effusion, Baker's) are **not** suppressed by omission (see Rule R1.4).

### 2.3 Precedence Order for Conflicts

Resolve conflicts in this descending order:

1. Explicit negation in the Impression.
2. Explicit positive diagnosis in the Impression.
3. Threshold rules (grade floors, severity floors, grade ceilings) applied to whatever survives steps 1–2.
4. Explicit statements in the Findings (subject to suppression rules).
5. Context or co-occurrence priors (soft scores only, never a hard 1).
6. Default: 0, with a low soft score.

An inference from an unqualified descriptive finding never outranks an explicit statement anywhere in the report.

### 2.4 P2 — Label Noise Exists; Do Not Learn From Single Anomalies

The ground truth contains cells that contradict the report text and contradict each other. The rule for handling them:

- Never build a rule from a single contradictory label.
- When a report pattern matches a known noisy area (Section 11), follow the rules, lower confidence, and raise a flag.
- Do not invent a clinical explanation to make a contradictory label look consistent.

---

## 3. Per-Report Procedure

1. **Detect language and structure.** Find the Impression/Conclusion (or equivalent header, numbered tail list, `--` delimiter, or repeated diagnostic tail). Note whether an Impression exists at all.
2. **Segment into sentences and attach each statement to its anatomy, side, and compartment.** Scope negations correctly (Section 4).
3. **Extract explicit diagnostic nouns per label family** using the lexicon in Section 9.
4. **Apply structural rules R1** (supremacy, suppression, no-Impression branch, fluid exception).
5. **Apply label rules R2–R7** (thresholds, ceilings, hedges, absorption).
6. **Apply cross-label interaction rules** (Section 8).
7. **Assign soft scores** using the evidence ladders (Section 10), keeping them ordered so stronger evidence always scores higher.
8. **Emit** the vector, short evidence strings, and flags (Section 12).

---

## 4. Parsing, Negation and Scope Rules

**N1 — Negation is local.** A negation applies only to the tissue, side, and compartment it modifies. "The medial meniscus is not torn" says nothing about the lateral meniscus.

**N2 — Boilerplate negations at the top of a section are weak.** A generic "No fracture is seen" at the start of an osseous section does not cancel a specific, later, structurally described fracture. A specific statement beats a generic one about the same structure.

**N3 — Watch for unheadered tails.** When a report has no headers, check whether the last sentences are a conclusion list. Do not let a negation in the last Findings sentence ("no fracture or bone contusion") bleed into the diagnoses that follow it.

**N4 — Telegraphic reports.** When Findings and Impression are identical or nearly identical, or when the report is a short diagnostic list with no distinct sections, treat the whole text as Impression-level.

**N5 — Shared-noun coordination (ellipsis).** In constructions such as "tear of the medial and lateral menisci" or "rotura de menisco interno y lateral", the noun applies to **both** coordinated items when the grammar carries it. Resolve semantically to both sides. Because annotators have sometimes dropped the first coordinated item, the first item gets a slight confidence reduction (about 0.80) rather than a full positive, and the cell gets a flag.

**N6 — Count duplicates once.** A finding repeated in Findings and Impression is one piece of evidence, not two.

**N7 — Dictation errors.** Resolve obvious misspellings and anatomical slips using surrounding anatomy and the Impression. For example, "medial cruciate ligament" next to an Impression naming the MCL refers to the MCL and must not trigger ACL.

**N8 — Hedge vocabulary.** "Suspected", "possible", "probable", "cannot exclude", "rule out (R/O)", "may correspond to", "suggestive of", "versus" are hedges. Their effect depends on the label family. They **kill** meniscus and ligament positives but **do not kill** fracture positives (see R3.3 and R6.3).

---

## 5. R1 — Report Structure Rules

**R1.1 — Impression supremacy.** Where Impression and Findings disagree, code from the Impression.

**R1.2 — Suppression of omitted findings.** A ligament, meniscus, or OA finding stated in the Findings but omitted from a *present* Impression codes 0 (soft score 0.15–0.30). Suppression applies to the label: an Impression line naming only "focal chondrosis" does not rescue an OA column for a compartment that the Findings called osteoarthritic.

**R1.3 — No-Impression branch.** If no Impression/Conclusion exists, the Findings stand on their own and are coded under R2–R7 without suppression.

**R1.4 — Fluid exception.** Effusion and Baker's cyst code from wherever they appear, subject to the severity ladder in R5.1 and the mention rule in R7.1. This exception is specific to fluid labels. It is not general leniency for the whole report.

---

## 6. Label Rules — OA, Menisci, Ligaments

### R2 — Compartment Osteoarthritis (Medial OA, Lateral OA, PF OA)

The OA labels are **threshold labels**: early cartilage disease does not count as OA.

**Compartment mapping:**
- Medial OA: medial femoral condyle, medial tibial plateau
- Lateral OA: lateral femoral condyle, lateral tibial plateau
- PF OA: patella (facets) and femoral trochlea

**R2.1 — Positive triggers (evidence must be in that compartment):**
1. Explicit "osteoarthritis", "arthrosis", "arthrotic/degenerative arthritic changes" for that compartment, carried into or stated in the Impression (or in the Findings of a no-Impression report).
2. Full-thickness cartilage loss, cartilage denudation, chondral ulcer, erosion, or exposed subchondral bone.
3. Grade 3–4 chondropathy (Outerbridge/ICRS) or grade IV chondromalacia.
4. A condylar or trochlear/patellar osteochondral defect or osteochondritis dissecans involving the joint surface.
5. Deep or delaminating fissures that are explicitly highlighted in the Impression.
6. Cartilage findings that the report itself folds into a named OA context (for example, "within the aforementioned osteoarthritic changes"), even at grade III.

**R2.2 — Grade ceiling (codes 0):** chondrosis or chondromalacia grade I–III without an OA statement or OA framing, including the words "mild", "moderate", "focal", "high-grade", "more than 50% thickness loss", "thinning", "heterogeneity", "fibrillation", or "low-grade fissuring". Soft score 0.20–0.40 depending on how close the wording is to the threshold.

**R2.3 — Osseous hallmarks are supporters, not triggers.** Osteophytes, subchondral cysts, and subchondral bone marrow edema **support** a compartment that already has cartilage-loss evidence or an OA statement. On their own, they code 0 (soft 0.35–0.50). When they accompany grade-ceiling cartilage disease in the same compartment, raise the soft score to about 0.50–0.60, but keep the binary call at 0 unless an OA statement exists.

**R2.4 — Global / tricompartmental statements.**
- If the Impression says "tricompartmental OA/chondrosis" **and** the Findings break down each compartment, verify each compartment separately. A compartment whose Findings show only grade-ceiling disease codes 0.
- If the Impression explicitly says "OA of all three compartments" (or "tricompartmental OA") **and** the Findings contain no compartment-by-compartment refutation, all three compartments code 1. This holds even with qualifiers such as "incipient", "early", or "mild", because the OA noun has been distributed to every compartment.
- "Tricompartmental chondrosis/chondropathy" without an OA noun distributes only chondrosis, so each compartment must still pass R2.1 by itself.

**R2.5 — Non-OA mimics (code 0):**
- Joint-space narrowing explained by meniscal extrusion or fracture, without cartilage or osteophyte prose.
- "Knee arthritis" that describes only synovial thickening or effusion (an inflammatory pattern) with no cartilage loss.
- Patellar or trochlear dysplasia (Wiberg types, hypoplastic trochlea) on its own.
- Focal chondral thinning at a fracture or impaction site (this belongs to the trauma, not to OA).
- A general degenerative context without compartment-specific cartilage prose.
- Osgood–Schlatter / tibial tuberosity sequelae (extra-capsular).

**R2.6 — Single-compartment telegraphic phrases.** An unqualified compartment chondropathy named in the Impression of a short telegraphic report (for example, "patellar chondropathy" as a standalone diagnosis) is promoted to that compartment's OA label (soft 0.75–0.85), because in telegraphic reports a named entity in the summary signals clinical significance.

### R3 — Menisci (Medial Meniscus, Lateral Meniscus)

**R3.1 — Positive triggers.** A definite tear noun attached to surfacing morphology codes 1: complete, horizontal, radial, vertical/longitudinal, oblique, bucket-handle, flap, root, complex, or a ramp lesion named as a definite finding. A tear that reaches an articular surface, or any "grade 3" tear, is surfacing. The Spanish terms "rotura" and "desgarro" are tear nouns, and so is "fisura" when it is used for the meniscus.

**R3.2 — Negatives (code 0):**
- Intrasubstance or degenerative signal "without surfacing tear", mucoid degeneration, myxoid change, fibrillation, Stoller grade 1–2 (including 2a), "sin caracteres propios de rotura".
- "Lesion" with a grade 1–2 description or explicitly not reaching the articular surface. The word "lesion" alone is not a tear noun.
- Truncation or blunting of the free edge, or post-meniscectomy shape change ("amputación marginal"), without a tear noun.
- Extrusion, maceration, or abnormal shape without a tear noun.
- Meniscocapsular strain or peripheral junction edema with preserved morphology.

**R3.3 — Suspicion kills meniscus positives.** A hedge without a definite tear noun or documented surface contact codes 0, **even when it appears in the Impression** (for example, "R/O ramp lesion", "suspected meniscocapsular separation", "root not clearly delineated"). Soft score 0.30–0.45.

**R3.4 — Suppression and the no-Impression branch.** These follow R1.2 and R1.3 exactly. A definite tear found only in the Findings, in a report whose Impression omits it, codes 0 (soft 0.25). In a report with no Impression, a definite tear in the Findings codes 1.

**R3.5 — Side assignment.** Assign each horn (anterior, body, posterior) and the root to the meniscus named. A tear from an out-of-schema structure (for example, a popliteomeniscal fascicle) does not by itself create a meniscus positive.

### R4 — Ligaments (ACL, MCL)

**R4.1 — Severity ladder:**

| Description | Label | Soft score |
|---|---|---|
| Complete tear, full rupture, grade 3, chronic complete tear, non-visualized ligament | 1 | 0.92–0.98 |
| Partial tear of moderate/high grade, grade 2 sprain | 1 | 0.85–0.92 |
| Structural instability attributed to the ligament (for example, anterior tibial translation for ACL) | 1 | 0.80–0.90 |
| Impression "consistent with / compatible with tear" | 1 | 0.85–0.90 |
| Unqualified "partial tear" (no grade) | 1 | 0.65–0.75 |
| Grade 1 sprain, "low-grade sprain", "low-grade partial tear" | 0 | 0.10–0.25 |
| Interstitial signal or "interstitial tear" with intact contour/fibres and no instability | 0 | 0.15–0.30 |
| Mucoid/myxoid degeneration, periligamentous edema, enthesopathic thickening | 0 | 0.05–0.15 |
| Hedged intrasubstance rupture ("may correspond to a small rupture") | 0 | 0.15–0.25 |
| Explicitly intact or normal | 0 | 0.00–0.05 |

**R4.2 — Intact-contour override.** When the report describes a low-grade injury and also states that the contour or fibres are normal or continuous, or that there is no translation, the ligament codes 0 even if the word "tear" appears.

**R4.3 — No-bleed perimeter (absolute).** PCL, LCL/FCL, posterolateral corner, popliteus, MPFL, retinacula, and the posteromedial oblique ligament never feed ACL or MCL. In particular:
- "Lateral collateral" (English LCL, Spanish *ligamento colateral externo / lateral externo*) never maps to MCL.
- Spanish *ligamento colateral interno* / *lateral interno* (LCI, LLI) and *ligamento colateral medial* (LCM) **are** the MCL.
- PCL (Spanish LCP, German HKB) never maps to ACL.

**R4.4 — Impression over Findings hedge.** If the Findings hedge ("suggestive of complete tear") and the Impression affirms ("complete ACL tear"), code 1.

## 7. Label Rules — Fluid, Bone, Cyst

### R5 — Effusion and Synovitis

These are **two independent columns**. Effusion is the most permissive label in the schema and Synovitis is the most restrictive. They are linked only by the dual triggers in R5.4. Co-occurrence alone is never a hard link.

#### R5.1 — Effusion severity × location ladder

| Evidence | Label | Soft score |
|---|---|---|
| Effusion in the Impression at any severity, including "small" or "minimal" | 1 | 0.88–0.97 |
| Moderate, large, extensive, or massive effusion anywhere | 1 | 0.88–0.97 |
| Hemarthrosis or lipohemarthrosis anywhere | 1 | 0.90–0.97 |
| Diffuse/widespread intracapsular fluid (joint plus suprapatellar recess) | 1 | 0.80–0.90 |
| Small/mild effusion in the Findings only, with an Impression that omits it | boundary | 0.45–0.65 |
| Small/mild effusion in the Findings of a report with no Impression | 1 | 0.75–0.85 |
| Trace, minimal, or "physiological" fluid in the Findings only | 0 | 0.10–0.25 |
| "No effusion", "no significant effusion" | 0 | 0.00–0.05 |

Effusion described in the Findings but replaced in the Impression by a different fluid-or-lining diagnosis (for example, only "synovitis") is a conflict cell: score it 0.40–0.55 and raise a flag.

#### R5.2 — Synovitis positive triggers (explicit synovial prose)

Synovitis requires prose that names the synovium or the disease: "synovitis", "reactive synovitis", "synovial thickening", "synovial hypertrophy", "synovial proliferation", "Reizsynovialitis", "sinovitis", or a joint phrase such as "effusion with synovitis".

#### R5.3 — Synovitis evidence ladder (keep this ordering)

| Evidence | Soft score |
|---|---|
| Explicit synovial prose in the Impression, chronic or reactive framing, or paired with effusion | 0.85–0.95 |
| Hemarthrosis, or haematoma at any site (dual trigger, R5.4) | 0.75–0.88 |
| Hoffa's fat-pad inflammation or impingement plus synovial thickening, named in the Impression | 0.80–0.88 |
| Hoffa's fat-pad inflammation or impingement named in the Impression without synovial prose | 0.60–0.70 |
| Explicit synovial prose in the Findings only (Impression present but silent) | 0.55–0.65 |
| Short boilerplate "mild synovitis" in an acute-trauma template report without chronic disease or hemarthrosis | 0.40–0.50 |
| No synovial prose, but effusion = 1 together with Baker's cyst plus OA, or with an acute tear plus bursitis | 0.45–0.55 |
| No synovial prose, effusion = 1, no supporting context | 0.30–0.40 |
| No synovial prose, effusion = 0 | 0.10–0.20 |
| Explicit "no synovitis" / "no synovial thickening" | 0.00–0.05 |

Binary threshold for Synovitis: call 1 only from the explicit-prose rows and the dual-trigger rows. The context rows exist only to order cases correctly in the soft score.

#### R5.4 — Dual triggers
- **Hemarthrosis** → Effusion 1 and Synovitis high (soft, see the ladder).
- **Haematoma at any site**, including an extra-articular haematoma at a ligament tear → Contusion 1, Synovitis high, and Effusion 1 if not already set.
- **Diffuse widespread intracapsular fluid** with acute internal derangement → Effusion 1, Synovitis moderately high (about 0.70).

#### R5.5 — Synovitis inference is forbidden as a hard call
- Effusion in a degenerative/OA context without synovial prose: Synovitis binary 0.
- Periarticular, prepatellar, infrapatellar, or subcutaneous soft-tissue edema: not synovitis.
- **Effusion is a weak predictor of Synovitis.** In a larger sample, the two columns agreed in only about 69% of studies. P(Synovitis | Effusion = 1) was about 0.6 and P(Synovitis | Effusion = 0) was about 0.2. Using the effusion score as the synovitis score ranked worse than a dedicated synovitis score (AUC about 0.57). Effusion may nudge the Synovitis score but must never dominate it.

#### R5.6 — Synovitis mimics (code 0)
Synovial cyst, plica, ganglion, bursitis (prepatellar, suprapatellar, pes anserine), Hoffa's fat-pad edema described only as edema, PVNS / tenosynovial giant cell tumour (a separate entity), and a joint "arthritis" phrase without synovial prose.

### R6 — Bone Trauma: Contusion and Fracture

Both bone columns are **noun-gated**: they code on what the report *calls* the finding, never on what the finding looks like.

#### R6.1 — Contusion positives (closed noun list)
"bone bruise", "bone contusion", "contusion" (as a positive diagnosis), "traumatic bone marrow edema", "haematoma/hematoma" at any site, or a named trauma pattern (pivot-shift, kissing contusion, impaction-bruise pattern named as contusion). Soft score 0.85–0.95.
- A positive "contusion" diagnosis in the Impression overrides a Findings-level "no bone contusion", even when the contusion is attributed to a non-osseous tissue such as muscle (soft 0.70–0.80, because this tissue-agnostic behaviour is only moderately supported).

#### R6.2 — Contusion negatives (inference blocked)
- Plain "bone marrow edema" or focal T2 signal **without** a trauma noun codes 0, even in a trauma workup and even when it is listed in the Impression (soft 0.40–0.55, boundary flag).
- Edema absorbed into an osteochondral impaction complex codes 0.
- Edema framed as degenerative, chondropathy-related, or subchondral beneath a cartilage defect codes 0 (it is absorbed into OA, see 8.1).
- Edema bound to a fracture ("adjacent bone marrow edema", "surrounding edema") codes 0 for Contusion (it is absorbed into Fracture).
- Soft-tissue, subcutaneous, or muscle *edema* (as opposed to a named contusion or haematoma) codes 0.

#### R6.3 — Fracture positives
A described fracture line or a structural cortical/trabecular fracture; a subchondral insufficiency fracture (SIFK) or "insufficiency fracture"; a named subchondral fracture; a bony avulsion (including a Segond-type avulsion fragment or *knöcherner Ausriss*); or any definite fracture noun in the Impression.
**Suspicion survives for fractures:** "in favour of subchondral fracture", or an Impression "R/O fracture" with supporting Findings prose, still codes 1 (soft 0.70–0.80). This deliberately reverses the meniscus hedge rule.

#### R6.4 — Fracture negatives (osteochondral / impaction family)
- "Osteochondral fracture", osteochondral defect, or osteochondral impaction **without** a displaced bony fragment or a fracture line through bone: Fracture 0. The finding belongs to the cartilage/OA family (or to nothing, if it is acute).
- Impaction nouns such as "subchondral impression", "impression injury", or "impaction-depression of the articular surface": Fracture 0.
- An explicit "no acute fracture" paired with only an impaction injury: Fracture 0.
- Tibial-tuberosity ossicles (Osgood–Schlatter): Fracture 0.
- Small ossicles or flakes next to an avulsion site whose bony origin is not stated are a boundary case (soft 0.45–0.60, first-instance flag).

#### R6.5 — Contusion and Fracture are separate columns
A report can be positive for both. Never convert a named contusion into a fracture, or a named fracture into a contusion.

### R7 — Baker's Cyst

**R7.1 — Mention rule.** Any positive mention of a Baker's or popliteal cyst (Baker cyst, popliteal cyst, quiste poplíteo, Baker-Zyste, Bakerova/poplitealna cista, Бейкерова киста, Baker kisti, Bakercyste) codes 1, at any size and in any section.

**R7.2 — Confidence by size and placement:**

| Evidence | Soft score |
|---|---|
| In the Impression, any size; or moderate/large anywhere; or ruptured/dissecting | 0.90–0.97 |
| Small cyst in the Findings only | 0.75–0.85 |
| "Trace" / "tiny" / "minimal" cyst in the Findings only, omitted from a present Impression | 0.55–0.65 (boundary) |

**R7.3 — Non-qualifying cystic findings (code 0):** perimeniscal/parameniscal cyst, ganglion cyst (including at the PCL or ACL insertion), subchondral cyst, intraosseous cyst, popliteal lymph node, and posteromedial fluid that is not named as a cyst.

**R7.4 — Explicit negatives and silence** code 0. Rupture or dissection of a cyst changes no other column by itself.

---

## 8. Cross-Label Interaction Rules

**8.1 — Absorption into OA.** Subchondral edema or cysts directly beneath a cartilage defect, in a compartment that is OA-positive on its own merits, are absorbed into that OA column and do not create a Contusion. Absorption requires the compartment to be positive first. Edema never creates the OA positive (R2.3).

**8.2 — Absorption into Fracture.** Edema syntactically bound to a fracture is absorbed into Fracture and does not create a Contusion.

**8.3 — Osteochondral lesions go to the cartilage family.** Articular-surface osteochondral defects and fragments support the compartment OA column (R2.1 item 4), not Fracture (R6.4).

**8.4 — Ligament no-bleed.** See R4.3. This rule is absolute.

**8.5 — Effusion ⟂ Synovitis.** Independent columns, except for the dual triggers in R5.4 and the weak soft prior in R5.3.

**8.6 — OA ⟂ Effusion.** Chronic OA does not imply effusion, and effusion does not imply OA.

**8.7 — One finding, one home.** Each piece of evidence goes to the single label family it best belongs to, according to the rules above. Do not count the same edema toward OA, Contusion, and Fracture at the same time.

---

## 9. Multilingual Lexicon (Term Families)

Work with **term families**, not single keywords. Always check the zero-coded neighbours before a positive call.

### 9.1 Section headers
| Role | Forms |
|---|---|
| Impression | Impression, IMP, Conclusion, Conclusión, Impresión, Diagnóstico, Beurteilung, Zusammenfassung, Zaključak, Mišljenje, Заключение, Conclusie, Sonuç, Kanaat |
| Findings | Findings, Hallazgos, Resultados, Befund(e), Nalaz, Находка/МР находки, Bevindingen, Bulgular |
| Technique only | Technique, Técnica, Technik, Tehnika — a report containing only technique plus findings has no Impression, so apply R1.3 |

### 9.2 Anatomy
| Structure | Forms | Never confuse with |
|---|---|---|
| ACL | ACL, LCA, ligamento cruzado anterior, VKB / vorderes Kreuzband, prednji križni ligament, предна кръстна връзка, voorste kruisband, ön çapraz bağ (ÖÇB) | PCL (LCP, HKB, stražnji križni, задна кръстна, achterste kruisband, arka çapraz bağ) |
| MCL | MCL, LCM, LLI, LCI (ligamento colateral/lateral interno), Innenband, medijalni kolateralni ligament, медиална колатерална връзка, mediale collaterale band, iç yan bağ | LCL/FCL (LLE, colateral externo, Außenband, lateralni kolateralni, латерална колатерална, laterale collaterale band, dış yan bağ) |
| Medial meniscus | medial meniscus, menisco interno/medial, Innenmeniskus, medijalni menisk, медиален менискус, mediale meniscus, iç menisküs | — |
| Lateral meniscus | lateral meniscus, menisco externo/lateral, Außenmeniskus, lateralni menisk, латерален менискус, laterale meniscus, dış menisküs | — |
| PF compartment | patella, rótula, Patella/Kniescheibe, patela/čašica, пателa, knieschijf, patella; trochlea, tróclea femoral, Trochlea, troklea | — |

### 9.3 Diagnostic families
| Family | Positive forms | Zero-coded neighbours |
|---|---|---|
| Tear | tear, rupture, rotura, desgarro, fisura (meniscus), Riss, Ruptur, ruptura, puknuće, разкъсване/скъсване, руптура, scheur, ruptuur, yırtık | lesion (alone), signal change, grade 1–2/2a, mucoid, myxoid, intrasubstance, degeneration, Degeneration, degeneratie, dejenerasyon, amputación marginal |
| Fracture | fracture, fractura, Fraktur, prijelom/fraktura, фрактура/счупване, fractuur/breuk, kırık, SIFK, insufficiency fracture, avulsion, knöcherner Ausriss | osteochondral fracture, osteochondral defect, impaction, subchondral impression, depression of plateau |
| Contusion | bone bruise, contusion, contusión (ósea), Kontusion/Knochenkontusion, kontuzija/nagnječenje, контузия, kneuzing/bone bruise, kontüzyon, traumatic bone marrow edema, haematoma/hematom | bone marrow edema (edema óseo, Knochenmarködem, edem koštane srži, костномозъчен оток, beenmergoedeem, kemik iliği ödemi) **without** a trauma noun |
| Effusion | effusion, derrame (articular), Erguss/Gelenkerguss, izljev, излив, hydrops/gewrichtsvocht, efüzyon/eklem sıvısı (artışı), hemarthrosis/hemartrosis/Hämarthros | trace, minimal, minimalni, physiological, no (significant) effusion |
| Synovitis | synovitis, sinovitis, Synovialitis/Reizsynovialitis, sinovitis/proliferacija sinovije, синовит, synovitis, sinovit, synovial thickening/hypertrophy/proliferation, engrosamiento sinovial | soft-tissue edema, effusion alone, bursitis, plica, synovial cyst, PVNS |
| OA | osteoarthritis, arthrosis, artrosis, Arthrose/Gonarthrose, artroza/artrotske/gonartrotske promjene, артроза/гонартроза, artrose, osteoartrit/gonartroz, espesor total, full-thickness, denudation, ulcer/úlcera condral, grade 3–4/IV | chondrosis, chondromalacia I–III, condropatía (unqualified, except per R2.6), Chondropathie/Chondromalazie I–III, hondromalacija, хондромалация, chondropathie, kondromalazi, thinning, "high grade", >50% |
| Cyst (Baker's) | Baker cyst, popliteal cyst, quiste de Baker/poplíteo, Baker-Zyste, Bakerova/poplitealna cista, Бейкерова киста, Bakercyste, Baker kisti | ganglion, parameniscal/perimeniscal cyst, subchondral cyst, lymph node |

### 9.4 Quantifier ladder (fluid, cysts)
`none < trace / minimal / minimalni / gering-minimal / spoor < small / mild / gering / leve / pequeño / mali / малък / klein / az < moderate / moderado / mäßig / umjeren / умерен / matig / orta < large / extensive / massive / abundante / ausgeprägt / velik / голям / groot / fazla / belirgin`

A quantifier that cannot be placed confidently on this ladder is a boundary case. Make the conservative call, keep the soft score at or below 0.65, and raise a trace-floor flag.

### 9.5 Hedge words
suspected, possible, probable, likely, cannot exclude, rule out / R/O, may represent, suggestive of, versus/vs; sospecha de, posible, probable, no se descarta; Verdacht auf, möglich, wahrscheinlich, DD; sumnja na, moguće, vjerojatno; съмнение за, вероятно; verdenking op, mogelijk, waarschijnlijk; şüpheli, olası, ekarte edilemedi.

---

## 10. Soft-Score Calibration

### 10.1 What a soft score means
It is the probability that **the dataset's** label is 1, not the severity of the disease and not clinical truth.

### 10.2 Generic bands

| Band | Score | Applies to |
|---|---|---|
| Explicit positive, well-scoped, no conflict | 0.85–0.98 | Named diagnosis in the Impression, or in the Findings of a no-Impression report |
| Strong positive with small ambiguity | 0.70–0.85 | Ungraded partial tear, Impression contusion in a non-bone tissue, telegraphic compartment chondropathy |
| Boundary-rule cell | 0.45–0.65 | Trace-floor fluid, grade-ceiling OA plus hallmarks, plain BME in trauma, suspicion hedge, first-instance mechanism |
| Conflict cell | 0.25–0.45 | Findings-versus-Impression tension, off-ladder quantifier, possible annotation noise |
| Suppressed (present in Findings, omitted from Impression) | 0.15–0.30 | R1.2 cells |
| No evidence | 0.05–0.15 | Silence |
| Explicit negation | 0.00–0.05 | "Intact", "no tear", "no effusion", "normal" |

### 10.3 Ranking rule (important for AUC)
AUC measures **ordering**, not exact values. Within each label, every report with stronger evidence must score higher than every report with weaker evidence. So:

- Never collapse different evidence tiers onto the same number. A report with a suppressed tear (0.25) must still rank above a report that is silent (0.10), which must rank above an explicit negation (0.02).
- Use the per-label ladders (R4.1, R5.1, R5.3, R7.2) as **ordinal tiers**. The exact numbers are starting points to re-fit on held-out data; the ordering is the rule.
- Context priors (for example, effusion for synovitis) may only reorder cases *within* a tier. They must never lift a case above a tier that has explicit prose.

### 10.4 Highest-risk labels (lowest current AUC) — apply extra care
1. **Lateral OA**: the most common errors are inheriting a global "tricompartmental" statement without lateral-specific evidence, and missing lateral-condyle or lateral-plateau cartilage findings. Map anatomy strictly (R2, 9.2) and grade each compartment independently.
2. **Synovitis**: use the R5.3 ladder. Never use the effusion score as a proxy.
3. **PF OA**: check patellar facets **and** trochlea. Respect R2.5 (dysplasia is not OA) and R2.6 (telegraphic promotion).
4. **Lateral Meniscus**: check for coordinated phrases (N5), lateral-horn tears hidden in complex descriptions, and suspicion hedges (R3.3).

For these four labels, use graded scores across the boundary band rather than snapping to 0 or 1. This is where ranking quality is won or lost.

---

## 11. Known Noisy Areas (Handle by Principle, Not by Memory)

These are *types* of situations where the ground truth has historically been inconsistent. When a new report falls into one of them, follow the rules, keep the score in the boundary or conflict band, and raise a flag.

| Noisy area | Rule-based handling |
|---|---|
| Malformed or unheadered reports where explicit tears appear in the tail | Code semantically per N3, reduce confidence by about 0.15, flag `structure` |
| Shared-noun coordination of both menisci | Both sides positive, first item about 0.80, flag `ellipsis` |
| "Synovitis" wording in acute-trauma template reports | R5.3 boilerplate tier (0.40–0.50), flag `synovitis-template` |
| Telegraphic reports listing "bone contusions" | Code Contusion per R6.1. Fracture stays 0 unless a fracture noun exists. Flag `bone-family` |
| OA cells contradicting explicit cartilage statements (for example, "cartilage normal") | Follow the text, flag `OA-anomaly` |
| Findings-only effusion replaced by a different Impression diagnosis | Conflict cell (R5.1), flag `fluid-conflict` |
| Large Baker's cyst in the Findings only | Code 1 per R7.1, flag `baker-section` |

Never encode a single anomalous label as a rule. A rule needs repeated, consistent evidence.

---

## 12. Flags

Every cell decided by a boundary rule rather than an explicit statement carries a flag that names its rule:

- `OA-anomaly` — R2.5 triggers, grade-ceiling cells with hallmarks, tricompartmental distribution questions
- `suspicion` — hedged tears, root lesions, separations (R3.3)
- `trace-floor` — fluid or cyst quantifiers at the language boundary (R5.1, R7.2, 9.4)
- `BME-boundary` — plain bone marrow edema in trauma framing (R6.2)
- `first-instance` — a mechanism not yet confirmed by ground truth (for example, ossicles next to an avulsion)
- `ellipsis`, `structure`, `synovitis-template`, `bone-family`, `fluid-conflict`, `baker-section` — Section 11

---

## 13. Output Format

```json
{
  "labels": {
    "ACL":              {"probability": 0.00, "binary": 0, "evidence": "<short quote-free reason>", "rule": "R4.1"},
    "MCL":              {"probability": 0.00, "binary": 0, "evidence": "", "rule": ""},
    "Medial Meniscus":  {"probability": 0.00, "binary": 0, "evidence": "", "rule": ""},
    "Lateral Meniscus": {"probability": 0.00, "binary": 0, "evidence": "", "rule": ""},
    "Medial OA":        {"probability": 0.00, "binary": 0, "evidence": "", "rule": ""},
    "Lateral OA":       {"probability": 0.00, "binary": 0, "evidence": "", "rule": ""},
    "PF OA":            {"probability": 0.00, "binary": 0, "evidence": "", "rule": ""},
    "Effusion":         {"probability": 0.00, "binary": 0, "evidence": "", "rule": ""},
    "Synovitis":        {"probability": 0.00, "binary": 0, "evidence": "", "rule": ""},
    "Baker's":          {"probability": 0.00, "binary": 0, "evidence": "", "rule": ""},
    "Contusion":        {"probability": 0.00, "binary": 0, "evidence": "", "rule": ""},
    "Fracture":         {"probability": 0.00, "binary": 0, "evidence": "", "rule": ""}
  },
  "report_structure": {"language": "", "has_impression": true, "telegraphic": false},
  "flags": [],
  "possible_annotation_mismatch": false
}
```

Values are placeholders. Unless the project specifies another threshold, the binary value is 1 when probability ≥ 0.50, with the exception that Synovitis follows its own binary rule in R5.3.

---

## 14. Pre-Output Self-Check

Before emitting, verify each of the following:

1. Did I find the Impression, or confirm that none exists?
2. Is every negation scoped to its own tissue, side, and compartment?
3. Did I suppress ligament, meniscus, and OA findings that the present Impression omitted, while **not** suppressing fluid findings?
4. Is every ligament call grade 2 or higher, and did nothing from PCL, LCL, or the posterolateral corner bleed into ACL or MCL?
5. Is every meniscus positive a definite surfacing tear with no hedge?
6. Did I grade each OA compartment independently, using OA nouns or full-thickness-family evidence rather than hallmarks alone?
7. Is Synovitis supported by explicit synovial prose or a dual trigger, rather than by effusion alone?
8. Are Contusion and Fracture each supported by their own nouns, with edema absorbed into the correct home?
9. Is Baker's a named popliteal cyst rather than a ganglion or subchondral cyst?
10. Are the soft scores **ordered** by evidence strength, with no two different tiers on the same number?
11. Does every boundary cell carry a flag?

---

## 15. Compact Instruction Prompt (for another model)

> You label knee MRI reports into 12 fixed labels: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture. Apply the attached Final Rulebook. Core rules: (1) Only named diagnoses or full-thickness-equivalent findings turn a label on; descriptive signs never do by themselves. (2) The Impression overrides the Findings. Ligament, meniscus, and OA findings omitted from a present Impression are suppressed; effusion and Baker's cyst are not. With no Impression, the Findings stand. (3) Ligaments need grade 2 or higher; PCL, LCL, and posterolateral injuries never map to ACL or MCL. (4) Menisci need a definite surfacing tear; hedges and intrasubstance signal are 0. (5) Grade each OA compartment independently; chondrosis grade I–III and hallmarks alone are 0; an Impression that distributes "OA" to all three compartments without refutation sets all three. (6) Effusion follows a severity × location ladder. Synovitis needs explicit synovial prose or a dual trigger (hemarthrosis, haematoma, diffuse fluid); effusion is only a weak prior. (7) Contusion and Fracture are noun-gated; plain bone marrow edema is not a contusion; osteochondral or impaction lesions are not fractures; suspicion keeps a fracture alive but kills a meniscus tear. (8) Baker's: any named popliteal cyst. Output calibrated probabilities that are strictly ordered by evidence tier, a binary call, a short evidence string, the rule applied, and flags for every boundary cell. Never build a rule from a single contradictory example.
