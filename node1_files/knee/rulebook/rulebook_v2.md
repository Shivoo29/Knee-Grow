# Knee MRI Report Labeling Rulebook (merged v1)

Base: rulebook 4 (cases #001-#016). Where the five source rulebooks contradict, rulebook 4 wins.
Points unique to rulebooks 1, 2, 3 and 5 are added where they do not contradict it.
This is a dataset-specific labeling guide reverse-engineered from expert labels, not clinical guidance.

## Task
Read one knee MRI report (any language) and return, for each of the 12 labels, the probability (0.00-1.00)
that the expert label is 1. Probability = confidence in the dataset label, NOT disease severity.
Labels, in order: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion,
Synovitis, Baker's, Contusion, Fracture.
Never create labels for PCL, LCL/FCL, popliteus, tendinosis, bursitis, Hoffa, loose bodies: they only help interpret the 12.

Probability bands:
- 0.95-1.00 explicit, well-scoped positive trigger
- 0.80-0.94 strong positive, small ambiguity
- 0.60-0.79 probable positive, real mapping/context doubt
- 0.40-0.59 genuinely ambiguous or known inconsistent annotation pattern (one-off cases below)
- 0.20-0.39 weak evidence, likely negative
- 0.00-0.19 explicit negative or no supported finding

## Rule 1: Report structure
- The Impression/Conclusion block (IMPRESSION, CONCLUSION, Impresión, Conclusie, Заключение, Συμπέρασμα, Sonuç,
  Zaključak, numbered list or `--` items at the end) overrides the Findings. A finding described in Findings but
  omitted from the Impression defaults to 0, unless an override below applies.
- Telegraphic reports where Findings and Impression repeat each other, or reports with no sections: treat the whole text as Impression-level.
- Unheadered tail: check whether the final sentences are a summary or still inside a preceding negation; do not let
  "no fracture or bone contusion" bleed onto later diagnoses.
- `[BILATERAL NOTE: two reports filed under this study]`: label ONLY from the FIRST report.
- Overrides where Findings count despite omission from the Conclusion:
  1. Telegraphic ACL conclusion (e.g. "Conclusion: ACL rupture") + explicit "bone contusion" of lateral femoral condyle
     and posterior lateral tibia in Findings -> Contusion = 1.
  2. Moderate/large effusion, or Baker's cyst > 5 cm, in Findings -> 1 even if omitted from a brief Conclusion.
- Hedged/speculative wording ("suspicious", "could be attributed to", "cannot exclude", "possible") -> 0,
  unless the Findings document a definite anatomical defect (e.g. "there is a gap in the posterior horn").
- Negation scope: tie every negation to its own sentence, tissue, side and compartment.
- A finding repeated in Findings and Impression counts once.

## Rule 2: OA by compartment (Medial OA, Lateral OA, PF OA)
Compartments: medial femoral condyle/medial tibial plateau = Medial; lateral condyle/plateau = Lateral;
patella/trochlea/patellar facets = PF.
Positive (1) when the compartment has:
1. Grade 3-4 chondropathy, full-thickness loss (espesor total, volledig kraakbeenverlies), chondral ulcer,
   osteochondral defect, or focal weight-bearing erosion with subchondral marrow edema plus an explicit OA diagnosis.
2. Deep or delaminating fissures highlighted in the Impression.
3. Secondary osseous hallmarks: marginal osteophytes, subchondral cysts, subchondral bone marrow edema, or marked
   joint-space narrowing (eklem aralığı belirgin daralmıştır).
Overrides:
- Conclusion explicitly says "OA of all three compartments" / "incipient OA of all three compartments" with no
  compartment-by-compartment refutation in Findings -> all three = 1, even with "incipient"/"early".
- Conclusion names one compartment's cartilage in the context of incipient OA (e.g. patellar) -> that compartment = 1.
- "Tricompartmental chondrosis/OA" WITH a granular Findings breakdown: score each compartment separately; a
  compartment with only mild thinning/low-grade fissuring and no bone changes = 0 (case #007: lateral with marrow
  edema = 1, PF with mild thinning = 0).
- Advanced global gonarthrosis (gevorderde gonartrose) with full-thickness loss and osteophytes -> all three = 1.
- Unqualified "condropatía rotuliana" in a short Spanish Impression -> PF OA = 1.
- If Findings say full-thickness and Impression says grade 3, either qualifies.
Negative (0):
- Superficial chondropathy (condropatía superficial, grade 1), even in the Impression.
- Generic mild joint-space thinning without osseous change.
- "Knee arthritis" meaning inflammation (synovial thickening/effusion) without cartilage loss or osteophytes.
- Acute traumatic osteochondral injury in an acute ACL-tear knee -> OA = 0 and Fracture = 0 (case #016).

## Rule 3: Menisci
Positive (1): explicit tear (tear, rotura, desgarro, ρήξη, yırtık, scheur, recurrent, bucket-handle, radial,
horizontal, longitudinal, root, complex) on that side, or a definite anatomical defect (gap in posterior horn/root).
Positive (1): degeneration WITH extrusion/subluxation (extrusion, υπεξάρθρημα, buiten de gewrichtsspleet) in a
compartment with severe grade 3-4 OA, or with a torn contralateral meniscus plus severe OA (cases #003, #006, #008, #011).
Negative (0):
- "Suspicious small ... tear" without a confirmed gap, even if repeated in the Conclusion.
- Intrasubstance degeneration / mucoid or degenerative signal without surfacing tear (sin caracteres propios de rotura) in a normal knee.
- Amputación marginal / blunting from prior meniscectomy without an explicit tear.
Ambiguous: "Rotura de menisco interno y lateral" style shared-noun phrase -> named-last meniscus 0.9, the other 0.55 (one-off case).
Synonyms: menisco interno = medial, menisco externo = lateral.

## Rule 4: Ligaments (ACL, MCL)
- ACL complete or subtotal tear (υφολική ρήξη) -> 1. Anterior tibial translation -> 1.
- Advanced mucoid degeneration of the ACL stated in the Conclusion (gevorderde mucoïde degeneratie) -> ACL = 1.
- Template "increased signal ... in favor of grade 2 injury" / "Grade 2 injury of ACL/MCL" WITHOUT the word tear -> 0
  (cases #005, #013).
- MCL explicit partial tear ("partial tear", "μερικής ρήξης (grade 2)", "grade II sprain (partial tear)") -> 1.
  Annotator variance: in an acute complete ACL tear + displaced bucket-handle tear knee it was left 0 once (#016):
  use 0.75 there, not 1.
- Grade 1 sprain, interstitial signal, "suspected grade 1 injury", intact contour, no tibial translation -> 0.
- Never map PCL/LCP to ACL, or LCL/LCI/FCL/posterolateral corner/popliteus to MCL.
  Synonyms: LCA = ACL; LCM, ligamento colateral medial, LLI = MCL.
- If Findings say "medial cruciate" but the Conclusion says medial collateral, follow the Conclusion.

## Rule 5: Effusion, Synovitis, Baker's
Effusion:
- Small / mild / minimal / trace effusion (Минимален ставен излив, manja količina izljeva, suprapatellar bursada
  sıvı artışı) -> 0, EVEN IF it appears in the Impression.
- Moderate, substantial, large, massive (μέτρια ποσότητα, ικανή ποσότητα, leve a moderado derrame, hydrops)
  -> 1, also from Findings alone.
- Effusion explicitly paired with synovial thickening/synovitis -> 1.
- Hemarthrosis / lipohemarthrosis -> Effusion 0.9.
Synovitis (run in order, stop at the first step that decides):
- S1 Bilateral note: first report only.
- S2 Definitive synovial term in the Impression -> 0.95. Terms: synovitis, synovial thickening/hypertrophy/proliferation,
  thickened enhancing synovium, Hoffa-synovitis; EL υμενίτιδα, πάχυνση του υμένα; ES sinovitis, engrosamiento sinovial;
  NL synovitis, synoviale verdikking; TR sinovit, sinoviyal kalınlaşma; BG синовит, задебеляване на синовията;
  HR sinovitis, zadebljanje sinovije. A definitive synovial term only in the Findings: remember it, continue.
- S3 Hedged synovial terms (possible, probable, suspicious for, suggestive of, cannot exclude, may represent, could be
  attributed to; posible, πιθανή, mogelijk, olası, възможен, mogući) count as absent.
- S4 Explicit negation (no synovitis, no synovial thickening, synovium unremarkable; sin sinovitis, χωρίς υμενίτιδα,
  geen synovitis, sinovit izlenmedi, без синовит, bez sinovitisa) -> 0.05.
- S5 Grade the effusion with the Effusion rule above.
- S6 Effusion = 1 -> 0.70; 0.85 if also (Baker's cyst + OA) or (acute high-grade ACL/MCL tear + bursitis).
- S7 Effusion = 0 -> 0.20; 0.60 if a definitive synovial term was found in the Findings (S2).
- Mimics that never trigger synovitis by themselves: Baker's/popliteal cyst, OA/chondropathy/osteophytes, synovial
  cyst, ganglion, parameniscal cyst, synovial fluid, plica/synovial fold, prepatellar/infrapatellar/pes anserine
  bursitis, Hoffa fat-pad edema, PVNS/tenosynovial giant cell tumor, synovial chondromatosis, lipoma arborescens.
  Hemarthrosis is graded like an effusion. MOAKS "effusion-synovitis": small grade -> S7, moderate/large -> S6.
Baker's:
- Baker's/popliteal cyst in the Impression, or moderate/large (> 5 cm) in Findings -> 1.
- Small incidental cyst (< 3 cm) or "trace" cyst only in Findings -> 0.
- Generic posterior/popliteal fluid without a cyst diagnosis -> 0.

## Rule 6: Contusion vs Fracture
- A: subchondral marrow edema or small regional subcortical fractures attached to a compartment's cartilage loss and
  omitted from the Impression -> absorbed into OA: Contusion 0, Fracture 0.
- B: definitive microtrabecular insufficiency fractures (SONK) WITHOUT articular collapse -> Contusion 1, Fracture 0.
  If speculative ("barely recognizable edema ... could be attributed to") -> both 0.
- Explicit "insufficiency fracture" / SIFK / "insufficient fracture of the medial femoral condyle" -> Fracture 1;
  its "adjacent bone marrow edema" is absorbed: Contusion 0.
- C: macro-cortical fracture (depressed tibial plateau, fibular head, intercondylar eminence) whose marrow edema is
  just peri-fracture reaction -> Fracture 1, Contusion 0.
- D (both 1): D1 subchondral fracture in one plateau + explicit contusions in several other bones in the Impression;
  D2 patellar-dislocation impaction fractures (impresijske frakture) + extensive surrounding bone edema;
  D3 extensive metaphyseal/epiphyseal bone bruise in an acute high-grade ACL + MCL tear.
- Explicit primary bone bruise / bone contusion / contusión ósea -> Contusion 1.
- "Contusion" as a positive diagnosis in the Impression (even "popliteus muscle contusion") overrides a Findings
  "no bone contusion" -> Contusion 0.8.
- Subcutaneous/prepatellar/fat-pad/muscle edema without the word contusion -> 0.
- Osteochondral defect or dislodged osteochondral fragment with "No fracture is seen" -> Fracture 0 (counts toward OA).
- Short Spanish "Contusiones óseas femorotibiales" (one-off case) -> Contusion 0.5, Fracture 0.5.

## Output
Return only JSON, labels in order:
{"ACL": p, "MCL": p, "Medial Meniscus": p, "Lateral Meniscus": p, "Medial OA": p, "Lateral OA": p, "PF OA": p,
 "Effusion": p, "Synovitis": p, "Baker's": p, "Contusion": p, "Fracture": p}
