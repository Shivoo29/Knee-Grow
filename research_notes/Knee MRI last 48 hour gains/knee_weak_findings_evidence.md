# Weak knee findings: plane/sequence, input-design and ensembling evidence for lateral tibiofemoral OA, patellofemoral OA, synovitis and lateral meniscus tears

**Team context (from the assignment).** Per-finding AUC vs expert labels:

| Finding | AUC |
|---|---|
| Lateral OA | ~0.81 |
| Synovitis | ~0.80 |
| PF OA | ~0.87 |
| Lateral meniscus | ~0.87–0.89 |
| All others | 0.95–0.99 |

Inputs: a 140 mm field and a 100 mm close-up tile, both at 256 px (≈0.55 and ≈0.39 mm/px). There are five slots: sagittal fat-sat, sagittal plain, coronal fat-sat, coronal plain, axial any.

**Scope.** This note complements `research_notes/Knee MRI prior work and tricks/knee_mri_literature.md`. It covers what that note did not:
- per-finding plane and sequence evidence
- plane-specialist ensembling
- failure modes
- cheap post-hoc changes

Numbers repeated from the earlier note are tagged [V-prior] for completeness.

**Legend.**
- **[V]**: number checked in the primary full text or abstract during this session.
- **[V-prior]**: verified in the earlier note.
- **[S]**: from a search snippet or secondary summary; the primary was not checked.
- **[U]**: unverified, or from memory.

**Abbreviations.** FS = fat-suppressed. IW = intermediate-weighted. PD = proton density. TSE/FSE = turbo/fast spin echo. GRE = gradient echo. BML = bone marrow lesion. LM/MM = lateral/medial meniscus. LMPRT = lateral meniscus posterior root tear. CE = contrast-enhanced. OOF = out-of-fold.

**Data handling.** No competition data or local text was sent to any service. All web queries were generic literature queries.

---

## Q1. Which planes and sequences carry each finding, for radiologists (MOAKS/WORMS, ACR/ESSR) and for deep-learning models?

### Takeaway
MOAKS ties each of the four findings to specific planes and sequences:

| Feature | Plane | Sequence |
|---|---|---|
| Meniscal horns | Sagittal | |
| Meniscal body and extrusion | Coronal | |
| Hoffa-synovitis | Mid-sagittal | FS fluid-sensitive |
| Effusion-synovitis | Axial | Fluid-sensitive |
| Patellar cartilage subregions | Axial | |
| Cartilage and BMLs | | FS T2/IW/PD TSE (gradient echo discouraged) |
| Osteophytes | | Non-FS short-TE |

Accuracy studies show that no single plane is enough for the hard sub-findings:
- **Lateral meniscal root tears** need a coronal sign plus a sagittal sign. Combined sensitivity is 79%; the best single sign reaches 62%.
- **Trochlear cartilage** reaches only 40–61% sensitivity for severe lesions on axial images alone.
- **Non-contrast Hoffa signal** is sensitive (71–88%) but not specific (30–55%) against contrast-enhanced synovitis.

**Slot policy:** the axial slot should hold a FS fluid-sensitive series. The lateral meniscus needs both the sagittal FS and coronal FS slots.

### Cited Findings

**MOAKS scoring rules (Hunter et al., Osteoarthritis Cartilage 2011;19:990–1002) [V]** — [PMC4058435](https://pmc.ncbi.nlm.nih.gov/articles/PMC4058435)
- **Cartilage.** Sequence choice is "still undergoing extensive review". MOAKS favours FS T2/IW/PD TSE and calls gradient echo (SPGR, FLASH, DESS) unsuitable for early OA. No plane is mandated.
- **Bone marrow lesions.** FS T2/IW/PD or STIR TSE in all three planes. Gradient echo is not recommended.
- **Osteophytes.** Non-FS short-TE images (T1 preferred over PD) or gradient echo, in all three planes. FS water-sensitive images are acceptable.
- **Meniscus.** Anterior and posterior horns are scored on sagittal, the body on coronal. Extrusion is measured on coronal (medial/lateral) and sagittal (anterior).
- **Hoffa-synovitis.** Diffuse hyperintensity in Hoffa's fat pad on FS T2/IW/PD, read on mid-sagittal slices, graded 0–3. MOAKS states it is a non-specific surrogate: post-arthroscopy change and Hoffa's disease mimic it.
- **Effusion-synovitis.** Scored on axial fluid-sensitive images:
  - 0: physiologic.
  - 1: small; fluid continuous in the retropatellar space.
  - 2: medium; slight convexity of the suprapatellar bursa.
  - 3: large; capsular distension.

  It was renamed from "effusion" because non-contrast images cannot separate fluid from synovial thickening.
- **Subregions.**
  - Patella: medial and lateral, defined on axial images; the apex counts as medial.
  - Femur: medial and lateral trochlea, central, and posterior. The trochlea/central boundary is a line tangent to the anterior tibial margin.
  - Tibia: anterior, central and posterior thirds on each side, plus a subspinous region.
- **Reliability.** 20 OAI knees, 2 readers; weighted κ (% agreement):

  | Feature | Intra-rater | Inter-rater |
  |---|---|---|
  | Hoffa-synovitis | 0.42 (55%) | 0.70 (75%) |
  | Effusion-synovitis | 0.90 (90%) | 0.72 (70%) |
  | Cartilage area, tibial | 0.73 (80%) | **0.36 (70%)** |
  | Cartilage area, patellar | 0.71 (75%) | 0.71 (75%) |
  | Cartilage depth, tibial | 0.83 (80%) | 0.73 (75%) |
  | Cartilage depth, patellar | 0.92 (90%) | **0.85 (80%)** |
  | Meniscal morphology, lateral | 0.91 (90%) | 0.95 (95%) |
  | Meniscal extrusion, lateral | 0.89 (90%) | 0.79 (80%) |
  | Osteophytes, tibial | 0.70 (75%) | 0.49 (55%) |
  | Osteophytes, patellar | 0.84 (85%) | 0.64 (65%) |

  The paper is internally inconsistent here: the Results text gives κ 0.42 to "infrapatellar synovitis", while Table 6 lists it as Hoffa-synovitis.
- **Sequences used in the reliability exercise (OAI).** Coronal IW TSE; sagittal 3D DESS with coronal and axial reformats; sagittal IW FS TSE.

**Lateral meniscus**
- **Updated meta-analysis** (75 studies, 8,507 patients, arthroscopy reference; Radiology, PMID 42048590) [S] — [ReachMD summary](https://reachmd.com/news/mri-accuracy-for-knee-meniscus-tears-updated-meta-analysis/2487402/), [Radiology abstract (DOI as listed in search; fetch returned 403)](https://pubs.rsna.org/doi/abs/10.1148/radiol.252288):

  | Measure | Lateral tears | Medial tears |
  |---|---|---|
  | Sensitivity | 78.5% (74.5–82.0) | 91.0% (89.3–92.4) |
  | Specificity | 94.0% (92.5–95.3) | 87.7% (85.2–89.8) |

  - Lateral sensitivity by reading criteria: 81.1% with surfacing signal plus meniscal distortion, 79.2% with signal alone, 60.8% when criteria were not reported (adjusted OR 3.74, 95% CI 1.37–10.18).
  - Lateral specificity fell across publication years (P = .01).
- **Lateral root-tear signs in ACL-injured knees** (Yoshihara et al., KSSTA 2025; 126 arthroscopy-confirmed LMPRT vs 126 matched controls, 1.5/3 T) [V] — [PMC12205407](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12205407/):

  | Sign (plane) | Sensitivity | Specificity |
  |---|---|---|
  | Cleft sign (coronal) | 60% | 98% |
  | Truncated triangle (sagittal) | 62% | 95% |
  | Ghost sign (sagittal) | 14% | 100% |
  | Lateral meniscal extrusion ≥2.2 mm (coronal) | 50% | 89% |
  | Cleft and/or truncated triangle (coronal + sagittal) | **79%** | **95%** |
  | Extrusion and/or cleft and/or truncated triangle | 87% | 86% |

  The paper's introduction cites LMPRT in 7–15% of ACL injuries.
- **Conflict on the ghost sign.** A 2025 meta-analysis of root-tear signs (8 studies) reports a ghost-sign summary AUC of 0.97 with pooled sensitivity ≥0.8, and pooled sensitivity 0.91 for cleft and/or truncated triangle [S] — [PMC12870275](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12870275/). This contradicts Yoshihara's 14% ghost-sign sensitivity.
- **Two planes vs one.** An ISAKOS 2019 abstract reports low sensitivity for root signs read on coronal or sagittal alone, rising with high specificity when both planes were read together [S] — [ISAKOS abstract](https://isakos.com/2019/Abstract/11483)
- **Readers vs DCNN inputs** (Fritz et al., Skeletal Radiol 2020) [V] — [PMC7299917](https://pmc.ncbi.nlm.nih.gov/articles/PMC7299917/):
  - Readers used coronal T1, coronal STIR, axial FS IW, and sagittal FS plus non-FS IW (Dixon).
  - The DCNN used only coronal STIR and sagittal FS IW, cropped around the menisci.

**Lateral tibiofemoral cartilage / OA**
- **Combined axial + coronal FS T2 FSE** (Bredella et al. 1999, 130 patients) [S] — [Applied Radiology review](https://appliedradiology.com/articles/mr-of-articular-cartilage-lesions-of-the-knee):
  - Cartilage defects vs arthroscopy: sensitivity 94%, specificity 99%, accuracy 98%.
  - Grading: 64% of lesions graded identically, 90% within one grade. MRI tended to under-grade.
  - For comparison, FSE PD in Potter et al. had 87% sensitivity.
- **OARSI Delphi MRI definition of OA** (Hunter et al., OAC 2011) [S] — [Oxford NDORMS record](https://www.ndorms.ox.ac.uk/publications/167725), [Penn State record](https://pure.psu.edu/en/publications/definition-of-osteoarthritis-on-mri-results-of-a-delphi-exercise/):
  - 11 propositions were accepted.
  - Single-feature C-statistics vs radiographic tibiofemoral OA: cartilage loss 0.73, BML 0.72, osteophyte 0.61.
- **Label reliability.** MOAKS inter-reader κ is lowest for tibial cartilage area (0.36), and tibial osteophyte κ is 0.49 (table above) [V] — [PMC4058435](https://pmc.ncbi.nlm.nih.gov/articles/PMC4058435)

**Patellofemoral cartilage / OA**
- **Axial vs 45° axial-oblique for the trochlea** (LaPrade et al., OJSM 2018; 99 knees, 3 T, T2 TSE vs arthroscopy) [V] — [PMC6178377](https://pmc.ncbi.nlm.nih.gov/articles/PMC6178377):

  | Trochlear lesions | Axial sens / spec | Axial-oblique sens / spec |
  |---|---|---|
  | Any grade (1–4) | 55% / 80% | 51% / 60% |
  | Severe (grade 3–4), whole trochlea | 61% / 95% | 52% / 87% |
  | Severe, proximal trochlea | 40% / 98% | 45% / 95% |
  | Severe, distal trochlea | 58% / 90% | 52% / 92% |

  - The planes did not differ significantly.
  - The authors conclude that all planes, including sagittal, should be used for the trochlea.
  - They quote a meta-analysis (Harris et al.) with 87% sensitivity for patellar cartilage lesions.
- **MOAKS.** Patellar subregions are defined on axial images. Patellar cartilage depth has the most reliable cartilage score in MOAKS (inter-reader κ 0.85) [V] — [PMC4058435](https://pmc.ncbi.nlm.nih.gov/articles/PMC4058435)

**Synovitis without contrast**
- **Hoffa signal vs contrast-enhanced synovitis** (Crema et al., MOST cohort, Osteoarthritis Cartilage; title/year not re-checked [U]) [V for the numbers] — [PMC3578385](https://pmc.ncbi.nlm.nih.gov/articles/PMC3578385):
  - Design: 393 knees. Hoffa signal was scored on non-contrast sagittal FS PD at 1.0 T. The reference was CE axial and sagittal FS T1 at 1.5 T, scored at 5 peripatellar sites.
  - Overall, 200 of 393 knees (50.9%) had CE synovial thickening at one or more sites.

  | Hoffa grade ≥1 vs CE synovitis at | Sensitivity | Specificity | Accuracy |
  |---|---|---|---|
  | Infrapatellar site | 71% | 55% | 58% |
  | Intercondylar site | 88% | 30% | 41% |

  - PPV ≈27% (infrapatellar) and ≈24% (intercondylar); NPV ≈89% and ≈91%. These were derived from the paper's 2×2 counts and are not printed in the paper.
  - Pain discrimination: C-statistic 0.66 for CE vs 0.59 for non-CE (p = 0.03).
  - Fully adjusted pain associations: CE grade 2 OR 3.2 (1.5–6.7); non-CE Hoffa grade 3 OR 1.3 (0.5–3.4, not significant).
- **Quantitative vs semiquantitative** (FNIH OA Biomarkers Consortium) [S] — [PMC9718123](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9718123/):
  - Effusion-synovitis volume correlated with the MOAKS effusion-synovitis score at r = 0.74.
  - Quantitative Hoffa-synovitis correlated with MOAKS Hoffa grades at r = 0.55.
- **Non-contrast quantitative qDESS** reached AUC ~0.96–0.98 against CE-graded synovitis. This is a research sequence, not deep learning [S] — [Arthritis Res Ther 2021, Table 5](https://arthritis-research.biomedcentral.com/articles/10.1186/s13075-021-02436-8/tables/5)

**What deep-learning papers actually fed in**
- **Fritz 2020 (meniscus):** coronal STIR plus sagittal FS IW only, cropped around the menisci [V] — [PMC7299917](https://pmc.ncbi.nlm.nih.gov/articles/PMC7299917/)
- **Germann 2020 (ACL):** only coronal and sagittal FS fluid-sensitive TSE. Input was 14 central coronal and 20 sagittal images, cropped to 128×320 px around the ACL [V] — [Europe PMC full text, PMC7343178](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7343178/fullTextXML)
- **Liu 2018 (cartilage lesions, 175 patients):** sagittal FS T2 FSE [S] — [Radiology](https://pubs.rsna.org/doi/10.1148/radiol.2018172986), [ISMRM 2018 abstract](https://archive.ismrm.org/2018/0200.html)
- **Astuto 2021:** a single sagittal 3D FSE CUBE (0.5 mm slices, 384×384, FOV 14 cm, resampled to 512×512). Collateral ligaments were not assessed because no coronal series was used [V] — [PMC8166108](https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108)
- **Raman 2022 (effusion):** three sagittal TSE slices (14, 19 and 24 of 37; 3 mm) packed as RGB at 128×128 [V] — [PMC8873489](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8873489/)
- **MRNet:** sagittal T2 (FS)/PD, coronal T1, axial PD, as described by Yang et al. 2025 [V] — [QIMS 2025](https://qims.amegroups.org/article/view/137915/html). Because MRNet's coronal series is T1, which is not fluid-sensitive, its "axial is best for meniscus" result is confounded with sequence.
- **Multi-condition models:** Eur Radiol 2025 used sagittal, coronal and axial PD-FS plus sagittal T1 [V-prior] — [PMC13035746](https://europepmc.org/article/PMC/PMC13035746). eClinMed 2025 used sagittal, coronal and axial PD-FS [V-prior] — [PMC12508577](https://europepmc.org/article/PMC/PMC12508577)

### Inferences
- **(a) Slot guidance**, derived from the MOAKS plane rules and the accuracy studies above:

  | Finding | Where the evidence lives | Slot implication |
  |---|---|---|
  | Lateral meniscus tear | Horns and the ghost/truncated-triangle root signs on sag FS. Body, cleft sign and extrusion on cor FS. | Keep both sag FS and cor FS filled. A lateral-meniscus specialist should never see only one plane. |
  | Lateral TF OA (high-grade cartilage loss) | Cartilage and BML on FS PD/IW/T2 TSE. Central weight-bearing subregions on coronal, anterior/posterior subregions on sagittal. Osteophytes on non-FS short-TE. | Cor FS is the primary slot. Cor plain and sag plain add osteophytes. |
  | PF OA | Patellar facets on axial FS. The trochlea also needs sagittal (axial alone gives 40–61% sensitivity for severe lesions). | Axial slot = FS fluid-sensitive (PD-FS, IW-FS, T2-FS or STIR) in preference to plain PD/T1. Mid-slices of sag FS cover the trochlea. |
  | Synovitis (non-contrast) | Effusion-synovitis on axial fluid-sensitive images (suprapatellar-bursa convexity defines grade 2). Hoffa-synovitis on mid-sagittal FS. | Axial slot = FS. Sagittal tiles must include the suprapatellar bursa and Hoffa's fat pad. Axial slice selection must include slices above the patella. |

- **Axial slot policy.** This is the strongest single argument the evidence gives. A plain axial PD/T1 loses both the fluid signal that defines effusion-synovitis and the FS cartilage contrast MOAKS recommends. No study compares FS and non-FS axial inputs for deep learning head-to-head.
- **Synovitis label ceiling.** Hoffa-synovitis has intra-rater κ 0.42 and only 30–55% specificity against contrast. An AUC of ~0.80 against expert labels may be close to what the labels allow, unless the expert reference is mostly effusion-driven (effusion-synovitis inter-reader κ 0.72).
- **Where the headroom is.** PF labels should be the most reliable of the four (patellar depth κ 0.85), and lateral tibial cartilage labels the least (tibial area κ 0.36). So per unit of effort, label noise should cap PF OA less than lateral OA, despite PF's higher current AUC.

### Gaps
- **ACR practice parameter.** The ACR–SPR–SSR knee MRI practice parameter (revised 2020) could not be retrieved; the PDF link redirected to a site builder — [ACR link](https://acr.org/-/media/ACR/Files/Practice-Parameters/MR-Knee.pdf?la=en). Its required planes, slice thickness and field-strength guidance are therefore unverified.
- **ESSR.** No ESSR knee-protocol consensus was found.
- **WORMS/BLOKS.** Their plane rules for synovitis and effusion were not retrieved.
- **OARSI definition text.** The exact OARSI MRI OA definition was not retrieved. From memory [U]: tibiofemoral OA combines a definite osteophyte and/or full-thickness cartilage loss with features such as subchondral BML or cyst, meniscal subluxation/maceration/degenerative tear, or partial-thickness cartilage loss. Patellofemoral OA requires a definite osteophyte plus partial- or full-thickness cartilage loss.
- **Missing comparisons:**
  - coronal vs sagittal accuracy for lateral tibial or femoral cartilage specifically (Bredella reports only the axial + coronal combination);
  - FS vs non-FS axial for patellar cartilage or synovitis.

---

## Q2. What do deep-learning papers report per finding, and which design changes moved these specific findings?

### Takeaway
Published deep-learning AUCs on routine 2D MRI:

| Finding | Published DL AUC | Team (vs expert labels) |
|---|---|---|
| Lateral meniscus tear | 0.77–0.84 (radiologists vs arthroscopy ~0.83) | 0.87–0.89 |
| Compartment cartilage | 0.74–0.85 | Lateral OA ~0.81 |
| PF / retropatellar cartilage | 0.80–0.85 | PF OA ~0.87 |
| Non-contrast synovitis | None published (an OAI effusion classifier reached only 61.5% accuracy) | ~0.80 |

The team's lateral meniscus and PF OA numbers already exceed every published routine-data deep-learning result. Lateral OA sits inside the cartilage band.

Design changes with a measured effect on these structures:
- **Localization.** Tack's ROI bounding-box crop or auxiliary bbox loss added +0.04 to +0.07 AUC for the lateral meniscus on 2D TSE.
- **Region crops.** Every strong clinical DCNN crops around the target.
- **Combined criteria.** Signal plus morphology, and coronal plus sagittal signs, raise sensitivity.

No paper isolates the effect of compartment crops on OA, synovitis or PF.

### Cited Findings

**Lateral meniscus**
- **Fritz 2020** (100 patients, arthroscopy reference) [V] — [PMC7299917](https://pmc.ncbi.nlm.nih.gov/articles/PMC7299917/):

  | Reader | Lateral AUC | Lateral sens / spec | Medial AUC |
  |---|---|---|---|
  | DCNN | 0.781 (0.687–0.858) | 58.3% / 92.1% | 0.882 |
  | MSK radiologist 1 | 0.828 | 70.8% / 94.7% | 0.918 |
  | MSK radiologist 2 | 0.827 | 66.7% / 98.7% | 0.913 |

  - Prevalence: lateral tears 24%, medial tears 57%.
  - DCNN AUC for any tear: 0.961.
- **Eur Radiol 2025** (internal / external) [V-prior] — [PMC13035746](https://europepmc.org/article/PMC/PMC13035746):
  - Lateral meniscus tear 0.84 / 0.77 (prevalence 8.5%).
  - Lateral meniscal extrusion 0.85 / 0.73.
  - Medial meniscus tear 0.88 / 0.83.
- **Tack 2021 (OAI 3D DESS).** Lateral meniscus AUC is 0.91–0.96 by horn. The posterior horn stays at 0.91 under every variant: full volume, crop, and bbox loss [V-prior] — [PMC8675251](https://europepmc.org/article/PMC/PMC8675251)
- **Earlier meniscus models** (medial + lateral pooled) [S, from the earlier note]:
  - Pedoia 2019: AUC 0.89.
  - SFR 2018 challenge, sagittal-only images: Mask R-CNN horn localization 0.906 — [Couteaux](https://www.em-consulte.com/article/1286640/automatic-knee-meniscus-tear-detection-and-orienta); Faster R-CNN 0.92 — [Roblot](https://www.em-consulte.com/article/1286641/artificial-intelligence-to-diagnose-meniscus-tears)

**Cartilage**
- **Liu 2018** (segmentation CNN plus patch classification CNN; sagittal FS T2 FSE; 175 patients) [S] — [Radiology](https://pubs.rsna.org/doi/10.1148/radiol.2018172986):

  | Evaluation | AUC | Sensitivity / specificity |
  |---|---|---|
  | 1 | 0.917 | 84.1% / 85.2% |
  | 2 | 0.914 | 80.5% / 87.9% |
- **Astuto 2021** (3D CUBE; V-Net ROIs → bounding-box crops → 17 hierarchical 3D CNNs; 1,435 studies from 294 patients) [V] — [PMC8166108](https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108):

  | Tissue | AUC |
  |---|---|
  | Cartilage | 0.93 (0.90–0.95); sensitivity 85%, specificity 89% |
  | Menisci | 0.93 |
  | Bone marrow edema | 0.83 |
  | ACL | 0.90 |

  Per-compartment AUCs exist only as an image table, which was not retrieved.
- **Zhong et al., ISMRM 2022** (OAI DESS, ~2,400 knees; MOAKS "healthy vs not"; U-Net mask crop plus 3D DenseNet) [V] — [ISMRM 2022 #4617](https://cds.ismrm.org/protected/22MProceedings/PDFfiles/4617.html):
  - Across compartments: accuracy 0.75–0.83, sensitivity 0.66–0.92, specificity 0.55–0.89.
  - Lateral tibial cartilage did better than patellar, femoral and medial tibial cartilage. The authors attribute this to better class balance.
- **Eur Radiol 2025, routine multi-sequence** (internal / external): tibial 0.85 / 0.74, femoral 0.84 / 0.77, retropatellar 0.85 / 0.80 [V-prior] — [PMC13035746](https://europepmc.org/article/PMC/PMC13035746)

**Infrapatellar fat pad (Hoffa)**
- eClinMed 2025 includes IFP abnormality among its "secondary" findings: group AUC 0.815 internal, 0.744 and 0.774 on the two external sets [V-prior] — [PMC12508577](https://europepmc.org/article/PMC/PMC12508577)
- A fat-pad radiomics signature for knee OA reached test AUC 0.78 [S] — [PMC10127010](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10127010/)

**Effusion / synovitis deep learning**
- **Raman 2022** [V] — [PMC8873489](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8873489/):
  - Setup: OAI, 1,628 knees; MOAKS effusion-synovitis 0 vs ≥1; three sagittal slices at 1.25 mm/px; dense network.
  - Test accuracy 61.5% (71% no-effusion, 52% effusion).
  - On a 36-knee subset, agreement with the OAI labels was 47.2% for the network and 41.7% for a radiologist. Network–reader κ was 0.29.
  - Fine-tuned VGG16 reached AUC ≤0.5.
  - The occlusion map highlighted the joint space and the suprapatellar bursa.
- **UCSF effusion-synovitis volume segmentation:** Dice 0.79, with moderate correlation to semiquantitative grades [S] — [eScholarship](https://escholarship.org/uc/item/7294c75t)
- **OMERACT 2024 validation:** deep-learning effusion volume vs expert semiquantitative scores, Kendall τ 0.34–0.43 [S] — [OMERACT PDF](https://omeract.org/wp-content/uploads/2024/04/23-00634-OM23-Validation-of-a-deep-learning-algorithm-FELFELIYAN.pdf)
- **ISMRM 2023 pipeline:** segments the suprapatellar capsule and fat pad on sagittal PD-FS, then classifies three synovitis types. The snippet gives no metrics [S] — [ISMRM 2023 #2270](https://cds.ismrm.org/protected/23MProceedings/PDFfiles/2270_ZsfsbTRRT.html)
- **Tolpadi et al. 2023:** a PatchGAN synthesized post-contrast images from pre-contrast scans, with synovial nRMSE 23.1 vs 26.2 for a U-Net. This was wrist rheumatoid arthritis (27 patients), not knee [S] — [Bioengineering 2023](https://www2.mdpi.com/2306-5354/10/5/516)

**Design changes with a measured effect on these structures**
- **ROI localization (Tack 2021).** Lateral meniscus AUC on OAI IW-TSE (2D, thicker slices) [V-prior] — [PMC8675251](https://europepmc.org/article/PMC/PMC8675251):

  | Variant | Anterior | Body | Posterior |
  |---|---|---|---|
  | Full volume | 0.88 | 0.85 | 0.85 |
  | Bounding-box crop | 0.92 | 0.90 | 0.90 |
  | Auxiliary bbox loss | 0.95 | 0.91 | 0.90 |

  - The gains are +0.04 to +0.07.
  - On DESS, the lateral meniscus gained only +0.01 to +0.02 and the posterior horn gained 0.00.
- **Cropping around the target** is standard in the best clinical DCNNs [V]:
  - menisci (Fritz) — [PMC7299917](https://pmc.ncbi.nlm.nih.gov/articles/PMC7299917/)
  - 128×320 around the ACL (Germann) — [Germann](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7343178/fullTextXML)
  - bounding-box subvolumes (Astuto) — [PMC8166108](https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108)
  - a meniscus-centred 256×256 crop plus an attention re-crop (eClinMed) — [PMC12508577](https://europepmc.org/article/PMC/PMC12508577)
- **Localization is hardest for lateral structures.** Astuto's bounding-box IoU [V] — [PMC8166108](https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108):

  | Structure | Medial | Lateral |
  |---|---|---|
  | Femoral condyle | 0.83 | 0.78 |
  | Tibia | 0.68 | 0.61 |
  | Meniscus posterior horn | 0.61 | 0.56 |
  | Meniscus anterior horn | 0.52 | 0.49 |

  Trochlea scored 0.69 and patella 0.68. The authors blame the smaller boxes.
- **Plane fusion for meniscus** (MRNet; Azcona): logistic-regression fusion 0.908 vs best single plane 0.882 (+0.026) [V-prior] — [arXiv 2010.01947](https://arxiv.org/pdf/2010.01947). Counter-examples are in Q3.
- **Reading criteria.**
  - Combining signal with morphology raised radiologists' lateral sensitivity from 60.8% to 81.1% [S] — [ReachMD](https://reachmd.com/news/mri-accuracy-for-knee-meniscus-tears-updated-meta-analysis/2487402/)
  - Combining coronal and sagittal root signs raised sensitivity from ≤62% to 79% at 95% specificity [V] — [PMC12205407](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12205407/)
- **Specialist vs general models.** Pathology-specific models averaged AUC 0.927 vs 0.898 for general-abnormality models across 52 studies. This comparison is confounded [V-prior] — [PMC12021734](https://europepmc.org/article/PMC/PMC12021734)
- **Resolution, not an ablation.** Astuto sampled at ≈0.27 mm/px (14 cm FOV resampled to 512) and reached cartilage AUC 0.93. Raman sampled at 1.25 mm/px and reached 61.5% effusion accuracy. The tasks and models differ, so this is not a controlled comparison [V for the parameters; the comparison is mine].

### Inferences
- **Lateral meniscus.** The team's 0.87–0.89 exceeds Fritz's two MSK radiologists (0.83 vs arthroscopy) and Eur Radiol's deep-learning result (0.84 / 0.77).
  - Remaining errors are probably dominated by tears that readers also cannot see (8 of 10 DCNN lateral false negatives were missed by at least one reader) and by label noise.
  - Expect small gains at most, concentrated in posterior-horn and root tears.
- **Lateral OA.** This is the best candidate for a lateral-compartment crop specialist.
  - Tack's +0.04 to +0.07 came from 2D TSE with thick slices, which resembles routine clinical data.
  - A 60–70 mm lateral-half crop at 256 px gives ≈0.23–0.27 mm/px, about Astuto's sampling, vs ≈0.39 mm/px in the current 100 mm tile. This is arithmetic, not a measured gain.
- **Synovitis.** There is no deep-learning benchmark. The physiologically supported features are:
  - suprapatellar and retropatellar fluid on axial and sagittal FS;
  - diffuse Hoffa hyperintensity on mid-sagittal FS.

  A sag-FS crop over the suprapatellar bursa plus Hoffa's fat pad would be the analogue of Tack's ROI trick. It is untested.

### Gaps
- **Not retrieved:**
  - Astuto's per-compartment AUCs (Table 5 is an image);
  - Liu 2018 results by location;
  - Zhong's per-compartment numbers (figure only);
  - medial vs lateral AUCs from Rizk 2021 (Phys Med 83:64–71).
- **No published benchmark:** no paper reports study-level AUCs for lateral tibiofemoral OA, patellofemoral OA or non-contrast synovitis on routine multi-sequence MRI.
- **No resolution ablation:** no controlled in-plane resolution study exists for any of the four findings.
- **KneeMRI** (Štajduhar 2017) is ACL-only, so it does not inform these findings [U].

---

## Q3. Does ensembling single-plane networks, with each other or with a joint model, beat a single joint model? By how much?

### Takeaway
All published evidence is on MRNet. It compares per-plane ensembles with the best single plane, never with a joint multi-plane model, and the effect is inconsistent:

| Study | Fusion method | Change vs best single plane |
|---|---|---|
| Azcona 2020 | Logistic regression | Meniscus +0.026, ACL +0.022, abnormal −0.012 |
| Belton 2021 | Element-wise max (MPFuseNet) | ACL +0.019, meniscus +0.005, abnormal −0.007 |
| Belton 2021 | Logistic regression | ACL −0.030 |
| Yang 2025 | Feature concatenation | Meniscus accuracy +0.008, ACL accuracy +0.033 |
| Kunitomi 2026 (thesis) | Hybrid fusion | ACL AUC −0.027 vs sagittal |

No paper tests adding plane specialists to a joint model. Expect 0 to +0.02 on a weak finding if the stacker is fit out-of-fold, and possible losses otherwise.

### Cited Findings
- **Azcona 2020** (ImageNet ResNet18 per plane, logistic-regression fusion; MRNet validation) [V-prior] — [arXiv 2010.01947](https://arxiv.org/pdf/2010.01947):

  | Task | Axial | Coronal | Sagittal | Fused |
  |---|---|---|---|---|
  | Meniscus | 0.882 | 0.852 | 0.781 | 0.908 |
  | ACL | 0.934 | 0.926 | 0.897 | 0.956 |
  | Abnormal | 0.949 | 0.913 | 0.950 | 0.938 (below the best single plane) |

- **Belton et al. 2021**, validation AUC [V] — [ar5iv 2108.08136](https://ar5iv.arxiv.org/html/2108.08136):

  | Model | ACL | Abnormal | Meniscus |
  |---|---|---|---|
  | Best single plane | 0.953 (axial) | 0.923 (coronal) | 0.862 (coronal) |
  | Multi-plane logistic regression (MPLR) | 0.923 | 0.952 | 0.862 |
  | MPFuseNet (element-wise max over planes) | 0.972 | 0.916 | 0.867 |
  | MP2 | 0.948 | 0.903 | 0.853 |

  - On the official MRNet validation set: MPFuseNet ACL 0.977 and meniscus 0.831; MPLR abnormal 0.957.
  - Why MPLR hurt: "single-plane models that had a lower validation AUC had a higher training AUC". The logistic regression therefore up-weighted overfit planes.
  - Multi-plane training took 3× the time and 3× the trainable parameters.
- **Yang, Shi, Li (QIMS 2025)**: MRNet; per-plane AlexNet extractors trained from scratch, features concatenated; validation set [V] — [QIMS](https://qims.amegroups.org/article/view/137915/html):

  | Model | Meniscus accuracy | Meniscus F1 | ACL accuracy |
  |---|---|---|---|
  | Sagittal | 0.633 | 0.639 | 0.892 |
  | Coronal | 0.750 | 0.717 | 0.842 |
  | Axial | 0.775 | 0.738 | 0.867 |
  | Three-plane fusion | 0.783 | 0.745 | 0.925 |

- **Kunitomi (Waseda master's thesis, 2026)** [S] — [PDF](https://www.ams.giti.waseda.ac.jp/data/pdf-files/2026_kunitomi_mt.pdf):
  - Meniscus: sagittal 0.791, axial 0.826.
  - ACL: sagittal 0.976 vs three-plane hybrid fusion 0.949.
- **Shared multi-label model vs separate models** [V-prior]:
  - Eur Radiol 2025 used one shared-encoder model over all sequences — [PMC13035746](https://europepmc.org/article/PMC/PMC13035746).
  - eClinMed 2025 used nine separate binary models with per-plane features — [PMC12508577](https://europepmc.org/article/PMC/PMC12508577).
  - Azcona, from scratch: per-task networks 0.858 vs one multi-task network 0.828 — [arXiv 2010.01947](https://arxiv.org/pdf/2010.01947).
- **Radiologist analogue for lateral root tears** [V] — [PMC12205407](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12205407/):
  - Best single sign: 62% sensitivity.
  - Coronal + sagittal signs: 79% sensitivity at the same 95% specificity.
  - Adding coronal extrusion: 87% sensitivity, 86% specificity.

### Inferences
- **(b) Plane specialists.** They are cheap (in Belton, a single-plane model costs about one-third of a three-plane model), but the evidence for them is weak.
- **Why a single-plane specialist should be weaker than the joint model.** For the four weak findings, the diagnostic signs are split across planes:
  - lateral meniscus: horns and body vs root signs;
  - synovitis: Hoffa on sagittal vs effusion-synovitis on axial;
  - PF: patella on axial vs trochlea on sagittal.

  Its only value is diversity inside an ensemble.
- **Expected size.** MRNet fusion gains (+0.02 to +0.03) were measured over single-plane baselines. The team's joint model already fuses planes, so the marginal gain from adding specialists should be smaller, plausibly ≤ +0.01 to +0.02 per weak finding.
- **Better-supported variant: region specialists that keep two planes.**
  - Lateral half of cor FS + sag FS, for lateral OA and lateral meniscus.
  - Axial FS + mid-sag FS, for PF OA and synovitis.
- **If plane specialists are added:**
  1. Fit the stacker (logistic regression or per-finding weights) only on out-of-fold predictions. Belton's −0.030 came from fitting on overfit training scores.
  2. When the OOF set is small, prefer a plain mean or a fixed small weight per specialist (for example 0.2–0.3) over a learned stacker. There are ≈4,400 studies, with few positives per weak finding. This rule is inferred from Belton's failure mode and has not been tested.
  3. Accept a specialist only if its OOF gain on the target finding exceeds the bootstrap SE. On gold-58 with ~10 positives, the SE is ≈0.04–0.07 (earlier note).

### Gaps
- No knee paper compares a joint model plus plane-specialist ensemble against the joint model alone.
- No paper reports per-finding stacking for OA, synovitis or PF.
- Belton's per-plane numbers and spatial-attention ablation appear only in figures.
- Kunitomi is an unreviewed thesis [S].

---

## Q4. Known failure modes (lateral vs medial, left/right orientation, partial volume, low field and heterogeneity) and remedies

### Takeaway
**Lateral structures are harder for radiologists and networks alike:**

| Measure | Lateral | Medial |
|---|---|---|
| Radiologist meniscal-tear sensitivity (meta-analysis) | 78.5% | 91% |
| DCNN meniscal AUC (Fritz) | 0.78 | 0.88 |
| Tibial ROI localization IoU (Astuto) | 0.61 | 0.68 |

**The lateral misses concentrate in the posterior horn and root:**
- sensitivity for posterior-horn tears is 73%;
- a Mayo series caught only 33% of root tears before surgery;
- meniscofemoral-ligament and popliteus-hiatus pseudotears mimic tears;
- magic angle brightens the posterior horn on short-TE images.

**Partial volume** costs small and horizontal tears and proximal-trochlea lesions (40% sensitivity). Multi-slice reading rules recover part of this: requiring two abnormal slices raised lateral PPV from 83% to 96%.

**Field strength vs heterogeneity.** 1.5 T vs 3 T made no difference in two DCNN studies. Site and protocol heterogeneity did:
- Germann's DCNN fell from 0.967 to 0.898 AUC on outside scans, while radiologists held steady.
- In Eur Radiol 2025, the largest external drops were lateral meniscal extrusion (−0.12) and tibial cartilage (−0.11).

**Left/right.** Published OA pipelines handle it by mirroring left knees into a canonical orientation.

### Cited Findings

**Lateral vs medial**
- **Meta-analysis:** lateral sensitivity 78.5% vs medial 91.0% [S] — [ReachMD](https://reachmd.com/news/mri-accuracy-for-knee-meniscus-tears-updated-meta-analysis/2487402/)
- **De Smet & Tuite 2006 (AJR), standard criterion:** lateral sensitivity 77% and specificity 89%; medial 95% and 85% [S] — [ScienceDaily](https://www.sciencedaily.com/releases/2006/12/061207160020.htm), [QxMD](https://www.qxmd.com/r/16985134)
- **De Smet & Mukherjee 2008** (AJR 190:22–26; 483 patients with MRI and arthroscopy) [S] — [AuntMinnie](https://www.auntminnie.com/clinical-news/mri/article/15585867/missed-turn-what-makes-mri-detour-in-diagnosing-meniscal-tears):
  - Lateral sensitivity was 100% when the tear involved more than one-third of the meniscus, and 73% for posterior-horn tears.
  - Many missed tears were longitudinal peripheral tears of the posterior horn.
- **ACL injury lowers sensitivity.**
  - A 2009 AJR study found sensitivity lowest for posterior-horn lateral tears, and acute ACL injury lowered overall MRI sensitivity [S] — [QxMD 19620451](https://www.qxmd.com/r/19620451)
  - De Smet & Graf: medial sensitivity fell from 94% to 69% with ACL injury [S] — [PMC2899202](https://pmc.ncbi.nlm.nih.gov/articles/PMC2899202)
- **Mayo root-tear series:** only 15 of 45 (33%) arthroscopically confirmed lateral root tears were diagnosed on preoperative MRI. Most were visible in retrospect, and 82% of patients had prior ACL reconstruction [S] — [Mayo](https://mayoclinic.elsevierpure.com/en/publications/high-rate-of-missed-lateral-meniscus-posterior-root-tears-on-preo)
- **Fritz DCNN errors** [V] — [PMC7299917](https://pmc.ncbi.nlm.nih.gov/articles/PMC7299917/):
  - Lateral meniscus: 6 false positives and 10 false negatives.
  - Of the 10 false negatives, 8 were also missed by at least one reader and 5 by both readers.
  - The authors' explanations: readers had more sequences; some tears were not visible on MRI; training labels came from NLP of reports (F1 0.97).
- **Pseudotears of the lateral posterior horn** [S]:
  - The meniscofemoral ligaments (Humphrey and Wrisberg) produced a pseudotear in up to 63% of patients in one study, with no association with real tears — [PubMed 17483944](https://pubmed.ncbi.nlm.nih.gov/17483944) (this attribution comes from a search summary and was not checked).
  - The popliteus tendon runs through the hiatus next to the posterior horn — [Radsource](https://radsource.us/?p=2837)
  - The radiologist remedy is to follow the structure across consecutive sagittal slices.
- **Magic angle** [S] — [Radiopaedia](https://radiopaedia.org/articles/1639), [RadiologyKey](https://radiologykey.com/mri-technical-considerations/):
  - Tightly bound collagen at ~54.7° to B0 appears bright on short-TE images (TE below ~32 ms: T1, PD, GRE).
  - This can involve the up-sloping lateral posterior horn. A long TE avoids it.
  - The primary source is Peterfy et al., AJR 1994;163:149–154. The citation is confirmed, but its numbers were not retrieved.

**Left/right orientation**
- An OAI-based MRI progression pipeline reorients scans and flips only left knees, so every knee shares one orientation. It then applies N4 bias correction and normalization [S] — [Arthritis Res Ther 2021, Fig. 1](https://arthritis-research.biomedcentral.com/articles/10.1186/s13075-021-02634-4/figures/1)

**Partial volume and thick slices**
- **Two-slice-touch rule** (De Smet & Tuite 2006) [S] — [ScienceDaily](https://www.sciencedaily.com/releases/2006/12/061207160020.htm), [Radiopaedia](https://images.radiopaedia.org/articles/two-slice-touch-rule-1):
  - Call a tear only when two or more images are abnormal.
  - Lateral PPV rose from 83% to 96% (p = 0.02); medial PPV from 91% to 94% (p = 0.37).
  - Root tears are the exception: one abnormal slice is accepted.
- **Tack 2021** attributes missed horizontal body tears to thick mediolateral slices. DESS (0.7 mm) beat 2D TSE for the medial meniscus [V-prior] — [PMC8675251](https://europepmc.org/article/PMC/PMC8675251)
- **Trochlea on axial T2 TSE:** severe-lesion sensitivity was 40% in the proximal trochlea and 58% in the distal trochlea [V] — [PMC6178377](https://pmc.ncbi.nlm.nih.gov/articles/PMC6178377)

**Field strength, scanner and protocol heterogeneity**
- **Germann 2020** (Invest Radiol; 512 subjects from 59 institutions; arthroscopy reference) [V] — [Europe PMC full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7343178/fullTextXML):

  | DCNN on | AUC | Sensitivity | Specificity |
  |---|---|---|---|
  | In-house MRI | 0.967 | 99.0% | 94.4% |
  | Outside MRI | 0.898 | 92.5% | 87.1% |

  - All three differences were significant (p < 0.05). Readers showed no in-house vs outside difference.
  - 1.5 T vs 3 T made no difference (all P ≥ 0.753).
  - The authors attribute the drop to training exposure: 3,295 of 5,802 training and testing exams were in-house. They add that NLP labels may have been less accurate on outside reports.
- **Fritz 2020:** 64 patients scanned at 1.5 T and 36 at 3 T, with no field-strength difference for the DCNN or the readers [V] — [PMC7299917](https://pmc.ncbi.nlm.nih.gov/articles/PMC7299917/)
- **Eur Radiol 2025, internal → external** [V-prior] — [PMC13035746](https://europepmc.org/article/PMC/PMC13035746):

  | Condition | Internal | External | Change |
  |---|---|---|---|
  | Lateral meniscal extrusion | 0.85 | 0.73 | −0.12 |
  | Tibial cartilage | 0.85 | 0.74 | −0.11 |
  | Lateral meniscus tear | 0.84 | 0.77 | −0.07 |
  | Femoral cartilage | 0.84 | 0.77 | −0.07 |
  | Retropatellar cartilage | 0.85 | 0.80 | −0.05 |
  | Effusion | 0.91 | 0.94 | +0.03 |

  The mean absolute drop was 0.05 ± 0.03.
- **Raman 2022:** effusion classification held up under added Gaussian noise (5–30% of maximum intensity; ~61% accuracy at 5%). No real low-field data was tested [V] — [PMC8873489](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8873489/)
- **AJR:** 1.5 T and 3 T gave similar meniscal-tear accuracy (headline only) [S] — [HealthExec](https://healthexec.com/topics/patient-care/precision-medicine/ajr-15t-3t-mri-deliver-similar-meniscal-tear-accuracy)

### Inferences
- **Laterality.** This is geometric reasoning, not a published result.
  - In coronal and axial images, the medial and lateral sides swap image position between left and right knees.
  - In a sagittal stack sorted by patient x, slices run lateral→medial for a right knee and medial→lateral for a left knee.
  - Without canonicalization, any medial/lateral head has to infer the side from anatomy (for example the fibular head).
  - Horizontal-flip augmentation without swapping the medial/lateral labels actively teaches the wrong mapping.
  - The published remedy is to mirror left knees (and reverse the sagittal slice order) before the network sees them. The team's canonicalization work matches this.
- **Partial volume.** A lateral-meniscus head that pools by max over slices behaves like the one-slice criterion (lateral PPV 83%).
  - Top-k (k = 2–3) mean or attention pooling over adjacent slices is the deep-learning analogue of two-slice-touch (PPV 96%).
  - Root tears are the exception and may need single-slice sensitivity.
  - This is untested.
- **Targeted fix.** The misses sit in the posterior horn and root. A sag-FS crop over the lateral posterior horn, plus the coronal root region (cleft sign), targets exactly that.
- **Heterogeneity, not field strength, is the documented deep-learning failure mode.** The weak findings (lateral extrusion, tibial cartilage, lateral meniscus) degrade most under domain shift. Per-series intensity normalization and site-balanced folds therefore matter more for them than for effusion.

### Gaps
- No deep-learning study measures laterality mislabeling or left/right confusion errors.
- No deep-learning study tests scanners below 1 T, or modern 0.55 T systems, for any of the four findings.
- The prevalence of magic-angle false positives in the lateral posterior horn on modern PD sequences was not retrieved.
- The source mapping for the meniscofemoral-ligament pseudotear figure is unverified.

---

## Q5. Which cheap inference-time or post-hoc changes have evidence?

### Takeaway
No knee paper tests test-time augmentation or post-hoc stacking for these four findings, so every cheap change below is supported only by adjacent evidence:

| Change | Adjacent evidence |
|---|---|
| Fit any plane/model stacker on out-of-fold predictions. Prefer mean or max fusion over in-sample logistic regression. | Belton: in-sample LR cost ACL −0.030; max fusion gave +0.005 to +0.019 |
| Canonicalize laterality before inference | OAI pipelines mirror left knees |
| Select the fat-suppressed axial for the axial slot | MOAKS rules for effusion-synovitis and patellar cartilage |
| Top-k slice pooling for the lateral meniscus | Two-slice-touch: lateral PPV 83% → 96% |
| Second-stage cross-finding stacker (effusion → synovitis; extrusion/BML → lateral OA) | How the labels are defined: MOAKS effusion-synovitis, OARSI |

AUC is unchanged by any monotone recalibration. Only rank-changing operations can help: ensembling, stacking, and input selection.

### Cited Findings
- **In-sample stacking hurt; max fusion helped.** In-sample logistic regression dropped ACL from 0.953 to 0.923. Max-fusion MPFuseNet gave ACL 0.972 and meniscus 0.867 vs 0.862 [V] — [ar5iv 2108.08136](https://ar5iv.arxiv.org/html/2108.08136)
- **Properly fused per-plane logistic regression:** meniscus +0.026, ACL +0.022 [V-prior] — [arXiv 2010.01947](https://arxiv.org/pdf/2010.01947)
- **Laterality:** left knees were flipped to the right-knee orientation in an OAI deep-learning pipeline [S] — [AR&T 2021](https://arthritis-research.biomedcentral.com/articles/10.1186/s13075-021-02634-4/figures/1)
- **Series selection rules (MOAKS):** effusion-synovitis on axial fluid-sensitive images; Hoffa-synovitis on mid-sagittal FS; cartilage on FS TSE, with gradient echo discouraged [V] — [PMC4058435](https://pmc.ncbi.nlm.nih.gov/articles/PMC4058435)
- **Two-slice-touch:** lateral PPV gain 83% → 96% [S] — [ScienceDaily](https://www.sciencedaily.com/releases/2006/12/061207160020.htm)
- **Cross-finding structure:**
  - MOAKS "effusion-synovitis" deliberately conflates fluid and synovium [V] — [PMC4058435](https://pmc.ncbi.nlm.nih.gov/articles/PMC4058435)
  - Hoffa signal flags 71–88% of CE synovitis at 30–55% specificity [V] — [PMC3578385](https://pmc.ncbi.nlm.nih.gov/articles/PMC3578385)
  - Individual OA features discriminate radiographic tibiofemoral OA with C = 0.61–0.73 [S] — [NDORMS](https://www.ndorms.ox.ac.uk/publications/167725)
  - Coronal lateral extrusion ≥2.2 mm is itself a root-tear sign: 50% sensitivity, 89% specificity [V] — [PMC12205407](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12205407/)

### Inferences
- **(c) Cheap changes, ranked by expected value per cost:**
  1. **Laterality canonicalization**, if not already complete. It costs nothing extra at inference if training used the same canonical orientation, and it removes a systematic medial/lateral error source.
  2. **FS-first axial selection.** When a study has both an FS and a plain axial, also run inference with each and average the predictions (series-swap TTA). This hedges the selection rule and changes rankings.
  3. **Out-of-fold stacker over all 12 finding logits** plus the plane- and tile-specific outputs, e.g. logistic regression or a small GBM per weak finding. Validate with fold-level bootstrap. Expected targets:
     - synovitis, from the effusion output;
     - lateral OA, from lateral-meniscus and extrusion-like signals;
     - PF OA, from effusion and patellar signals.
  4. **Top-2/3 slice pooling** for the lateral meniscus head. This works at inference only if the head already exposes per-slice scores; otherwise it needs retraining.
  5. **Inference-only crops are not supported by the evidence.** Tack's crop gains came from models trained on crops; cropping only at test time would be out of distribution.
- Temperature scaling, Platt calibration and threshold tuning are monotone transforms, so they cannot change per-finding AUC. They matter only if the metric changes.

### Gaps
- No knee-MRI paper reports how test-time augmentation (flips, multi-crop, slice jitter) affects AUC for these findings.
- No paper tests cross-finding stacking or series-swap TTA on knee MRI.
- No study gives effusion and synovitis co-occurrence rates in expert-labelled clinical (non-OAI) knee MRI.
