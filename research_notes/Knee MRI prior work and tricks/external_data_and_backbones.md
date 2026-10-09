# External knee MRI datasets and pretrained MRI backbones for the RSNA Knee Abnormality Detection competition (licences, cost, evidence)

Research date: 2026-10-08. Competition context comes from the brief: 4,407 studies, 58 gold-labelled, weak findings are synovitis, fracture, lateral OA, contusion and MCL. Prize rule: the winning solution must be open-sourced under an OSI licence that does not limit commercial use. External data and models must be "publicly available and equally accessible to all participants at no cost". These rules were taken from the brief and not re-checked against the Kaggle rules page.

Notation for our 12 findings: ACL, MCL, MM (medial meniscus tear), LM (lateral meniscus tear), MOA / LOA / PFOA (medial / lateral / patellofemoral OA), EFF (effusion), SYN (synovitis), BAK (Baker's cyst), BME (bone contusion/BME), FX (fracture).
Bandwidth: 7 MB/s is about 25 GB/hour. All download-time figures below are my arithmetic from that rate.
"UNVERIFIED" marks claims I could not confirm against a primary source in this session.

---

## Q1. Which external knee MRI datasets exist, what is in them, and how do their labels map onto our 12 findings? (Table A1)

### Takeaway
Only three sources have radiologist-level labels for many of our 12 findings. fastMRI+ boxes on 1,172 coronal fastMRI volumes cover ACL, MCL, meniscus, effusion, cysts, BME and fracture/contusion. OAI MOAKS readings cover ACL, medial/lateral meniscus, compartment OA, effusion-synovitis, Hoffa-synovitis, Baker's cyst and BML. MRNet and KneeMRI only add ACL and meniscus labels, and our labels for those two are already relatively strong.

### Cited Findings

**Table A1: content and label mapping**

| Dataset | Size | Sequences / planes | Labels provided | Maps onto our 12 findings |
|---|---|---|---|---|
| **MRNet** (Stanford) | 1,370 exams (1,130 train / 120 val / 120 hidden test) — [MRNet page](https://stanfordmlgroup.github.io/competitions/mrnet/) | Protocol includes cor T1, cor T2-FS, sag PD, sag T2-FS, ax PD-FS; 56.6% at 3T — [MRNet page](https://stanfordmlgroup.github.io/competitions/mrnet/). Released data has one sagittal, one coronal and one axial array per exam — [HF re-upload schema](https://huggingface.co/datasets/AUMLProject/mrnet-acl) | Exam-level binary: abnormal 1,104, ACL tear 319, meniscal tear 508, all extracted from clinical reports — [AIMI page](https://aimi.stanford.edu/datasets/mrnet-knee-mris) | ACL yes. Meniscus yes but **no medial/lateral split**. "Abnormal" is too coarse for anything else. |
| **KneeMRI** (Štajduhar 2017, Rijeka) | 917 volumes of left or right knees, 12-bit — [Zenodo READ.ME](https://zenodo.org/records/14789903) | "proton density weighted fat suppression", Siemens Avanto 1.5T, 2006–2014 — [Zenodo READ.ME](https://zenodo.org/records/14789903). The sagittal plane comes from third-party descriptions — [ELNet](https://arxiv.org/pdf/2005.02706). A search snippet calls the series "T1-weighted", which **conflicts** with the READ.ME. | ACL healthy / partial / complete rupture plus a ROI box — [Zenodo](https://zenodo.org/records/14789903). Counts 690 / 172 / 55 — [ELNet via search snippet](https://arxiv.org/pdf/2005.02706) | ACL only. Partial vs complete ACL could map to our tear definition. |
| **fastMRI knee raw** (NYU) | 1,594 exams: train 973, val 199, multicoil test 118, singlecoil test 108, challenge 196 (held back) — [Knoll 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6996599/) | Coronal PD and coronal PD-FS (~36 slices), 1.5T/3T — [Knoll 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6996599/); [fastMRI site](https://fastmri.med.nyu.edu/) | No labels: "we are currently not providing any diagnostic labeling…" — [Knoll 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6996599/) | None directly. Use with fastMRI+ below. |
| **fastMRI+** (knee) | 16,154 boxes over 22 knee categories. All 1,172 train+val knee studies annotated by one MSK radiologist — [fastMRI+ Sci Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC8983757/); [arXiv](https://arxiv.org/abs/2109.03812) | Coronal (same volumes as fastMRI) | Boxes (count / subjects): Meniscus tear 5,658/663; Displaced meniscal tissue 232/56; Subchondral edema 986/196; Bone lesion 183/29; **Fracture/Contusion/Dislocation 1,060/119**; Cartilage full-thickness 615/122; partial-thickness 2,985/588; ACL high-grade 678/101; ACL low-mod 765/153; **MCL high 11/4; MCL low-mod 285/121**; PCL 18/3 and 142/40; LCL 14/3 and 130/48; Joint effusion 1,311/142; Joint bodies 38/11; **Periarticular cysts 864/161**; Muscle strain 65/11; Soft-tissue lesion 90/10; Patellar retinaculum high-grade 24/4 — [Sci Data Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC8983757/) | ACL (high-grade, or high + low-mod), MCL, meniscus (no side label, so medial vs lateral must be inferred from box x-position and knee laterality), EFF, BAK (inside the "periarticular cysts" class), BME (subchondral edema), FX + contusion (one combined class), OA proxy (cartilage loss without a compartment label). **No synovitis.** |
| **fastMRI knee DICOM** (NYU) | 10,012 clinical studies from 9,290 patients — [Knoll 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6996599/) | All five clinical sequences: cor PD, cor PD-FS, ax T2-FS, sag PD, sag T2-FS. Lossless JPEG2000 — [Knoll 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6996599/) | **No labels** — [Knoll 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6996599/) | None. Unlabelled but domain-matched clinical knee MRI, usable for SSL or pseudo-labels. |
| **OAI** (Osteoarthritis Initiative) | 4,796 participants, ~26.6M images — [NDA OAI page via search snippet](https://nda.nih.gov/oai). Bilateral knee MRI at baseline and annual visits, ages 45–79 — [NYU data catalog](https://datacatalog.med.nyu.edu/dataset/10162) | 3T. Sag 3D DESS, cor IW 2D TSE, sag IW FS TSE — [FNIH MOAKS paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5105263/). DESS 160 sections, 0.7 mm, Siemens Trio — [search snippet, OAI protocol](https://pmc.ncbi.nlm.nih.gov/articles/PMC3048821/) | MOAKS semi-quantitative readings. FNIH Biomarkers Consortium: **600 knees** (406 cases, 194 controls) at baseline and 24 months covering cartilage, osteophytes, BML, meniscal damage/extrusion, Hoffa-synovitis, effusion-synovitis — [PMC5105263](https://pmc.ncbi.nlm.nih.gov/articles/PMC5105263/). Total MOAKS-read knees across all OAI projects: UNVERIFIED. 3DReasonKnee uses **7,970** OAI 3D volumes with MOAKS-based grades — [arXiv 2510.20967](https://arxiv.org/abs/2510.20967) | MOAKS items: ACL (complete tear only), MM/LM (tears per horn, medial vs lateral), MOA/LOA/PFOA (cartilage + BML across 14 subregions, osteophytes across 12 regions), EFF/SYN (effusion-synovitis, Hoffa-synovitis), BAK (popliteal cyst present/absent), BME (BML, mostly degenerative). **No MCL/LCL and no fractures in MOAKS** — [Radiopaedia MOAKS](https://radiopaedia.org/articles/mri-osteoarthritis-knee-score-moaks) |
| **3DReasonKnee** (labels for OAI) | 7,970 3D knee MRI volumes, 494k QA "quintuples" — [arXiv](https://arxiv.org/abs/2510.20967) | OAI DESS. **Images are not included**: "download the original raw DICOM files from https://nda.nih.gov/oai/query-download" — [HF card](https://huggingface.co/datasets/rajpurkarlab/3DReasonKnee) | "taxonomymatch csv files with MOAKS scores", nnU-Net masks of subregions and lesions (files include ACL, Effusion-Synovitis, Hoffa-Synovitis, medial/lateral meniscus, BML masks), train/val/test splits — [HF card/file list](https://huggingface.co/datasets/rajpurkarlab/3DReasonKnee) | Same as OAI MOAKS, already joined to image paths. Saves the step of finding and joining MOAKS files on NDA. |
| **MOST** (Multicenter OA Study) | Knee radiographs and knee MRIs through the 168-month visit — [AgingResearchBiobank news](https://agingresearchbiobank.nia.nih.gov/news/23) | UNVERIFIED (sequence and field strength not read). A "MRI 1.5T Gadolinium" dataset description exists — [ARB doc title](https://agingresearchbiobank.nia.nih.gov/studies/most/documents/download/Dataset_Descriptions/MOSTDatasetDescriptionV2MRI15TGadolinium.pdf/) | MOAKS readings, e.g. V7/V9 — [ARB reading protocol doc](https://agingresearchbiobank.nia.nih.gov/studies/most/documents/download/Dataset_Descriptions/DatasetDescriptionReadingProtocol_V79MOAKS.pdf/) (contents not read) | Like OAI MOAKS. A contrast-enhanced subset could carry synovitis grades (UNVERIFIED). |
| **SKM-TEA** (Stanford) | 155 patients, ~25,000 slices — [arXiv 2203.06823](https://arxiv.org/abs/2203.06823) | 3D qDESS — [DATASET.md](https://github.com/StanfordMIMI/skm-tea/blob/main/DATASET.md) | Segmentation of 4 tissues (patellar, femoral, tibial cartilage, meniscus). Bounding boxes for 16 pathologies in COCO JSON (examples given: meniscal tears, ligament tears, cartilage lesions, effusion) — [arXiv](https://arxiv.org/abs/2203.06823); [DATASET.md](https://github.com/StanfordMIMI/skm-tea/blob/main/DATASET.md) | Possibly ACL, meniscus, EFF and cartilage (OA proxy). Too small (155) to matter. |
| **K2S** (MICCAI 2022, UCSF) | 300 train/val + 50 test — [K2S data page](https://k2s.grand-challenge.org/Data/) | GE 3T 3D CUBE, 8× undersampled k-space — [K2S](https://k2s.grand-challenge.org/Data/) | Bone and cartilage segmentation only (6 classes) — [search snippet](https://k2s.grand-challenge.org/Data/) | None of our findings. |
| **OAI-ZIB / OAIZIB-CM** | 507 MRIs (404 train + 103 test) as NIfTI — [HF](https://huggingface.co/datasets/YongchengYAO/OAIZIB-CM) | OAI 3D DESS | Segmentation: femur, femoral cartilage, tibia, medial and lateral tibial cartilage — [HF](https://huggingface.co/datasets/YongchengYAO/OAIZIB-CM) | No pathology labels. Medial/lateral cartilage masks are only an indirect OA proxy. |
| **KneeCoT** (HF, 2026) | 25K 3D knee MRI volumes, 600K VQA, 1.05 TB — [HF card](https://huggingface.co/datasets/YiHui0124/KneeCoT) | Not stated | Meniscus, ligaments (ACL, PCL, collaterals), cartilage, bone lesions, effusion, damage grades — [HF card](https://huggingface.co/datasets/YiHui0124/KneeCoT) | Broad coverage on paper, but see access below. |
| **MeniMV** (arXiv 2025) | 3,000 exams, 750 patients, 3 centres — [arXiv](https://arxiv.org/html/2512.18437v1) | T2-FS sagittal + coronal | Meniscal horn grade 0–3 — [arXiv](https://arxiv.org/html/2512.18437v1) | Meniscus only. No public download URL found. |

Other Hugging Face hits from a [search of HF datasets for "knee"](https://huggingface.co/api/datasets?search=knee):
- `AUMLProject/mrnet-acl` is an unofficial MRNet re-upload: 1,130/120 exams, 32×224×224 fp16, 7.25 GB — [HF](https://huggingface.co/datasets/AUMLProject/mrnet-acl).
- `lisherlock/13-structure-segmentation-of-knee-osteoarthritis` holds OAI-derived masks under CC-BY-SA-4.0 — [HF](https://huggingface.co/datasets/lisherlock/13-structure-segmentation-of-knee-osteoarthritis).
- `charwisreno/OAI-Knee-MRI` is manually gated, licence "other".
- Several repos whose names suggest RSNA-knee competition re-uploads (`kevlar-ethereal-SOTA/rsna-knee-336-12slices`, `Zera123098/RSNA-Knee-336px-npz`, `ragunath-ravi/rsna-knee-features`). I did not inspect them.

### Inferences
- For our weak findings, fastMRI+ and OAI-MOAKS are the only sources that matter (see Q3). MRNet and KneeMRI duplicate ACL and meniscus, which are not where we are weak.
- fastMRI+ is coronal-only and gives no meniscus side. Mapping MM vs LM, or medial vs lateral cartilage loss, needs box x-position plus laterality. Laterality may not be in raw HDF5 headers (UNVERIFIED). This is a real ingest cost.
- OAI comes from a 45–79-year-old OA cohort on one vendor and protocol. MOAKS BMLs are mostly degenerative, so they are a weak proxy for traumatic contusion. Acute fractures are essentially absent.
- RSNA-looking HF re-uploads are competition data, not external data (Rule 4.b). Do not use them.

### Gaps
- Exact list of SKM-TEA's 16 pathology classes, and total number of knees with MOAKS across all OAI projects: not retrieved.
- Zenodo API keyword search returned no hits (probably a query-syntax issue), so I found no other Zenodo-hosted labelled knee MRI sets beyond KneeMRI and OAIZIB-CM.
- Kaggle-hosted knee MRI datasets were not enumerated. The Kaggle site is JS-rendered and only general web search was used.
- Whether fastMRI single-coil files share file names with the multicoil volumes annotated by fastMRI+ is UNVERIFIED. It is likely, since single-coil data are emulated from multicoil. Check `knee_file_list.csv` in the fastMRI+ repo.

---

## Q2. Is each dataset "publicly available, equally accessible at no cost", and is its licence compatible with open-sourcing the winning solution under an OSI licence that allows commercial use? (Table A2)

### Takeaway
Every labelled knee MRI dataset that is actually obtainable is non-commercial or research-only: MRNet, KneeMRI, fastMRI/fastMRI+, SKM-TEA, OAIZIB-CM, KneeCoT. OAI's own terms could not be read without logging in. MOST and KneeCoT need institutional or hospital agreements, and K2S is no longer distributed, so those three fail the "equally accessible" test. Using any non-commercial dataset puts the prize at risk unless the hosts or Kaggle's winner-licence clause explicitly allow it.

### Cited Findings

**Table A2: access, licence, cost, eligibility**

| Dataset | Access procedure and approval time | Licence terms (commercial / redistribution / derivatives) | Download size | Free and equally accessible? | Licence risk for prize |
|---|---|---|---|---|---|
| MRNet | Now distributed through Stanford AIMI and Redivis (https://stanford.redivis.com/datasets/4a2c-4cpkzrn2c); web login required — [AIMI MRNet](https://aimi.stanford.edu/datasets/mrnet-knee-mris). Approval time UNVERIFIED. | Research Use Agreement "prohibiting commercial use, redistribution, and re-identification" — [AIMI page via search snippet](https://aimi.stanford.edu/datasets/mrnet-knee-mris); full RUA text could not be rendered. AIMI datasets are "shared broadly under a license that is restricted to non-commercial purposes". The commercial licence costs **$70,000 per dataset per year**, with up to 4 weeks of review — [AIMI commercial use](https://aimi.stanford.edu/datasets-commercial-use) | Official zip size UNVERIFIED. An HF re-upload of the processed arrays is 7.25 GB — [HF](https://huggingface.co/datasets/AUMLProject/mrnet-acl) | Free for non-commercial research with registration | **High.** Non-commercial; the commercial route is paid. The HF re-upload breaches "no redistribution". |
| KneeMRI | Open Zenodo download, instant — [Zenodo](https://zenodo.org/records/14789903) | **CC BY-NC-ND 4.0**; must cite Štajduhar 2017 and acknowledge CHC Rijeka — [READ.ME](https://zenodo.org/records/14789903) | 3.7 GB (`volumetric_data.7z`), about 9 min | Yes | **High.** NonCommercial and NoDerivatives. Whether trained weights count as "adapted material" is legally unclear. |
| fastMRI raw + DICOM | Online application with affiliation and intended use. A separate DSA for each dataset type; the download link is e-mailed — [fastMRI site](https://fastmri.med.nyu.edu/). "No requirement to submit a research proposal" — [Knoll 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6996599/). Approval time UNVERIFIED. | DSA: "download link … without charge for **internal research or educational purposes only**". "Not DISTRIBUTE, PUBLISH, REPRODUCE, RETRANSMIT, COMMERCIALLY EXPLOIT, COPY OR OTHERWISE TRANSFER any portion or all of the fastMRI Dataset **or any subsequent variables or data files derived from the fastMRI Dataset**…". "I will NOT SHARE THE DOWNLOAD LINK" — [fastMRI site](https://fastmri.med.nyu.edu/). Paper: "we do not allow any commercial use of the dataset itself" but "we do not explicitly discourage the development or testing of software, algorithms, or other intellectual property with the dataset" — [Knoll 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6996599/). The site says MLPerf benchmark use counts as non-commercial; prize competitions are not mentioned — [fastMRI site](https://fastmri.med.nyu.edu/) | Single-coil: train 88 GB, val 19 GB, test 7 GB. Multicoil: train 931 GB, val 192 GB, test 109 GB. DICOM: 134 + 30 = **164 GB** — [Knoll 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6996599/) | Free with click-through DSA | **Medium-high.** Non-commercial, and the "derived data files" clause could be read as covering trained weights. Kaggle prize use is not addressed. |
| fastMRI+ annotations | GitHub/Synapse, no gate — [repo](https://github.com/microsoft/fastmri-plus) | Repo **MIT** — [GitHub licence API](https://github.com/microsoft/fastmri-plus/blob/main/LICENSE). Paper CC BY 4.0 — [Sci Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC8983757/). Images still need the fastMRI DSA, "download and use the data for academic purposes after signing the data sharing agreement" — [repo](https://github.com/microsoft/fastmri-plus) | CSV, under 10 MB (estimate) | Yes (labels) | Labels are fine. The images carry the fastMRI risk above. |
| OAI | Log in to NDA through RAS (eRA Commons, **Login.gov**, or PIV/CAC), create or link an NDA account, then "agree to OAI data access terms". Listed as "Free to All" — [NYU data catalog](https://datacatalog.med.nyu.edu/dataset/10162). "Clinical and imaging datasets can be instantly downloaded through the central NDA portal" — [NIAMS page via search snippet](https://www.niams.nih.gov/grants-funding/funded-research/osteoarthritis-initiative/osteoarthritis-initiative-application) (page not fetched). Generic NDA controlled access requires a DUC co-signed by an institutional official with an FWA, while open-access groups need only an account — [NDA access info via search snippet](https://nda.nih.gov/nda/access-data-info). | OAI-specific terms text **UNVERIFIED** (behind login). Commercial-use and redistribution clauses unknown. | Imaging size UNVERIFIED. ~26.6M images in total; selective download per series is required. | Probably yes: Login.gov is open to individuals and the terms are click-through. **Verify by logging in today.** | **Unknown** until the OAI terms are read |
| 3DReasonKnee (OAI labels) | HF, **ungated** — [HF API/card](https://huggingface.co/datasets/rajpurkarlab/3DReasonKnee) | **CC BY 4.0** — [HF card](https://huggingface.co/datasets/rajpurkarlab/3DReasonKnee) | about 1.09 GB original files — [HF datasets-server size](https://datasets-server.huggingface.co/size?dataset=rajpurkarlab/3DReasonKnee) | Yes | Low for the labels. Images inherit the OAI terms. |
| MOST | Request for Materials through the NIA AgingResearchBiobank — [ARB news](https://agingresearchbiobank.nia.nih.gov/news/23). Must be submitted by a PI or senior researcher; "A Signing Official from your institution will be asked to provide their signature"; agreements expire after 12 months — [ARB How to make a request via search snippet](https://agingresearchbiobank.nia.nih.gov/how-to-make-a-request) | Governed by ARB terms (OHRTA/OHDTA). Text UNVERIFIED. | UNVERIFIED | **No**: needs an institutional signature and weeks of review | Moot |
| SKM-TEA | Stanford AIMI portal (Azure) — [DATASET.md](https://github.com/StanfordMIMI/skm-tea/blob/main/DATASET.md) | Code MIT — [repo](https://github.com/StanfordMIMI/skm-tea). Data terms presumably the AIMI non-commercial RUA (UNVERIFIED for SKM-TEA specifically) — [AIMI](https://aimi.stanford.edu/datasets-commercial-use) | 900 GB compressed / 1.6 TB expanded — [DATASET.md](https://github.com/StanfordMIMI/skm-tea/blob/main/DATASET.md) | Free for non-commercial use | High, and impractical |
| K2S | "WE ARE NOT SHARING ANY DATA AFTER THE CHALLENGE HAS BEEN CONCLUDED" — [K2S](https://k2s.grand-challenge.org/Data/) | CC BY-NC-ND — [K2S](https://k2s.grand-challenge.org/Data/) | n/a | **No** (unavailable) | Moot |
| OAIZIB-CM | HF, no gate — [HF](https://huggingface.co/datasets/YongchengYAO/OAIZIB-CM); Zenodo DOI [10.5281/zenodo.14934086](https://doi.org/10.5281/zenodo.14934086) | **CC BY-NC 4.0**; citing CartiMorph is compulsory — [HF](https://huggingface.co/datasets/YongchengYAO/OAIZIB-CM) | 12.7 GB | Yes | High (non-commercial) and low value |
| KneeCoT | Contact-info gate; users must "sign a formal ethical use agreement with the affiliated hospital" — [HF](https://huggingface.co/datasets/YiHui0124/KneeCoT) | CC-BY-NC-4.0 | 1.05 TB | **No** (hospital agreement) | Moot |

The Stanford AIMI commercial licence is $70,000 per year per dataset and "negotiated fees and discounts are not available" — [AIMI](https://aimi.stanford.edu/datasets-commercial-use). Commercial-grade access to MRNet is therefore not "at no cost".

ImageNet itself is non-commercial: "Researcher shall use the Database only for non-commercial research and educational purposes" — [image-net.org](https://www.image-net.org/download.php). This shows that dataset-level non-commercial terms on pretraining data are normal. The weights trained on ImageNet are licensed separately (see Q5).

A competition precedent: on DrivenData, an organiser ruled that DINOv3-based models were not prize-eligible because the winners' MIT licence and the DINOv3 Agreement conflict — [DrivenData forum](https://community.drivendata.org/t/is-dinov3-allowed-to-use/11488).

### Inferences
- Datasets that pass "free and equally accessible": KneeMRI, fastMRI, fastMRI+ labels, MRNet (registration plus RUA), OAI (very likely, via Login.gov), 3DReasonKnee labels, OAIZIB-CM. Fail: MOST, KneeCoT, K2S.
- Whether "non-commercial research only" covers a prize competition is ambiguous for all of them. The fastMRI DSA's "derived data files" clause is the strongest reason a trained model might be caught. The safe action is a host ruling on the forum before relying on any of these for a prize-eligible submission.

### Gaps
- Kaggle's standard rules may say that winners need not grant an open-source licence for third-party input data or pretrained models with incompatible licences. I recall this wording but **could not verify it** (search found nothing). Check the competition's Rules tab, Winner License / External Data sections, and the forum. This one clause decides whether non-commercial external data is a prize risk.
- OAI data access terms text, and whether an account with Login.gov only and no institution is enough in practice, are UNVERIFIED.
- Full MRNet RUA text was not rendered (AIMI and Redivis pages are JS).

---

## Q3. Which datasets cover our weakest findings (synovitis, fracture, bone contusion/BME, lateral OA, MCL)?

### Takeaway
For FX, BME/contusion and MCL, fastMRI+ is the only accessible source with expert boxes, although with modest subject counts (119, 196 and 125 subjects). For SYN and LOA, OAI-MOAKS is the only accessible source (Hoffa-synovitis, effusion-synovitis, lateral-compartment cartilage, BML and osteophyte scores), with labels conveniently packaged by 3DReasonKnee. Nothing accessible covers all five.

### Cited Findings
- **Synovitis**: MOAKS grades Hoffa-synovitis (sagittal, 0–3) and effusion-synovitis (axial, 0–3) — [Radiopaedia](https://radiopaedia.org/articles/mri-osteoarthritis-knee-score-moaks); [FNIH paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5105263/). 3DReasonKnee ships "Hoffa-Synovitis" and "Effusion-Synovitis" masks per OAI case — [HF file list](https://huggingface.co/datasets/rajpurkarlab/3DReasonKnee). fastMRI+ has **no** synovitis class — [Sci Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC8983757/). MOST has a 1.5T gadolinium MRI dataset (contents UNVERIFIED) but is not accessible — [ARB](https://agingresearchbiobank.nia.nih.gov/studies/most/documents/download/Dataset_Descriptions/MOSTDatasetDescriptionV2MRI15TGadolinium.pdf/).
- **Fracture**: fastMRI+ "Bone-Fracture/Contusion/Dislocation" has 1,060 boxes in 119 subjects; this one class mixes fracture and contusion — [Sci Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC8983757/). MOAKS does not score fractures — [Radiopaedia](https://radiopaedia.org/articles/mri-osteoarthritis-knee-score-moaks).
- **Bone contusion / BME**: fastMRI+ "Bone-Subchondral Edema" has 986 boxes in 196 subjects, plus the fracture/contusion class above — [Sci Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC8983757/). MOAKS scores BML size in 14–15 subregions — [Radiopaedia](https://radiopaedia.org/articles/mri-osteoarthritis-knee-score-moaks).
- **Lateral OA**: MOAKS scores cartilage and BML per subregion (lateral femur and tibia) and osteophytes in 12 regions — [Radiopaedia](https://radiopaedia.org/articles/mri-osteoarthritis-knee-score-moaks). fastMRI+ cartilage boxes (partial 588 subjects, full 122) carry no compartment label — [Sci Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC8983757/). OAIZIB-CM has lateral tibial cartilage masks only — [HF](https://huggingface.co/datasets/YongchengYAO/OAIZIB-CM).
- **MCL**: fastMRI+ MCL low-moderate sprain has 285 boxes in 121 subjects, and high-grade 11 boxes in 4 subjects — [Sci Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC8983757/). MOAKS has **no** MCL/LCL items — [Radiopaedia](https://radiopaedia.org/articles/mri-osteoarthritis-knee-score-moaks). KneeCoT lists collaterals but is not accessible — [HF](https://huggingface.co/datasets/YiHui0124/KneeCoT).
- Also covered: fastMRI+ joint effusion (142 subjects) and periarticular cysts (161 subjects) — [Sci Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC8983757/). MOAKS has a popliteal (Baker's) cyst item — [Radiopaedia](https://radiopaedia.org/articles/mri-osteoarthritis-knee-score-moaks).

### Inferences
- fastMRI+ high-grade MCL (4 subjects) is useless as a "tear" label. Low-moderate sprain (121 subjects) only helps if our MCL definition includes sprains or partial tears. Check our label spec.
- The fastMRI+ fracture/contusion class lumps both together. Training a combined "bone injury" auxiliary head, or masking FX/BME separately, is safer than assigning these boxes to FX alone.
- OAI synovitis labels come from non-contrast surrogates (Hoffa, effusion-synovitis). If our "synovitis" label comes from report text, the semantics may only partly match. For example, radiology reports often say "synovitis" for any thickened synovium.

### Gaps
- No accessible dataset with explicit fracture-only or MCL-tear labels in meaningful numbers was found.
- No subtype breakdown of fastMRI+ "periarticular cysts" (Baker's vs ganglion vs parameniscal) was found.

---

## Q4. Practical cost: size, format, approval time, and how to stay within ~60 GB and 7 MB/s

### Takeaway
fastMRI+ needs about 107 GB of single-coil train+val downloads, about 4.3 h at 7 MB/s. The ground-truth reconstructions are inside the HDF5 files, so the data can be stream-extracted to under 10 GB without keeping the raw k-space. OAI needs an NDA package build plus a selective image download whose size and time I could not pin down. MRNet and KneeMRI are small (under 10 GB). fastMRI DICOM (164 GB, ~6.5 h), SKM-TEA (1.6 TB) and KneeCoT (1.05 TB) are expensive or infeasible.

### Cited Findings
- fastMRI train/val HDF5 files hold reconstructed targets. The official loader reads `reconstruction_esc` (single-coil) or `reconstruction_rss` (multicoil) "if self.recons_key in hf" — [fastMRI `mri_data.py`](https://github.com/facebookresearch/fastMRI/blob/main/fastmri/data/mri_data.py). So no k-space reconstruction is needed.
- fastMRI sizes: single-coil train 88 GB / val 19 GB; multicoil train 931 GB / val 192 GB; DICOM 134 + 30 GB, lossless JPEG2000 — [Knoll 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6996599/).
- KneeMRI: 3.7 GB 7z of Python pickles (`.pck`) — [Zenodo](https://zenodo.org/records/14789903).
- MRNet: an HF re-upload of the processed arrays is 7.25 GB — [HF](https://huggingface.co/datasets/AUMLProject/mrnet-acl). Official size UNVERIFIED.
- OAIZIB-CM: 12.7 GB NIfTI — [HF](https://huggingface.co/datasets/YongchengYAO/OAIZIB-CM). 3DReasonKnee labels: about 1.09 GB — [HF size API](https://datasets-server.huggingface.co/size?dataset=rajpurkarlab/3DReasonKnee).
- SKM-TEA: 900 GB compressed — [DATASET.md](https://github.com/StanfordMIMI/skm-tea/blob/main/DATASET.md). KneeCoT: 1.05 TB — [HF](https://huggingface.co/datasets/YiHui0124/KneeCoT).
- OAI DESS is 160 sections at 0.7 mm — [search snippet, OAI protocol](https://pmc.ncbi.nlm.nih.gov/articles/PMC3048821/). Images are downloaded from https://nda.nih.gov/oai/query-download — [3DReasonKnee card](https://huggingface.co/datasets/rajpurkarlab/3DReasonKnee).
- Approval times: KneeMRI and the HF sets are instant. OAI is "instantly downloaded" per NIAMS ([snippet](https://www.niams.nih.gov/grants-funding/funded-research/osteoarthritis-initiative/osteoarthritis-initiative-application), UNVERIFIED in practice). fastMRI is a form plus e-mailed link, timing UNVERIFIED ([site](https://fastmri.med.nyu.edu/)). MRNet is a Redivis login plus RUA, timing UNVERIFIED. MOST needs an institutional signature (weeks) ([ARB](https://agingresearchbiobank.nia.nih.gov/how-to-make-a-request)). The AIMI commercial licence takes up to 4 weeks plus $70k ([AIMI](https://aimi.stanford.edu/datasets-commercial-use)).

### Inferences (my estimates; arithmetic, not sourced)
- fastMRI+ ingest plan: stream `knee_singlecoil_train.tar.gz` (88 GB, ~3.5 h) and `..._val` (19 GB, ~45 min) through Python `tarfile` in streaming mode (`r|gz`). Keep only `reconstruction_esc`/`reconstruction_rss` per file, cast to uint8 or fp16 at 320×320, and discard k-space. Footprint: about 1,172 vol × 36 slices × 320² × 2 B ≈ 8.6 GB fp16, or about 4.3 GB uint8. The 88 GB archive never needs to land on the NFS.
- fastMRI DICOM (164 GB, ~6.5 h) can also be streamed and down-sampled. 10,012 studies × 5 series × ~16 resampled slices at 224² uint8 comes to about 40 GB. This is tight against the 60 GB budget, so subsample (for example 2 series per study).
- Multicoil (931 GB, ~37 h), SKM-TEA (~36 h) and KneeCoT (~42 h) do not fit the window at 7 MB/s.
- OAI per-series size is UNVERIFIED. Rough guess: 2D TSE series of ~10–20 MB each. 3,000 sagittal IW-TSE-FS series would then be about 30–60 GB (~1.5–2.5 h), plus NDA package-building overhead.

### Gaps
- Real turnaround times for fastMRI and Redivis/MRNet approval were not found. Submit the forms today if either is on the table.
- OAI image package sizes and NDA download tooling speed were not found.

---

## Q5. Pretrained backbones for 2D/2.5D/3D MRI: licence, input type, and evidence of gains over ImageNet init (Table B)

### Takeaway
No medical or MRI backbone has strong, independent evidence of beating good ImageNet-21k/12k timm weights on knee MRI classification. RadImageNet's knee gains (+4.8% AUC on ACL, +4.5% on meniscus) come from its own paper on 2D slices. Two later studies found ImageNet equal or better. DINOv2 matches ImageNet-21k supervised ViT-L on an MRI benchmark. Licence-clean options (OSI-compatible) are timm IN-21k/12k weights (Apache-2.0/MIT, except ConvNeXt-V2 FCMAE, which is CC-BY-NC), DINOv2, MedSAM, SAM-Med3D, MedicalNet, RadImageNet, Rad-DINO and BiomedCLIP (MIT), and MRI-CORE (Apache-2.0). Avoid DINOv3 (custom), MedSigLIP (HAI-DEF), Curia (research-only RAIL-M), Models Genesis (non-commercial) and ConvNeXt-V2 FCMAE.

### Cited Findings

**Table B: backbones**

| Backbone | Licence (verified where linked) | Input / pretraining data | Evidence on MRI classification vs ImageNet |
|---|---|---|---|
| **timm ImageNet-21k/12k weights** | Apache-2.0: `coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k`, `convnext_base.fb_in22k_ft_in1k`, `vit_base_patch16_224.augreg2_in21k_ft_in1k`, `tf_efficientnetv2_m.in21k_ft_in1k`. MIT: `eva02_base_patch14_448.mim_in22k_ft_in22k_in1k`. **CC-BY-NC-4.0: `convnextv2_base.fcmae_ft_in22k_in1k`** — [HF API, e.g. timm/convnextv2](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k); [timm/coatnet](https://huggingface.co/timm/coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k). timm code Apache-2.0 — [GitHub](https://github.com/huggingface/pytorch-image-models) | 2D RGB natural images | Baseline. ViT-L/16 IN-21k linear probe on brain-tumour MRI reaches AUROC 0.975 vs ResNet152 IN-1k 0.948 — [Baharoon et al.](https://arxiv.org/html/2312.02366v4) |
| **RadImageNet** (ResNet50, DenseNet121, InceptionV3, IRv2) | Repo **MIT**; weights on Google Drive (TF and PyTorch); the dataset itself is by request — [GitHub](https://github.com/BMEII-AI/RadImageNet). No separate weights licence stated (presumably MIT, UNVERIFIED). HF third-party copy tagged MIT — [Lab-Rasool/RadImageNet](https://huggingface.co/Lab-Rasool/RadImageNet) | 2D, 1.35M CT/MRI/US images, 131,872 patients; MRI includes knee — [Mei 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/) | Own paper on 2D knee slices (1,021 ACL-tear and 4,201 meniscal-tear images): ACL AUC 0.97 vs 0.91 IN (+4.8%); meniscus 0.96 vs 0.92 (+4.5%); "superiority … more evident on small datasets" — [Mei 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/). **Contradicted**: "ResNet50 models pre-trained on ImageNet tend to outperform those trained on RadImageNet" (7 tasks) — [Juodelyte 2023](https://arxiv.org/abs/2302.08272). No evidence RadImageNet beats ImageNet for ACL tear — [Frees et al. 2025](https://arxiv.org/abs/2508.17567) |
| **MedicalNet / Med3D** (Tencent) | **MIT** — [LICENSE](https://github.com/Tencent/MedicalNet/blob/master/LICENSE); [HF](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18) | 3D ResNets pretrained on multi-dataset 3D segmentation (corpus details UNVERIFIED) | No knee MRI classification evidence found |
| **Models Genesis** | **Non-commercial only**: "solely for non-commercial use … no licenses to use the Work for commercial or for-profit purposes are granted"; patent pending — [LICENSE](https://github.com/MrGiovanni/ModelsGenesis/blob/master/LICENSE) | 3D U-Net-style self-supervised (CT-based; specifics UNVERIFIED) | Claims to beat 3D training from scratch — [GitHub](https://github.com/MrGiovanni/ModelsGenesis). No knee MRI evidence. **Incompatible.** |
| **Rad-DINO** (Microsoft) | **MIT** LICENSE file, with commit "Use MIT license (#9)" on 2025-06-27 after an earlier "Change license (#1)" on 2024-08-19 — [HF commits](https://huggingface.co/microsoft/rad-dino/commits/main); [LICENSE](https://huggingface.co/microsoft/rad-dino). README still says "RAD-DINO is shared for research purposes only" — [card](https://huggingface.co/microsoft/rad-dino). The original licence (reportedly MSRLA) is UNVERIFIED. | 2D ViT-B/14 (DINOv2), 518 px, **chest X-ray only** (882,775 images) — [card](https://huggingface.co/microsoft/rad-dino) | No MRI evidence. Wrong modality. |
| **BiomedCLIP** | **MIT**; "Any deployed use case of the model --- commercial or otherwise --- is currently out of scope" — [card](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | 2D ViT-B/16 at 224, PMC-15M figure-caption pairs | No knee MRI evidence found |
| **MedSAM** | **Apache-2.0** — [GitHub](https://github.com/bowang-lab/MedSAM); [HF](https://huggingface.co/wanglab/medsam-vit-base) | 2D SAM ViT-B (segmentation) | No classification evidence found |
| **SAM-Med3D (turbo)** | Code **Apache-2.0** — [GitHub](https://github.com/uni-medical/SAM-Med3D). HF weights repo has no licence field — [HF](https://huggingface.co/blueyo0/SAM-Med3D) | 3D, 128³; SA-Med3D-140K (143K masks, 245 categories) | Not evaluated for classification — [GitHub](https://github.com/uni-medical/SAM-Med3D) |
| **DINOv2** | **Apache-2.0** — [HF facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base); [timm](https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m) | 2D ViT S/B/L/g, natural images (LVD-142M) | Brain-tumour MRI linear probe AUROC: ViT-g 0.976, L 0.974, B 0.972, S 0.962 vs IN-21k ViT-L 0.975 and DenseNet201 0.960 — [Baharoon et al.](https://arxiv.org/html/2312.02366v4). Another comparison found DINOv2 weaker than ImageNet CNNs on clinical MRI sets — [summary](https://astro.paperswithcode.com/paper/comparative-analysis-of-imagenet-pre-trained) |
| **DINOv3** | **Custom "DINOv3 License"**: commercial use allowed; redistribution must include the Agreement and display "Built with DINOv3"; ITAR/military/weapons uses prohibited; terminable — [Meta licence](https://ai.meta.com/resources/models-and-libraries/dinov3-license). Not OSI. HF `facebook/` repos manually gated; `timm/vit_base_patch16_dinov3.lvd1689m` ungated with the same licence — [HF](https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m). DrivenData ruled it not prize-eligible — [forum](https://community.drivendata.org/t/is-dinov3-allowed-to-use/11488) | 2D ViT / ConvNeXt, LVD-1689M | OrthoFoundation (DINOv3 backbone, 1.2M knee X-ray+MRI images) "matching supervised baselines using only 50% of labeled data"; weights not found; paper CC BY-NC-ND — [arXiv 2601.18250](https://arxiv.org/abs/2601.18250) |
| **MRI-CORE** (Duke, 2025) | "The code and models are released under the **Apache 2.0 License**"; weights on Google Drive — [GitHub README](https://github.com/mazurowski-lab/mri_foundation). GitHub's licence API finds no LICENSE file in the repo, so this is a minor discrepancy. | 2D ViT-B with SAM init, DINOv2 objective, **1024×1024** slices; Duke-110K: 116,806 volumes, 18 body locations — [arXiv](https://arxiv.org/html/2506.12186v2) | Classification probes include KneeMRI, but **no ImageNet baseline numbers**; compared only with SAM/MedSAM — [arXiv](https://arxiv.org/html/2506.12186v2) |
| **MARS** (Nature BME 2026, HKUST) | Code **Apache-2.0**; weights on Google Drive with **no weights licence stated** — [GitHub](https://github.com/zqiuak/MARS) | 3D Swin (B/L/H), 336,476 multi-sequence MRI volumes, head-to-knee — [SmartX page](https://smartx.cse.ust.hk/2026/07/15/mars/); [GitHub](https://github.com/zqiuak/MARS) | Best on 41/44 benchmarks, "notably strong results on knee abnormality diagnosis and osteoarthritis grading"; numbers not retrieved — [SmartX](https://smartx.cse.ust.hk/2026/07/15/mars/) |
| **OrthoDiffusion** (2026) | Weights availability not stated — [arXiv](https://arxiv.org/abs/2602.20752) | 3 orientation-specific 3D diffusion models, 15,948 unlabelled knee MRIs | Detects 8 knee abnormalities; keeps precision with 10% of labels (abstract) — [arXiv](https://arxiv.org/abs/2602.20752) |
| **Curia** (Raidium) | **"research-only RAIL-M"**, gated — [HF](https://huggingface.co/raidium/curia) | 2D ViT (~86M), 150k CT/MRI exams | Has a "kneeMRI" task head; no metrics shown — [HF](https://huggingface.co/raidium/curia). **Incompatible.** |
| **MedSigLIP** (Google) | **Health AI Developer Foundations terms** (custom), gated — [HF](https://huggingface.co/google/medsiglip-448); [terms](https://developers.google.com/health-ai-developer-foundations/terms) | 2D SigLIP, 448 | Not checked for MRI |
| **MedImageInsight** (Microsoft) | Third-party HF copy tagged MIT — [lion-ai/MedImageInsights](https://huggingface.co/lion-ai/MedImageInsights). Official HF repo not publicly reachable through the API; official licence UNVERIFIED | 2D, multi-modality | Not checked |

### Inferences
- For a team already on timm IN-12k CoAtNet (Apache-2.0), the expected gain from switching to a medical init is small and uncertain. RadImageNet's knee gains come from its own small 2D-slice setting, and independent replications did not reproduce them. A 7–8 h training run is better spent elsewhere.
- MRI-specific foundation models (MRI-CORE, MARS) are licence-clean or close to it. Their inputs, however, are 2D slices at 1024² (MRI-CORE) or 3D Swin (MARS), which fit poorly into the existing pipeline and 12–24 GB GPUs within days. They also lack published knee-classification comparisons against IN-21k.
- Kaggle commonly accepts ImageNet-pretrained weights even though ImageNet data terms are non-commercial. This suggests the weights' licence is what matters. That reasoning does not clearly carry over to datasets whose own terms reach "derived data files" (fastMRI).

### Gaps
- No head-to-head knee-MRI study comparing DINOv2/DINOv3/MRI-CORE/MARS against IN-21k ConvNeXt/CoAtNet was found.
- MARS knee numbers and the MARS weights licence were not found.
- No evidence was found on 2.5D or 3D medical inits (MedicalNet, SAM-Med3D) for knee classification.

---

## Q6. Evidence that external data helps, and what is worth the time in the 14-day window (Oct 8 → merge Oct 15 → final Oct 22)

### Takeaway
Published evidence that external knee data improves a different multi-site target is thin. The clearest data point shows a domain-shift cost: MRNet trained at Stanford scored ACL AUC 0.824 on Rijeka data with no retraining, versus 0.911 when trained on Rijeka data. **Recommendation:**
1. Keep timm Apache/MIT ImageNet-12k/21k backbones.
2. Do not spend GPU runs on medical inits.
3. Post a host-ruling request today on non-commercial datasets.
4. If the G2 gate says go, ingest **fastMRI+ only**, as masked auxiliary labels for FX/BME/MCL/EFF/BAK (about 4.5 h download, ~1–1.5 days of work).
5. Treat OAI-MOAKS (the only synovitis and lateral-OA source) as a stretch goal for one owner. Do it only if NDA access proves instant.

### Cited Findings
- MRNet → KneeMRI: "AUC of 0.824 … with no additional training". Trained on Rijeka data: 0.911. MRNet used ImageNet-initialised AlexNet; internal AUCs were abnormal 0.937, ACL 0.965, meniscus 0.847 — [Bien et al. 2018, PLoS Med](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699).
- OAI-MOAKS has been used to train detectors: BML detection on 1,899 OAI exams (1,524/182/193 split) with a modified MRNet on sagittal IW-FS images — [medRxiv, details via search snippet](https://www.medrxiv.org/content/10.1101/2022.09.08.22279696v1.full). Effusion estimation from MOAKS-ES reached only ~62% average accuracy on the larger test set — [Sci Rep 2022 via search snippet](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8873489/).
- Knee foundation-model papers report better label efficiency, but not gains on a fixed external benchmark against IN-21k: OrthoFoundation (50% of labels) — [arXiv](https://arxiv.org/abs/2601.18250); OrthoDiffusion (10% of labels) — [arXiv](https://arxiv.org/abs/2602.20752).
- RadImageNet's knee gains, and the contradicting replications, are in Q5 — [Mei 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/); [Juodelyte 2023](https://arxiv.org/abs/2302.08272); [Frees 2025](https://arxiv.org/abs/2508.17567).

### Inferences: ranked plan for the 14-day window
1. **Today, no GPU (highest value per hour):**
   - Post on the competition forum asking whether non-commercial and research-only external datasets (fastMRI/fastMRI+, OAI, MRNet) are allowed for prize-eligible solutions, and how the winner-licence clause treats them.
   - Submit the fastMRI DSA form now so the link is ready if the answer is yes.
   - Create a Login.gov + NDA account and read the actual OAI terms.
2. **Backbones:**
   - Keep timm IN-12k/21k Apache/MIT weights.
   - Remove any ConvNeXt-V2 FCMAE (CC-BY-NC), DINOv3, Curia, MedSigLIP or Models Genesis weights from the plan.
   - If a free GPU slot appears, the cheapest licence-clean ablation is a DINOv2 ViT-B/14 (Apache) or RadImageNet DenseNet121 slice encoder in the existing 2.5D pipeline. Expect at most a small, uncertain gain.
3. **fastMRI+ (go if G2 is yes and the licence is accepted):**
   - Gives 1,172 coronal radiologist-annotated volumes covering FX+contusion (119 subjects), subchondral edema (196), MCL sprain (121), effusion (142), cysts (161), ACL and meniscus.
   - Cost: about 107 GB streamed (~4.3 h at 7 MB/s), about 5–9 GB kept, 1–1.5 days to map boxes to study labels with masked loss. Medial vs lateral must be inferred from box position.
   - Risks: coronal-only; magnitude reconstructions differ from vendor DICOM; non-commercial DSA with a "derived files" clause.
   - Timeline: download Oct 9–10, ingest Oct 10–11, first student with external data Oct 11–12. That fits the G2 → G3 gates.
4. **OAI-MOAKS (stretch):**
   - The only source for SYN and LOA/PFOA compartment labels plus BAK and BML. Labels are free and ungated through 3DReasonKnee (CC BY 4.0, ~1 GB).
   - Images need NDA package building and a selective download of size unknown to me. The domain gap is large: one 3T research protocol, an OA cohort, almost no acute trauma.
   - Worth it only if one person can own it end to end by about Oct 12 and access really is instant.
5. **Skip in this window:**
   - MRNet: duplicates ACL/meniscus, non-commercial, and commercial use costs $70k/yr.
   - KneeMRI: ACL only, CC BY-NC-ND.
   - SKM-TEA: 1.6 TB, 155 cases.
   - K2S: unavailable.
   - MOST: institutional signature, weeks of review.
   - KneeCoT: hospital agreement, 1 TB.
   - OAIZIB-CM: segmentation only.
   - MeniMV: not released.
   - Any HF re-upload of MRNet or RSNA data: licence breach or Rule 4.b.
6. **Optional, higher-upside, licence-gated:** the fastMRI knee DICOM set (10,012 unlabelled clinical studies in the same 5 sequence types) is the best domain match for the planned noisy-student/pseudo-label rounds.
   - Cost: 164 GB (~6.5 h) and on-the-fly down-sampling to stay under ~40 GB.
   - Under the same DSA as fastMRI+, so only after a licence ruling.
   - No published evidence of gains in this exact setup was found.
7. **Hygiene:** if any external dataset is used, run a perceptual-hash near-duplicate check against the competition train/test images. Contributing sites are not public to me, and an overlap (for example NYU or Stanford data) would inflate CV. This is speculative.

### Gaps
- No published study shows that adding fastMRI+, OAI or MRNet improves a multi-site, report-labelled knee MRI classifier like this competition's. Any gain must be measured on our gold-58, where SE is about ±0.02 macro-AUC, so only large gains will be detectable.
- No forum or host ruling on external data for this competition was checked here; that is out of my scope and covered by another research task.
- Whether Kaggle's winner-licence clause exempts incompatible-licence input data or models is UNVERIFIED (see Q2). It decides whether non-commercial data is a prize risk or only a disclosure item.
