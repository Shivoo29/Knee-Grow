The evidence points to **label quality and label meaning** as the main problem, more than to the number of positive studies.

- **The training labels are noisy proxies.** The project notes estimate substantial Synovitis label noise, because reports often do not mention it. Its hybrid report-derived labels scored only **0.804 AUC on the 74-study checked set**, and the JEV text ensemble alone scored **0.717**. The notes also say some evaluation labels were identified from images, which text-derived training labels cannot reliably capture. [Label results and limitations](/Users/gulatisukaran/Developer/knee-rsna/Knee-Grow/data/labels/README.md:78) [OOF label notes](/Users/gulatisukaran/Developer/knee-rsna/Knee-Grow/purify_labels_oof.py:1)

- **“Synovitis” covers varied findings.** The project’s definition includes Hoffa fat-pad synovitis, while reports may use the term for synovial thickening, inflammation, or findings associated with effusion. That makes positive examples visually and linguistically inconsistent. [Label conventions](/Users/gulatisukaran/Developer/knee-rsna/Knee-Grow/data/labels/README.md:108)

- **The image signal may be subtle or peripheral.** A whole-study model has to find it among other knee findings; a centered 140 mm crop may also lose some edge coverage in off-center studies. That could contribute, but we have no crop-size ablation showing it is the main cause.

One caveat: the binary file’s 1,801 positives are **thresholded report-derived labels**, not 1,801 image-confirmed cases. More positives alone therefore may not make the model learn Synovitis better.
