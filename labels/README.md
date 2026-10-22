# Label Tables Directory (`labels/`)

This directory collects all major candidate label tables evaluated on the **Gold-58 ground truth** for the RSNA Knee Abnormality Detection competition. All files contain all 4,407 studies and 12 continuous finding columns.

---

### Files Overview

| File Name | Convenient Alias | Description | Gold-58 Macro-AUC |
| :--- | :--- | :--- | :---: |
| `labels_clef_sev.csv` | — | Cloudflare Llama 3.3 70B mapped to 4-tier clinical severity | `0.8900` |
| `labels_sev_v1.csv` | — | Clef Severity + JEV Hybrid Lateral OA spliced in (used for active ConvNeXt fleet) | `0.8940` |
| `HYBRID_labels_scores.csv` | `jev_hybrid.csv` | Kaggle forum hard-swap blend by riadmohamed42 | `0.9020` |
| `FINAL2_labels_scores.csv` | `jev_final2.csv` | OpenJev 6-prompt text-ensemble scores only | `0.8745` |
| `llm_labels_v4_blend.csv` | `stevenleehans_v4.csv` | Kaggle forum heuristic/LLM blend by stevenleehans | `0.8927` |
| `llm_labels_v2.csv` | `pilkwang_v2.csv` | Public label set by pilkwang (strong on ACL, MCL, Fracture) | `0.8873` |
| `result_labels.csv` | — | Newly generated local few-shot Gemma run | `0.8639` |

---

### Quick Blending Example

```python
import pandas as pd

# Load any of the tables
clef = pd.read_csv("labels/labels_clef_sev.csv").set_index("StudyInstanceUID")
jev = pd.read_csv("labels/jev_hybrid.csv").set_index("StudyInstanceUID")
res = pd.read_csv("labels/result_labels.csv").set_index("StudyInstanceUID")

# Example simple blend:
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
       "Medial OA", "Lateral OA", "PF OA", "Effusion",
       "Synovitis", "Baker's", "Contusion", "Fracture"]

blend = 0.50 * jev[LAB] + 0.40 * clef[LAB] + 0.10 * res[LAB]
blend.to_csv("labels/my_custom_blend.csv")
```
