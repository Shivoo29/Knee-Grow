"""[1] label_llm.py: read each radiology report with local Qwen, emit 12 finding probabilities.

Needs `qwen start` (llama.cpp server on localhost:8080). Reports never leave this machine.
Resumable: answers are appended to a JSONL cache, so a crash or Ctrl+C loses nothing.
Prints only AUC numbers vs the 58 gold studies, never report text.

    .venv/bin/python label_llm.py            # label all 4,407 studies (~3-4 h)
    .venv/bin/python label_llm.py --limit 40 # quick trial
"""
import argparse, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np, pandas as pd, requests
from sklearn.metrics import roc_auc_score

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
          'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

# Definitions follow the gold annotation style: severity-thresholded, borderline graded negative.
SYSTEM = """You are a musculoskeletal radiologist. Read one knee MRI report (any language) and
estimate, for each finding, the probability (0.0-1.0) that an expert would label it PRESENT.
Borderline, mild or equivocal findings count as mostly absent. Not mentioned = absent (low).

ACL: ACL tear, partial or complete, including torn reconstruction graft. Intact graft = absent.
MCL: MCL tear or high-grade sprain. Mild periligamentous edema / grade 1 sprain = mostly absent.
Medial Meniscus: tear of the medial meniscus. Degeneration or intrasubstance signal alone = absent.
