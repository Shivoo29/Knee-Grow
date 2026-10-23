#!/usr/bin/env python3
"""tools/make_r3_labels.py - round-3 base labels: report labels with Lateral OA + PF OA from Gemma 4 + rulebook_final.md.

Merges the Gemma 4 CSVs (node1 all_gemma4_final.csv + molab out_*.csv; rows with ok=0 are ignored), swaps the two
OA columns into the report labels, writes the new base and a 5-fold split. Then run make_r2_labels.py with --base on it.
    uv run -q --with numpy --with pandas --with pyarrow python -I tools/make_r3_labels.py rulebook/all_gemma4_final.csv rulebook/molab/out_*.csv
"""
import json
import sys

import numpy as np
import pandas as pd

SWAP = ["Lateral OA", "PF OA"]
base = pd.read_parquet("pcseq/labels_sev_v1.parquet")
idx = "StudyInstanceUID"
base = base.set_index(idx) if idx in base.columns else base
g = pd.concat([pd.read_csv(p) for p in sys.argv[1:]])
g = g[g["ok"] == 1].drop_duplicates(idx).set_index(idx)
have = base.index.intersection(g.index)
print(f"gemma rows ok: {len(g)} | base studies: {len(base)} | swapped: {len(have)} | missing (keep report label): {len(base) - len(have)}")
for c in SWAP:
    print(f"  {c}: report mean {base[c].mean():.3f} -> gemma mean {g.loc[have, c].mean():.3f}")
    base.loc[have, c] = g.loc[have, c].astype(base[c].dtype).values
base.reset_index().to_parquet("pcseq/labels_r3_base.parquet", index=False)

# the 58 gold studies get the same swap, so make_r2_labels.py --gold_report measures it on expert labels
gold = pd.read_csv("labels/labels_clef_sev.csv").set_index(idx)
gold = gold[[c for c in gold.columns if not c.startswith("Unnamed")]]
gh = gold.index.intersection(g.index)
for c in SWAP:
    gold.loc[gh, c] = g.loc[gh, c].astype(float).values
gold.reset_index().to_csv("labels/labels_r3_gold_report.csv", index=False)
print(f"gold report labels: {len(gh)} studies swapped -> labels/labels_r3_gold_report.csv")

# 5 folds over the same train studies as folds_rand4 (gold excluded), seed 0
ids = sorted(json.load(open("pcr2/folds_rand4.json"))["folds"])
fold = np.random.RandomState(0).permutation(len(ids)) % 5
json.dump({"folds": {u: int(f) for u, f in zip(ids, fold)}, "note": "random 5 folds, seed 0, train studies only (gold excluded)"},
          open("pcr2/folds_rand5.json", "w"))
print("wrote pcseq/labels_r3_base.parquet, pcr2/folds_rand5.json")
