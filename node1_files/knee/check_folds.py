import json
from collections import Counter

with open("/home/s05814802723/projects/knee/ds/folds_site.json") as f:
    d = json.load(f)

folds = d["folds"]
if isinstance(folds, dict):
    counts = Counter(folds.values())
else:
    counts = Counter([x["fold"] for x in folds])

print("Total studies:", len(folds))
print("Fold counts:", dict(sorted(counts.items())))
