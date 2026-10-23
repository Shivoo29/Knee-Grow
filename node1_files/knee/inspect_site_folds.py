import json
import pandas as pd

folds_path = "/home/s05814802723/projects/knee/ds/folds_site.json"
with open(folds_path) as f:
    folds = json.load(f)
    print("folds_site keys:", list(folds.keys()))
    if isinstance(folds, dict):
        for k, v in list(folds.items())[:5]:
            print(f"Fold/Key {k}: count={len(v) if isinstance(v, (list, dict)) else v}")

meta_path = "/home/s05814802723/projects/knee/ds/site_metadata.csv"
df_meta = pd.read_csv(meta_path)
print("\nsite_metadata.csv shape:", df_meta.shape)
print("columns:", list(df_meta.columns))
print(df_meta.head(3))
