import pandas as pd

df = pd.read_csv("/home/s05814802723/projects/knee/result_labels.csv")
uid_col = df.columns[0]
df_indexed = df.set_index(uid_col)
out_path = "/home/s05814802723/projects/knee/result_labels.parquet"
df_indexed.to_parquet(out_path)
print(f"Successfully converted result_labels.csv to {out_path} with shape {df_indexed.shape}")
