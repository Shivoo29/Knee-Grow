import pandas as pd
import json

print('=== labels_avg.parquet ===')
df_avg = pd.read_parquet('/home/s05814802723/projects/knee/labels_avg.parquet')
print('Shape:', df_avg.shape)
print('Cols:', df_avg.columns.tolist())
print(df_avg.describe().round(3))

print('\n=== gold_58.csv ===')
df_gold = pd.read_csv('/home/s05814802723/projects/knee/ds/gold_58.csv')
print('Shape:', df_gold.shape)
print('Cols:', df_gold.columns.tolist())
print(df_gold.describe().round(3))

print('\n=== llm_labels_full.csv ===')
df_llm = pd.read_csv('/home/s05814802723/projects/knee/ds/rsna-knee-llm-report-labels/llm_labels_full.csv')
print('Shape:', df_llm.shape)
print('Cols:', df_llm.columns.tolist())
print(df_llm.head(2))

