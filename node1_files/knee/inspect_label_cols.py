import pandas as pd

p1 = "/home/s05814802723/projects/knee/labels_avg.parquet"
p2 = "/home/s05814802723/projects/knee/ds/rsna-knee-labels/labels_llm_soft.parquet"
p3 = "/home/s05814802723/projects/knee/labels_distil_v1.parquet"

df1 = pd.read_parquet(p1)
print(f"labels_avg ({df1.shape}):\n  cols: {list(df1.columns)}")
print("  first row:")
print(df1.iloc[0])

df2 = pd.read_parquet(p2)
print(f"\nlabels_llm_soft ({df2.shape}):\n  cols: {list(df2.columns)}")
print("  first row:")
print(df2.iloc[0])

if os.path.exists(p3):
    df3 = pd.read_parquet(p3)
    print(f"\nlabels_distil_v1 ({df3.shape}):\n  cols: {list(df3.columns)}")
