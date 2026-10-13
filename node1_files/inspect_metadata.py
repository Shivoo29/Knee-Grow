import os
import pandas as pd
import numpy as np

KNEE = "/home/s05814802723/projects/knee"

train_path = os.path.join(KNEE, "train.csv")
if os.path.exists(train_path):
    df_train = pd.read_csv(train_path)
    print("train.csv shape:", df_train.shape)
    print("train.csv columns:", list(df_train.columns))
    print(df_train.head(2))

series_path = os.path.join(KNEE, "ds", "comp", "train_series.csv")
if os.path.exists(series_path):
    df_series = pd.read_csv(series_path)
    print("\ntrain_series.csv shape:", df_series.shape)
    print("train_series.csv columns:", list(df_series.columns))
    print(df_series.head(3))

labels_path = os.path.join(KNEE, "labels_avg.parquet")
if os.path.exists(labels_path):
    df_labels = pd.read_parquet(labels_path)
    print("\nlabels_avg.parquet shape:", df_labels.shape)
    print("labels_avg.parquet columns:", list(df_labels.columns))
    print(df_labels.head(2))
