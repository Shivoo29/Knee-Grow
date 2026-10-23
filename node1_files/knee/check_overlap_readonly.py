import pandas as pd
import numpy as np

train = pd.read_csv('/home/s05814802723/projects/knee/train.csv')
LAB = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
       'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
gold = train.dropna(subset=LAB)
df = pd.read_parquet('/home/s05814802723/projects/knee/labels_distil_v1.parquet')

print('Train shape:', train.shape)
print('Gold shape:', gold.shape)
print('Parquet shape:', df.shape)
print('Gold UIDs sample:', gold['StudyInstanceUID'].head(3).tolist())
print('Parquet cols:', df.columns.tolist())
uid_col = 'StudyInstanceUID' if 'StudyInstanceUID' in df.columns else df.columns[0]
print(f'Parquet UID col is "{uid_col}", sample:', df[uid_col].head(3).tolist())

gold_set = set(gold['StudyInstanceUID'].astype(str))
parquet_set = set(df[uid_col].astype(str))
overlap = gold_set.intersection(parquet_set)
print(f'OVERLAP between Gold 58 and Parquet ({len(df)}): {len(overlap)}')

# Check total unique studies in train.csv
all_train_studies = set(train['StudyInstanceUID'].astype(str))
print('Total unique studies in train.csv:', len(all_train_studies))
print('Total unique studies in train.csv minus Gold 58:', len(all_train_studies - gold_set))

# Check where the 58 gold studies are
non_gold = all_train_studies - gold_set
print('Overlap between Parquet and non-gold train studies:', len(parquet_set.intersection(non_gold)))
