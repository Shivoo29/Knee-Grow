import os, pandas as pd
HERE = os.path.expanduser('~/projects/knee')
df = pd.read_csv(os.path.join(HERE, 'train.csv'))
LAB = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
       'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
print('train.csv shape:', df.shape)
has_labels = df.dropna(subset=LAB)
print('Rows with all 12 labels non-null:', len(has_labels))
print('Sample gold rows:')
print(has_labels[['StudyInstanceUID'] + LAB[:3]].head(3))
