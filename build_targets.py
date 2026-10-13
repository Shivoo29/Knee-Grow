"""Distillation targets: per-finding mix of the base954 teacher and our report labels.

    teacher = 0.7 * Raptor + 0.3 * compact     (same weights as the base954 blend, but on probabilities)
    target  = a_c * teacher + (1 - a_c) * labels_avg

a_c = 0.3 where the best report-label table beats the teacher on gold-58 by > 0.03 AUC (Lateral OA, PF OA),
else 0.6. Gold studies are not in labels_avg, so they never get a target (they stay validation-only).

    .venv/bin/python build_targets.py            # needs teacher/c{0,1,2,3}_{raptor_raw,compact}.csv
"""
import numpy as np
import pandas as pd

L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
     'Synovitis', "Baker's", 'Contusion', 'Fracture']
ALPHA = {c: 0.6 for c in L} | {'Lateral OA': 0.3, 'PF OA': 0.3}


def load(kind):
    parts = [pd.read_csv(f'teacher/c{k}_{kind}.csv', dtype={'StudyInstanceUID': str}) for k in range(4)]
    return pd.concat(parts).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')[L]


labels = pd.read_parquet('labels_avg.parquet').set_index('StudyInstanceUID')[L]
teacher = 0.7 * load('raptor_raw') + 0.3 * load('compact')
missing = labels.index.difference(teacher.index)
assert len(missing) == 0, f'{len(missing)} training studies have no teacher prediction'
teacher = teacher.loc[labels.index]
assert teacher.notna().all().all() and ((teacher >= 0) & (teacher <= 1)).all().all()

target = pd.DataFrame({c: ALPHA[c] * teacher[c] + (1 - ALPHA[c]) * labels[c] for c in L}, index=labels.index)
target.reset_index().to_parquet('labels_distil_v1.parquet', index=False)
print('rows', len(target), '| mean target vs labels:',
      ' '.join(f'{c[:5]} {target[c].mean():.2f}/{labels[c].mean():.2f}' for c in L))
