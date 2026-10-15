"""OOF relabelling, stage 2: training targets = per-finding mix of OUT-OF-FOLD image predictions and report labels.

    target_c = a_c * oof_image_c + (1 - a_c) * labels_avg_c

oof_image comes from the 4 fold models (run_pc_oof.sh / run_node1_oof.sh): every study is predicted by the model
that did NOT train on it, so unlike the in-sample base954 teacher it adds image information the reports lack.
On gold-58, 0.6 * (our CoAtNet) + 0.4 * (report labels) scored 0.939 vs 0.899 for report labels alone.
a_c: --alpha for every finding, or --alpha_json {"finding": a} to override per finding.

    .venv/bin/python build_targets_v2.py [--alpha 0.6] [--alpha_json alphas.json] [--out labels_oofmix_v1.parquet]
"""
import argparse, glob, json
import numpy as np
import pandas as pd

L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
     'Synovitis', "Baker's", 'Contusion', 'Fracture']


def load_oof(tag='oof4', n_folds=4):
    parts = []
    for k in range(n_folds):
        hits = glob.glob(f'raptor_oof_{tag}_f{k}_swa_fold{k}.npz') or glob.glob(f'raptor_oof_{tag}_f{k}_fold{k}.npz')
        assert hits, f'fold {k}: no OOF file for tag {tag}'
        z = np.load(hits[0], allow_pickle=True)
        parts.append(pd.DataFrame(z['pred'], index=[str(u) for u in z['ids']], columns=L))
    oof = pd.concat(parts)
    assert not oof.index.duplicated().any(), 'a study is in two folds'
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=0.6)
    ap.add_argument('--alpha_json', default=None)
    ap.add_argument('--out', default='labels_oofmix_v1.parquet')
    a = ap.parse_args()
    alpha = {c: a.alpha for c in L} | (json.load(open(a.alpha_json)) if a.alpha_json else {})
    lab = pd.read_parquet('labels_avg.parquet').assign(StudyInstanceUID=lambda d: d.StudyInstanceUID.astype(str))
    lab = lab.set_index('StudyInstanceUID')[L]
    oof = load_oof()
    missing = lab.index.difference(oof.index)
    assert len(missing) == 0, f'{len(missing)} training studies have no OOF prediction'
    oof = oof.loc[lab.index]
    assert oof.notna().all().all() and ((oof >= 0) & (oof <= 1)).all().all()
    tgt = pd.DataFrame({c: alpha[c] * oof[c] + (1 - alpha[c]) * lab[c] for c in L}, index=lab.index)
    tgt.rename_axis('StudyInstanceUID').reset_index().to_parquet(a.out, index=False)
    print(a.out, 'rows', len(tgt), '| alpha', {c[:6]: alpha[c] for c in L})
    print('mean target vs report label:', ' '.join(f'{c[:5]} {tgt[c].mean():.2f}/{lab[c].mean():.2f}' for c in L))


if __name__ == '__main__':
    main()
