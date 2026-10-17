#!/bin/bash
# Build the small private Kaggle dataset shivoo29/knee-cloud-bundle that cloud/run_fold.sh downloads: code,
# labels, fold split, train.csv without the Report column, and (once node1 has made them) the fastMRI teacher
# labels. Creates the dataset the first time, adds a version after that.
#   cloud/make_bundle.sh [fastmri_teacher_sev.parquet]        (B=<dir> NO_UPLOAD=1 for a local build only)
set -e
cd "$(dirname "$0")/.."
B=${B:-kaggle_ds/knee-cloud-bundle}
mkdir -p "$B"
cp train_knee.py roi_localize.py run_cnxt_fm_folds.sh build_corpus96_stream.py cloud/run_fold.sh "$B/"
cp tools/fastmri_corpus.py tools/fetch_ranges.py "$B/"
cp labels_sev_v1.parquet label_weights_sev_v1.parquet folds_rand4.json "$B/"
.venv/bin/python -I -c "
import sys, pandas as pd
lab = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
       'Effusion', 'Synovitis', \"Baker's\", 'Contusion', 'Fracture']
pd.read_csv('data/train.csv', usecols=['StudyInstanceUID'] + lab).to_csv(sys.argv[1], index=False)" "$B/train.csv"
if [ -n "${1:-}" ]; then cp "$1" "$B/fastmri_teacher_sev.parquet"; fi
echo '{"title": "knee-cloud-bundle", "id": "shivoo29/knee-cloud-bundle", "licenses": [{"name": "CC0-1.0"}]}' > "$B/dataset-metadata.json"
[ -n "${NO_UPLOAD:-}" ] && exit 0
if kaggle datasets status shivoo29/knee-cloud-bundle >/dev/null 2>&1; then
  kaggle datasets version -p "$B" -m "${MSG:-update}" -t -q
else
  kaggle datasets create -p "$B" -t -q
fi
echo BUNDLE_UPLOADED
