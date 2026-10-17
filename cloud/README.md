# Student folds on the RTX Pro 6000 instances

One instance trains one fold of the fastMRI noisy student (ConvNeXt-T, competition studies with Clef severity
labels + ~5.8k fastMRI knee exams labelled by our own ConvNeXt teacher). node1 trains fold 0; the three
instances take folds 1, 2 and 3.

## Before the first run (once)

1. **Kaggle access.** The instance downloads two private datasets owned by `shivoo29`:
   `shivoo29/knee-cloud-bundle` (code, labels, teacher labels) and `shivoo29/knee-c192-u8d` (our competition
   corpus, 16 shards, ~61 GB). If the instance uses another Kaggle account, add that account as a collaborator
   on both datasets (Kaggle → dataset → Settings → Sharing). The token goes in `~/.kaggle/kaggle.json`
   (or `KAGGLE_USERNAME` + `KAGGLE_KEY`).
2. **fastMRI links.** Put the two knee DICOM links from the NYU email into a private file, one per line:
   ```
   knee_DICOMs_batch1 https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_DICOMs_batch1.tar.xz?...
   knee_DICOMs_batch2 https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_DICOMs_batch2.tar.xz?...
   ```
   then `chmod 600` it. The script reads the links from this file only (never from a command line), and builds
   fastMRI on the instance itself: fastMRI data is never uploaded anywhere.
3. **Disk.** About 150 GB free under `WORK` (peak while batch1 is built; 136 GB at the end).

## Run

Get the script from the bundle and start it:
```bash
kaggle datasets download shivoo29/knee-cloud-bundle -f run_fold.sh -p .
FOLD=1 WORK=/data FASTMRI_URLS=~/fastmri_urls.txt bash run_fold.sh 2>&1 | tee run_fold_1.log
```
In a molab (marimo) notebook, run it from one cell and keep the cell busy, so the session never looks idle
(molab stops after 90 idle minutes):
```python
import os, subprocess
env = {**os.environ, "FOLD": "1", "WORK": "/data", "FASTMRI_URLS": "/root/fastmri_urls.txt"}
p = subprocess.Popen(["bash", "run_fold.sh"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end="")
p.wait()
```

Steps, each skipped when its output already exists (a rerun resumes):

| step | what | time (4 CPUs) |
|---|---|---|
| 1 | bundle: code, labels, fold split, train.csv without reports | seconds |
| 2 | fastMRI batch2 + first 64 GB of batch1 from S3, built into `WORK/fm` | download + ~1.5 h |
| 3 | corpus192 shards from Kaggle into `WORK/run/vols` | ~15-30 min |
| 4 | waits (checks every 10 min) until the bundle has `fastmri_teacher_sev.parquet` | — |
| 5 | trains the fold, batch 8 x accumulation 2 (same effective batch as the PC runs) | ~3 h |
| 6 | publishes `<your user>/knee-cnxt-fm-f<FOLD>` (SWA checkpoint, OOF, gold predictions, log) | minutes |

Steps 2 and 3 run while the teacher labels are still being made on node1, so an instance can start before them.
