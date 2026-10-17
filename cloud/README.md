# Student folds on the RTX Pro 6000 instances

One instance trains one fold of the fastMRI noisy student (ConvNeXt-T, competition studies with Clef severity
labels + ~5.8k fastMRI knee exams labelled by our own ConvNeXt teacher). node1 trains fold 0; the three
instances take folds 1, 2 and 3.

## Before the first run (once)

1. **Kaggle access.** The instance downloads two private datasets owned by `shivoo29`:
   `shivoo29/knee-cloud-bundle` (code, labels, teacher labels) and `shivoo29/knee-c192-u8d` (our competition
   corpus, 16 shards, 71 GB). If the instance uses another Kaggle account, add that account as a collaborator
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
### molab

molab notebooks are public but undiscoverable: anyone with the link sees the cell code and outputs and can fork
the notebook (the docs don't say whether a fork gets uploaded files). So the Kaggle key and the fastMRI links
never go in a cell, an uploaded file (`.env` included) or an output. Per instance:
1. New notebook, attach the GPU.
2. ☰ → **Pair with an agent**; give Claude the URL and token it shows. Keep the tab open for the whole run
   (pairing and the log refresh need it).
3. Claude runs the required `help(cm)` call, then
   `python3 -I cloud/molab_cells.py <FOLD> | MARIMO_TOKEN=<token> bash .agents/skills/marimo-pair/scripts/execute-code.sh --url <url> -`.
   That writes `run_fold.sh` to `/tmp/knee` and adds a password form, a launcher and a log view refreshed every
   minute (keeps the session from looking idle; molab stops after 90 idle minutes).
4. Paste the Kaggle key and the two fastMRI links into the form and press **Start fold N**. The values stay in
   kernel memory, the run's environment and `/tmp/knee/fastmri_urls.txt` (600; molab keeps nothing in `/tmp`).

The run is its own background process, so an interrupted cell or a dropped pairing call doesn't stop it. If it
stops with an error, fill in the form and press Start again: finished steps are skipped while the session is alive.

Steps, each skipped when its output already exists (a rerun resumes):

| step | what | time on molab (20 CPUs, ~100 MB/s) |
|---|---|---|
| 1 | bundle: code, labels, fold split, train.csv without reports | seconds |
| 2 | fastMRI batch2 + first 64 GB of batch1 from S3, built into `WORK/fm` | ~30 min |
| 3 | corpus192 shards from Kaggle into `WORK/run/vols` | ~15 min |
| 4 | waits (checks every 10 min) until the bundle has `fastmri_teacher_sev.parquet` | — |
| 5 | trains the fold, batch 8 x accumulation 2 (same effective batch as the PC runs), 12 loader workers | ~2-3 h |
| 6 | publishes `<your user>/knee-cnxt-fm-f<FOLD>` (SWA checkpoint, OOF, gold predictions, log) | minutes |

Steps 2 and 3 run while the teacher labels are still being made on node1, so an instance can start before them.
Checked on molab (Oct 18): 20 CPUs, 156 GB RAM free, the root disk is a real disk (a 4 GB write left RAM
untouched), ~100 MB/s downloads. Start sessions after the corpus upload is confirmed; a session that starts
by ~05:00 has 12 h for prep (~1 h), the wait for the teacher (~07:00-08:30) and training (~2-3 h).
