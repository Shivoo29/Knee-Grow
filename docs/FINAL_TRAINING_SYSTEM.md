# Final Training System — RSNA Knee Abnormality Detection

Last updated: Oct 20 2026, 22:30 IST. Competition deadline: **Oct 23 2026, 05:29 IST** (Oct 22 23:59 UTC).
Owner: shivoo29. This file is the single source of truth for the last training round. Part 2 is a step-by-step
guide for teammates who run backup training on their own molab accounts.

---

## Part 1 — What we are doing

### 1.1 Goal

Train two model families on improved labels and submit them **without** the public 0.958 ensemble:

| # | Kaggle submission | What it contains |
|---|---|---|
| 1 | CoAtNet only | 5-fold CoAtNet (one model per fold) |
| 2 | ConvNeXt only | 5-fold ConvNeXt-Tiny |
| 3 | Ensemble | 1 + 2 blended with per-finding weights fitted on held-out predictions |

The three scores tell us how strong our own models are and whether an external model is needed at all.

### 1.2 Pipeline

```
train.csv reports ──► Gemma 4 31B + rulebook_final.md (Lateral OA, PF OA only)
                          │  4,407 reports, 0 failures (node1 + molab, local Ollama)
                          ▼
report labels (labels_sev_v1) with Lateral OA + PF OA swapped ──► labels_r3_base
                          │  mixed with image-model out-of-fold predictions, report weight 0.4
                          ▼
                labels_r3.parquet  (4,349 train studies, 58 gold studies held out)
                          │
     ┌────────────────────┼─────────────────────────────┐
     ▼                    ▼                             ▼
5-fold ConvNeXt-T    5-fold CoAtNet-rmlp-2-384     backup seeds (PC, node1, teammates)
     │                    │                             │
     └──── each fold publishes: checkpoint (SWA), out-of-fold + gold-58 predictions, log ────► Kaggle datasets
                          ▼
         3 Kaggle notebooks (CoAtNet / ConvNeXt / ensemble), runtime < 8 h, user submits
```

### 1.3 Labels (round 3)

* Source of truth for the rulebook: `rulebook/rulebook_final.md`. The labeller is `rulebook/label_llm.py` (Ollama,
  temperature 0, JSON of 12 numbers). Only **Lateral OA** and **PF OA** are taken from Gemma; on the 25-study test,
  Gemma's other columns were worse than the current labels (Effusion 0.611 vs 0.897).
* Build: `tools/make_r3_labels.py` (swap) → `tools/make_r2_labels.py --base pcseq/labels_r3_base.parquet
  --gold_report labels/labels_r3_gold_report.csv --alpha 0.4` → `pcr2/labels_r3.parquet`.
* Gold-58 AUC of the training labels at alpha 0.4 (expert labels as truth):

| | Round 2 (old) | Round 3 (new) |
|---|---|---|
| Macro, 12 findings | 0.9343 | **0.9405** |
| Lateral OA | 0.832 | **0.878** |
| PF OA | 0.918 | **0.946** |
| Other 10 findings | unchanged | unchanged |

  Caveat: 33 of the 58 gold studies were read while writing the rulebook, so the gold gain is partly optimistic.
  The clean check is the 5-fold out-of-fold score after training.
* Folds: `pcr2/folds_rand5.json`, 5 random folds (seed 0) of the 4,349 train studies, 870 per fold. Gold-58 is in no
  fold; every fold reports gold-58 AUC each epoch.
* Loss weights: `pcr2/label_weights_pu.parquet` (idea 9: Synovitis negatives the report never mentions weigh 0.3;
  measured gold 0.9296 → 0.9323).

### 1.4 Model recipes

Both use the same 2.5D attention-MIL reader (`train_knee.py`) on corpus192: 5 slots (sagittal FS, sagittal plain,
coronal FS, coronal plain, axial) = 40 + 40 + 36 + 36 + 40 slices, two-scale tiles, 15 epochs, patience 5, bs 2 ×
accum 8, gradient checkpointing, SWA of the best 3 epochs.

| | ConvNeXt (`PRESET=cnxt`) | CoAtNet (`PRESET=coat`) |
|---|---|---|
| Backbone | `convnext_tiny.in12k_ft_in1k` | `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` |
| Resolution | 256 | 384 |
| Tiles train / eval | 48 / 380 | 24 / 128 (Kaggle runtime) |
| LR backbone / head | 1e-4 / 1e-3 | 4e-5 / 2e-3 |
| Time per fold | ~2.8 h measured (PC, molab, node1 ~2.4 h) | unknown, measured after epoch 0 |
| Expected gold-58 per fold | ~0.932–0.936 (round 2: 0.930–0.935) | ~0.915–0.925 (older CoAtNet runs) |

Why these sizes: in our own runs bigger was not better (ConvNeXt-S 0.918, ConvNeXt-B 0.895, ConvNeXt-T 0.930–0.935
gold-58). CoAtNet-3/4 have no usable pretrained weights in timm (CoAtNet-4: none; CoAtNet-3: 224-only).

Seeds: `seed = SEED_BASE + fold`. Main run 110, our backup 210, teammates 310 and 410. Same folds, different seeds,
so every finished model is useful: a backup either replaces a lost fold or becomes a second seed in the ensemble.

### 1.5 Who trains what

| Machine | Reliable | Trains | Seed base | How |
|---|---|---|---|---|
| molab C (new) | no | ConvNeXt folds 0–4, 2 at a time | 110 | `cloud/molab_train.sh` (Part 2) |
| molab D (new) | no | CoAtNet folds 0–4, 2 at a time | 110 | `cloud/molab_train.sh` (Part 2) |
| PC, RTX 4070 SUPER 12 GB | yes | ConvNeXt folds 0→4, one at a time | 210 | `pcr2/run_r3.sh` |
| node1, L4 24 GB | yes | CoAtNet folds 4→0, one at a time (reverse order) | 210 | `pcr2/run_r3.sh` |
| teammate molab (backup) | no | whatever is missing, see 2.6 | 310 / 410 | `cloud/molab_train.sh` |

node1 runs in reverse order so that if molab D dies late, the folds it had not reached are the ones node1 did first.
If CoAtNet on node1 is slower than ~10 h per fold, node1 switches to ConvNeXt-T at higher resolution instead.

Local commands (PC / node1):

```bash
# PC (repo root)
PRESET=cnxt SEED_BASE=210 PY=$PWD/.venv/bin/python RAPTOR_INPUT=$PWD/ds/corpus192 nohup pcr2/run_r3.sh > pcr2/r3_pc.log 2>&1 &
# node1: first free the GPU from Ollama (ollama stop gemma4:31b), copy the 5 files listed in run_r3.sh into a run dir
PRESET=coat SEED_BASE=210 FOLDS="4 3 2 1 0" PY=~/miniconda3/envs/knee/bin/python RAPTOR_INPUT=<corpus192 dir> ./run_r3.sh
```

### 1.6 Timeline (IST)

| When | What |
|---|---|
| Oct 20 22:30 | Round-3 labels final (done) |
| Oct 20 ~23:00 | PC + node1 start; molab C/D created, 30 min data download, then training |
| Oct 21 ~07:00 | molab ConvNeXt 5 folds done (estimate) |
| Oct 21 ~13:00 | PC ConvNeXt seed 210 done (estimate) |
| Oct 21, after CoAtNet epoch 0 | decide CoAtNet tiles; target: all 5 CoAtNet folds by Oct 22 ~10:00 |
| Oct 22 10:00–18:00 | ensemble weights from out-of-fold predictions; 3 Kaggle notebooks; test commits; user submits |
| Oct 23 05:29 | deadline |

### 1.7 Kaggle notebooks (after training)

* Test-time additions already validated: the axial slot prefers the fat-suppressed series (PF OA 0.878 → 0.893 on
  97 fold-0 studies); a Lateral OA scanner prior on the model's Lateral OA column (gold 0.807 → 0.867).
* Runtime target under 8 h on 2× T4 for ~1,300 hidden studies (estimates: ConvNeXt 5 folds ~1.3 h, CoAtNet 5 folds
  ~2.5–4 h, ensemble ~4–5 h). Each notebook is first committed on the 3 visible test studies: complete, no NaNs, s/study
  × 1,300 under 8 h.
* Ensemble weights: per finding, chosen on the 5-fold out-of-fold predictions, checked on gold-58.

### 1.8 What can go wrong

| Risk | What we do |
|---|---|
| A molab instance crashes | Finished folds are already on Kaggle. Re-run the cell with only the missing folds. Backups (PC, node1, teammates) cover the rest. |
| CoAtNet too slow | After epoch 0, if 5 folds cannot finish by Oct 22 10:00, restart all CoAtNet folds with `--k 12`, same for every fold. |
| Two folds at once are slower than one after another | `PARALLEL=1`. Rule: if each epoch takes more than 2× the single-fold time, parallel does not pay. |
| molab quota runs out | The user checks the remaining quota; teammates' accounts take over the missing folds. |
| Kaggle notebook over time | Drop flip TTA, then drop folds; the kill guard stops our part at 8.4 h. |

Rules that hold throughout: no fastMRI- or OAI-derived models; no Kaggle keys in cells, files or chat (molab
Secrets only); only the owner submits.

---

## Part 2 — Setting up a molab training machine (step by step, no experience needed)

You need: a Kaggle account that is on our competition team (or has accepted the competition rules) and a molab
account. Total hands-on time ~10 minutes; then it runs for many hours by itself.

### 2.1 One-time: get access to the data (teammates only)

1. Send the owner (shivoo29) your **Kaggle username**.
2. The owner shares three private datasets with you (Kaggle → dataset page → **Settings → Sharing** → add your
   username): `shivoo29/knee-r3-labels`, `shivoo29/knee-cloud-bundle`, `shivoo29/knee-c192-u8d`.
3. Check: open `https://www.kaggle.com/datasets/shivoo29/knee-r3-labels` while logged in. If you see the files, you
   have access.

### 2.2 One-time: your Kaggle API token

1. kaggle.com → click your profile picture → **Settings** → section **API** → **Create New Token**.
2. Copy the token (it starts with `KGAT_`). Keep it private: never paste it into a notebook cell, a file or a chat.

### 2.3 Create the molab notebook

1. Go to **molab.marimo.io**, sign in, click **New notebook**. Name it, e.g. `knee-convnext` or `knee-coatnet`.
2. Top bar → **Compute** → choose **RTX Pro 6000** (if several, pick the one with the most CPU and RAM).
3. Wait until the bottom-right shows the kernel as connected (green).

### 2.4 Add your secrets

1. In the bottom panel click **Secrets** → **Add Secret**.
2. Key `KAGGLE_API_TOKEN`, value: paste your token. Location `.env`. Click **Add Secret**.
3. Add a second secret: key `KAGGLE_USERNAME`, value: your Kaggle username.
4. Restart the kernel so it sees them: **⚙ (top right) → Restart kernel**.

### 2.5 Start training

Click **+ Python** to make a new cell, paste this, edit the two lines marked `EDIT`, press **Shift+Enter**:

```python
import os, subprocess
assert os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY"), "add the Kaggle secret, then restart the kernel"
assert os.environ.get("KAGGLE_USERNAME"), "add the KAGGLE_USERNAME secret, then restart the kernel"
PRESET = "cnxt"        # EDIT: "cnxt" on the ConvNeXt machine, "coat" on the CoAtNet machine
SEED_BASE = 110        # EDIT: owner 110, owner backup 210, teammate A 310, teammate B 410
FOLDS = "0 1 2 3 4"    # only change this when re-running after a crash (list the missing folds)
PARALLEL = 2           # folds trained at the same time on the one GPU
cmd = f"""
set -e; mkdir -p /tmp/r3 && cd /tmp/r3
pip install -q kaggle timm
for f in run_r2.sh molab_train.sh; do
  python -m kaggle datasets download shivoo29/knee-r3-labels -f $f -p . -q --force
  [ -f $f.zip ] && python -c "import zipfile; zipfile.ZipFile('$f.zip').extractall('.')" && rm $f.zip
done
PRESET={PRESET} SEED_BASE={SEED_BASE} FOLDS="{FOLDS}" PARALLEL={PARALLEL} bash molab_train.sh
"""
subprocess.Popen(["bash", "-c", cmd], stdout=open("/tmp/r3_train.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
print("Started. Watch it in the Terminal tab:  tail -5 /tmp/r3_train.log")
```

It prints `Started.` immediately. The first ~30 minutes download the data (4,407 MRI studies); then training starts.

### 2.6 Check that it is working

Open the **Terminal** tab in the bottom panel and run:

```bash
tail -5 /tmp/r3_train.log                                    # which folds started / ALL_DONE at the end
grep -h "GOLD macro" /tmp/knee/run/train_*.log | tail -4     # one line per finished epoch
nvidia-smi                                                   # GPU should be busy
```

What good looks like:
* `/tmp/r3_train.log` shows `started fold 0`, then `started fold 1` once the data is downloaded.
* Each epoch line looks like `ep3 loss 0.93 | GOLD macro-AUC 0.90 (best 0.90) | ... | 2800s`. The last number is
  seconds since that fold started. The gold AUC climbs to ~0.92–0.93 (ConvNeXt) by the last epochs.
* A finished fold appears on Kaggle as your private dataset `knee-r3-<preset>-s<seed>-f<fold>`.

Send the owner a screenshot of the epoch lines after the first epoch: that is when the speed is known.

### 2.7 If something goes wrong

| You see | Do this |
|---|---|
| `add the Kaggle secret...` | Do step 2.4 again, then restart the kernel and run the cell again. |
| `403` / `Forbidden` / `not found` in `/tmp/r3_train.log` | The datasets are not shared with you yet (step 2.1). |
| `Option B: API token file` | The secret was not loaded: restart the kernel, run the cell again. |
| `CUDA out of memory` | Set `PARALLEL = 1` and run the cell again. |
| The notebook disconnected / machine restarted | Check Kaggle for finished folds. Run the cell again with `FOLDS` = only the missing folds. A fold that was half done starts over. |
| `fold N stopped during download` | Run the cell again; downloads resume. |

Keep the molab tab open while it trains.

### 2.8 When it is done

1. `/tmp/r3_train.log` ends with `ALL_DONE`.
2. Teammates: open each finished dataset `knee-r3-...` on Kaggle → **Settings → Sharing** → add `shivoo29`.
3. Tell the owner which folds and seed you finished.
