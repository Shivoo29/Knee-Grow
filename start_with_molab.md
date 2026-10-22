# Start with molab + Kaggle

How we train models on molab (free cloud GPUs) and get results onto Kaggle.

## 1. Before you start

- A molab account (molab.marimo.io) and a Kaggle account.
- Your Kaggle API key: kaggle.com → Settings → API → "Create New Token". Keep it secret.

## 2. Open a molab notebook (the right size)

1. On molab, create a new notebook.
2. Click **Compute** (top right) and set:
   - **CPU Cores:** the highest option (20)
   - **Memory:** the highest option (**160 GiB**). The job stores ~70 GB of scans in memory; with 32 GiB it dies at once.
   - **GPU:** RTX Pro 6000
3. Click **Save and restart**.

## 3. Hand it to Claude

1. In the notebook, get the pairing link: the "pair" prompt with a URL like `https://sb-....sb.molab.run/` and a token.
2. Send both to Claude. Claude adds the cells: a title, a key box, a **Start** button and a live log.

## 4. Start the job

1. Paste your Kaggle key into the password box. Never paste it into a code cell.
2. Press **Start**.
3. The live log shows: scan download (~30–40 min), then one line per epoch, then "DONE".
4. When it finishes, the job uploads its results to a private Kaggle dataset by itself.

## 5. Rules that keep it alive

- **At most 2 molab notebooks open at once.** A third one kills another, even an idle one. Close extra tabs fully.
- **Keep the tab open and the laptop awake** (no sleep) until "DONE".
- If the memory line in the log says "DANGER", tell Claude.

## 6. If a notebook dies ("kernel not found")

1. Don't panic: after every epoch the job saves a checkpoint to Kaggle (`knee-ckpt-...`).
2. Open a fresh notebook, set Compute again (step 2) and send Claude the new pairing link.
3. Paste the key and press **Start**. It continues from the last saved epoch.

## 7. Kaggle

- **Results:** your Kaggle profile → Datasets. Our model files are private datasets named `knee-...`.
- **Submitting:** only the account owner submits, on the competition's **Submissions** page. Claude prepares the notebook; a person clicks Submit.
- **Scores take time:** a single model scores in ~1–2 h; the full blend with the public models takes up to ~8 h.
- **Final choice:** by the deadline, select **2** submissions as finals: one safe, one bold.

## 8. Never

- Never put keys or tokens in cells, files or chat messages to other services.
- Never share competition images or report text outside our machines.
- Never create extra accounts to get more GPUs; it breaks the terms and risks a ban.
