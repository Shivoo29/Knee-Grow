# RULES FOR EVERY DELEGATED MODEL (read first, obey exactly)

Project: Kaggle RSNA Knee Abnormality Detection (knee MRI -> 12 findings, macro ROC-AUC). Root: ~/Documents/still_learning/rsna-knee-abnormality-detection
Python: .venv/bin/python. Current best leaderboard score 0.954 (public notebook fork). Deadline Oct 22.

## Data security (violating this can disqualify the team)
1. NEVER open, print, quote or send the `Report` column of any CSV (data/train.csv, yehezkielhaganta etc.). Load CSVs with `usecols=` that excludes it.
2. NEVER print pixel arrays or save/send images. Print only shapes, counts, AUCs, timings.
3. NEVER upload anything anywhere. Do not paste file contents into web tools. Reading public web pages is fine.
4. Do not copy data/, teacher/, labels_*.parquet, *.npy, *.pt outside this repo.

## Actions you must NOT take
- No `git push`, no `git commit` (the owner commits), no `git reset/rebase`.
- No `kaggle competitions submit`, no `kaggle kernels push`, no dataset upload/delete.
- No ssh to node1 and no killing/restarting any process (training jobs are running on this PC and on node1). Never use `pkill -f`.
- Never edit existing project files. Write ONLY inside `delegation/results/` unless the task says otherwise.
- Never touch the Azure machine. Never use sudo.

## Reporting style
- Facts with numbers and file:line. No adjectives ("breakthrough", "record", "champion"). No predictions of leaderboard scores.
- Gold-58 AUC differences under ~0.02 are noise; per-finding under ~0.06 are noise. Say "noise" when it is.
- If you did not verify something, write "NOT VERIFIED". If a command failed, quote the error exactly.
- First 15 lines of your results file = SUMMARY (the reviewer reads only this): verdict, key numbers, list of problems with file:line, what you did NOT check.
- Time-box: stop at the time written in the task and report what you have.
