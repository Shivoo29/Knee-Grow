# T03 - DRAFT (do not run) the end-of-competition backup + cleanup script (~30 min)

After Oct 22 the owner wants to push everything useful to a private GitHub repo and then empty the PC and node1 (ssh alias `node1`, dir ~/projects/knee).
GitHub limits: 100 MB per file, no git-lfs allowed. So: code/logs/small csv/parquet/json -> git; model weights (*.pt, ~40 files, ~15 GB) -> a private HuggingFace repo
via `huggingface_hub` upload API (no git-lfs); competition data (ds/, corpus*, raw zip, *.npy, *.npz) -> NOT backed up anywhere, just deleted (competition rule).
Write ONLY a draft script delegation/results/endgame_cleanup.sh that: (1) prints a dry-run listing with sizes of what would be uploaded / deleted on PC and on node1,
(2) requires the flag --really-delete AND an interactive typed confirmation before any rm, (3) never touches anything outside the project dirs, (4) never kills processes,
(5) checks no train_knee.py process is running before deleting. Also write delegation/results/T03.md: a checklist of the exact order of steps. Do NOT execute the script.
