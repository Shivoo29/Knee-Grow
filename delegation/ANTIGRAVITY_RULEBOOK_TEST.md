# Task: score the final report rulebook on the 25 held-out gold studies (node1, local models only)

Repo: /home/codemaster29/Documents/still_learning/rsna-knee-abnormality-detection (work in rulebook/).
Read chat.md section 79 first. Do not commit with Co-Authored-By lines.

## Hard rules
- Report text NEVER leaves node1: no hosted AI, never print/log report text, never copy train.csv off node1.
- Only local Ollama models on node1 (`ssh node1`): gemma4:31b and qwen3.8:latest are already installed.
- One GPU job at a time on node1. Never `pkill -f` over ssh (it can kill your own shell); kill by PID.
- Do not delete anything on node1 without asking the user.

## Steps
1. Copy the rulebook: `cp ~/Downloads/final_rulebook.md rulebook/rulebook_final.md`, then
   `scp rulebook/rulebook_final.md node1:projects/knee/rulebook/`.
2. On node1 (`cd ~/projects/knee/rulebook`), run each model on the 25 unread gold studies (~10 min each):
   `RULEBOOK=rulebook_final.md python3 label_llm.py gemma4:31b test_ids.txt test_gemma4_final.csv`
   `RULEBOOK=rulebook_final.md python3 label_llm.py qwen3.8:latest test_ids.txt test_qwen3.8_final.csv`
   (label_llm.py forces a flat JSON of 12 numbers; the rulebook's richer output format is ignored, which is fine.)
   Check the last column `ok` is 1 for all rows.
3. Copy both CSVs back to rulebook/ and build three mixes (current labels = ../labels/labels_clef_sev.csv):
   - test_final_mix_g.csv = (gemma_final + current) / 2
   - test_final_mix_all.csv = (gemma_final + qwen_final + current) / 3
   - test_final_oa_g.csv = current labels with Lateral OA and PF OA replaced by gemma_final
4. Score (from rulebook/):
   `uv run -q --with numpy --with pandas --with scikit-learn python -I score.py test_gemma4.csv test_gemma4_final.csv test_qwen3.8_final.csv test_mix.csv test_final_mix_g.csv test_final_mix_all.csv test_final_oa_g.csv`

## Decision gate (fixed in advance)
Previous best = 0.881 (current labels with Lateral OA + PF OA from Gemma 4 v1); current labels = 0.870.
- Best new row >= 0.885 AND no finding drops more than 0.03 vs current -> tell the user it passed. The next step
  (only if the user approves GPU time) is labeling all 4,407 reports with that recipe, building round-3 labels with
  tools/make_r2_labels.py (--base = the new mixed labels), and a fold-0 retrain on the PC compared with round-2 fold 0.
- Otherwise -> stop; the rulebook is not used for the final submission.
Caveat to state: 25 studies (only 4 Lateral OA positives), so +/-0.01 is noise.

## Deliverable
Paste the score table to the user with the verdict, and add chat.md section 80 (or the next free number):
models, rulebook file, scores, verdict.
