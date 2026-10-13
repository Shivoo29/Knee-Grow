# T01 - Review the two new Kaggle notebooks for failure modes (READ-ONLY, ~30 min)

Files: kaggle_students/build_nb.py (generator), kaggle_students/students_only/students_only.ipynb,
kaggle_students/base954_students/base954_students.ipynb, kaggle_base954_fleet/fleet_v6.py (the inference script both run).
Context: notebooks run on Kaggle (2x T4, no internet, 9 h hard limit, hidden test ~1,300 studies) and must write /kaggle/working/submission.csv
with columns StudyInstanceUID + 12 finding columns, rows in the same order as test.csv. A past bug silently produced all-0.5 predictions.

Find anything that could (a) crash, (b) silently produce wrong/empty predictions, (c) exceed 9 h, (d) write a wrong submission format.
Check specifically: path discovery (/kaggle/input/... for dataset `shivoo29/knee-students`, files raptor_ft_v6d_s7_swa.pt / raptor_ft_v6d_s8_swa.pt),
RAPTOR_CKPT / RAPTOR_OUTPUT_CSV env handling in fleet_v6.py, test_ids ordering, NaN guards, what happens if one student fails,
and the time guard (STOP_AFTER_H). Do NOT run the notebooks. You may run `python -c` syntax checks and read code.
Output: delegation/results/T01.md (summary first, then each problem: file:line, why, proposed 1-line fix; do not apply fixes).
