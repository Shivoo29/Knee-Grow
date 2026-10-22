# CoAtNet runs behind the 0.949 / 0.945 weights (nartaa/rsna-knee-publication-swa-weights-20261007)

Both runs: `train_knee.py` (sha256 fffd79ab…, our edit of dreaddevelopment's public Raptor training notebook) run UNCHANGED
through `ext_trainer.py` (sha256 a0899b97…), which loads the trainer by path, patches four module globals, and appends the
external OAI knees to the TRAINING split only. Gold-58 and every RSNA row are untouched.

- External part: 2,399 OAI knees (2,399 trained, 0 all-NaN dropped), training rows 4,349 RSNA + 2,399 = 6,748.
- External cells supervise only `PF OA, Lateral OA, Synovitis` (pre-registered MOAKS cuts); every other external cell is NaN = masked
  out of the loss. Effusion, Contusion, Fracture and MCL can never be enabled. External cell weight 1.0.
- `pos_weight` is pinned to the RSNA label file (`--pw_labels labels_v1_blendjev.parquet`), not to the merged table.
- Labels: `labels_v1_blendjev.parquet` (see ../labels/).
- SWA: average of the top-3 epochs **ranked by Gold-58 AUC** (train_knee.py ~l.985–1001). For a leakage-controlled
  rerun, replace this with selection inside your inner fold.

| Model | sha256 (SWA) | arch / res | crop | SWA epochs |
|---|---|---|---|---|
| 0.949 (scored 320 crop + mirror TTA) | 7e5315da… | coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k / 384 | 1.0 | 14, 11, 12 |
| 0.945 efficiency | 3394fdd7… | coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k / 224 | 0.8 (`RAPTOR_CROP_FRAC=0.8`) | 15, 13, 12 |

Exact argv (384; the 224 run differs only in --arch, --res, --tag and RAPTOR_CROP_FRAC=0.8):

```
python ext_trainer.py --trainer train_knee.py --out <out> --ext_dir <oai_corpus96> --ext_weight 1.0 \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --norm imagenet --workers 4 --k 12 --k_eval 24 --bs 1 --accum 16 \
  --win_chunk 4 --epochs 16 --seed 42 --bb_lr 4.01e-05 --head_lr 0.00195 --wd 0.0588 \
  --labels labels_v1_blendjev.parquet --tag alldata_t16_blendjev_oai_d96_r384
```
RSNA input: 96-slice / 384 px cached stacks (`RAPTOR_INPUT`), built with the Raptor corpus geometry.

The OAI part (`ext_ids.npy`, volumes, masks, MOAKS-derived labels) is subject-level OAI data and is NOT included. The exact
input manifest is filed as a restricted manifest with NDA Study #3407 (https://nda.nih.gov/study.html?id=3407,
DOI 10.15154/0hcg-f676); request it through NDA with an approved OAI Data Use Certification.
