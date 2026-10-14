# T09: Sequence head (bi-GRU + auxiliary attention loss) and label down-weighting

Read `delegation/PREAMBLE.md` first and obey it. Exceptions under "Where you work". Time-box: 5 hours.

## Why (sources)
- RSNA 2024 lumbar 1st place (Avengers writeup): bi-LSTM over ordered slice embeddings + attention MIL + an auxiliary
  attention loss moved private LB 0.4259 -> 0.4062 (their metric: lower is better).
- RSNA 2024 lumbar 2nd place (brendanartley GitHub README): dropping / down-weighting training samples whose model
  prediction strongly disagrees with the given label helped with noisy labels.
- Our head today (`train_knee.py`, class `RaptorClassifier`, `head()`): attention MIL, permutation-invariant; it does
  not see slice order. Training window centers are SHUFFLED (`pick_centers(..., train=True)` shuffles the pool).

## Build A: `--seq_head` (train_knee.py AND kaggle_base954_fleet/fleet_v6.py)
1. When on, the dataset returns window centers SORTED ascending (train and eval), and returns the center indices.
   Refuse `--seq_head` together with `--roi_mode compartment` (raise ValueError: 2 crops per center break ordering).
2. In `RaptorClassifier` (both files, identical code): `self.gru = nn.GRU(F_dim, F_dim // 2, num_layers=2,
   batch_first=True, bidirectional=True, dropout=0.1)` applied to the (B, K, F) window features (after adding
   `slot_emb` if that flag is on), then a residual add (`feats = feats + gru_out`), then the existing
   LayerNorm -> attention -> per-finding classifier.
3. Auxiliary head: a second, independent copy of the existing attention-MIL head (own LayerNorm/att/clsW/clsb) on the
   PRE-GRU features. `forward` returns main logits; add `forward_with_aux(x, slots=None)` returning (main, aux).
   Training loss = `main_bce + a.aux_w * aux_bce`, new flag `--aux_w` default 0.3. Eval / inference use main only.
4. Save `"seq_head": bool`, `"aux_w": float` in every checkpoint dict next to `"slots"`. `fleet_v6.py` builds the
   GRU + aux modules when `ck.get("seq_head")` (aux unused at inference but must exist for `load_state_dict`), and
   feeds windows in ascending center order (eval centers from `pick_centers` are already ascending; assert it).
5. Without `--seq_head`: byte-identical behaviour and identical state_dict keys to today.

## Build B: `--label_weights PATH` (train_knee.py only) + `tools/build_label_weights.py`
1. `tools/build_label_weights.py --thr 0.7 --out label_weights_v1.parquet`:
   teacher = 0.7 * Raptor + 0.3 * compact from `teacher/c{0,1,2,3}_raptor_raw.csv` and `teacher/c{0..3}_compact.csv`
   (same recipe as `build_targets.py`; dedupe on StudyInstanceUID); labels from `labels_avg.parquet`.
   Weight per (study, finding): 0.0 if `|teacher - label| > thr`, else 1.0. Columns: StudyInstanceUID, w_<finding>.
   Print per-finding fraction set to 0 (counts only). Gold studies are not in labels_avg; do not add them.
2. In training, when `--label_weights` is given: BCE with `reduction="none"`, multiply by the per-element weights,
   then mean over elements whose weight > 0. Keep the existing `pos_weight`. Without the flag: identical loss.
3. Save `"label_weights": basename or None` in the checkpoint dicts and JSON.

## Tests (`tests/test_seq_head.py`, plain `def test_*` + `__main__` runner; run `.venv/bin/python tests/test_seq_head.py`)
- seq_head off: state_dict keys identical to a model built by the pre-change code (`git show bd76bdc:train_knee.py`).
- seq_head on: output shape (B,12); permuting the input windows CHANGES the output (order-aware); aux returns (B,12).
- train/fleet `RaptorClassifier` source identical (extract class source text from both files).
- A seq_head checkpoint saved by train_knee loads in fleet_v6 with `strict=True` and gives the same logits on a
  random CPU input (tiny backbone: build with `timm.create_model("resnet18", num_classes=0)`, no download needed if
  `pretrained=False`).
- `--seq_head` + compartment ROI raises ValueError.
- label weights: weighted loss equals plain BCE when all weights are 1; ignores elements with weight 0.
- build_label_weights on a tiny synthetic teacher/labels pair gives the expected 0/1 mask.
CPU only. Do NOT run training or any GPU job (a training job is running on this PC).

## Where you work
`git worktree add ../knee-seq -b seq` from the project root (base = current main, commit bd76bdc or later); edit
only there. Do not commit. Results: `delegation/results/T09_seq_head.md` in the MAIN folder; first 15 lines = SUMMARY.
