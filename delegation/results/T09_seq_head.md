# SUMMARY (lines 1-15; reviewer reads only this)
1. Verdict: seq_head + label_weights built in worktree ../knee-seq (branch seq, no commit); 7/7 CPU tests pass; no training/GPU job run.
2. Changed: unified RaptorClassifier in train_knee.py + fleet_v6.py (identical source); dataset sorted-centers + centers/weights return; aux loss + weighted BCE; 3 flags; build_label_weights.py; test_seq_head.py.
3. Key numbers: order-aware perm_diff 1.56e-02 vs base 1.79e-07; strict ckpt roundtrip maxdiff 0.00e+00; all-ones weighted==plain diff 0.00e+00; real weights 4349 rows, zero-frac 0.001-0.019/finding.
4. Problems: none blocking; GRU residual needs even F_dim (all backbones even); test-only f-string/slicing fixes, no build changes.
5. file:line (worktree): class train_knee.py:415-492 = fleet_v6.py:105-182; dataset 270-400; loss 401-410/798-816; ckpts +6 sites; fleet setup 1131-1145, assert 711-713.
6. NOT checked: real training run with either flag; Kaggle end-to-end; GPU numerics; odd-F_dim behavior.
7. NOT VERIFIED: aux_w=0.3/thr=0.7 optimality; per-finding gains; hidden-test runtime of seq models.
8. Main folder untouched (git status: no modifications); main .venv used read-only; training jobs untouched (no ssh/pkill, CPU only).
9. Old path: seq-off state_dict keys identical to bd76bdc pre-change code; non-seq forward op-order unchanged.
10. Safe pick note: label-trained v5/v6 path unaffected when flags off; students path unchanged.
11. Backfill precedent: tools/add_roi_keys.py pattern reused for ckpt keys (seq/label keys load via ck.get with defaults in fleet).
12. Real-data weights: teacher=0.7R+0.3C deduped, 4349/4349 label rows covered, UIDs unique, values {0,1}, gold excluded by construction.
13. No UIDs/pixels/Report text in tests or this file; CSVs loaded with usecols excluding Report; shapes/counts/scalars only.
14. Noise rule: no AUC claims here (exact-equality thresholds 1e-6/1e-3 as specified); no LB predictions.

## Build A: --seq_head (worktree ../knee-seq, branch seq)
- Unified class: `RaptorClassifier` source identical in `train_knee.py:415-492` and `fleet_v6.py:105-182` (test extracts both). Base kept train's form (class `win_chunk=0`, checkpointed `encode`); fleet behavior preserved by `m.win_chunk = WIN_CHUNK` in setup (`fleet_v6.py:1145`) and explicit `F_dim`.
- GRU: `nn.GRU(F_dim, F_dim//2, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)` on post-slot_emb (B,K,F), residual add, then unchanged norm/att/per-finding classifier (`train_knee.py:483-493`). Aux head is an independent LayerNorm/att/clsW/clsb copy on pre-GRU feats (`:495-504`); `forward` returns main; `forward_with_aux` returns (main, aux) (`:506-512`).
- Modules built only when seq on (None otherwise), so seq-off state_dict keys are exactly today's (verified vs `git show bd76bdc:train_knee.py`).
- Dataset (`train_knee.py:270-400`): `seq_head` flag sorts centers ascending in train AND eval (`:331`); returns center indices (LongTensor, trailing, after slots) ; `ValueError` on seq+compartment-ROI (`:296-297`, checked before the boxes RuntimeError). `collate` stacks all trailing elements generically (`:393-398`); the 6 batch consumers now gate slots on `a.slot_emb` (slots stay index 2 iff emitted): `:741,798,856,978,1002,1048`.
- Flags: `--seq_head`, `--aux_w` (default 0.3), wired into datasets via `_roi_kw` (`:653-657`), model ctor (`:704-705`), loss `main + aux_w*aux` (`:798-816`, weighted too when both flags on). Eval/OOF use `forward` (main only).
- Checkpoints: `"seq_head": bool, "aux_w": float` next to `"slots"` in all 4 .pt dicts. Fleet builds GRU+aux from `ck.get("seq_head")` (`fleet_v6.py:1131-1132`), loads old checkpoints unchanged, asserts ascending `win_centers` when seq on (`:711-713`).

## Build B: --label_weights + tools/build_label_weights.py
- Script (`tools/build_label_weights.py`): `teacher = 0.7*raptor_raw + 0.3*compact` over `teacher/c{0..3}_*.csv` (concat + `drop_duplicates`, same as `build_targets.py:19-21`; `usecols` excludes Report), labels from `labels_avg.parquet`; `w=0.0 iff |teacher-label|>thr else 1.0`; writes `StudyInstanceUID, w_<finding>` x12.
- Real run (`--thr 0.7`): 4349 rows, 12 cols, UIDs unique, values {0.0,1.0}; zero fractions ACL 0.002(7), MCL 0.004(18), MedMen 0.009(39), LatMen 0.006(24), MedOA 0.003(13), LatOA 0.002(7), PFOA 0.004(17), Effusion 0.003(11), Synovitis 0.001(3), Bakers 0.009(37), Contusion 0.005(23), Fracture 0.019(81).
- Training (`train_knee.py`): `--label_weights PATH` loads uid->float32[12] (`:659-668`, raises on missing train ids); train dataset emits weights last; loss `weighted_bce` (`:401-410`: none-reduction BCE*weights, mean over weight>0, same pos_weight); flag off = identical `lossf` call. `"label_weights": basename|None` in 4 .pt dicts + 2 JSON sidecars.

## Tests (`tests/test_seq_head.py`, main .venv `python -I`, CPU, 0.5s total)
- Exact output: `PASS seq_off_keys keys=9 identical / PASS seq_on_order_aware out=(2,12) perm_diff=1.56e-02 base_diff=1.79e-07 aux_ok / PASS class_source_identical chars=3451 identical / PASS ckpt_roundtrip_strict maxdiff=0.00e+00 / PASS seq_compartment_refused (85ch) / PASS weighted_loss ones_diff=0.00e+00 zero_ignored=1.1616 / PASS build_label_weights_synthetic rows=3 mask_ok / done 7/7 pass`. No command failed.
- Roundtrip used `timm resnet18, pretrained=False` (no download); synthetic UIDs SYN0-2 (no real UIDs); pydicom/cv2 stubbed (decode paths only).
