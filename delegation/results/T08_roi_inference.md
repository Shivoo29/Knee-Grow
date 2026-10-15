# SUMMARY (lines 1-15; reviewer reads only this)
1. Verdict: ROI inference built in worktree ../knee-roi (branch roi, no commit); 6/6 CPU tests pass; no GPU job run.
2. Changed: fleet_v6.py +285/-24 (ROI block, _build_windows, predict_study rewire, ckpt attrs); train_knee.py 6 sites (+roi keys); new tools/add_roi_keys.py, tests/test_roi_inference.py.
3. Key numbers: ROI windows (94,3,384,384) maxdiff 0.00e+00 vs StudyWindows (norm none+imagenet); boxes 0 mismatches on 3 studies; legacy path maxdiff 0.00e+00.
4. Problems: none blocking; adaptations: helper return vs loop continue (AST-normalized in test); norm applied pre-.to(dev) (IEEE-identical elementwise).
5. file:line (worktree): ROI block fleet_v6.py:385-660; predict_study 668-695; ckpt attrs 1107-1110; ckpt dicts train_knee.py:790,819,835,944 + json 808,880.
6. NOT checked: end-to-end run on a real spec_oa checkpoint (none exists yet); Kaggle runtime; GPU numerics.
7. NOT VERIFIED: hidden-test DICOM volume dims (code reads H,W dynamically; tested on 384 corpus + 64 synthetic).
8. Main folder untouched (git status: no modifications); all edits in worktree; main .venv used read-only for tests.
9. Data: boxes computed at inference from the same uint8 volume via the copied function; 59 GB vols + npz read mmap from main checkout (absent in worktree).
10. Old-checkpoint path: behavior equal to main-branch source (exec git-show code, CPU, res 64/384, TTA off/on).
11. Backfill: tools/add_roi_keys.py CKPT --roi_mode compartment -> *_roi.pt, keeps all other keys (tested on fake ckpt).
12. No UIDs/pixels/Report text in tests or this file; shapes/ints/scalars/timings only; usecols rule unneeded (no CSVs opened).
13. Noise rule: no AUC claims in this report (exact-equality tests, thresholds 1e-6 / 0 mismatches).
14. Time-box: ~1h of 4h used; PC/node1 training jobs untouched (no ssh, no pkill, CPU only).

## Build (worktree ../knee-roi, branch roi; `git worktree list` confirms c0f8758 base)
- `kaggle_base954_fleet/fleet_v6.py:385-660`: ROI block. `_roi_square_box` (:406) and `_roi_compartments` (:480) are byte copies of `roi_localize.py:11-82` / `:85-117` (only the def name carries the `_roi_` prefix train_knee.py itself uses); `Tuple`/`Optional` stand-ins (:395-402) avoid any new import so the file stays Kaggle self-contained. `_roi_slice_box` (:515) is the verbatim per-slice body of `build_roi_boxes.py:37-79` (signature carries the loop context: vol, mask, c, sboxes, cens, i_local, W, H, grids). `_roi_resize` (:565) mirrors `StudyWindows._resize` (`train_knee.py:307-313`).
- `_build_windows` (`fleet_v6.py:574-665`): single window builder. roi=None reproduces the legacy inline loop statement-for-statement; ROI branch reproduces `StudyWindows.__getitem__` eval mode (`train_knee.py:315-357`): centers via `pick_centers(..., slots=...)` without cap (as train), box from CENTER slice, compartment = 2 crops/center, truncate to k, slot id per tile. Caller passes `(1,D,4)` box arrays with i_local=0 to match the copied indexing.
- `predict_study` (`fleet_v6.py:668-695`): builds roi_cfg from model attrs (defaults mode=compartment, pad=0.06, overlap=0.12), calls `_build_windows`, then unchanged device/TTA/model path (TTA uses per-window centers, required for 2-crops/center).
- Model setup (`fleet_v6.py:1107-1110`): `m.roi/roi_mode/roi_pad/roi_overlap` from `ck.get` with the task's defaults; checkpoints without roi keys get roi=False and follow the legacy branch.
- `train_knee.py` (worktree): added `"roi": a.roi, "roi_mode": a.roi_mode, "roi_pad": a.roi_pad, "roi_overlap": a.roi_overlap` next to `"slots": a.slots` in all 4 .pt dicts (:790 best, :819 topk, :835 running SWA, :944 final SWA); extended the 2 json sidecars (:808 partial, :880 final) with pad/overlap. Nothing else changed.
- `tools/add_roi_keys.py` (new): `torch.load` CKPT on CPU, sets the 4 keys, `torch.save` to `*_roi.pt`; exits nonzero on missing/non-dict input.

## Tests (`tests/test_roi_inference.py`, CPU only, main .venv `python -I`)
- t1 (`:75`): `inspect.getsource` equality for both roi_localize functions (modulo def-name prefix); AST equality of `_roi_slice_box` body vs the `range(D)` loop body with bare-continue/bare-return normalized; plus text+identity check that train_knee imports those same objects (`train_knee.py:84`).
- t2 (`:127`): rows [0, 1000, 4406] of `ds/corpus96/all_vols.npy` (mmap (4407,96,384,384) u8) through the fleet copy vs `roi_boxes.npz` sbox (4407,96,4)/cen (4407,96,2): 0 mismatched ints.
- t3 (`:159`, none+imagenet): row 0, k=94, slots=(2,3), clean_windows, compartment; fleet `_build_windows` vs `StudyWindows(train=False)` with npz boxes: shape (94,3,384,384), max abs diff 0.00e+00, slot ids equal, 94 windows.
- t4 (`:210`): legacy functions exec'd from `git show main:...fleet_v6.py`; constants SLOT_BOUNDS/MAXS/IMG/K_EVAL equal; fake-model `predict_study` old-vs-new on synthetic (96,64,64) u8, res 64/384, TTA off/on: maxdiff 0.00e+00.
- t5 (`:244`): backfill script on a tiny fake ckpt in tmpdir: 4 keys added, rest kept, `*_roi.pt` written.
- Exact output: `PASS t1 ... (0.0s) / PASS t2 ... mismatches=0 (0.1s) / PASS t3 ... maxdiff=0.00e+00 (0.3-0.4s) / PASS t4 ... maxdiff=0.00e+00 / PASS t5 ... / done 6/6 pass (2.1s)`. No command failed.

## Notes / deviations
- `_roi_slice_box` uses `return` where the loop body used `continue` (no enclosing loop in a helper); the test normalizes exactly this and nothing else.
- Norm moved before `.to(dev)` in the rewire; float32 elementwise (x-mean)/std is IEEE-identical on CPU/CUDA; CPU-tested at 0.00e+00.
- pydicom/cv2 are absent from .venv; the test stubs them (decode paths only, never called by window/ROI code).
- `train_coat_spec_oa_s21.log` (untracked, running specialist job) was not touched; the run started before step 5, hence the backfill script.
