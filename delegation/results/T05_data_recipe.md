# SUMMARY
- Verdict: PASS. All deliverables implemented, tested, and verified in worktree `../knee-recipe`.
- Tests: 7/7 unit tests passed in `.venv/bin/python tests/test_data_recipe.py` (0 failures).
- Parity: `slot_of`, `pick_centers`, `check_fallback_guard`, `SLOT_BOUNDS` byte-identical across `train_knee.py` and `fleet_v6.py`.
- Clean windows: `--clean_windows` enforces intra-slot windows on unmasked slices (`train_knee.py:200`, `fleet_v6.py:353`).
- Augmentation: `--mirror_p` anatomical mirror and `--contrast_aug` random gamma in [0.8, 1.25] in `StudyWindows` (`train_knee.py:228,322,345`).
- Slot embedding: `--slot_emb` learned `nn.Embedding(5, F_dim)` initialized to exact zeros in `RaptorClassifier` (`train_knee.py:416`, `fleet_v6.py:106`).
- Checkpoint schema: `"clean_windows"` and `"slot_emb"` saved across all checkpoint dicts and JSONs (`train_knee.py:783,798,812,828,870,937`).
- Inference fallback guard: `check_fallback_guard` strictly raises RuntimeError if fallback rate > 3% (`fleet_v6.py:381,935`).
- Backward compatibility: Zero new flags preserves 100% byte-identical training and inference; zero-init embedding outputs identical logits.
- Problems / Blockers: NONE.
- NOT CHECKED: `./run_pc_dinov2_s11.sh` or GPU training run (prohibited by T05; GPU job running on PC).
- NOT CHECKED: 64-study node1 speed check with `--win_chunk 0` vs 14 (reserved for Claude on node1).
- NOT CHECKED: CoAtNet version 6 run on node1 (reserved for Claude on node1).
- NOT CHECKED: LB score impact (prohibited by PREAMBLE rules).

---

## 1. Deliverable Overview & File Locations

All development and test execution took place strictly within the git worktree `../knee-recipe` (`/home/codemaster29/Documents/still_learning/knee-recipe`) on branch `recipe`. The main working directory and other worktrees (`../knee-lat`) were untouched. No running training processes were disturbed.

| Component | Target File | Status | Description |
| :--- | :--- | :--- | :--- |
| **Clean Windows** | `train_knee.py:200-225`<br>`fleet_v6.py:353-378` | Verified | `pick_centers` validates `c-1, c, c+1` are all real (`mask > 0`) and in identical slot (`slot_of(c-1) == slot_of(c) == slot_of(c+1)`). |
| **Anatomical Mirror** | `train_knee.py:228-244` | Verified | `mirror_volume` reverses real slices in sagittal slots (0, 1) and flips coronal/axial slots (2, 3, 4) horizontally along `axis=-1`. |
| **Contrast Aug** | `train_knee.py:344-348` | Verified | Study-level random gamma $\gamma \sim [0.8, 1.25]$ on $[0, 1]$ window floats before brightness jitter. |
| **Slot Embedding** | `train_knee.py:416-452`<br>`fleet_v6.py:106-140` | Verified | `nn.Embedding(5, F_dim)` initialized to exact zeros, added to window features before LayerNorm. |
| **Self-Contained Parity** | `fleet_v6.py:340-399` | Verified | `SLOT_BOUNDS`, `slot_of`, `pick_centers`, `check_fallback_guard` byte-identical between files. |
| **Fallback Guard** | `train_knee.py:247-259`<br>`fleet_v6.py:381-393,935` | Verified | Raises `RuntimeError` if fallback rate $> 3\%$ of total studies prior to writing output CSV. |
| **Unit Test Suite** | `tests/test_data_recipe.py` | Verified | 7 comprehensive unit tests executed via `.venv/bin/python tests/test_data_recipe.py`. |

---

## 2. Test Execution & Output

The full unit test suite was executed in `../knee-recipe` using `.venv/bin/python tests/test_data_recipe.py`.

```
$ .venv/bin/python tests/test_data_recipe.py
======================================================================
Running Task T05 Test Suite (Data Recipe)
======================================================================
PASS: test_clean_centers_no_straddle_and_no_mask
PASS: test_legacy_centers_unchanged_without_flag
PASS: test_mirror_volume_real_reversed_and_identity_twice
PASS: test_slot_of_matches_center
PASS: test_forward_signature_and_zero_init_identity
PASS: test_source_parity_slot_of_and_pick_centers
PASS: test_fallback_guard_raises_at_4pct_passes_at_2pct
======================================================================
ALL 7 TESTS PASSED SUCCESSFULLY!
======================================================================
```

All 7 test functions passed with exit code 0.

---

## 3. Implementation Details

### 3.1 Clean Windows & Slot Boundaries (`train_knee.py` & `fleet_v6.py`)
- **Slot Boundaries**: Fixed cumulative ranges for the 96 slices derived from $[26, 22, 18, 12, 18]$:
  - Slot 0: $[0, 26)$ (Sagittal fluid)
  - Slot 1: $[26, 48)$ (Sagittal non-fluid)
  - Slot 2: $[48, 66)$ (Coronal fluid)
  - Slot 3: $[66, 78)$ (Coronal non-fluid - empty in 25.5% studies)
  - Slot 4: $[78, 96)$ (Axial)
- **`slot_of(idx: int) -> int`**: Pure function mapping any slice index $0..95$ to its corresponding slot $0..4$, with bounds clamping.
- **`pick_centers(valid, count, train=False, clean_windows=False, mask=None, max_d=96)`**:
  - When `clean_windows=True` and `mask is not None`: Candidate center $c$ is admitted if and only if `mask[c - 1] > 0 and mask[c] > 0 and mask[c + 1] > 0` AND `slot_of(c - 1) == slot_of(c) == slot_of(c + 1)`.
  - When `clean_windows=False`: Preserves legacy logic `cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]`.

### 3.2 Study-Level Augmentations (`train_knee.py`)
- **`--mirror_p FLOAT` (default 0.0)**:
  - Applied per study in `StudyWindows.__getitem__` during training: `if self.train and self.mirror_p > 0.0 and random.random() < self.mirror_p: vol = mirror_volume(vol, m)`.
  - `mirror_volume`: In Sagittal slots 0 and 1, only real slices (`mask > 0`) are reversed in place (`out[start:start+n_real] = out[start:start+n_real][::-1]`), keeping padded slices intact at the end. In Coronal/Axial slots 2, 3, 4, slices are flipped horizontally (`axis=-1`). Double-mirroring is mathematically an identity operation.
- **`--contrast_aug` (default False)**:
  - Applied per study in `StudyWindows.__getitem__` during training:
    ```python
    if self.contrast_aug:
        gamma = random.uniform(0.8, 1.25)
        x = x.clamp(0, 1).pow(gamma)
    ```
    Followed by existing brightness jitter and clamped to $[0, 1]$.

### 3.3 Learned Slot Embedding (`train_knee.py` & `fleet_v6.py`)
- **`--slot_emb` (default False)**:
  - `RaptorClassifier.__init__`: Initializes `self.slot_emb = nn.Embedding(5, F_dim)` with exact zeros (`nn.init.zeros_(self.slot_emb.weight)`).
  - `RaptorClassifier.head`: If `self.slot_emb is not None and slots is not None`, adds `self.slot_emb(slots)` directly to window features before LayerNorm.
  - Forward signature: `forward(x, slots=None)` maintains backward compatibility.
  - Zero-initialization guarantees that at initialization, the model output with slot embeddings is byte-for-byte identical to the model output without slot embeddings.
  - Dataset: `StudyWindows` returns `(x, y, s_tensor)` when `slot_emb=True`, and `collate` stacks slot IDs into shape `(B, K)`.

### 3.4 Checkpoint Persistence (`train_knee.py`)
Both `"clean_windows": bool(a.clean_windows)` and `"slot_emb": bool(a.slot_emb)` are persisted next to `"res": a.res, "norm": a.norm` across all checkpoint structures:
1. `best_state` dict (`train_knee.py:783`)
2. Intermediate JSON `raptor_ft_{tag}.json` (`train_knee.py:798`)
3. `topk` entries (`train_knee.py:812`)
4. Running SWA checkpoint `raptor_ft_{tag}_swa.pt` (`train_knee.py:828`)
5. Final JSON `raptor_ft_{tag}.json` (`train_knee.py:870`)
6. Final SWA checkpoint `raptor_ft_{tag}_swa.pt` (`train_knee.py:937`)

### 3.5 Kaggle Fleet Parity & Strict Fallback Guard (`fleet_v6.py`)
- Reads `ck.get("clean_windows", False)` and `ck.get("slot_emb", False)` upon loading model checkpoints.
- Models instantiate `RaptorClassifier(..., slot_emb=has_slot)`, and `predict_study` passes slot IDs if `has_slot_emb` is True.
- Legacy checkpoints without these keys default to `False` and execute identical legacy inference.
- **Strict Fallback Guard**:
  ```python
  def check_fallback_guard(fallback_count: int, total_studies: int, max_rate: float = 0.03):
      if total_studies <= 0:
          return
      rate = fallback_count / float(total_studies)
      if rate > max_rate:
          raise RuntimeError(
              f"FATAL GUARD: fallback count {fallback_count}/{total_studies} "
              f"({rate:.2%}) exceeded allowed limit of {max_rate:.1%}!"
          )
  ```
  Called at `fleet_v6.py:935` before writing `submission.csv`. If fallbacks exceed 3%, execution terminates immediately with `RuntimeError`.

---

## 4. Verification & Parity Checks

1. **Clean Windows Verification**:
   - Tested on synthetic volume with empty slot 3 (`mask[66:78] = 0`) and padding.
   - All 30 selected center windows satisfy: $c-1, c, c+1$ are all unmasked and in the same slot.
   - Zero windows straddle slot boundaries; zero windows touch slot 3.
2. **Legacy Parity Verification**:
   - Compared `pick_centers(..., clean_windows=False)` against an untouched reference copy of today's legacy `_centers`.
   - Verified across ranges $[0, 96]$, $[5, 75]$, $[10, 30]$, $[20, 22]$, $[1, 5]$, and $[0, 50]$ for $K \in \{12, 24, 42, 94\}$: exact match across 100% of cases.
3. **Mirror Verification**:
   - Sagittal real slices reversed in-place; padded slices untouched.
   - Coronal and Axial slices horizontally flipped.
   - Double mirror verified: `np.array_equal(mirror_volume(mirror_volume(vol, mask), mask), vol)` is True.
4. **Source Code Parity**:
   - `inspect.getsource(train_knee.slot_of) == inspect.getsource(fleet_v6.slot_of)`: True.
   - `inspect.getsource(train_knee.pick_centers) == inspect.getsource(fleet_v6.pick_centers)`: True.
   - `inspect.getsource(train_knee.check_fallback_guard) == inspect.getsource(fleet_v6.check_fallback_guard)`: True.
   - `train_knee.SLOT_BOUNDS == fleet_v6.SLOT_BOUNDS`: True.
5. **Fallback Guard Verification**:
   - At 4% fallback rate (4/100): raises `RuntimeError`.
   - At 2% fallback rate (2/100): passes cleanly.
   - At 3% fallback rate (3/100): passes cleanly.

---

## 5. What Was NOT Checked

As specified in `delegation/PREAMBLE.md` and `delegation/tasks/T05_data_recipe.md`:
- **NOT CHECKED**: `./run_pc_dinov2_s11.sh` or GPU training run (a training job is currently running on this PC; GPU training was strictly prohibited).
- **NOT CHECKED**: 64-study node1 speed check with `--win_chunk 0` vs 14 (reserved for Claude on node1).
- **NOT CHECKED**: CoAtNet version 6 training run on node1 (reserved for Claude on node1 after seed 9 finishes).
- **NOT CHECKED**: Kaggle leaderboard score impact.
