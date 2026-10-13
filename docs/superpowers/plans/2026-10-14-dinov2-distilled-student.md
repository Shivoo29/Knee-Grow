# DINOv2 Distilled Student (Experiment A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one distilled student with a different backbone family (DINOv2 ViT-Base) on `labels_distil_v1.parquet`, and make the Kaggle inference notebooks able to run it, so it can join the blend with diverse errors.

**Architecture:** No new training code. `train_knee.py` already takes `--arch/--res/--freeze_blocks` and stores `arch`/`res` in every checkpoint; `fleet_v6.py` already rebuilds the model from `ck["arch"]`. Only gap: input normalisation (DINOv2 wants ImageNet mean/std; `fleet_v6.py` applies none). Add `norm` to checkpoints and apply it at inference. Then train, gate on gold-58, upload weights, build a Kaggle notebook.

**Tech Stack:** PyTorch, timm 1.0.30 (has `vit_base_patch14_reg4_dinov2`, `vit_base_patch16_dinov3`), Kaggle notebooks (T4 x2, offline), existing `kaggle_students/build_nb.py`.

**Spec:** none written. Source of intent: chat of 2026-10-14 (decision: Experiment A before Experiment B). Evidence base: `STATUS.md`, `train_knee.py:296-362`, `kaggle_base954_fleet/fleet_v6.py:84-96,745-770`.

## Global Constraints

- Never open or print the Report column; never paste images or report text (competition rule 4.b).
- Never touch the Azure MinisAi machine; never kill the user's `finetune.py` on the PC; no systemd auto-restart on node1.
- Only the user clicks Submit on Kaggle and pushes to GitHub. Local commits only, **no Co-Authored-By lines**.
- Deadline: final submission Oct 22 23:59 UTC. 5 submissions/day (reset 05:30 IST). Kaggle GPU quota about 11 h left until Oct 17.
- Kaggle inference: T4 x2, no internet, 9 h hard limit. Training must keep the CoAtNet recipe's data path (`ds/corpus96`, `--k 12 --k_eval 94`).
- User-facing text uses full model names ("CoAtNet version 6 seed 9"), not short tags.
- GPUs are busy: node1 runs CoAtNet seed 9 (`tmux distil24`), PC runs CoAtNet seed 10 until about Oct 14 02:00 IST. Do not start a second training job on a busy GPU.

## Review Focus

- A ViT checkpoint trained with ImageNet norm but run at inference without it gives silently wrong (not crashing) scores. Pinned by Task 1 test.
- `res` not a multiple of the ViT patch size (14 for DINOv2) breaks or silently crops. Pinned by Task 2 smoke.
- Old CoAtNet checkpoints have no `norm` key; inference must keep treating them as `none`. Pinned by Task 1 test.
- Inference time on T4 for 94 windows with ViT-Base could exceed the CoAtNet cost and hit the 9 h limit. Pinned by Task 4 timing gate.
- DINOv2 weights need internet at training time only (node1/PC have it); inference loads from the checkpoint with `pretrained=False`. Pinned by Task 4.

---

### Task 1: Carry `norm` through checkpoints and inference

**Files:**
- Modify: `train_knee.py` (checkpoint dicts around lines 676, 703, 718: add `"norm": a.norm`)
- Modify: `kaggle_base954_fleet/fleet_v6.py` (read `norm` after line 754, apply it where window tensors are built)
- Test: `tests/test_norm_contract.py`

**Interfaces:**
- Consumes: `a.norm` in `train_knee.py` (`"none"` or `"imagenet"`); checkpoint dict `ck`.
- Produces: checkpoint key `ck["norm"]` (str); in `fleet_v6.py` a function `apply_norm(x, norm)` taking a float tensor `(..., 3, H, W)` in [0,1] and returning the normalised tensor (identity for `"none"`).

- [ ] **Step 1: Find where inference builds the model input**

Run: `grep -n "WIN_CHUNK\|def score\|autocast" kaggle_base954_fleet/fleet_v6.py | head -30`
Note the line where window chunks are moved to the device; `apply_norm` is applied there.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_norm_contract.py
import importlib.util, torch
spec = importlib.util.spec_from_file_location("fleet_v6", "kaggle_base954_fleet/fleet_v6.py")
src = open("kaggle_base954_fleet/fleet_v6.py").read()
ns = {}
start = src.index("def apply_norm")
end = src.index("\n\n", start)
exec("import torch\n" + src[start:end], ns)
apply_norm = ns["apply_norm"]

def test_none_is_identity():
    x = torch.rand(2, 3, 8, 8)
    assert torch.equal(apply_norm(x, "none"), x)

def test_imagenet_matches_timm_constants():
    x = torch.full((1, 3, 4, 4), 0.5)
    y = apply_norm(x, "imagenet")
    exp = (0.5 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    assert torch.allclose(y[0, :, 0, 0], exp, atol=1e-5)
```

- [ ] **Step 3: Run it, expect FAIL**

Run: `.venv/bin/python -m pytest tests/test_norm_contract.py -q`
Expected: FAIL (`ValueError: substring not found`, no `apply_norm` yet).

- [ ] **Step 4: Implement**

In `fleet_v6.py`, above `build_backbone`:

```python
def apply_norm(x, norm):
    if norm != "imagenet":
        return x
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    return (x - mean) / std
```

After `res = int(ck.get("res", 384))` add `norm = ck.get("norm", "none")`, and apply `chunk = apply_norm(chunk, norm)` at the line found in Step 1. In `train_knee.py` add `"norm": a.norm,` next to every `"res": a.res,` in the three saved dicts. Check `train_knee.py` `_mean/_std` constants equal the ImageNet values above (`grep -n "_mean\|_std" train_knee.py`); if they differ, copy those exact values into `apply_norm` and the test.

- [ ] **Step 5: Run test, expect PASS**

Run: `.venv/bin/python -m pytest tests/test_norm_contract.py -q`
Expected: 2 passed.

- [ ] **Step 6: Commit (no co-author)**

```bash
git add tests/test_norm_contract.py train_knee.py kaggle_base954_fleet/fleet_v6.py
git commit -m "Carry input norm in checkpoints and apply it at inference"
```

---

### Task 2: Smoke-test the DINOv2 backbone and measure cost

**Files:**
- Create: `run_pc_dinov2_s11.sh`

**Interfaces:**
- Consumes: Task 1 (checkpoints carry `norm`); `train_knee.py --smoke`; free GPU (PC after seed 10 ends).
- Produces: measured seconds per epoch, peak VRAM, and the final launcher `run_pc_dinov2_s11.sh` used by Task 3.

- [ ] **Step 1: Write the launcher**

```bash
#!/bin/bash
# Distilled student, DINOv2 ViT-Base (frozen first 6 blocks), seed 11, PC (RTX 4070 SUPER 12 GB). Same effective batch as the CoAtNet students.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RAPTOR_INPUT="$PWD/ds"
exec systemd-inhibit --what=sleep:idle --why="knee dinov2 s11 training" \
  .venv/bin/python train_knee.py --ckpt timm --arch vit_base_patch14_reg4_dinov2 \
    --res 378 --norm imagenet --bs 1 --accum 16 --k 12 --k_eval 94 --epochs 16 --patience 5 \
    --win_chunk 8 --grad_ckpt --freeze_blocks 6 --bb_lr 2e-5 \
    --labels labels_distil_v1.parquet --seed 11 --tag dinov2_s11 "$@"
```

`378 = 27 x 14` (multiple of the patch size). Run `chmod +x run_pc_dinov2_s11.sh`.

- [ ] **Step 2: Confirm the PC GPU is free**

Run: `pgrep -af train_knee.py; nvidia-smi --query-gpu=memory.used --format=csv,noheader`
Expected: no `train_knee.py` process, memory under 1 GB. If seed 10 is still running, STOP and wait; do not start.

- [ ] **Step 3: Run the smoke**

Run: `./run_pc_dinov2_s11.sh --smoke 2>&1 | tail -30`
Expected: model loads (`[raptor] timm-pretrained backbone: vit_base_patch14_reg4_dinov2`), one step runs, no OOM, no shape error. If the `--freeze_blocks` message reports 0 frozen tensors, the block name pattern did not match; open `train_knee.py:529-540` and fix the pattern before continuing.

- [ ] **Step 4: Measure real cost**

Run: `./run_pc_dinov2_s11.sh --limit 64 --epochs 1 2>&1 | grep -E "^ep|sec|VRAM|peak" ; nvidia-smi --query-gpu=memory.used --format=csv,noheader`
Compute full-epoch time = (time for 64 studies) x (train size / 64). Write the number into `STATUS.md`.
Gate: full run (16 epochs) must finish by Oct 18 22:00 IST. If epoch time x 16 exceeds that, lower `--epochs` to fit and edit the launcher; if a single epoch exceeds 2 h, switch the arch to `vit_small_patch14_dinov2` (same flags) and re-measure.

- [ ] **Step 5: Commit**

```bash
git add run_pc_dinov2_s11.sh STATUS.md
git commit -m "PC launcher for DINOv2 ViT-Base distilled student (seed 11) with measured cost"
```

---

### Task 3: Train the student and gate on gold-58

**Files:**
- Create (by run): `train_coat_dinov2_s11.log`, `raptor_ft_dinov2_s11.json`, `raptor_ft_dinov2_s11_swa.pt` (the `.pt` stays gitignored)

**Interfaces:**
- Consumes: `run_pc_dinov2_s11.sh` from Task 2.
- Produces: `raptor_ft_dinov2_s11_swa.pt` (checkpoint with `arch`, `res=378`, `norm="imagenet"`), best gold-58 AUC in the JSON.

- [ ] **Step 1: Launch detached**

Run: `nohup setsid ./run_pc_dinov2_s11.sh > train_coat_dinov2_s11.log 2>&1 &`
Expected: `pgrep -af train_knee.py` shows the process within 30 s.

- [ ] **Step 2: Check progress every few hours**

Run: `grep "^ep" train_coat_dinov2_s11.log | tail -3 | cut -c1-90`
Expected: gold AUC rising over the first epochs. Do not kill it; the learning-rate schedule is fixed at launch.

- [ ] **Step 3: Apply the gate when it finishes**

Read the best gold-58 AUC from `raptor_ft_dinov2_s11.json`.
- At least 0.915: go to Task 4.
- Below 0.915: stop here, record the result in `STATUS.md`, and fall back to Plan B (public-notebook blend). CoAtNet students reached 0.9279 and 0.9296; gold-58 noise is about 0.02, so this is a floor to rule out a broken run, not a ranking.

- [ ] **Step 4: Commit the logs**

```bash
git add train_coat_dinov2_s11.log raptor_ft_dinov2_s11.json STATUS.md
git commit -m "DINOv2 ViT-Base distilled student seed 11: training log and result"
```

---

### Task 4: Kaggle notebook and timing check

**Files:**
- Modify: `kaggle_students/build_nb.py` (CKPTS dict at line 18; add the ViT checkpoint)
- Modify: `kaggle_students_upload/` (add the weight; new dataset version of `shivoo29/knee-students`)

**Interfaces:**
- Consumes: `raptor_ft_dinov2_s11_swa.pt`; `build_nb.py` produces `students_only/` and `base954_students/` notebooks from `CKPTS`.
- Produces: kernels `shivoo29/rsna-knee-students-only` and `shivoo29/rsna-knee-base954-students` running CoAtNet seeds 7, 8 plus the DINOv2 student.

- [ ] **Step 1: Add the checkpoint**

Change line 18 to:

```python
CKPTS = {'s7': 'raptor_ft_v6d_s7_swa.pt', 's8': 'raptor_ft_v6d_s8_swa.pt', 'dino11': 'raptor_ft_dinov2_s11_swa.pt'}
```

Also add a separate output so the DINOv2 student's own LB can be read: build a third notebook variant `dino_only` is NOT worth it (YAGNI); instead the blend notebook weights stay at W and the LB of "students only (3)" vs the earlier 0.948 shows the effect.

- [ ] **Step 2: Upload the weight to the Kaggle dataset**

Copy `raptor_ft_dinov2_s11_swa.pt` into `kaggle_students_upload/`, then run:
`cd kaggle_students_upload && kaggle datasets version -p . -m "add DINOv2 ViT-Base student seed 11" --dir-mode zip`
Expected: "Dataset version is being created". The user approves the upload (it sends weights to an external service).

- [ ] **Step 3: Rebuild and push the notebooks**

Run: `.venv/bin/python kaggle_students/build_nb.py && kaggle kernels push -p kaggle_students/students_only && kaggle kernels push -p kaggle_students/base954_students`
Expected: both kernels queued; wait for the commit run on the 3 visible studies to say "Succeeded".

- [ ] **Step 4: Timing gate**

Read the log line `STUDENT_DONE dino11 <seconds>s` (printed by `build_nb.py` RUN code) from the commit run. Hidden test has many more studies than the 3 visible; scale: per-student time on the hidden set = commit seconds x (hidden studies / 3) is NOT valid because of fixed startup cost, so compare against the already-known CoAtNet student cost: run time of `STUDENT_DONE s7` in the same log. Accept if `dino11` seconds is at most 1.5 x `s7` seconds. Otherwise set `STOP_AFTER_H` lower in `build_nb.py` (already skips late students) and rebuild.

- [ ] **Step 5: Commit**

```bash
git add kaggle_students/build_nb.py
git commit -m "Students notebooks: add DINOv2 ViT-Base student"
```

(The user submits to the leaderboard themselves; compare the new LB against 0.948 for students only and against the base954 + students 0.15 result.)

---

## Self-review

- Spec coverage: backbone swap (Task 2, 3), diversity (Task 4 blend), norm risk (Task 1), cost/limits (Task 2 gate, Task 4 gate), safety constraints (Global Constraints). No gap found.
- Placeholders: none. Two commands depend on a value measured in the run (epoch count, `apply_norm` insertion line); the plan says exactly how to find each.
- Type consistency: checkpoint key `norm` (str) written in Task 1, read in Task 1, relied on in Tasks 3 and 4. Launcher name `run_pc_dinov2_s11.sh` and tag `dinov2_s11` are the same across tasks. `raptor_ft_dinov2_s11_swa.pt` follows the existing `raptor_ft_<tag>_swa.pt` naming.
- Unverified assumptions to check during execution: `train_knee.py` ImageNet mean/std constants equal the values in `apply_norm`; `--freeze_blocks` matches ViT `blocks.N.` names; the `--limit` flag still allows timing a partial epoch.
