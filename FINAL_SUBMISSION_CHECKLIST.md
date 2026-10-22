# FINAL SUBMISSION CHECKLIST — RSNA Knee Abnormality Detection
**Deadline: Oct 22, 2026 23:59 UTC (~55 hours from now)**  
**Current best public: 0.958 (heliosli 0.9586 / goodpjw2008 0.958)**  
**Target: 0.960-0.962 (gold 0.964, top-10 0.967)**

---

## ✅ WHAT'S READY (Validated, No Retraining Needed)

| Improvement | Evidence | Expected Gain |
|-------------|----------|---------------|
| Round-2 labels (report + image OOF mix) | 4 folds done, gold 0.9296→0.9318 | Base for our model |
| Scanner prior for Lateral OA | +0.010 report, +0.062 gold | +0.001-0.003 |
| Axial slot prefers FS series | PF OA 0.878→0.893 on affected | +0.0003-0.0005 |
| 5 all-data seeds (310-314) | Gold 0.9257-0.9346 | +0.001-0.002 |
| Flip TTA (anatomical mirror) | In heliosli 0.9586 | Already in public |
| Idea 9: Synovitis "not mentioned" down-weight | Synovitis 0.806→0.822 gold | +0.001-0.003 |

**Total expected: ~0.960-0.962** (not top-10, but medal territory)

---

## 📋 SUBMISSION PLAN (Next 48 Hours)

### TODAY (Oct 20) — Build & Test
| Time | Task | Owner |
|------|------|-------|
| Now | Build test corpus for our model on Kaggle | You (run notebook cell) |
| +2h | Run final_blend_complete.py on Kaggle (GPU) | You (push + run) |
| +6h | Check submission score (queue 4-6h) | Wait |
| Evening | If score ≥ 0.959, prepare bold version (OUR_WEIGHT=0.35) | You |

### TOMORROW (Oct 21) — Tune & Submit
| Time | Task |
|------|------|
| Morning | Read score, adjust OUR_WEIGHT (0.20 vs 0.35) |
| Midday | Submit v2 (bold if v1 good, else conservative) |
| Evening | Submit v3 (fine-tuned per-finding weights if time) |

### Oct 22 (Deadline Day) — Final Two
| Time | Task |
|------|------|
| Morning | Select best 2 submissions (safe + bold) |
| 18:00 UTC | Submit final 2 (allows 6h queue) |

---

## 🔧 WHAT YOU NEED TO DO NOW

### 1. Build Test Corpus for Our Model (on Kaggle)
Our model needs a test corpus at `/kaggle/input/knee-c192-u8d-test/` with:
- `all_ids.npy` — study UIDs
- `all_vols.npy` — (N, 192, 384, 384) uint8
- `all_masks.npy` — (N, 192) uint8

**Action**: Run this cell in a Kaggle notebook (GPU T4, internet off):
```python
# Build test corpus from DICOMs (takes ~30 min)
!git clone https://github.com/your-repo/rsna-knee.git
%cd rsna-knee
!python build_corpus96_stream.py --zip /kaggle/input/rsna-knee-abnormality-detection/test.zip \
    --series_csv /kaggle/input/rsna-knee-abnormality-detection/test_series.csv \
    --out_dir /kaggle/input/knee-c192-u8d-test --workers 8
```
*Or use the existing `build_corpus96_stream.py` from the repo with test data.*

### 2. Upload Required Datasets to Kaggle (if not already)
| Dataset | Purpose | Status |
|---------|---------|--------|
| `knee-cnxt-r2-folds` | Our 4-fold round-2 checkpoints | ✅ Exists |
| `knee-c192-u8d-test` | Our test corpus | ❌ **Build needed** |
| `knee-cloud-bundle` | Scanner prior + labels | ✅ Exists |
| `rsna-knee-publication-swa-weights-20261007` | Raptor weights | ✅ Public |
| `rsna-orthofoundation-fold0-epoch7-20261004` | R6 code | ✅ Public |
| `rsna-knee-r6-frozen-fe055406` | R6 weights v1 | ✅ Public |
| `rsna-knee-publication-swa-weights-20261007` | Raptor weights | ✅ Public |

### 3. Run Final Blend Notebook
Upload `cloud/final_blend_complete.py` as a Kaggle notebook with:
- **Accelerator**: GPU T4 × 2
- **Internet**: Off
- **Datasets**: All 7 above
- **Timeout**: 8.3 hours
- **OUR_WEIGHT**: 0.20 (safe) or 0.35 (bold)

**Expected runtime**: ~5-6 hours (Raptor 384: 2h, Raptor 224: 2h, R6: 2h, Our model: 1h, Blend: <30min)

---

## 🎯 TWO FINAL SUBMISSIONS

| Version | OUR_WEIGHT | Per-Finding Overrides | Rationale |
|---------|------------|----------------------|-----------|
| **Safe** | 0.20 | Lateral OA 0.1, Synovitis 0.1, PF OA 0.3 | Conservative, robust to private LB shuffle |
| **Bold** | 0.35 | Same | Maximizes our model contribution if it's truly better |

**Per-finding weights (fixed, not tuned on gold):**
- Lateral OA: 0.1 (R6 stronger)
- PF OA: 0.3 (Raptor stronger via OAI)
- Synovitis: 0.1 (Raptor stronger)
- Baker's: 0.3 (R6 stronger)
- All others: 0.2 (safe) / 0.35 (bold)

---

## ⚠️ RISKS & MITIGATIONS

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Test corpus build fails | Medium | Use heliosli's DICOM processing as base; our corpus is simpler (5 slots vs 6) |
| 9h timeout | Medium | Cut Raptor 224 stage first (saves 2h) |
| Submission queue >6h | High | Submit early Oct 21; last submit by 18:00 UTC Oct 22 |
| Private LB reshuffle | Certain | Two submissions hedged; don't over-tune to public |

---

## 📁 FILES TO UPLOAD (if not already)

```
/home/codemaster29/Documents/still_learning/rsna-knee-abnormality-detection/
├── cloud/final_blend_complete.py          # Main notebook script
├── cloud/final_blend_v2.py                # Simplified blend
├── scanner_latoa_prior.json               # Scanner prior for Lateral OA
├── folds_rand4.json                       # Fold definitions
├── train_knee.py                          # Our training/inference code
└── build_corpus96_stream.py               # Corpus builder
```

---

## 🚀 IMMEDIATE NEXT STEP

**You do this now:**
1. Open Kaggle, create new notebook
2. Set: GPU T4×2, Internet Off, 8.3h timeout
3. Attach all 7 datasets above
4. Paste `cloud/final_blend_complete.py` as the notebook
5. Set `OUR_WEIGHT = 0.20`
6. Run All
6. When done, **Submit to Competition** with description: `"Final blend: Raptor+R6+Our Round-2 @ 0.2, scanner prior, axial-FS"`

**Then wait for score. Report back here.**

---

## 📊 EXPECTED OUTCOMES

| Scenario | Public Score | Action |
|----------|--------------|--------|
| ≥ 0.960 | Great! | Submit bold (0.35) next |
| 0.959 | Good | Submit bold (0.35) next |
| 0.958 | Baseline | Try per-finding tune, resubmit |
| 0.957 | Below baseline | Check test corpus, our model weight |

**Top-10 (0.967) unlikely in 48h. Medal (top ~10%) realistic at 0.960+.**

---

*Report generated: 2026-10-20 12:00 IST | Based on 48h research + all validation runs*