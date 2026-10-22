import marimo._code_mode as cm
import base64

cell9_code = """# Cell 9: Run Feasibility Evaluation Loop & Compute Metrics
from PIL import Image as _PILImage
import numpy as _np
from sklearn.metrics import roc_auc_score as _roc_auc_score

_yes_id = mg_processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
_no_id = mg_processor.tokenizer.encode("No", add_special_tokens=False)[0]

_num_studies = len(mg_ids)
pred_probs = _np.zeros((_num_studies, 12), dtype=_np.float32)
study_times = []

print(f"Starting evaluation of {_num_studies} studies on {mg_model_id}...")

for _i in range(_num_studies):
    _t_study = time.time()
    print(f"\\n--- Study {_i+1}/{_num_studies} (ID: {mg_ids[_i]}) ---")
    
    for _j, _finding in enumerate(FINDINGS):
        _cfg = FINDING_CONFIGS[_finding]
        _slice_indices = _cfg["slices"]
        _pil_imgs = [_PILImage.fromarray(mg_imgs[_i, s]).convert("RGB") for s in _slice_indices]
        
        _content = [{"type": "text", "text": _cfg["header"]}]
        for _ in _pil_imgs:
            _content.append({"type": "image"})
        _content.append({"type": "text", "text": _cfg["question"]})
        _msgs = [{"role": "user", "content": _content}]
        
        _prompt = mg_processor.apply_chat_template(_msgs, add_generation_prompt=True, tokenize=False)
        _inputs = mg_processor(text=_prompt, images=_pil_imgs, return_tensors="pt").to(mg_model.device)
        
        with torch.no_grad():
            _out = mg_model(**_inputs)
            _logits = _out.logits[0, -1]
            _pair = _logits[[_yes_id, _no_id]].float()
            _p = torch.softmax(_pair, dim=0)[0].item()
        
        pred_probs[_i, _j] = _p
    
    _dt_study = time.time() - _t_study
    study_times.append(_dt_study)
    print(f"  [+] Finished study {_i+1} in {_dt_study:.2f}s | Lat OA: p={pred_probs[_i, 5]:.2f} (gt={mg_truth[_i, 5]}) | PF OA: p={pred_probs[_i, 6]:.2f} (gt={mg_truth[_i, 6]})")

mean_sec_per_study = float(_np.mean(study_times))
total_sec = float(_np.sum(study_times))

# Extrapolate to 1,300 studies on Kaggle T4 (5x slower)
kaggle_t4_sec_per_study = mean_sec_per_study * 5.0
kaggle_t4_total_hours = (1300 * kaggle_t4_sec_per_study) / 3600.0

# Compute AUC per finding
auc_per_finding = {}
for _j, _finding in enumerate(FINDINGS):
    _y_true = mg_truth[:, _j]
    _y_pred = pred_probs[:, _j]
    if len(_np.unique(_y_true)) > 1:
        try:
            _auc = _roc_auc_score(_y_true, _y_pred)
            auc_per_finding[_finding] = float(_auc)
        except Exception as _e:
            auc_per_finding[_finding] = None
    else:
        auc_per_finding[_finding] = None

macro_auc = float(_np.mean([v for v in auc_per_finding.values() if v is not None])) if auc_per_finding else 0.0
"""

cell10_code = """# Cell 10: Results & Decision Rule Report
_rows = []
for _j, _finding in enumerate(FINDINGS):
    _y_true = mg_truth[:, _j]
    _n_pos = int(_np.sum(_y_true))
    _n_neg = len(_y_true) - _n_pos
    _auc_str = f"**{auc_per_finding[_finding]:.4f}**" if auc_per_finding.get(_finding) is not None else "N/A (1 class)"
    _rows.append(f"| {_finding} | {_auc_str} | {_n_pos} pos / {_n_neg} neg | {pred_probs[:, _j].mean():.3f} |")

_decision_lat = "PASS (> 0.85)" if (auc_per_finding.get("Lateral OA") or 0) > 0.85 else "FAIL (<= 0.85)"
_decision_pf = "PASS (> 0.85)" if (auc_per_finding.get("PF OA") or 0) > 0.85 else "FAIL (<= 0.85)"
_decision_time = "PASS (< 2.0h)" if kaggle_t4_total_hours < 2.0 else "FAIL (>= 2.0h)"

_overall_decision = "RECOMMENDED FOR BLEND" if (((auc_per_finding.get("Lateral OA") or 0) > 0.85 or (auc_per_finding.get("PF OA") or 0) > 0.85) and kaggle_t4_total_hours < 2.0) else "DO NOT ADD TO BLEND"

_table_md = "\\n".join([
    f"## Feasibility Study: {mg_model_id} (Zero-Shot)",
    f"**Macro AUC (evaluable classes):** **`{macro_auc:.4f}`** | **Speed:** **`{mean_sec_per_study:.2f}s / study`** (12 questions)\\n",
    "| Finding | Zero-Shot AUC | Positive / Negative | Mean Pred Prob |",
    "| :--- | :--- | :--- | :--- |",
    *_rows,
    "",
    "---",
    "### Runtime & Kaggle Inference Feasibility",
    f"- **Speed on Molab GPU:** `{mean_sec_per_study:.2f}` sec / study ({mean_sec_per_study / 12:.2f}s per question)",
    f"- **Total Time for 12 Studies:** `{total_sec:.1f}` seconds ({total_sec / 60:.2f} minutes)",
    f"- **Estimated Speed on Kaggle T4 (5x slower):** `{kaggle_t4_sec_per_study:.2f}` sec / study",
    f"- **Projected Kaggle Runtime for 1,300 Studies:** **`{kaggle_t4_total_hours:.2f} hours`** ({kaggle_t4_total_hours * 60:.1f} minutes)",
    "",
    "---",
    "### Decision Rule Check",
    f"- **Lateral OA AUC:** `{auc_per_finding.get('Lateral OA', 'N/A')}` vs Target > 0.85 (Current Trained Model: 0.835) -> {_decision_lat}",
    f"- **PF OA AUC:** `{auc_per_finding.get('PF OA', 'N/A')}` vs Target > 0.85 (Current Trained Model: 0.862) -> {_decision_pf}",
    f"- **Kaggle T4 Time for 1,300 Studies:** `{kaggle_t4_total_hours:.2f}h` vs Target < 2.0h -> {_decision_time}",
    f"- **Final Verdict:** {_overall_decision}"
])

mo.md(_table_md)
"""

async def add_cells():
    async with cm.get_context() as ctx:
        cid9 = ctx.create_cell(cell9_code, hide_code=False)
        cid10 = ctx.create_cell(cell10_code, hide_code=False)
        ctx.run_cell(cid9)
        ctx.run_cell(cid10)
        print(f"Cells created and queued: Cell 9 ({cid9}), Cell 10 ({cid10})")

await add_cells()
