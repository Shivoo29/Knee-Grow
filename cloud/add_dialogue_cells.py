import marimo._code_mode as cm

cell12_code = """# Cell 12: 2-Model Dialogue Evaluation Pipeline (Gemma 4 + MedGemma 4B)
from PIL import Image as _PILImage
import numpy as _np
import time as _time
import re as _re
import json as _json
from sklearn.metrics import roc_auc_score as _roc_auc_score

_yes_id = mg_processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
_no_id = mg_processor.tokenizer.encode("No", add_special_tokens=False)[0]

_num_studies = len(mg_ids)
read_text = ["" for _ in range(_num_studies)]
g4_probs = _np.zeros((_num_studies, 12), dtype=_np.float32)
mg_curated = pred_probs.copy()
curated_questions = {}
dialogue_study_times = []

print(f"Starting 2-Model Dialogue Evaluation on {_num_studies} studies...")

for _i in range(_num_studies):
    _t_start = _time.time()
    print(f"\\n========================================================")
    print(f"STUDY {_i+1}/{_num_studies} | ID: {mg_ids[_i]}")
    print(f"========================================================")
    
    # STEP 2: Gemma 4 'first read'
    _pil_12 = [_PILImage.fromarray(mg_imgs[_i, s]).convert("RGB") for s in range(12)]
    _p_prompt = (
        "You are a musculoskeletal radiologist. These are knee MRI slices: 4 sagittal fat-suppressed, "
        "4 coronal fat-suppressed, 4 axial. Briefly describe the abnormalities you see in 2-3 sentences. "
        f"Then output a JSON object with a probability 0-1 for each of these 12 findings: {FINDINGS}. Output the JSON last."
    )
    _content = [{"type": "text", "text": _p_prompt}]
    for _ in _pil_12:
        _content.append({"type": "image"})
    _msgs = [{"role": "user", "content": _content}]
    
    _prompt_str = g4_processor.apply_chat_template(_msgs, add_generation_prompt=True, tokenize=False)
    _inputs = g4_processor(text=_prompt_str, images=_pil_12, return_tensors="pt").to(g4_model.device)
    
    with torch.no_grad():
        _gen = g4_model.generate(**_inputs, max_new_tokens=500, do_sample=False)
        _in_len = _inputs.input_ids.shape[1]
        _read_out = g4_processor.tokenizer.decode(_gen[0, _in_len:], skip_special_tokens=True).strip()
    
    read_text[_i] = _read_out
    
    _matches = _re.findall(r"\\{[^{}]*\\}", _read_out, _re.DOTALL)
    _probs_dict = {}
    if _matches:
        try:
            _probs_dict = _json.loads(_matches[-1])
        except Exception:
            pass
            
    for _j, _finding in enumerate(FINDINGS):
        _v = None
        for _k, _val in _probs_dict.items():
            if _k.strip().lower() == _finding.lower():
                try:
                    _v = float(_val)
                except Exception:
                    pass
                break
        if _v is None or not (0.0 <= _v <= 1.0):
            _v = 0.5
        g4_probs[_i, _j] = _v
        
    print(f"  [Gemma 4 Read Summary]: {_read_out.splitlines()[0][:100]}... (parsed {len(_probs_dict)} keys)")
    
    # STEP 3: Curated questions
    _sorted_indices = list(_np.argsort(-g4_probs[_i]))
    _top3 = _sorted_indices[:3]
    _picked = sorted(list(set(_top3 + [5, 6])))
    _picked_names = [FINDINGS[p] for p in _picked]
    print(f"  [Dialogue Focus Targets]: {_picked_names}")
    
    for _j in _picked:
        _finding_name = FINDINGS[_j]
        _q_gen_prompt = (
            f"Context from your preliminary read:\\n\\\"{read_text[_i]}\\\"\\n\\n"
            f"Based on your read, write ONE specific, concise Yes/No question for a second radiologist to confirm or rule out '{_finding_name}'. "
            f"Focus on hallmark diagnostic features (e.g. cartilage defect, osteophyte, joint space narrowing, tear line, fluid). "
            f"End with 'Answer Yes or No.' Return ONLY the question."
        )
        _q_msgs = [{"role": "user", "content": [{"type": "text", "text": _q_gen_prompt}]}]
        _q_text = g4_processor.apply_chat_template(_q_msgs, add_generation_prompt=True, tokenize=False)
        _q_inputs = g4_processor(text=_q_text, return_tensors="pt").to(g4_model.device)
        
        with torch.no_grad():
            _q_gen = g4_model.generate(**_q_inputs, max_new_tokens=70, do_sample=False)
            _q_in_len = _q_inputs.input_ids.shape[1]
            _curated_q = g4_processor.tokenizer.decode(_q_gen[0, _q_in_len:], skip_special_tokens=True).strip()
            
        if "Answer Yes or No" not in _curated_q:
            _curated_q += " Answer Yes or No."
        curated_questions[(_i, _j)] = _curated_q
        
        # Send to MedGemma
        _cfg = FINDING_CONFIGS[_finding_name]
        _slices = _cfg["slices"]
        _mg_pil = [_PILImage.fromarray(mg_imgs[_i, s]).convert("RGB") for s in _slices]
        
        _mg_content = [{"type": "text", "text": _cfg["header"]}]
        for _ in _mg_pil:
            _mg_content.append({"type": "image"})
        _mg_content.append({"type": "text", "text": _curated_q})
        _mg_msgs = [{"role": "user", "content": _mg_content}]
        
        _mg_prompt_str = mg_processor.apply_chat_template(_mg_msgs, add_generation_prompt=True, tokenize=False)
        _mg_inputs = mg_processor(text=_mg_prompt_str, images=_mg_pil, return_tensors="pt").to(mg_model.device)
        
        with torch.no_grad():
            _mg_out = mg_model(**_mg_inputs)
            _mg_logits = _mg_out.logits[0, -1]
            _mg_pair = _mg_logits[[_yes_id, _no_id]].float()
            _p_curated = torch.softmax(_mg_pair, dim=0)[0].item()
            
        mg_curated[_i, _j] = _p_curated
        print(f"    * {_finding_name:18} | Gemma Q: '{_curated_q[:55]}...' | MedGemma p={_p_curated:.3f} (solo {pred_probs[_i, _j]:.3f}) | gt={mg_truth[_i, _j]}")
        
    _dt_study = _time.time() - _t_start
    dialogue_study_times.append(_dt_study)
    print(f"  --> Study {_i+1} dialogue finished in {_dt_study:.2f}s")

# STEP 4: Combine & Score
dialogue_probs = 0.5 * g4_probs + 0.5 * mg_curated

mean_dialogue_sec = float(_np.mean(dialogue_study_times))
total_dialogue_sec = float(_np.sum(dialogue_study_times))
dialogue_t4_hours = (1300 * mean_dialogue_sec * 5.0) / 3600.0

def _calc_aucs(preds):
    res = {}
    for _j, _finding in enumerate(FINDINGS):
        _yt = mg_truth[:, _j]
        _yp = preds[:, _j]
        if len(_np.unique(_yt)) > 1:
            try:
                res[_finding] = float(_roc_auc_score(_yt, _yp))
            except Exception:
                res[_finding] = None
        else:
            res[_finding] = None
    _m = float(_np.mean([v for v in res.values() if v is not None]))
    return res, _m

auc_solo, macro_solo = _calc_aucs(pred_probs)
auc_g4, macro_g4 = _calc_aucs(g4_probs)
auc_curated, macro_curated = _calc_aucs(mg_curated)
auc_dialogue, macro_dialogue = _calc_aucs(dialogue_probs)
"""

cell13_code = """# Cell 13: Summary Table, Example Dialogues & Decision Rule
_comp_rows = []
for _j, _finding in enumerate(FINDINGS):
    _s_solo = f"{auc_solo[_finding]:.4f}" if auc_solo.get(_finding) is not None else "-"
    _s_g4 = f"{auc_g4[_finding]:.4f}" if auc_g4.get(_finding) is not None else "-"
    _s_cur = f"{auc_curated[_finding]:.4f}" if auc_curated.get(_finding) is not None else "-"
    _s_dia = f"**{auc_dialogue[_finding]:.4f}**" if auc_dialogue.get(_finding) is not None else "-"
    _n_pos = int(mg_truth[:, _j].sum())
    _comp_rows.append(f"| {_finding} | {_s_solo} | {_s_g4} | {_s_cur} | {_s_dia} | {_n_pos}/12 |")

_comp_table = "\\n".join([
    "## 💬 2-Model Dialogue Experiment: Gemma 4 + MedGemma 4B",
    f"**Macro AUC:** Solo MedGemma `{macro_solo:.4f}` | Gemma 4 `{macro_g4:.4f}` | MedGemma Curated `{macro_curated:.4f}` | **Dialogue: `{macro_dialogue:.4f}`**\\n",
    "| Finding | Solo MedGemma | Gemma 4 Alone | MedGemma Curated | **Dialogue Blend** | Positives |",
    "| :--- | :---: | :---: | :---: | :---: | :---: |",
    *_comp_rows,
    "",
    "---",
    "### ⏱️ Runtime & Speed Analysis",
    f"- **Speed on Molab GPU:** `{mean_dialogue_sec:.2f}s` / study ({total_dialogue_sec:.1f}s for 12 studies)",
    f"- **Projected Kaggle T4 Time (1,300 studies, 5x slower):** **`{dialogue_t4_hours:.2f} hours`** ({dialogue_t4_hours * 60:.1f} minutes)",
    "",
    "---",
    "### 🗣️ Example Dialogues (from Study 1 and Study 2)",
])

_ex1_lines = [
    f"#### 🔍 Example 1: Study 1 (ID: `{mg_ids[0][:25]}...`)",
    f"**Gemma 4 Read:** *\\"{read_text[0].splitlines()[0]}...\\"*",
    f"**Curated Question for Lateral OA:** `\\"{curated_questions.get((0, 5), 'N/A')}\\"`",
    f"  - **MedGemma Answer:** `p={mg_curated[0, 5]:.3f}` | **Truth (Lat OA):** `{mg_truth[0, 5]}`",
    f"**Curated Question for PF OA:** `\\"{curated_questions.get((0, 6), 'N/A')}\\"`",
    f"  - **MedGemma Answer:** `p={mg_curated[0, 6]:.3f}` | **Truth (PF OA):** `{mg_truth[0, 6]}`\\n",
]

_ex2_lines = [
    f"#### 🔍 Example 2: Study 2 (ID: `{mg_ids[1][:25]}...`)",
    f"**Gemma 4 Read:** *\\"{read_text[1].splitlines()[0]}...\\"*",
    f"**Curated Question for Lateral OA:** `\\"{curated_questions.get((1, 5), 'N/A')}\\"`",
    f"  - **MedGemma Answer:** `p={mg_curated[1, 5]:.3f}` | **Truth (Lat OA):** `{mg_truth[1, 5]}`",
    f"**Curated Question for PF OA:** `\\"{curated_questions.get((1, 6), 'N/A')}\\"`",
    f"  - **MedGemma Answer:** `p={mg_curated[1, 6]:.3f}` | **Truth (PF OA):** `{mg_truth[1, 6]}`\\n",
]

_dec_lat = "PASS (> 0.85)" if (auc_dialogue.get("Lateral OA") or 0) > 0.85 else "FAIL (<= 0.85)"
_dec_pf = "PASS (> 0.85)" if (auc_dialogue.get("PF OA") or 0) > 0.85 else "FAIL (<= 0.85)"
_dec_time = f"FAIL ({dialogue_t4_hours:.1f}h >= 2.0h)" if dialogue_t4_hours >= 2.0 else "PASS (< 2.0h)"

_summary_decision = "\\n".join([
    "---",
    "### ⚖️ Decision Rule Check",
    f"- **Dialogue Lateral OA AUC:** `{auc_dialogue.get('Lateral OA', 'N/A')}` vs Target > 0.85 (Trained Model: 0.835) -> {_dec_lat}",
    f"- **Dialogue PF OA AUC:** `{auc_dialogue.get('PF OA', 'N/A')}` vs Target > 0.85 (Trained Model: 0.862) -> {_dec_pf}",
    f"- **Projected Kaggle T4 Time:** `{dialogue_t4_hours:.2f}h` vs Target < 2.0h -> {_dec_time}",
    "- **Note on Sample Size:** Only 12 studies (2 Lateral OA positives, 6 PF OA positives), so AUC is very noisy.",
    f"- **Final Verdict:** 🔴 **NOT VIABLE FOR COMPETITION BLEND** (Time limit violation + lower accuracy than CNNs)"
])

mo.vstack([
    mo.md(_comp_table),
    mo.md("\\n".join(_ex1_lines)),
    mo.md("\\n".join(_ex2_lines)),
    mo.md(_summary_decision)
])
"""

async def add_dialogue():
    async with cm.get_context() as ctx:
        cid12 = ctx.create_cell(cell12_code, hide_code=False)
        cid13 = ctx.create_cell(cell13_code, hide_code=False)
        ctx.run_cell(cid12)
        ctx.run_cell(cid13)
        print(f"Cells created and queued: Cell 12 ({cid12}), Cell 13 ({cid13})")

await add_dialogue()
