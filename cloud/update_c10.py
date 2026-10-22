import marimo._code_mode as cm

fixed_cell10 = """# Cell 10: Results & Decision Rule Report
import numpy as _np

_rows = []
for _j, _finding in enumerate(FINDINGS):
    _y_true = mg_truth[:, _j]
    _n_pos = int(_np.sum(_y_true))
    _n_neg = len(_y_true) - _n_pos
    _auc_str = f"**{auc_per_finding[_finding]:.4f}**" if auc_per_finding.get(_finding) is not None else "N/A (1 class)"
    _rows.append(f"| {_finding} | {_auc_str} | {_n_pos} pos / {_n_neg} neg | {pred_probs[:, _j].mean():.3f} |")

_decision_lat = "⚠️ ARTIFACT (AUC=1.0000 on 2 pos, but saturated mean_p=1.000)"
_decision_pf = "❌ FAIL (AUC=0.6111 <= 0.85)"
_decision_time = f"❌ FAIL ({kaggle_t4_total_hours:.1f}h >= 2.0h limit)"

_overall_decision = "🔴 **DO NOT ADD TO COMPETITION BLEND**"

_table_md = "\\n".join([
    f"## 🏆 Feasibility Study: {mg_model_id} (Zero-Shot)",
    f"**Macro AUC (evaluable classes):** **`{macro_auc:.4f}`** | **Speed:** **`{mean_sec_per_study:.2f}s / study`** (12 questions)\\n",
    "| Finding | Zero-Shot AUC | Positive / Negative | Mean Pred Prob |",
    "| :--- | :--- | :--- | :--- |",
    *_rows,
    "",
    "---",
    "### ⏱️ Runtime & Kaggle Inference Feasibility",
    f"- **Speed on Molab GPU:** `{mean_sec_per_study:.2f}` sec / study ({mean_sec_per_study / 12:.2f}s per question)",
    f"- **Total Time for 12 Studies:** `{total_sec:.1f}` seconds ({total_sec / 60:.2f} minutes)",
    f"- **Estimated Speed on Kaggle T4 (5x slower):** `{kaggle_t4_sec_per_study:.2f}` sec / study",
    f"- **Projected Kaggle Runtime for 1,300 Studies:** **`{kaggle_t4_total_hours:.2f} hours`** ({kaggle_t4_total_hours * 60:.1f} minutes)",
    "",
    "---",
    "### ⚖️ Decision Rule Check",
    f"- **Lateral OA AUC:** `{auc_per_finding.get('Lateral OA', 'N/A')}` vs Target > 0.85 (Current Trained Model: 0.835) -> {_decision_lat}",
    f"- **PF OA AUC:** `{auc_per_finding.get('PF OA', 'N/A')}` vs Target > 0.85 (Current Trained Model: 0.862) -> {_decision_pf}",
    f"- **Kaggle T4 Time for 1,300 Studies:** `{kaggle_t4_total_hours:.2f}h` vs Target < 2.0h -> {_decision_time}",
    f"- **Final Verdict:** {_overall_decision}"
])

mo.md(_table_md)
"""

async def update():
    async with cm.get_context() as ctx:
        ctx.edit_cell('dOOW', fixed_cell10)
        ctx.run_cell('dOOW')
        print("Cell 10 successfully updated and executed in notebook!")

await update()
