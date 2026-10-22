import marimo._code_mode as cm

cell11_code = """# Cell 11: Load Gemma 4 (google/gemma-4-E2B-it)
from transformers import AutoProcessor as _AutoProc, AutoModelForImageTextToText as _AutoMod
import time as _time

_t0 = _time.time()
g4_model_id = "google/gemma-4-E2B-it"
_tok = key_form.value["hf_token"].strip() if key_form.value else None
assert _tok, "HF Token missing"

g4_processor = _AutoProc.from_pretrained(g4_model_id, token=_tok)
g4_model = _AutoMod.from_pretrained(
    g4_model_id,
    token=_tok,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
_g4_load_time = _time.time() - _t0
_vram_gb = torch.cuda.memory_allocated() / (1024**3)

mo.md(f\"\"\"
### 🤖 Gemma 4 E2B Initialized for Dialogue
- **Model:** `{g4_model_id}` (bfloat16) loaded in **{_g4_load_time:.1f}s**
- **Total VRAM Allocated:** **{_vram_gb:.2f} GB**
\"\"\")
"""

async def add_cell11():
    async with cm.get_context() as ctx:
        cid11 = ctx.create_cell(cell11_code, hide_code=False)
        ctx.run_cell(cid11)
        print(f"Cell 11 created with ID: {cid11}")

await add_cell11()
