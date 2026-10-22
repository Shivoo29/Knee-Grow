#!/usr/bin/env python3
"""
Final submission script.
Creates two submission notebooks:
1. Safe: heliosli 0.9586 recipe + our model at 0.2 weight
2. Bold: same + our model at 0.35 weight

This script prepares the notebooks for upload to Kaggle.
"""
import os, json, shutil
from pathlib import Path

# Source notebooks
SRC_HELIOSLI = "/home/codemaster29/Documents/still_learning/rsna-knee-abnormality-detection/research_notes/Knee MRI last 48 hour gains/kaggle_oct20/kernels/heliosli__this-is-best-public-dataset-new-version-0-9586/this-is-best-public-dataset-new-version-0-9586.ipynb"

OUT_DIR = Path("/home/codemaster29/Documents/still_learning/rsna-knee-abnormality-detection/final_submission/notebooks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load heliosli notebook
with open(SRC_HELIOSLI) as f:
    nb = json.load(f)

def create_blend_notebook(our_weight, suffix):
    """Create a blend notebook with our model at given weight."""
    new_nb = json.loads(json.dumps(nb))  # deep copy
    
    # Modify the blend weights in the notebook
    # Find the blend cell (Stage 3) and update weights
    for cell in new_nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "W_RAPTOR_C" in src and "W_RAPTOR" in src:
                # Replace the weight values
                src = src.replace('OUR_WEIGHT = 0.20', f'OUR_WEIGHT = {our_weight}')
                src = src.replace('OUR_WEIGHT = 0.35', f'OUR_WEIGHT = {our_weight}')
                cell["source"] = src.splitlines(keepends=True)
            if "OUR_WEIGHT = 0.20" in src:
                src = src.replace('OUR_WEIGHT = 0.20', f'OUR_WEIGHT = {our_weight}')
                cell["source"] = src.splitlines(keepends=True)
            if "OUR_WEIGHT = 0.35" in src:
                src = src.replace('OUR_WEIGHT = 0.35', f'OUR_WEIGHT = {our_weight}')
                cell["source"] = src.splitlines(keepends=True)
    
    # Update title
    for cell in new_nb["cells"]:
        if cell["cell_type"] == "markdown":
            src = "".join(cell["source"])
            if "0.9586" in src:
                src = src.replace("0.9586", f"0.9586+our_{our_weight}")
                cell["source"] = src.splitlines(keepends=True)
                break
    
    out_path = OUT_DIR / f"final_blend_our_{our_weight}.ipynb"
    with open(out_path, "w") as f:
        json.dump(new_nb, f, indent=1)
    print(f"Created: {out_path}")

if __name__ == "__main__":
    OUT_DIR = Path("/home/codemaster29/Documents/still_learning/rsna-knee-abnormality-detection/final_submission/notebooks")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Copy heliosli notebook as base
    shutil.copy(SRC_HELIOSLI, OUT_DIR / "heliosli_09586_base.ipynb")
    
    # Create safe version (our weight 0.20)
    create_blend_notebook(0.20, "safe")
    
    # Create bold version (our weight 0.35)
    create_blend_notebook(0.35, "bold")
    
    print("Done. Notebooks in:", OUT_DIR)
