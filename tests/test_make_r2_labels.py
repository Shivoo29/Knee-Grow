import numpy as np
import pandas as pd


def compute_alpha_cell(base_alpha, k, d, min_alpha=0.15):
    return float(np.clip(base_alpha + k * d, min_alpha, base_alpha))


def test_alpha_cell_clipping():
    base_alpha = 0.4
    k = 2.0
    # Positive delta should be capped at base_alpha
    assert compute_alpha_cell(base_alpha, k, 0.05) == 0.4
    assert compute_alpha_cell(base_alpha, k, 0.0) == 0.4
    
    # Negative delta should lower alpha
    assert round(compute_alpha_cell(base_alpha, k, -0.05), 3) == 0.300
    assert round(compute_alpha_cell(base_alpha, k, -0.10), 3) == 0.200
    
    # Large negative delta should clip to 0.15
    assert compute_alpha_cell(base_alpha, k, -0.15) == 0.15
    assert compute_alpha_cell(base_alpha, k, -0.30) == 0.15


def test_lang_mapping():
    # Languages with < 60 mapped to 'other'
    sample_langs = ["en"] * 100 + ["es"] * 70 + ["de"] * 30 + ["fr"] * 10
    s = pd.Series(sample_langs)
    counts = s.value_counts()
    mapped = s.map(lambda l: l if counts[l] >= 60 else "other")
    assert set(mapped.unique()) == {"en", "es", "other"}
    assert (mapped == "other").sum() == 40
