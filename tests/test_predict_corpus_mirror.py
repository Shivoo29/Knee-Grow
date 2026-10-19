import json
import os
import numpy as np
import train_knee as tk
from tools.predict_corpus import MirroredVolsWrapper, load_study_ids


def test_mirrored_vols_wrapper():
    np.random.seed(42)
    vols = np.random.randint(0, 255, (4, 96, 64, 64), dtype=np.uint8)
    masks = np.ones((4, 96), dtype=np.uint8)
    
    mirror_rows = {1, 3}
    wrapper = MirroredVolsWrapper(vols, masks, mirror_rows)
    
    assert wrapper.shape == vols.shape
    assert len(wrapper) == 4
    
    # Row 0 (not mirrored) should match exactly
    assert np.array_equal(wrapper[0], vols[0])
    # Row 1 (mirrored)
    m1 = tk.mirror_volume(vols[1], masks[1])
    assert np.array_equal(wrapper[1], m1)
    # Row 2 (not mirrored)
    assert np.array_equal(wrapper[2], vols[2])
    # Row 3 (mirrored)
    m3 = tk.mirror_volume(vols[3], masks[3])
    assert np.array_equal(wrapper[3], m3)


def test_load_study_ids(tmp_path):
    # Test json with folds
    fold_file = tmp_path / "folds.json"
    data = {"folds": {"s1": 0, "s2": 1, "s3": 0, "s4": 2}}
    fold_file.write_text(json.dumps(data))
    
    f0 = load_study_ids(f"{fold_file}:0")
    assert sorted(f0) == ["s1", "s3"]
    
    f1 = load_study_ids(f"{fold_file}:1")
    assert f1 == ["s2"]
    
    # Test plain list
    txt_file = tmp_path / "studies.txt"
    txt_file.write_text("s1\ns2\n# comment\ns5\n")
    loaded = load_study_ids(str(txt_file))
    assert loaded == ["s1", "s2", "s5"]
