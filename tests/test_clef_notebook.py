#!/usr/bin/env python3
"""tests/test_clef_notebook.py - runs the molab grading cell on CPU with Clef's real tokenizer and a fake model.
    uv run --with marimo --with transformers==5.10.2 --with torch --with pandas --with pyarrow \
        python -I tests/test_clef_notebook.py
Synthetic reports only. The fake model raises out-of-memory on batches over 3 records to exercise the split.
"""
import importlib.util
import io
import os
import sys
import tempfile
import time
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WT, "molab"))
import clef_grades as nb  # noqa: E402

LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
       'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


class FakeClef(torch.nn.Module):
    def __init__(self, interrupt_at=None):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.calls = []
        self.interrupt_at = interrupt_at

    def forward(self, batch):
        recs = batch["records"]
        self.calls.append(len(recs))
        if len(self.calls) == self.interrupt_at:
            raise KeyboardInterrupt
        if len(recs) > 3:
            raise torch.cuda.OutOfMemoryError("fake")
        g = torch.Generator().manual_seed(len(recs))
        return [[torch.randn(len(q.option_ids), generator=g) for q in r.questions] for r in recs]


def _setup():
    spec = importlib.util.spec_from_file_location("jsm", hf_hub_download("Cloudflare/clef", "joint_schema_model.py"))
    jsm = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = jsm
    spec.loader.exec_module(jsm)
    tok = AutoTokenizer.from_pretrained("Cloudflare/clef")
    phrases = ["Moderate joint effusion.", "Complete ACL tear.", "Small Baker's cyst.", "Mild chondromalacia patellae."]
    reports = [" ".join(phrases[: 1 + i % 4]) * (1 + i % 5) for i in range(40)]
    for i in (3, 17, 29):
        reports[i] = "" if i != 17 else None
    csv = pd.DataFrame({"StudyInstanceUID": [f"1.2.{i}" for i in range(40)], "ACL": 0, "Report": reports})
    return jsm, tok, csv.to_csv(index=False).encode()


def _grade(smoke, tmp, model=None):
    jsm, tok, data = _setup()
    model = model or FakeClef()
    tempfile.tempdir = tmp  # the notebook writes its part files to tempfile.gettempdir()
    try:
        _, defs = nb.grade_reports.run(grade=NS(value=True), grader=NS(value="Cloudflare/clef"), jsm=jsm, model=model,
                                       processor=NS(tokenizer=tok), smoke=NS(value=smoke), torch=torch,
                                       upload=NS(contents=lambda: data))
    finally:
        tempfile.tempdir = None
    return defs["grades"], defs["out_path"], model


def test_questions():
    q = nb.controls.run()[1]["QUESTIONS"]
    assert list(q) == LAB
    assert all(v["type"] == "choice" and "not_mentioned" in v["criteria"] for v in q.values())
    return f"12 findings in label order, {sum(len(v['criteria']) for v in q.values())} options"


def test_smoke_run():
    with tempfile.TemporaryDirectory() as td:
        grades, path, model = _grade(True, td)
        assert len(grades) == 32 and max(model.calls) > 3  # every row graded although batches over 3 always fail
        for f in LAB:
            s = grades[[c for c in grades.columns if c.startswith(f + "|")]].sum(axis=1)
            assert np.allclose(s, 1.0, atol=1e-4), f
        back = pd.read_parquet(path)
        assert back.shape == grades.shape and back.columns.tolist() == grades.columns.tolist()
        assert path.name == "grades_clef_smoke.parquet"
    return f"32 rows, probabilities sum to 1 per finding, OOM split used, {grades['n_tokens'].mean():.0f} tokens/report"


def test_full_run_alignment():
    with tempfile.TemporaryDirectory() as td:
        grades, path, _ = _grade(False, td)
        want = [f"1.2.{i}" for i in range(40) if i not in (3, 17, 29)]
        assert grades["StudyInstanceUID"].tolist() == want and path.name == "grades_clef_part1.parquet"
    return "37 rows (3 empty reports skipped), ids in file order despite length-sorted batches"


def test_interrupt_and_resume():
    with tempfile.TemporaryDirectory() as td:
        g1, p1, _ = _grade(False, td, FakeClef(interrupt_at=10))  # mid batch 2 (a batch of 10 takes 7 split calls)
        g2, p2, _ = _grade(False, td)
        ids1, ids2 = set(g1["StudyInstanceUID"]), set(g2["StudyInstanceUID"])
        assert 0 < len(ids1) < 37 and not ids1 & ids2 and len(ids1 | ids2) == 37
        assert (p1.name, p2.name) == ("grades_clef_part1.parquet", "grades_clef_part2.parquet")
    return f"interrupt kept {len(ids1)} rows in part1; resume graded the other {len(ids2)} into part2, no overlap"


TESTS = [("questions", test_questions), ("smoke_run", test_smoke_run), ("full_run_alignment", test_full_run_alignment),
         ("interrupt_and_resume", test_interrupt_and_resume)]


def main():
    fails = 0
    t0 = time.time()
    for name, fn in TESTS:
        t = time.time()
        try:
            print(f"PASS {name} {fn()} ({time.time()-t:.1f}s)", flush=True)
        except Exception as e:
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e} ({time.time()-t:.1f}s)", flush=True)
    print(f"done {len(TESTS)-fails}/{len(TESTS)} pass ({time.time()-t0:.1f}s)", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
