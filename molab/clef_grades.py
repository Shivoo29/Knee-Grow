# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "torch",
#     "torchvision",
#     "transformers==5.10.2",
#     "accelerate",
#     "huggingface_hub",
#     "safetensors",
#     "pandas",
#     "pyarrow",
#     "pillow",
# ]
# ///
"""Task 9 on molab: grade every finding of every train report with Clef (Cloudflare/clef, Apache-2.0).

One forward pass per report returns a probability for every option of all 12 questions (Clef's joint schema
head; no generated text). The notebook shows only timings and option counts, never report text or study ids;
the grades parquet goes to the browser through a lazy download, so it is not embedded in the notebook page.
Three plain cells, so they can be pasted into a new molab notebook (GPU on): upload train.csv, press 1, then 2.
"""
import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def controls():
    import importlib.util
    import io
    import shutil
    import sys
    import tempfile
    import time
    from pathlib import Path

    import marimo as mo

    PREAMBLE = "Knee MRI radiology report:\n"
    TOKENS_PER_BATCH = 32768  # padded tokens per forward pass; an out-of-memory batch is split in half
    MAX_BATCH = 16

    def _q(what, criteria, note=""):
        return {"type": "choice", "criteria": criteria,
                "instructions": f"Based only on this knee MRI report, which option best describes {what}?{note}"}

    def _meniscus(side):
        return _q(f"the {side} meniscus", {
            "not_mentioned": f"The report does not mention the {side} meniscus.",
            "normal": f"The {side} meniscus is described as intact or normal.",
            "degeneration_only": "Intrasubstance degeneration or signal that does not reach an articular surface.",
            "possible_tear": "Possible, suspected, or indeterminate tear.",
            "definite_tear": ("Definite tear: signal reaching an articular surface, or a named tear such as horizontal, "
                              "oblique, radial, vertical, complex, flap, bucket-handle, or root tear."),
            "morphologic_abnormality": ("Macerated, truncated, blunted, or displaced meniscal tissue, or marked "
                                        "extrusion, without a separately described tear."),
            "post_surgical": "Postoperative meniscus (meniscectomy or repair) without a new tear described.",
        })

    def _oa(compartment):
        return _q(f"the cartilage and osteoarthritis of the {compartment}", {
            "not_mentioned": "The report does not describe the cartilage or osteoarthritis of this compartment.",
            "normal": "The cartilage of this compartment is described as normal or preserved.",
            "low_grade": ("Only mild or low-grade change: chondromalacia, fibrillation, signal change, mild thinning, "
                          "partial-thickness loss under half the thickness, or mild osteoarthritis."),
            "high_grade_small": ("High-grade cartilage loss (over half the thickness, or full thickness) limited to a "
                                 "small focal area, under about 1 cm."),
            "high_grade_large": ("High-grade cartilage loss (over half the thickness, or full thickness) over a moderate "
                                 "or large area, about 1 cm or more, or moderate, severe, or advanced osteoarthritis "
                                 "of this compartment."),
            "unspecified": "Cartilage loss or osteoarthritis of this compartment is mentioned without depth or extent.",
        }, " Tricompartmental or diffuse osteoarthritis applies to every compartment.")

    # Options follow the host rubric (discussion 733343) and keep borderline and silent cases apart,
    # so the grade -> label mapping can be chosen afterwards without another GPU pass.
    QUESTIONS = {
        "ACL": _q("the anterior cruciate ligament (ACL)", {
            "not_mentioned": "The report does not mention the ACL.",
            "intact": "The ACL is described as intact or normal.",
            "sprain_or_low_grade": "ACL sprain, mucoid degeneration, or a low-grade partial tear (under half of the fibers).",
            "high_grade_partial": "High-grade partial ACL tear (over half of the fibers torn).",
            "complete_tear": "Complete ACL tear or rupture.",
            "possible_tear": "Possible, suspected, or indeterminate ACL tear.",
            "graft_intact": "ACL reconstruction with an intact graft.",
            "graft_torn": "ACL reconstruction with a torn or failed graft.",
        }),
        "MCL": _q("the medial collateral ligament (MCL)", {
            "not_mentioned": "The report does not mention the MCL.",
            "intact": "The MCL is described as intact or normal.",
            "low_grade_or_chronic": ("Low-grade MCL sprain, periligamentous edema, thickening, or a chronic, old, "
                                     "or healed MCL injury."),
            "moderate_partial": "Moderate (grade 2) MCL sprain or partial tear not described as high grade.",
            "acute_high_grade_partial": "Acute high-grade partial MCL tear with fiber disruption.",
            "acute_complete_tear": "Acute complete MCL tear.",
            "possible_tear": "Possible or indeterminate MCL tear.",
        }),
        "Medial Meniscus": _meniscus("medial"),
        "Lateral Meniscus": _meniscus("lateral"),
        "Medial OA": _oa("medial tibiofemoral compartment (medial femoral condyle and medial tibial plateau)"),
        "Lateral OA": _oa("lateral tibiofemoral compartment (lateral femoral condyle and lateral tibial plateau)"),
        "PF OA": _oa("patellofemoral compartment (patella and trochlea)"),
        "Effusion": _q("the knee joint effusion", {
            "not_mentioned": "The report does not mention joint fluid or effusion.",
            "none_or_physiologic": "No effusion, or only physiologic fluid.",
            "trace_or_small": "Trace, minimal, or small joint effusion.",
            "moderate": "Moderate joint effusion.",
            "large": "Large joint effusion, including a large hemarthrosis or lipohemarthrosis.",
            "unspecified": "Joint effusion, hemarthrosis, or lipohemarthrosis without a stated size.",
        }),
        "Synovitis": _q("synovitis", {
            "not_mentioned": "The report does not mention synovitis or the synovium.",
            "none": "The report states there is no synovitis.",
            "possible_or_mild": "Mild, possible, or reactive synovitis, or mild synovial thickening.",
            "definite": "Synovitis, synovial thickening, or synovial proliferation is described.",
        }),
        "Baker's": _q("a popliteal (Baker's) cyst", {
            "not_mentioned": "The report does not mention a popliteal (Baker's) cyst.",
            "none": "The report states there is no Baker's cyst.",
            "small": "Small or tiny Baker's cyst.",
            "moderate_or_large": "Moderate or large Baker's cyst.",
            "ruptured": "Ruptured or leaking Baker's cyst.",
            "unspecified": "Baker's cyst present without a stated size.",
        }),
        "Contusion": _q("bone marrow edema or bone contusion", {
            "not_mentioned": "The report does not mention bone marrow edema or contusion.",
            "none": "The report states there is no bone marrow edema or contusion.",
            "traumatic_contusion": ("Bone contusion, bone bruise, or traumatic marrow edema (for example a pivot-shift "
                                    "pattern) without a fracture line."),
            "degenerative_edema": "Subchondral marrow edema related to osteoarthritis, cartilage loss, or degeneration.",
            "other_edema": "Stress reaction, transient, reactive, or nonspecific marrow edema.",
            "edema_with_fracture": "Marrow edema only around a fracture.",
        }),
        "Fracture": _q("a fracture", {
            "not_mentioned": "The report does not mention a fracture.",
            "none": "The report states there is no fracture.",
            "acute_fracture": ("Acute fracture with a cortical break or fracture line, including avulsion, impaction, "
                               "osteochondral, tibial plateau, or Segond fracture."),
            "insufficiency_or_stress": "Subchondral insufficiency fracture or stress fracture.",
            "old_or_healed": "Old, healed, healing, or chronic fracture, or fracture fixation hardware.",
            "possible": "Possible, suspected, or indeterminate fracture.",
        }),
    }

    upload = mo.ui.file(filetypes=[".csv"], label="Upload train.csv (only StudyInstanceUID and Report are read)")
    grader = mo.ui.dropdown(["Cloudflare/clef", "Cloudflare/clef-flash"], value="Cloudflare/clef", label="Grader")
    load = mo.ui.run_button(label="1. Load grader")
    smoke = mo.ui.checkbox(value=True, label="Smoke test: first 32 reports only")
    grade = mo.ui.run_button(label="2. Grade reports")
    mo.vstack([upload, grader, load, smoke, grade])
    return (MAX_BATCH, PREAMBLE, Path, QUESTIONS, TOKENS_PER_BATCH, grade, grader, importlib, io, load, mo,
            shutil, smoke, sys, tempfile, time, upload)


@app.cell
def load_grader(Path, grader, importlib, load, mo, shutil, sys, tempfile, time):
    mo.stop(not load.value, mo.md("Press **1. Load grader** (downloads 56 GB for clef, 19 GB for clef-flash)."))
    import accelerate  # noqa: F401  (imported here so a missing package shows up before the download)
    import torch
    import torchvision  # noqa: F401  (the Qwen image processor needs it even for text-only records)
    import transformers  # noqa: F401

    mo.stop(not torch.cuda.is_available(), mo.md("No GPU: turn the GPU on in the notebook specs, then restart."))
    _need = 60 if grader.value == "Cloudflare/clef" else 22
    _root = max([Path(tempfile.gettempdir()), Path.cwd()], key=lambda p: shutil.disk_usage(p).free)
    _free = shutil.disk_usage(_root).free / 1e9
    mo.stop(_free < _need, mo.md(f"Only {_free:.0f} GB free at `{_root}`; {grader.value} needs about {_need} GB."))
    from huggingface_hub import snapshot_download

    _t0 = time.time()
    _path = Path(snapshot_download(grader.value, cache_dir=str(_root / "hf"), max_workers=16))
    _t_dl = time.time() - _t0
    _spec = importlib.util.spec_from_file_location("jsm_" + grader.value.split("/")[-1].replace("-", "_"),
                                                   _path / "joint_schema_model.py")
    jsm = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = jsm  # its dataclasses look the module up while it executes
    _spec.loader.exec_module(jsm)
    model, processor = jsm.load_release_model(_path, device="cuda")
    mo.md(f"Loaded **{grader.value}** on {torch.cuda.get_device_name()} | download {_t_dl / 60:.1f} min | "
          f"load {(time.time() - _t0 - _t_dl) / 60:.1f} min | weights {torch.cuda.memory_allocated() / 1e9:.0f} GB "
          f"| {_free:.0f} GB disk was free at `{_root}` | transformers {transformers.__version__}")
    return jsm, model, processor, torch


@app.cell
def grade_reports(MAX_BATCH, PREAMBLE, Path, QUESTIONS, TOKENS_PER_BATCH, grade, grader, io, jsm, mo, model,
                  processor, smoke, tempfile, time, torch, upload):
    mo.stop(not grade.value, mo.md("Press **2. Grade reports** (smoke test first)."))
    mo.stop(upload.contents() is None, mo.md("Upload train.csv first."))
    import pandas as pd

    _t0 = time.time()
    _df = pd.read_csv(io.BytesIO(upload.contents()), usecols=["StudyInstanceUID", "Report"], dtype=str)
    _n_all = len(_df)
    _df = _df[_df["Report"].fillna("").str.strip() != ""]
    _n_reports = len(_df)
    if smoke.value:
        _df = _df.head(32)
    _tok = processor.tokenizer
    _enc = [jsm.encode_record(_tok, {"state": PREAMBLE + r, "questions": QUESTIONS}) for r in _df["Report"]]
    _order = sorted(range(len(_enc)), key=lambda i: len(_enc[i].input_ids))
    _batches, _cur = [], []
    for _i in _order:  # ascending length, so the newest record sets the padded length
        if _cur and ((len(_cur) + 1) * len(_enc[_i].input_ids) > TOKENS_PER_BATCH or len(_cur) == MAX_BATCH):
            _batches.append(_cur)
            _cur = []
        _cur.append(_i)
    _batches.append(_cur)
    _dev = next(model.parameters()).device

    def _run(idx):
        try:
            _b = jsm.collate_records([_enc[i] for i in idx], _tok.pad_token_id, _dev)
            with torch.inference_mode():
                return list(zip(idx, model(_b)))
        except torch.cuda.OutOfMemoryError:
            if len(idx) == 1:
                raise
            torch.cuda.empty_cache()
            return _run(idx[: len(idx) // 2]) + _run(idx[len(idx) // 2:])

    _rows = [None] * len(_enc)
    for _b in mo.status.progress_bar(_batches, title=f"Grading {len(_enc)} reports", show_eta=True, show_rate=True):
        for _i, _logits in _run(_b):
            _row = {}
            for _q, _ql in zip(_enc[_i].questions, _logits):
                _row.update(zip([f"{_q.question_id}|{o}" for o in _q.option_ids], _ql.float().softmax(-1).tolist()))
            _rows[_i] = _row
    grades = pd.DataFrame(_rows).astype("float32")
    grades.insert(0, "StudyInstanceUID", _df["StudyInstanceUID"].values)
    grades["n_tokens"] = [len(e.input_ids) for e in _enc]
    _secs = time.time() - _t0
    out_path = Path(tempfile.gettempdir()) / (f"grades_{grader.value.split('/')[-1]}"
                                              + ("_smoke" if smoke.value else "") + ".parquet")
    grades.to_parquet(out_path, index=False)
    _lines = [f"**{len(grades)} reports graded in {_secs / 60:.1f} min** ({_secs / len(grades):.2f} s per report; "
              f"full run of {_n_reports} about {_secs / len(grades) * _n_reports / 60:.0f} min) | "
              f"{_n_all - _n_reports} rows without a report skipped | tokens per report: mean "
              f"{grades['n_tokens'].mean():.0f}, max {grades['n_tokens'].max()}", "", "Most likely option, count of reports:"]
    for _f in QUESTIONS:
        _cols = [c for c in grades.columns if c.startswith(_f + "|")]
        _top = grades[_cols].idxmax(axis=1).str.split("|", n=1).str[1].value_counts()
        _lines.append(f"- **{_f}**: " + ", ".join(f"{k} {v}" for k, v in _top.items()))
    mo.vstack([mo.md("\n".join(_lines)),
               mo.download(data=lambda: out_path.read_bytes(), filename=out_path.name,
                           mimetype="application/octet-stream", label=f"Download {out_path.name}")])
    return grades, out_path


if __name__ == "__main__":
    app.run()
