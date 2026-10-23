#!/usr/bin/env python3
"""tests/test_series_selection.py - series-count robustness of the 96-slot selection.

Metadata-only: reads data/train_series.csv with pinned columns (the Report column
lives in train.csv and is never opened). No DICOM, no pixel data, no writes, no network.

Runs the REAL selection function (rsna_preproc.pick_series_for_slot) over every train
study with the caller loop mirrored from build_corpus96_stream.py:131-152 and
sovereign_fast_dual_fleet.py:138-148, plus the corpus-builder clone for divergence.

Run:  env -u PYTHONPATH -u PYTHONHOME .venv/bin/python tests/test_series_selection.py
"""
import importlib
import random
import sys
import types
from collections import Counter
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The selection functions read dict rows only; cv2/pydicom are unused locally and
# may be missing. Stub them so the repo modules import, but make any real use fail loudly.
for _name, _attrs in (("cv2", ("resize",)),
                      ("pydicom", ("dcmread",)),
                      ("pydicom.pixel_data_handlers.util", ("apply_modality_lut",))):
    try:
        importlib.import_module(_name)
    except ImportError:
        parts = _name.split(".")
        for i in range(len(parts) - 1):
            sys.modules.setdefault(".".join(parts[:i + 1]), types.ModuleType(".".join(parts[:i + 1])))
        stub = types.ModuleType(_name)
        for attr in _attrs:
            setattr(stub, attr, lambda *a, _attr=attr, **k: (_ for _ in ()).throw(
                RuntimeError(f"{_name}.{_attr} called in metadata-only test")))
        sys.modules[_name] = stub

from rsna_preproc import SLOTS, pick_series_for_slot  # noqa: E402
from build_corpus96_stream import _pick_series_for_slot as pick_corpus  # noqa: E402

SERIES_CSV = ROOT / "data" / "train_series.csv"
SAFE_COLS = ["StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive",
             "Fat_Suppression", "Anatomical_Plane"]
SLOT_NAMES = ["Sag-FS", "Sag-nonFS", "Cor-FS", "Cor-nonFS", "Ax"]


def fluid_of(row):
    # NB: deliberately NOT `int(float(v or -1))` — that idiom maps a legit 0 to -1,
    # which is exactly the bug under test in rsna_preproc.py:120.
    try:
        f = float(row.get("Fluid_Sensitive", None))
    except (TypeError, ValueError):
        return None
    return None if f != f else int(f)  # NaN-safe


def load_studies():
    df = pd.read_csv(SERIES_CSV, usecols=SAFE_COLS)
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    df["SeriesInstanceUID"] = df["SeriesInstanceUID"].astype(str)
    print(f"[meta] {len(df)} series rows / {df.StudyInstanceUID.nunique()} studies")
    print(f"[meta] Fluid_Sensitive NaN: {int(df.Fluid_Sensitive.isna().sum())}, "
          f"plane values: {sorted(df.Anatomical_Plane.astype(str).unique())}")
    studies = {sid: g.to_dict("records")
               for sid, g in df.groupby("StudyInstanceUID", sort=False)}  # CSV row order kept
    return studies


def run_selection(rows, picker=pick_series_for_slot):
    """Caller loop: slots in fixed SLOTS order, chosen series removed from pool."""
    used, chosen = set(), []
    for plane, fluid, _k in SLOTS:
        r = picker(rows, plane, fluid, used)
        if r is None:
            chosen.append(None)
        else:
            uid = str(r["SeriesInstanceUID"])
            used.add(uid)
            chosen.append((uid, fluid_of(r)))
    return chosen


def uid_key(chosen):
    return tuple(c[0] if c else None for c in chosen)


def fmt(chosen):
    parts = []
    for name, c in zip(SLOT_NAMES, chosen):
        parts.append(f"{name}={c[0][-6:]}(fs{c[1]})" if c else f"{name}=EMPTY")
    return " ".join(parts)


def probe(picker, rows, plane, fluid, label):
    try:
        r = picker(list(rows), plane, fluid, set())
        print(f"  [probe] {label}: chosen={r['SeriesInstanceUID'][-6:] if r else None}")
    except Exception as e:
        print(f"  [probe] {label}: RAISED {type(e).__name__}: {e}")


def main():
    studies = load_studies()

    # 1) Baseline selection on every study, CSV row order.
    base = {sid: run_selection(rows) for sid, rows in studies.items()}

    # 2) Empty slot groups, per group + wrong-contrast fallback fills.
    empty = [0] * 5
    fallback = [0] * 5
    all_empty = any_empty = 0
    empty_hist = Counter()
    for chosen in base.values():
        n_empty = sum(c is None for c in chosen)
        empty_hist[n_empty] += 1
        all_empty += n_empty == 5
        any_empty += n_empty > 0
        for i, c in enumerate(chosen):
            if c is None:
                empty[i] += 1
            elif SLOTS[i][1] in (0, 1) and c[1] != SLOTS[i][1]:
                fallback[i] += 1
    print("\n[section 2] empty slot groups (study picked no series for the slot)")
    for i, name in enumerate(SLOT_NAMES):
        print(f"  {name:9s}: {empty[i]:5d} studies empty "
              f"({100.0 * empty[i] / len(base):.2f}%) | filled with WRONG fluid contrast: {fallback[i]}")
    print(f"  studies with >=1 empty slot group: {any_empty} | all 5 groups empty: {all_empty}")
    print(f"  empty-groups-per-study histogram: {dict(sorted(empty_hist.items()))}")

    # 3) Order non-determinism: two independent shuffles per study, compared to each
    #    other and to the CSV-order result.
    s_vs_s = s_vs_base = 0
    for sid, rows in studies.items():
        a = list(rows); random.Random(f"{sid}|s1").shuffle(a)
        b = list(rows); random.Random(f"{sid}|s2").shuffle(b)
        ua, ub = uid_key(run_selection(a)), uid_key(run_selection(b))
        s_vs_s += ua != ub
        s_vs_base += (ua != uid_key(base[sid])) or (ub != uid_key(base[sid]))
    print("\n[section 3] selection vs series row order")
    print(f"  studies where two shuffled runs disagree:            {s_vs_s}")
    print(f"  studies where any shuffled run != CSV-order pick:    {s_vs_base}")

    # 4) Corpus-builder clone vs canonical picker (CSV order), with a breakdown of
    #    where and how they disagree.
    diff_slots = Counter()
    diff_examples = []
    clone_diff = 0
    for sid, rows in studies.items():
        a, b = base[sid], run_selection(rows, pick_corpus)
        if uid_key(a) == uid_key(b):
            continue
        clone_diff += 1
        for i, (ca, cb) in enumerate(zip(a, b)):
            if (ca[0] if ca else None) != (cb[0] if cb else None):
                diff_slots[i] += 1
                if len(diff_examples) < 5:
                    diff_examples.append(
                        f"study={sid[-6:]} slot={SLOT_NAMES[i]}(fluid={SLOTS[i][1]}) "
                        f"preproc={ca[0][-6:]}(fs{ca[1]}) corpus={cb[0][-6:]}(fs{cb[1]})")
    print(f"\n[section 4] build_corpus96_stream._pick_series_for_slot vs rsna_preproc "
          f"on clean CSV data: {clone_diff} studies differ")
    print(f"  differing picks by slot: "
          f"{ {SLOT_NAMES[i]: n for i, n in diff_slots.items()} }")
    for line in diff_examples:
        print(f"  e.g. {line}")

    # 5) Dirty-value probes (synthetic rows, in-memory only).
    print("\n[section 5] dirty-value probes (synthetic rows)")
    nan_row = {"SeriesInstanceUID": "1.2.826.NANTEST", "Anatomical_Plane": "Sagittal",
               "Fluid_Sensitive": float("nan")}
    probe(pick_series_for_slot, [nan_row], "Sagittal", 1, "NaN fluid, slot wants FS (preproc)")
    probe(pick_corpus, [nan_row], "Sagittal", 1, "NaN fluid, slot wants FS (corpus clone)")
    probe(pick_series_for_slot, [nan_row], "Sagittal", -1, "NaN fluid, Ax slot (preproc)")
    empty_row = {"SeriesInstanceUID": "1.2.826.EMPTY01", "Anatomical_Plane": "Sagittal",
                 "Fluid_Sensitive": ""}
    fs0_row = {"SeriesInstanceUID": "1.2.826.FS0REAL", "Anatomical_Plane": "Sagittal",
               "Fluid_Sensitive": 0}
    fs1_row = {"SeriesInstanceUID": "1.2.826.FS1REAL", "Anatomical_Plane": "Sagittal",
               "Fluid_Sensitive": 1}
    probe(pick_series_for_slot, [fs1_row, fs0_row], "Sagittal", 0,
          "slot wants fluid=0, rows=[FS,nonFS] (preproc)")
    probe(pick_corpus, [fs1_row, fs0_row], "Sagittal", 0,
          "slot wants fluid=0, rows=[FS,nonFS] (corpus clone)")
    probe(pick_series_for_slot, [empty_row, fs0_row], "Sagittal", 0, "'' fluid vs real 0 (preproc)")
    probe(pick_corpus, [empty_row, fs0_row], "Sagittal", 0, "'' fluid vs real 0 (corpus clone)")
    lower_row = {"SeriesInstanceUID": "1.2.826.LOWRPLN", "Anatomical_Plane": "sagittal",
                 "Fluid_Sensitive": 1}
    probe(pick_series_for_slot, [lower_row], "Sagittal", 1, "lowercase plane (preproc)")
    probe(pick_corpus, [lower_row], "Sagittal", 1, "lowercase plane (corpus clone)")

    # 6) Extreme study sizes: 3-series and >=10-series, chosen UIDs tail-6 only.
    print("\n[section 6] chosen series for 3-series studies")
    for sid, rows in sorted(studies.items(), key=lambda kv: len(kv[1])):
        if len(rows) != 3:
            continue
        print(f"  study={sid[-6:]} n={len(rows)} | {fmt(base[sid])}")
    print("[section 6] chosen series for >=10-series studies (n, study tail-6)")
    n_big = 0
    for sid, rows in sorted(studies.items(), key=lambda kv: len(kv[1])):
        if len(rows) < 10:
            continue
        n_big += 1
        print(f"  study={sid[-6:]} n={len(rows)} | {fmt(base[sid])}")
    print(f"  total >=10-series studies: {n_big}")

    print("\n[summary] studies=%d | empty-groups Sag-FS/Sag-nonFS/Cor-FS/Cor-nonFS/Ax = %s"
          % (len(base), empty))
    print(f"[summary] any-empty={any_empty} all-empty={all_empty} "
          f"shuffle-vs-shuffle={s_vs_s} shuffle-vs-csv={s_vs_base} clone-diff={clone_diff}")


def test_series_selection():
    main()


if __name__ == "__main__":
    main()
