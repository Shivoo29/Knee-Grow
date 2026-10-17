#!/usr/bin/env python3
"""Second report labeler: a local Gemma (any OpenAI-compatible server) labels every train.csv report,
learning how report wording maps to expert labels from the 58 expert-labelled reports given as examples.
Runs on the PC that hosts Gemma. Python 3 standard library only; prints counts only, never report text.

Gold rows are labelled 2-fold (examples from the other half only), so gold-58 can still score this label
source honestly; the other 4,349 rows see all 58 examples. Gold rows run first, so an early partial file
is already enough for that check.

    python3 gemma_labels.py --train train.csv --model gemma4:27b --base_url http://localhost:11434/v1 \
        --out gemma_labels.csv            # add --limit 70 for a quick test (58 gold + 12 others)

The server context must hold the examples (about 25k tokens): Ollama `OLLAMA_CONTEXT_LENGTH=32768`,
LM Studio context length 32768, llama.cpp `-c 32768`. Rerun the same command to resume.

Output gemma_labels.csv, one row per report, columns: StudyInstanceUID, the 12 findings (probability 0-1
that the expert readers marked it positive), shots (A or B = which gold half was shown, for gold rows;
all = the other rows), ok (1 parsed, 0 not parsed: findings left empty).
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import time
import urllib.request

LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
       'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
RUBRIC = """You label knee MRI radiology reports. Each study was also read on the images by expert radiologists \
(two readers, a third adjudicating), who marked 12 findings positive with this rubric, grading borderline \
findings negative to favour specificity:
- ACL: high-grade partial tear (more than half of the fibres) or complete tear.
- MCL: acute high-grade partial or complete tear with fibre disruption and edema. Low-grade or chronic change is negative.
- Medial Meniscus, Lateral Meniscus: signal that definitely reaches the surface on at least two images, or a \
morphologic abnormality. Intrasubstance degeneration is negative.
- Medial OA, Lateral OA, PF OA: a moderate or large area (about 1 cm or more) of high-grade cartilage loss \
(more than half the thickness) in that compartment: medial tibiofemoral, lateral tibiofemoral, patellofemoral \
(patella and trochlea). Tricompartmental osteoarthritis concerns all three.
- Effusion: moderate or large joint effusion.
- Synovitis: inflammation and thickening of the synovial lining.
- Baker's: moderate or large popliteal cyst.
- Contusion: edema-like bone marrow signal without a fracture line.
- Fracture: acute cortical break or fracture line.
The experts read the images, not the report, so the two can disagree: reports often leave out synovitis, \
effusion size and subtle fractures, and they mention mild changes the rubric counts as negative. The worked \
examples below are real reports with the experts' labels (1 positive, 0 negative). Study them to learn how \
report wording maps to the expert labels, including the cases where the report is silent.
For the new report, estimate for each finding the probability, as an integer from 0 to 100, that the experts \
marked it positive. Reply with one JSON object only, using exactly these keys: """ + json.dumps(LAB)


def read_rows(path):
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_gold(row):
    return all((row.get(k) or "").strip() != "" for k in LAB)


def system_prompt(examples):
    parts = [RUBRIC]
    for i, r in enumerate(examples, 1):
        labels = {k: int(float(r[k])) for k in LAB}
        parts.append(f"\n\nExample {i}\nReport:\n{r['Report'].strip()}\nExpert labels: {json.dumps(labels)}")
    return "".join(parts)


def plan(rows, seed=0):
    """(row, shots) jobs, gold first. A gold row only ever sees examples from the other gold half."""
    rows = [r for r in rows if (r.get("Report") or "").strip()]
    gold = sorted((r for r in rows if is_gold(r)), key=lambda r: r["StudyInstanceUID"])
    random.Random(seed).shuffle(gold)
    half = {"A": gold[: len(gold) // 2], "B": gold[len(gold) // 2:]}
    systems = {"A": system_prompt(half["A"]), "B": system_prompt(half["B"]), "all": system_prompt(gold)}
    jobs = [(r, "B") for r in half["A"]] + [(r, "A") for r in half["B"]]
    jobs += [(r, "all") for r in rows if not is_gold(r)]
    return jobs, systems


def ask(base_url, model, system, report, timeout=600):
    body = {"model": model, "temperature": 0, "max_tokens": 1024,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": "New report:\n" + report.strip()}]}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"] or ""


def parse(text):
    """Last JSON object with all 12 keys -> probabilities 0-1 (accepts 0-100 or 0-1 answers), else None."""
    for blob in reversed(re.findall(r"\{[^{}]*\}", text)):
        try:
            d = json.loads(blob)
            v = [float(d[k]) for k in LAB]
        except (ValueError, KeyError, TypeError):
            continue
        scale = 1.0 if max(v) <= 1.0 and any(x % 1 for x in v) else 100.0
        return [min(max(x / scale, 0.0), 1.0) for x in v]
    return None


def main(argv=None, ask_fn=ask):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base_url", default="http://localhost:11434/v1")
    ap.add_argument("--out", default="gemma_labels.csv")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    jobs, systems = plan(read_rows(a.train))
    done = {r["StudyInstanceUID"] for r in read_rows(a.out)} if os.path.exists(a.out) else set()
    todo = [(r, s) for r, s in jobs if r["StudyInstanceUID"] not in done][: a.limit or None]
    print(f"{len(jobs)} reports, {len(done)} already labelled, {len(todo)} to do", flush=True)
    new = not os.path.exists(a.out)
    t0, n_ok = time.time(), 0
    with open(a.out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["StudyInstanceUID"] + LAB + ["shots", "ok"])
        for i, (r, shots) in enumerate(todo, 1):
            probs = None
            for attempt in range(3):
                try:
                    probs = parse(ask_fn(a.base_url, a.model, systems[shots],
                                         r["Report"] + ("" if attempt == 0 else "\n\nReply with the JSON object only.")))
                except OSError as e:  # server down or timeout: retry, then stop (rerun resumes)
                    if attempt == 2:
                        sys.exit(f"server error after retries ({type(e).__name__}); rerun to resume")
                    time.sleep(10)
                    continue
                if probs is not None:
                    break
            n_ok += probs is not None
            w.writerow([r["StudyInstanceUID"]] + (["%.3f" % p for p in probs] if probs else [""] * len(LAB))
                       + [shots, int(probs is not None)])
            f.flush()
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"{i}/{len(todo)} | parsed {n_ok}/{i} | {el / i:.1f} s/report | "
                      f"ETA {(len(todo) - i) * el / i / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
