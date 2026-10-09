"""[1] label_llm.py: read each radiology report with local Qwen, emit 12 finding probabilities.

Needs `qwen start` (llama.cpp server on localhost:8080). Reports never leave this machine.
Resumable: answers are appended to a JSONL cache, so a crash or Ctrl+C loses nothing.
Prints only AUC numbers vs the 58 gold studies, never report text.

    .venv/bin/python label_llm.py            # label all 4,407 studies (~3-4 h)
    .venv/bin/python label_llm.py --limit 40 # quick trial
"""
import argparse, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np, pandas as pd, requests
from sklearn.metrics import roc_auc_score

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
          'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

# Definitions follow the gold annotation style: severity-thresholded, borderline graded negative.
SYSTEM = """You are a musculoskeletal radiologist. Read one knee MRI report (any language) and
estimate, for each finding, the probability (0.0-1.0) that an expert would label it PRESENT.
Borderline, mild or equivocal findings count as mostly absent. Not mentioned = absent (low).

ACL: ACL tear, partial or complete, including torn reconstruction graft. Intact graft = absent.
MCL: MCL tear or high-grade sprain. Mild periligamentous edema / grade 1 sprain = mostly absent.
Medial Meniscus: tear of the medial meniscus. Degeneration or intrasubstance signal alone = absent.
Lateral Meniscus: tear of the lateral meniscus. Same rule.
Medial OA: moderate-to-severe osteoarthritis / cartilage loss of the medial tibiofemoral compartment.
Lateral OA: same, lateral tibiofemoral compartment.
PF OA: same, patellofemoral compartment (patella or trochlea).
Effusion: moderate or large joint effusion. Trace/small/physiologic = mostly absent.
Synovitis: synovitis or synovial thickening/proliferation (incl. Hoffa synovitis).
Baker's: popliteal (Baker's) cyst of any size.
Contusion: bone contusion / bone bruise / traumatic bone marrow edema.
Fracture: any fracture (avulsion, osteochondral, insufficiency, plateau, etc.).

Answer only with the JSON object."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": LABELS,
          "properties": {l: {"type": "number", "minimum": 0, "maximum": 1} for l in LABELS}}


def ask(report, url):
    r = requests.post(url, timeout=300, json={
        "model": "qwen", "temperature": 0, "max_tokens": 300,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": report}],
        "chat_template_kwargs": {"enable_thinking": False},  # answer directly, no hidden reasoning
        "response_format": {"type": "json_schema", "json_schema": {"name": "labels", "schema": SCHEMA}},
    })
    r.raise_for_status()
    out = json.loads(r.json()["choices"][0]["message"]["content"])
    return [float(np.clip(out[l], 0, 1)) for l in LABELS]


def gold_auc(pred, train):
    """Macro AUC of a label table vs the 58 gold studies."""
    g = train.dropna(subset=['ACL']).set_index('StudyInstanceUID')[LABELS]
    p = pred.reindex(g.index)[LABELS]
    ok = p.notna().all(axis=1)
    aucs = {l: roc_auc_score(g.loc[ok, l], p.loc[ok, l]) for l in LABELS if g.loc[ok, l].nunique() > 1}
    return ok.sum(), float(np.mean(list(aucs.values()))), aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/train.csv')
    ap.add_argument('--cache', default='data/llm_qwen.jsonl')
    ap.add_argument('--out', default='data/llm_qwen.parquet')
    ap.add_argument('--url', default='http://127.0.0.1:8080/v1/chat/completions')
    ap.add_argument('--workers', type=int, default=4)  # = llama-server slots
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    train = pd.read_csv(a.train)
    done = set()
    if os.path.exists(a.cache):
        done = {json.loads(l)['StudyInstanceUID'] for l in open(a.cache)}
    # gold studies first, so a partial run can already be scored
    todo = train.assign(g=train.ACL.isna()).sort_values('g')
    todo = todo[~todo.StudyInstanceUID.isin(done)]
    if a.limit:
        todo = todo.head(a.limit)
    print(f'{len(done)} cached, {len(todo)} to label', flush=True)

    with open(a.cache, 'a') as f, ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(ask, str(r.Report), a.url): r.StudyInstanceUID for r in todo.itertuples()}
        for i, fu in enumerate(as_completed(futs), 1):
            uid = futs[fu]
            try:
                f.write(json.dumps({'StudyInstanceUID': uid, 'p': fu.result()}) + '\n'); f.flush()
            except Exception as e:  # one bad report must not kill a 4-hour run; rerun picks it up
                print(f'fail {uid[-8:]}: {type(e).__name__}', flush=True)
            if i % 100 == 0:
                print(f'{i}/{len(todo)}', flush=True)

    rows = [json.loads(l) for l in open(a.cache)]
    pred = pd.DataFrame([r['p'] for r in rows], columns=LABELS,
                        index=pd.Index([r['StudyInstanceUID'] for r in rows], name='StudyInstanceUID'))
    pred = pred[~pred.index.duplicated()]
    pred.reset_index().to_parquet(a.out, index=False)
    n, macro, aucs = gold_auc(pred, train)
    print(f'saved {len(pred)} -> {a.out} | gold n={n} macro-AUC {macro:.4f} (v4_blend = 0.8927)')
    print(' '.join(f'{l[:8]}={v:.2f}' for l, v in aucs.items()))


if __name__ == '__main__':
    main()
