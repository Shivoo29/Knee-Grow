"""Label knee MRI reports with a LOCAL Ollama model + rulebook.md (rule 4.b: report text never leaves node1).
    python3 label_llm.py MODEL IDS_FILE OUT_CSV          (resumes: skips ids already in OUT_CSV)"""
import csv, json, os, sys, urllib.request
MODEL, IDS, OUT = sys.argv[1:4]
F = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion", "Fracture"]
here = os.path.dirname(os.path.abspath(__file__))
rules = open(os.path.join(here, os.environ.get("RULEBOOK", "rulebook.md"))).read()
csv.field_size_limit(10 ** 9)
rep = {r["StudyInstanceUID"]: r["Report"] for r in csv.DictReader(open(os.path.expanduser("~/projects/knee/train.csv")))}
ids = [l.strip() for l in open(IDS) if l.strip()]
done = {r["StudyInstanceUID"] for r in csv.DictReader(open(OUT))} if os.path.exists(OUT) else set()
schema = {"type": "object", "properties": {f: {"type": "number"} for f in F}, "required": F}
w = open(OUT, "a", newline=""); out = csv.writer(w)
if not done:
    out.writerow(["StudyInstanceUID", *F, "ok"])
for k, u in enumerate(i for i in ids if i not in done):
    body = {"model": MODEL, "stream": False, "format": schema, "think": False,
            "options": {"temperature": 0, "num_ctx": 16384},
            "messages": [{"role": "system", "content": rules},
                         {"role": "user", "content": "Report:\n" + rep[u] + "\n\nApply the rulebook. Return only the JSON."}]}
    try:
        req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
        d = json.loads(json.loads(urllib.request.urlopen(req, timeout=900).read())["message"]["content"])
        out.writerow([u, *[min(max(float(d[f]), 0.0), 1.0) for f in F], 1])
    except Exception as e:  # never print report text; just mark the row
        out.writerow([u, *[0.5] * 12, 0]); print("fail", u[-8:], type(e).__name__, flush=True)
    w.flush(); print(f"{k + 1} done", flush=True)
