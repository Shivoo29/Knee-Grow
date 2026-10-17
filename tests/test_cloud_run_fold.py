#!/usr/bin/env python3
"""tests/test_cloud_run_fold.py - cloud/run_fold.sh end to end on CPU, with a fake `kaggle` command, a local
HTTP server standing in for fastMRI's S3, a synthetic corpus192 with all 4,407 real ids (constant volumes) and a
--smoke training run. Needs pydicom + opencv + timm (not in .venv), about 15 minutes:
    <python with pydicom + cv2 + timm> -I tests/test_cloud_run_fold.py
"""
import http.server
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import threading
import time

import numpy as np
import pandas as pd

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WT, "tools"))
sys.path.insert(0, os.path.join(WT, "tests"))
import export_corpus as ec  # noqa: E402
import fastmri_corpus as fc  # noqa: E402
import test_fastmri_corpus as tfc  # noqa: E402

LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
       'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
FAKE_KAGGLE = f"""#!{sys.executable}
import os, shutil, sys
a, mirror = sys.argv[1:], os.environ["FAKE_KAGGLE_MIRROR"]
if a[:2] == ["datasets", "download"]:
    src, dst = os.path.join(mirror, a[2].split("/")[1]), a[a.index("-p") + 1]
    os.makedirs(dst, exist_ok=True)
    names = [a[a.index("-f") + 1]] if "-f" in a else os.listdir(src)
    for n in names:
        if not os.path.exists(os.path.join(src, n)):
            sys.exit(1)
        shutil.copy(os.path.join(src, n), dst)
elif a[:2] in (["datasets", "create"], ["datasets", "version"]):
    with open(os.path.join(mirror, "published.txt"), "a") as f:
        f.write(" ".join(sorted(os.listdir(a[a.index("-p") + 1]))) + "\\n")
else:
    sys.exit(2)
"""


def _server(data):
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            s, e = self.headers["Range"].split("=")[1].split("-")
            s, e = int(s), min(int(e), len(data) - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {s}-{e}/{len(data)}")
            self.send_header("Content-Length", str(e - s + 1))
            self.end_headers()
            self.wfile.write(data[s:e + 1])

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _c192(d):
    os.makedirs(d, exist_ok=True)
    ids = pd.read_csv(os.path.join(WT, "data", "train.csv"), usecols=["StudyInstanceUID"])["StudyInstanceUID"].astype(str).values
    blob = ec.encode(np.full((192, 384, 384), 100, np.uint8))
    for k, rows in enumerate(np.array_split(np.arange(len(ids)), 16)):
        with tarfile.open(os.path.join(d, f"vols_{k:02d}.u8dtar"), "w") as tf:
            for r in rows:
                ti = tarfile.TarInfo(f"{ids[r]}.u8d"); ti.size = len(blob)
                tf.addfile(ti, io.BytesIO(blob))
    np.save(os.path.join(d, "all_ids.npy"), ids.astype(object))
    np.save(os.path.join(d, "all_masks.npy"), np.ones((len(ids), 192), np.uint8))
    json.dump({"format": "u8d", "shape": [192, 384, 384]}, open(os.path.join(d, "corpus.json"), "w"))


def test_run_fold_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        mirror, work, binp = (os.path.join(td, x) for x in ("mirror", "work", "bin"))
        os.makedirs(binp)
        open(os.path.join(binp, "kaggle"), "w").write(FAKE_KAGGLE)
        os.chmod(os.path.join(binp, "kaggle"), 0o755)
        teacher = os.path.join(td, "teacher.parquet")
        df = pd.DataFrame({"StudyInstanceUID": [fc.study_id("knee_batch/studyA"), fc.study_id("knee_batch/studyB")]})
        for c in LAB:
            df[c] = 0.3
        df.to_parquet(teacher, index=False)
        env = dict(os.environ, B=os.path.join(mirror, "knee-cloud-bundle"), NO_UPLOAD="1")
        subprocess.run(["bash", os.path.join(WT, "cloud", "make_bundle.sh"), teacher], env=env, check=True)
        _c192(os.path.join(mirror, "knee-c192-u8d"))
        arch = os.path.join(td, "fm.tar.xz")
        tfc._archive(arch, [("studyA", tfc.SERIES_A), ("studyB", tfc.SERIES_B)])
        srv = _server(open(arch, "rb").read())
        urls = os.path.join(td, "urls")
        open(urls, "w").write("".join(f"knee_DICOMs_batch{b} http://127.0.0.1:{srv.server_port}/b{b}?sig=x\n" for b in (1, 2)))
        env = dict(os.environ, PATH=binp + os.pathsep + os.environ["PATH"], FAKE_KAGGLE_MIRROR=mirror,
                   KAGGLE_USERNAME="tester", FOLD="0", WORK=work, FASTMRI_URLS=urls, FM_GB="0", PY=sys.executable,
                   EXTRA="--smoke", BS="2", ACCUM="1", WORKERS="0", CUDA_VISIBLE_DEVICES="", SKIP_INSTALL="1")
        try:
            r = subprocess.run(["bash", os.path.join(WT, "cloud", "run_fold.sh")], env=env, capture_output=True,
                               text=True, timeout=5400)
        finally:
            srv.shutdown()
        tail = (r.stdout + r.stderr)[-3000:]
        assert r.returncode == 0 and "DONE fold 0 -> tester/knee-cnxt-fm-f0" in r.stdout, tail
        assert "fastMRI corpus: 2 studies" in r.stdout, tail
        pub = open(os.path.join(mirror, "published.txt")).read()
        assert "raptor_ft_cnxt_fm_f0_swa.pt" in pub and "train_cnxt_fm_f0.log" in pub, pub
        log = open(os.path.join(work, "run", "train_cnxt_fm_f0.log")).read()
        assert "[extra] +2 studies from RAPTOR_EXTRA_CORPUS" in log, log[-2000:]
    return "bundle -> fastMRI from fake S3 (u8d) -> 16 corpus192 shards -> smoke fold 0 with 2 extra studies -> published"


TESTS = [("run_fold_end_to_end", test_run_fold_end_to_end)]


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
