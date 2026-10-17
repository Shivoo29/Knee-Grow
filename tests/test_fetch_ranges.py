#!/usr/bin/env python3
"""tests/test_fetch_ranges.py - tools/fetch_ranges.py against a local HTTP server with range support.
    .venv/bin/python -I tests/test_fetch_ranges.py
"""
import contextlib
import http.server
import io
import os
import sys
import tempfile
import threading
import time

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WT, "tools"))
import fetch_ranges as fr  # noqa: E402

DATA = os.urandom(5 * (1 << 20) + 12345)


class _Range(http.server.BaseHTTPRequestHandler):
    fail_once = set()

    def do_GET(self):
        a, b = self.headers["Range"].split("=")[1].split("-")
        a, b = int(a), min(int(b), len(DATA) - 1)
        if a in _Range.fail_once:                      # first request for this chunk dies mid-way
            _Range.fail_once.discard(a)
            self.send_response(206); self.send_header("Content-Length", str(b - a + 1)); self.end_headers()
            self.wfile.write(DATA[a:a + 100])
            return
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {a}-{b}/{len(DATA)}")
        self.send_header("Content-Length", str(b - a + 1))
        self.end_headers()
        self.wfile.write(DATA[a:b + 1])

    def log_message(self, *a):
        pass


def _serve():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Range)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _run(argv):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        fr.main(argv)
    return out.getvalue()


def test_full_partial_resume():
    srv = _serve()
    try:
        with tempfile.TemporaryDirectory() as td:
            urls = os.path.join(td, "urls")
            open(urls, "w").write(f"a http://127.0.0.1:{srv.server_port}/x?sig=SECRET\n")
            out = os.path.join(td, "a.bin")
            fr.time.sleep = lambda s: None                         # no backoff waits in the test
            _Range.fail_once = {2 << 20}
            log = _run(["--urls", urls, "--archive", "a", "--out", out, "--chunk_mb", "1", "--threads", "3", "--timeout", "5"])
            assert open(out, "rb").read() == DATA and "[retry] chunk 2" in log and "SECRET" not in log
            os.remove(out + ".done")
            open(out, "r+b").write(b"\0" * 1000)                   # damage, then a rerun with no record refetches
            _run(["--urls", urls, "--archive", "a", "--out", out, "--chunk_mb", "1", "--threads", "3"])
            assert open(out, "rb").read() == DATA
            log = _run(["--urls", urls, "--archive", "a", "--out", out, "--chunk_mb", "1"])
            assert "0 to fetch" in log
            part = os.path.join(td, "p.bin")
            _run(["--urls", urls, "--archive", "a", "--out", part, "--chunk_mb", "1", "--max_gb", "0.003"])
            assert open(part, "rb").read() == DATA[:3_000_000]
    finally:
        srv.shutdown()
    return "byte-exact full file, chunk retried after a cut, resume skips done chunks, prefix-only download"


TESTS = [("full_partial_resume", test_full_partial_resume)]


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
