#!/usr/bin/env python3
"""Resumable parallel download of one archive (or its first --max_gb GB) by HTTP range requests.

The URL comes from a private '<name> <url>' file and never appears on a command line (ps shows
argv to every user of a shared machine). Chunks land at their offsets in a preallocated file;
finished chunk numbers go to <out>.done, so a rerun only fetches what is missing. A connection
that stalls for --timeout seconds is dropped and the chunk retried.

    python tools/fetch_ranges.py --urls .urls --archive knee_DICOMs_batch1 --out /tmp/fastmri_dl/b1.tar.xz --max_gb 64
"""
import argparse
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def total_size(url, timeout):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return int(r.headers["Content-Range"].split("/")[-1])


def fetch(url, out, lock, done_path, i, start, end, timeout, tries=50):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(out, "r+b") as f:
                f.seek(start)
                got = 0
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    got += len(b)
            if got != end - start + 1:
                raise IOError(f"short read {got}")
            with lock, open(done_path, "a") as d:
                d.write(f"{i}\n")
            return got
        except Exception as e:  # stall, reset, short read, 5xx: retry the whole chunk
            print(f"[retry] chunk {i} attempt {attempt + 1}: {type(e).__name__}", flush=True)
            time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f"chunk {i} failed {tries} times")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_gb", type=float, default=0, help="only the first N GB (0 = whole file)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--chunk_mb", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=60)
    a = ap.parse_args(argv)
    urls = dict(line.split(None, 1) for line in open(os.path.expanduser(a.urls)) if line.strip())
    url = urls[a.archive].strip()
    size = total_size(url, a.timeout)
    if a.max_gb:
        size = min(size, int(a.max_gb * 1e9))
    if not os.path.exists(a.out) or os.path.getsize(a.out) != size:
        with open(a.out, "ab") as f:
            f.truncate(size)
    ch = a.chunk_mb << 20
    chunks = [(i, s, min(s + ch, size) - 1) for i, s in enumerate(range(0, size, ch))]
    done_path = a.out + ".done"
    done = {int(x) for x in open(done_path)} if os.path.exists(done_path) else set()
    todo = [c for c in chunks if c[0] not in done]
    print(f"{a.archive}: {size / 1e9:.1f} GB in {len(chunks)} chunks, {len(todo)} to fetch", flush=True)
    lock, t0, got = threading.Lock(), time.time(), 0
    with ThreadPoolExecutor(a.threads) as ex:
        for k, n in enumerate(ex.map(lambda c: fetch(url, a.out, lock, done_path, *c, a.timeout), todo), 1):
            got += n
            if k % 8 == 0 or k == len(todo):
                print(f"{k}/{len(todo)} chunks | {got / 1e6 / (time.time() - t0):.1f} MB/s", flush=True)
    print(f"DOWNLOAD_DONE {a.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
