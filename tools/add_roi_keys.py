#!/usr/bin/env python3
"""tools/add_roi_keys.py - backfill ROI keys into a specialist checkpoint.

The run in progress was started before train_knee.py saved roi keys, so patch
its .pt after the fact:

    .venv/bin/python tools/add_roi_keys.py CKPT [--roi_mode compartment]
                                               [--roi_pad 0.06] [--roi_overlap 0.12]

Loads CKPT (a .pt checkpoint dict), sets roi=True plus roi_mode/roi_pad/
roi_overlap, keeps every other key unchanged, and saves to CKPT with suffix
`_roi.pt` (foo.pt -> foo_roi.pt). CPU only; prints shapes/counts only.
"""
import argparse
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser(description="Backfill ROI keys into a checkpoint")
    ap.add_argument("ckpt", help="Input .pt checkpoint file")
    ap.add_argument("--roi_mode", default="compartment",
                    choices=["tight", "safe", "compartment"])
    ap.add_argument("--roi_pad", type=float, default=0.06)
    ap.add_argument("--roi_overlap", type=float, default=0.12)
    a = ap.parse_args()

    if not os.path.isfile(a.ckpt):
        print(f"missing input: {a.ckpt}", flush=True)
        return 2
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model" not in ck:
        print(f"not a checkpoint dict (keys: {len(ck) if isinstance(ck, dict) else type(ck)})",
              flush=True)
        return 2

    before = len(ck)
    ck["roi"] = True
    ck["roi_mode"] = a.roi_mode
    ck["roi_pad"] = float(a.roi_pad)
    ck["roi_overlap"] = float(a.roi_overlap)
    out = a.ckpt[:-3] + "_roi.pt" if a.ckpt.endswith(".pt") else a.ckpt + "_roi.pt"
    torch.save(ck, out)
    model_keys = len(ck["model"]) if isinstance(ck["model"], dict) else -1
    print(f"keys {before}->{len(ck)} model_tensors {model_keys} "
          f"roi=True mode={a.roi_mode} pad={a.roi_pad} overlap={a.roi_overlap} "
          f"wrote {os.path.basename(out)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
