"""Audit a generated dataset before anything is trained on it.

Reads every clip once, so a truncated or corrupt ``.npz`` fails here rather than
hours into feature extraction, and prints what the full run should look like:
counts per process x stratum, frame shapes, merge events, clipping, and clips
with an empty frame.

Usage (remote, after generation; a few minutes for 2000 clips):
    uv run python scripts/audit_dataset.py data/dynamics-v1
"""

from __future__ import annotations

import collections
import glob
import json
import sys

import numpy as np


def scan_clips(files: list[str]) -> dict:
    """Read every clip in full -- decompressing ``frames`` is what surfaces a
    truncated file -- and collect per-clip checks."""
    bad, shapes, clipped, empty = [], collections.Counter(), [], []
    for f in files:
        try:
            z = np.load(f)
            frames, radii = z["frames"], z["radii"]
        except Exception as e:  # any read failure is the finding, whatever its type
            bad.append((f, repr(e)))
            continue
        shapes[(frames.shape, str(frames.dtype))] += 1
        full_scale = np.iinfo(frames.dtype).max
        clipped.append(((frames == 0) | (frames == full_scale)).mean())
        if ((radii > 0).sum(axis=0) == 0).any():
            empty.append(f.split("/")[-1])
    return {"bad": bad, "shapes": shapes, "clipped": clipped, "empty": empty}


def main() -> None:
    root = sys.argv[1]
    files = sorted(glob.glob(f"{root}/clip_*.npz"))
    with open(f"{root}/manifest.jsonl") as fh:
        manifest = [json.loads(line) for line in fh]
    print(f"clips on disk: {len(files)}   manifest lines: {len(manifest)}")
    scan = scan_clips(files)
    cells = collections.Counter((m["process"], m["stratum"]) for m in manifest)
    merges = sum(m["n_merges"] for m in manifest if m["process"] == "coalescence")
    print(f"unreadable files: {len(scan['bad'])} {scan['bad'][:3]}")
    print(f"frame shapes: {dict(scan['shapes'])}")
    print(f"by process x stratum: {dict(sorted(cells.items()))}")
    print(f"coalescence merges total: {merges}")
    print(f"black/saturated pixel fraction: mean {np.mean(scan['clipped']):.4%}  "
          f"worst {np.max(scan['clipped']):.4%}")
    print(f"clips with a frame containing no particles: {len(scan['empty'])} "
          f"{scan['empty'][:3]}")


if __name__ == "__main__":
    main()
