"""Generate a synthetic LPTEM dataset: known physics -> realistic frames + ground truth.

Wires the tested pieces together — trajectories/radii from ``sim.dynamics`` through
the ``sim.render_bridge`` image-formation model — and writes one ``.npz`` per clip
plus a JSONL manifest. Every clip ships with the exact positions, radii, and the
physical parameters that produced it, so the evaluation always has an answer key.

Each ``.npz`` holds:
    frames    uint8  (T, H, W)   the LPTEM video (what V-JEPA sees)
    positions float  (N, T, 2)   particle centres (row, col), ground truth
    radii     float  (N, T)      particle radii in px, ground truth (0 = absent)
and clip-level scalars (process, exponent, D, ...) mirrored into the manifest.

Usage (local smoke):
    uv run python scripts/generate_dynamics_dataset.py --out data/synthetic/smoke \
        --n-videos 5 --n-frames 48 --size 512
Heavy generation runs on the remote (Step 5); this script is the same code path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jepa_lptem.sim.dataset import PROCESSES, build_clip
from jepa_lptem.sim.render_bridge import normalize_uint8, render_video


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-videos", type=int, default=5)
    ap.add_argument("--n-frames", type=int, default=48)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--processes", nargs="+", default=list(PROCESSES), choices=PROCESSES)
    ap.add_argument("--mtf", type=Path, default=Path("code/LPTEMsimulator/MTFmatrix512.mat"),
                    help="MTF .mat; used only when --size matches its resolution")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    size = (args.size, args.size)
    mtf = None
    if args.mtf.exists():
        from scipy.io import loadmat
        m = loadmat(str(args.mtf))["MTFmatrix"]
        if m.shape == size:
            mtf = m
        else:
            print(f"[warn] MTF is {m.shape} but size is {size}; rendering without MTF")

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    manifest = args.out / "manifest.jsonl"
    with manifest.open("w") as mf:
        for i in range(args.n_videos):
            process = args.processes[i % len(args.processes)]
            positions_t, radii_t, meta, pos, radii = build_clip(process, args.n_frames, size, rng)
            video = render_video(positions_t, radii_t, size, mtf=mtf, rng=rng)
            frames = np.stack([normalize_uint8(f) for f in video])  # per-clip contrast

            name = f"clip_{i:04d}_{process}.npz"
            np.savez_compressed(args.out / name, frames=frames, positions=pos, radii=radii,
                                **{k: v for k, v in meta.items() if v is not None})
            record = {"file": name, **meta, "shape": list(frames.shape)}
            mf.write(json.dumps(record) + "\n")
            print(f"[{i + 1}/{args.n_videos}] {name}  frames {frames.shape}  "
                  f"process={process} N={meta['n_particles']}")

    print(f"\nWrote {args.n_videos} clips + manifest to {args.out}")


if __name__ == "__main__":
    main()
