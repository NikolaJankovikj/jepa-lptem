"""Generate a synthetic LPTEM dataset: known physics -> realistic frames + ground truth.

Wires the tested pieces together — trajectories/radii from ``sim.dynamics`` through
the ``sim.render_bridge`` image-formation model — and writes one ``.npz`` per clip
plus a JSONL manifest. Every clip ships with the exact positions, radii, and the
physical parameters that produced it, so the evaluation always has an answer key.

Each ``.npz`` holds:
    frames    uint8  (T, H, W)   the LPTEM video (what V-JEPA sees)
    positions float  (N, T, 2)   particle centres (row, col), ground truth
    radii     float  (N, T)      particle radii in px, ground truth (0 = absent)
and clip-level scalars (process, exponent, D, drift, ...) mirrored into the
manifest, plus ``merge_events`` (M, 3) as ``[frame, survivor, absorbed]`` -- the
exact coalescence times Step 12 grades anticipation lead-time against.

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

from jepa_lptem.sim.dataset import EXPERIMENT_EVERY, PROCESSES, build_clip
from jepa_lptem.sim.render_bridge import calibrate_mtf, render_video, to_uint8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-videos", type=int, default=5)
    ap.add_argument("--n-frames", type=int, default=48)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--processes", nargs="+", default=list(PROCESSES), choices=PROCESSES)
    ap.add_argument("--mtf", type=Path, default=Path("code/LPTEMsimulator/MTFmatrix512.mat"),
                    help="measured MTF .mat; resampled to --size at detector calibration")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    size = (args.size, args.size)
    mtf = None
    if args.mtf.exists():
        from scipy.io import loadmat
        # calibrate_mtf resamples the measured matrix onto this render's frequency
        # grid at the empirically anchored detector scale (see MTF_FREQ_SCALE) --
        # applied verbatim, the matrix blurs 2x harder than the real detector.
        mtf = calibrate_mtf(loadmat(str(args.mtf))["MTFmatrix"], args.size)
    else:
        print(f"[warn] {args.mtf} not found; rendering without MTF")

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    manifest = args.out / "manifest.jsonl"
    with manifest.open("w") as mf:
        for i in range(args.n_videos):
            process = args.processes[i % len(args.processes)]
            # Walk the experiment stratum diagonally across the (round, process)
            # grid. Two failed versions are why this is not a one-liner:
            # `i % EXPERIMENT_EVERY` is in lockstep with `i % len(processes)` when
            # those are equal (they both are, at 5), so it locks the experiment
            # stratum onto one single process forever; cycling on rounds instead
            # unlocks the process but puts the first experiment clip at index 20, so
            # a short smoke run gets *none* -- and containment is checked on smoke
            # runs, against the one stratum that exists to bracket real data.
            # The diagonal gives 1-in-EXPERIMENT_EVERY from the first round and
            # reaches every process for any process count, since each one takes its
            # turn at `round == -proc (mod EXPERIMENT_EVERY)`.
            proc_idx, round_ = i % len(args.processes), i // len(args.processes)
            stratum = ("experiment"
                       if (round_ + proc_idx) % EXPERIMENT_EVERY == 0
                       else "physics")
            positions_t, radii_t, meta, pos, radii = build_clip(
                process, args.n_frames, size, rng, stratum=stratum)
            video = render_video(positions_t, radii_t, size, mtf=mtf, rng=rng,
                                 pixel_size_nm=meta["pixel_size_nm"],
                                 beam_flux=meta["beam_flux"],
                                 thickness_sd_nm=meta["thickness_sd_nm"])
            # one fixed physical scale for the whole dataset, so absolute contrast
            # survives and overall brightness cannot leak particle size
            frames = np.stack([to_uint8(f, beam_flux=meta["beam_flux"]) for f in video])

            name = f"clip_{i:04d}_{process}_{stratum}.npz"
            np.savez_compressed(args.out / name, frames=frames, positions=pos, radii=radii,
                                **{k: v for k, v in meta.items() if v is not None})
            record = {"file": name, **meta, "shape": list(frames.shape)}
            mf.write(json.dumps(record) + "\n")
            print(f"[{i + 1}/{args.n_videos}] {name}  frames {frames.shape}  "
                  f"{stratum:10s} N={meta['n_particles']:2d}  "
                  f"flux={meta['beam_flux']:5.1f}  px={meta['pixel_size_nm']:.1f}nm  "
                  f"r_max={radii.max():.1f}px  merges={meta['n_merges']}  "
                  f"drift={meta['drift_per_step']:.2f}/step")

    print(f"\nWrote {args.n_videos} clips + manifest to {args.out}")


if __name__ == "__main__":
    main()
