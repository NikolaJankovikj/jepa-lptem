"""How fast does a particle's position stop being predictable at token scale?

Ground truth only: no rendering, no encoder. A per-token forecast can keep a particle
only where most of the possible futures still put it in the same token. The
L1-optimal deterministic forecast is the per-token median, and where fewer than half
the futures have the particle there, that median is background. So the fraction of
particles still in their token after h latent steps bounds how long any
deterministic token-level forecaster can keep particles sharp
(``experiments/core-perframe/``, the fade).

Windows match the Gate-2 follow-up and forecast-probe snippets: center crop, windows
starting at video frames 0 and 36, one latent frame per tubelet, horizons measured
from the last context frame (latent frame ``CTX - 1``). Per particle visible at both
ends:

- ``same_cell``  -- its centre is in the same 16 px token cell
- ``inside``     -- the old centre still lies inside the particle (a particle covers
  several cells, so this is the looser bound)
- ``displacement`` -- how far its centre moved, px

Runs on the Mac in seconds:
    uv run python scripts/token_persistence.py --out experiments/token-persistence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from extract_features import TUBELET, crop_and_align_gt

from jepa_lptem.eval.centroid_eval import GRID
from jepa_lptem.sim.dataset import PROCESSES, build_clip

FULL_SIZE, CROP, WINDOW, STARTS = 512, 256, 64, (0, 36)
N_FRAMES = 100
CTX = 8  # context latent frames, as in the follow-up snippets
HORIZONS = (1, 2, 3, 5, 10, 24)
CELL = CROP / GRID  # one token, px


def gt_windows(n_per_process: int, seed: int) -> list[dict]:
    """Physics-stratum ground truth, crop-aligned at latent-frame times."""
    rng = np.random.default_rng(seed)
    rows = []
    for clip, process in enumerate(p for p in PROCESSES for _ in range(n_per_process)):
        _, _, _, pos, rad = build_clip(
            process, N_FRAMES, (FULL_SIZE, FULL_SIZE), rng
        )
        for start in STARTS:
            p, r = crop_and_align_gt(pos, rad, start, WINDOW, FULL_SIZE, CROP)
            rows.append(
                {"process": process, "clip": clip, "start": start, "pos": p, "rad": r}
            )
    return rows


def persistence(pos: np.ndarray, rad: np.ndarray, ref: int, h: int) -> dict:
    """Per particle visible at ``ref`` and ``ref + h``: same cell, old centre inside
    the particle, displacement."""
    seen = (rad[:, ref] > 0) & (rad[:, ref + h] > 0)
    a, b = pos[seen, ref], pos[seen, ref + h]
    moved = np.linalg.norm(b - a, axis=-1)
    return {
        "same_cell": (np.floor(a / CELL) == np.floor(b / CELL)).all(axis=-1),
        "inside": moved < rad[seen, ref],
        "displacement": moved,
    }


def summarize(rows: list[dict]) -> dict:
    out = {}
    for group in (*PROCESSES, "all"):
        sel = [r for r in rows if group in (r["process"], "all")]
        out[group] = {}
        for h in HORIZONS:
            parts = [persistence(r["pos"], r["rad"], CTX - 1, h) for r in sel]
            cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            out[group][f"+{h}"] = {
                "n_particles": int(cat["same_cell"].size),
                "same_cell": float(cat["same_cell"].mean()),
                "inside": float(cat["inside"].mean()),
                "median_displacement_px": float(np.median(cat["displacement"])),
            }
    return out


def print_report(summary: dict) -> None:
    print(f"{'':12s}{'':22s}" + "".join(f"{'+' + str(h):>7s}" for h in HORIZONS))
    for group, by_h in summary.items():
        for i, (key, label) in enumerate((
            ("same_cell", "same 16 px cell"),
            ("inside", "old centre inside"),
            ("median_displacement_px", "median displacement"),
        )):
            name = group if i == 0 else ""
            print(
                f"{name:12s}{label:22s}"
                + "".join(f"{by_h[f'+{h}'][key]:7.2f}" for h in HORIZONS)
            )


def write_metrics(args, n_windows: int, summary: dict) -> None:
    config = {
        "n_per_process": args.n_per_process,
        "seed": args.seed,
        "stratum": "physics",
        "n_windows": n_windows,
        "crop": CROP,
        "cell_px": CELL,
        "window_starts": STARTS,
        "reference_latent_frame": CTX - 1,
        "frames_per_latent": TUBELET,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(
        json.dumps({"config": config, **summary}, indent=1)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-per-process", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rows = gt_windows(args.n_per_process, args.seed)
    summary = summarize(rows)
    print_report(summary)
    write_metrics(args, len(rows), summary)
    print(f"\nwrote {args.out}/metrics.json")


if __name__ == "__main__":
    main()
