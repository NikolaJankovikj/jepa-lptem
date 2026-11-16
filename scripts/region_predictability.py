"""Does a coarse region's own context predict its future beyond the frame average?

Ground truth only. A regional forecaster is only worth building if a region's future
departs from the frame average in a way its own context predicts. Per window, the
crop is cut into ``cell``-px regions; per region and latent frame, the particle count
and total area (centre-assigned, pi r^2). The target is the region's *deviation* from
the frame mean at ``CTX - 1 + h``. Two ridge fits, split by clip:

- ``frame``  -- the frame's count and area over the 8 context frames
- ``region`` -- the same plus the region's own count and area over the context

Since the frame features alone cannot locate anything, ``frame`` is the pooled
forecaster's best case and the R2 of ``region`` is the spatial signal on offer.
Noise-free readings, so this is an upper bound.

    uv run python scripts/region_predictability.py --out experiments/region-persistence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from token_persistence import CROP, CTX, gt_windows

from jepa_lptem.sim.dataset import PROCESSES

HORIZONS = (1, 5, 10, 24)
CELLS = (64, 128)
RIDGE = 1e-2
N_BOOT = 200


def region_stats(pos: np.ndarray, rad: np.ndarray, cell: int) -> np.ndarray:
    """(T, n_regions, 2): count and total area per region, centre-assigned."""
    side = CROP // cell
    seen = rad > 0
    idx = np.clip(np.floor(pos / cell).astype(int), 0, side - 1)
    flat = idx[..., 0] * side + idx[..., 1]  # (N, T)
    out = np.zeros((rad.shape[1], side * side, 2))
    for t in range(rad.shape[1]):
        r = flat[seen[:, t], t]
        np.add.at(out[t, :, 0], r, 1.0)
        np.add.at(out[t, :, 1], r, np.pi * rad[seen[:, t], t] ** 2)
    return out


def design(rows: list[dict], cell: int, h: int) -> tuple:
    """Per region sample: frame features, region features, deviation targets, clip."""
    frame_x, region_x, y, clips = [], [], [], []
    for row in rows:
        s = region_stats(row["pos"], row["rad"], cell)
        frame = s.mean(axis=1)  # (T, 2), per-region average
        ctx_frame = frame[:CTX].reshape(-1)
        dev = s[CTX - 1 + h] - frame[CTX - 1 + h]  # (n_regions, 2)
        for k in range(s.shape[1]):
            frame_x.append(ctx_frame)
            region_x.append(s[:CTX, k].reshape(-1))
            y.append(dev[k])
            clips.append(row["clip"])
    return (np.array(frame_x), np.array(region_x), np.array(y), np.array(clips))


def ridge_predict(x_tr, y_tr, x_te):
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-9
    a = np.c_[(x_tr - mu) / sd, np.ones(len(x_tr))]
    b = np.c_[(x_te - mu) / sd, np.ones(len(x_te))]
    w = np.linalg.solve(a.T @ a + RIDGE * len(a) * np.eye(a.shape[1]), a.T @ y_tr)
    return b @ w


def r2(y, pred):
    return 1.0 - ((y - pred) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def two_fold_predictions(x, y, clips):
    pred = np.zeros_like(y)
    for fold in (0, 1):
        te = clips % 2 == fold
        pred[te] = ridge_predict(x[~te], y[~te], x[te])
    return pred


def clip_bootstrap_ci(y, pred, clips, rng) -> tuple:
    """95 % interval of R2 per target column, resampling whole clips."""
    uniq = np.unique(clips)
    members = {c: np.flatnonzero(clips == c) for c in uniq}
    boot = []
    for _ in range(N_BOOT):
        sel = np.concatenate([members[c] for c in rng.choice(uniq, len(uniq))])
        boot.append(r2(y[sel], pred[sel]))
    return np.percentile(boot, [2.5, 97.5], axis=0)


def evaluate(rows: list[dict], cell: int, h: int, rng) -> dict:
    frame_x, region_x, y, clips = design(rows, cell, h)
    frame_pred = two_fold_predictions(frame_x, y, clips)
    region_pred = two_fold_predictions(np.c_[frame_x, region_x], y, clips)
    lo, hi = clip_bootstrap_ci(y, region_pred, clips, rng)
    out = {
        stat: {
            "r2_frame": float(r2(y, frame_pred)[j]),
            "r2_region": float(r2(y, region_pred)[j]),
            "r2_region_ci": [float(lo[j]), float(hi[j])],
        }
        for j, stat in enumerate(("count", "area"))
    }
    out["mean_particles_per_region"] = float(
        np.mean([
            region_stats(r["pos"], r["rad"], cell)[CTX - 1, :, 0].mean() for r in rows
        ])
    )
    return out


def report_line(key: str, result: dict) -> str:
    return f"{key:28s} n/region {result['mean_particles_per_region']:.2f}" + "".join(
        f"  {s} R2 {result[s]['r2_region']:.2f}"
        f" [{result[s]['r2_region_ci'][0]:.2f}, {result[s]['r2_region_ci'][1]:.2f}]"
        f" (frame {result[s]['r2_frame']:.2f})"
        for s in ("count", "area")
    )


def run_grid(rows: list[dict], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    results = {}
    for cell in CELLS:
        for group in (*PROCESSES, "all"):
            sel = [r for r in rows if group in (r["process"], "all")]
            for h in HORIZONS:
                key = f"{cell}/{group}/+{h}"
                results[key] = evaluate(sel, cell, h, rng)
                print(report_line(key, results[key]))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-per-process", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    results = run_grid(gt_windows(args.n_per_process, args.seed), args.seed)
    config = {
        "n_per_process": args.n_per_process,
        "seed": args.seed,
        "cells": CELLS,
        "horizons": HORIZONS,
        "ridge": RIDGE,
        "reference_latent_frame": CTX - 1,
        "readings": "noise-free",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "predictability.json").write_text(
        json.dumps({"config": config, **results}, indent=1)
    )


if __name__ == "__main__":
    main()
