"""Is a clip's growth predictable from its context, at the quality a probe reads it?

Ground truth only. Mean radius over the crop-visible particles, per latent frame, on
the same windows as ``token_persistence.py``. The context is latent frames
0..CTX-1. A forecast at +h is scored against the true mean radius at frame
CTX-1+h. The "readings" are the true context values plus Gaussian noise of
``--reading-sd`` px, a stand-in for what a probe reads off encoded frames. Every
extrapolator sees only those readings:

- ``persistence``   -- the last reading, held
- ``pooled_trend``  -- last reading x the median growth ratio at +h over all training
  windows: an h-aware readout of one frame that cannot tell the processes apart
  (the forecast-probe persistence baseline)
- ``line_fit``      -- a straight line through the context readings, extrapolated
- ``ridge``         -- a ridge regression from the context readings to log(r(+h)/last),
  fitted on training clips with all processes pooled and no process label: the most
  a forecaster can take from the context at probe quality
- ``process_trend`` -- like ``pooled_trend`` but with the true process given (an
  oracle reference, not a forecaster)

Train/test split by clip, so both windows of a clip stay on one side. Also reported:
how well the context slope tells growing (diffusion, reaction) from non-growing clips.

Runs on the Mac in seconds:
    uv run python scripts/growth_identifiability.py --out experiments/growth-identifiability
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from token_persistence import CTX, gt_windows

from jepa_lptem.sim.dataset import PROCESSES

HORIZONS = np.array([1, 5, 10, 24])
# A probe's mean-radius MAE on the true future is 0.61-0.70 px at +24 on diffusion and
# reaction (experiments/forecast-probe/, floor). For Gaussian error MAE = 0.80 sd,
# so sd ~0.8 px. 0 gives the noise-free bound.
READING_SD = (0.0, 0.8)
RIDGE_LAMBDA = 1.0
GROWING = ("diffusion", "reaction")


def mean_radius(rad: np.ndarray) -> np.ndarray:
    """``(N, T)`` radii, 0 = absent -> ``(T,)`` mean over visible, NaN if none."""
    count = (rad > 0).sum(axis=0)
    return np.where(count > 0, rad.sum(axis=0) / np.maximum(count, 1), np.nan)


def context_features(readings: np.ndarray, ctx: int = CTX) -> np.ndarray:
    """Context readings relative to the last one, their squares, and log size."""
    rel = readings[:, :ctx] / readings[:, ctx - 1 : ctx]
    return np.c_[np.ones(len(readings)), rel, rel**2, np.log(readings[:, ctx - 1])]


def ridge_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.solve(x.T @ x + RIDGE_LAMBDA * np.eye(x.shape[1]), x.T @ y)


def extrapolate(
    truth: np.ndarray, readings: np.ndarray, process: np.ndarray, train
) -> dict:
    """Every extrapolator's ``(W, len(HORIZONS))`` forecast of the mean radius."""
    last = readings[:, CTX - 1 : CTX]
    future = truth[:, CTX - 1 + HORIZONS]
    ratio = future / truth[:, CTX - 1 : CTX]
    x = context_features(readings)
    slope = np.polyfit(np.arange(CTX), readings[:, :CTX].T, 1)[0][:, None]
    level = readings[:, :CTX].mean(axis=1, keepdims=True) + slope * (CTX - 1) / 2
    by_process = {
        p: np.median(ratio[train & (process == p)], axis=0) for p in set(process)
    }
    return {
        "persistence": np.repeat(last, len(HORIZONS), axis=1),
        "pooled_trend": last * np.median(ratio[train], axis=0),
        "line_fit": level + slope * HORIZONS,
        "ridge": last
        * np.exp(x @ ridge_fit(x[train], np.log(future[train] / last[train]))),
        "process_trend": last * np.array([by_process[p] for p in process]),
    }


def growth_detection(readings: np.ndarray, process: np.ndarray, train) -> dict:
    """Test accuracy of "growing" iff the relative context slope exceeds a threshold
    picked on train."""
    rel = readings[:, :CTX] / readings[:, CTX - 1 : CTX]
    slope = np.polyfit(np.arange(CTX), rel.T, 1)[0]
    growing = np.isin(process, GROWING)
    best = max(
        np.unique(slope[train]),
        key=lambda c: ((slope[train] > c) == growing[train]).mean(),
    )
    share = growing[~train].mean()
    return {
        "accuracy": float(((slope[~train] > best) == growing[~train]).mean()),
        "chance": float(max(share, 1 - share)),
    }


def score(truth, forecasts: dict, process: np.ndarray, test) -> dict:
    """Mean-radius MAE per process, method and horizon on the test clips."""
    future = truth[:, CTX - 1 + HORIZONS]
    return {
        p: {
            name: dict(
                zip(
                    (f"+{h}" for h in HORIZONS),
                    np.abs(f - future)[test & (process == p)].mean(axis=0).tolist(),
                    strict=True,
                )
            )
            for name, f in forecasts.items()
        }
        for p in PROCESSES
    }


def split_windows(rows: list[dict], rng) -> tuple:
    """True mean-radius trajectories, process labels and a by-clip train mask;
    windows with an empty frame are dropped."""
    truth = np.array([mean_radius(r["rad"]) for r in rows])
    keep = ~np.isnan(truth).any(axis=1)
    process = np.array([r["process"] for r in rows])[keep]
    clip = np.array([r["clip"] for r in rows])[keep]
    clips = rng.permutation(np.unique(clip))
    return truth[keep], process, np.isin(clip, clips[: len(clips) // 2])


def run(rows: list[dict], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    truth, process, train = split_windows(rows, rng)
    out = {"n_windows": len(truth), "n_dropped_empty": len(rows) - len(truth)}
    for sd in READING_SD:
        readings = truth + rng.normal(0.0, sd, truth.shape)
        forecasts = extrapolate(truth, readings, process, train)
        out[f"sd_{sd}"] = {
            "mae_px": score(truth, forecasts, process, ~train),
            "growth_detection": growth_detection(readings, process, train),
        }
    return out


def print_report(result: dict) -> None:
    for sd in READING_SD:
        block = result[f"sd_{sd}"]
        print(
            f"\nreading sd {sd} px: mean-radius MAE (px) on test clips"
            + "".join(f"{'+' + str(h):>7s}" for h in HORIZONS)
        )
        for p, methods in block["mae_px"].items():
            for i, (name, by_h) in enumerate(methods.items()):
                print(
                    f"  {p if i == 0 else '':12s}{name:16s}"
                    + "".join(f"{v:7.2f}" for v in by_h.values())
                )
        print(
            f"  growing vs not from the context slope: accuracy {block['growth_detection']['accuracy']:.2f}"
            f" (chance {block['growth_detection']['chance']:.2f})"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-per-process", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    result = run(gt_windows(args.n_per_process, args.seed), args.seed)
    print_report(result)
    config = {
        "n_per_process": args.n_per_process,
        "seed": args.seed,
        "context": CTX,
        "reading_sd": READING_SD,
        "ridge_lambda": RIDGE_LAMBDA,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(
        json.dumps({"config": config, **result}, indent=1)
    )
    print(f"\nwrote {args.out}/metrics.json")


if __name__ == "__main__":
    main()
