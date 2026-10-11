"""Can any forecaster that sees only the context meet Gate 2''s exponent bar?

PLAN Step 10, *Exponent ceiling from the context*, fixed 2026-10-11 before it ran.
Ground truth only: the simulated physics windows of ``token_persistence.gt_windows``,
the mean radius of the crop-visible particles, split by clip. Readings are the true
context plus N(0, sd) px. Each method forecasts the mean radius at every forecast
frame, and the forecast's growth exponent is fitted against the clip's physical time,
as Gate 2' does:

- ``ridge``  -- growth_identifiability's features, log(r(+h) / last reading)
- ``knn``    -- k nearest training contexts (relative to their last reading)
- ``process_given`` / ``process_start_given`` -- train-median ratio per process (and
  window start): references that know what a forecaster would have to infer
- ``truth``  -- the true future in the crop, the value the bar is measured against

The bar is reachable at context 8 iff at sd 0.8 ridge or knn has median exponents
within +-0.1 of the truth on both diffusion and reaction. Context 16 (forecast frames
16-31) is reported, not judged.

Runs on the Mac in seconds:
    uv run python scripts/exponent_ceiling.py --out experiments/exponent-ceiling
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from eval_forecasts import EXPONENT_TOL, snippet_exponent
from growth_identifiability import GROWING, READING_SD, mean_radius
from token_persistence import TUBELET, gt_windows

from jepa_lptem.sim.dataset import EXPONENT

CONTEXTS = (8, 16)
JUDGED_CONTEXT = 8
KNN_K = 25
RIDGE_LAMBDA = 1.0
MIN_READING_PX = 1.0
CONTEXT_ONLY = ("ridge", "knn")


def trajectories(rows: list[dict], seed: int) -> dict:
    """True mean radius ``(W, 32)``, physical times, process, start and a by-clip
    train mask; windows with an empty frame are dropped."""
    truth = np.array([mean_radius(r["rad"]) for r in rows])
    keep = ~np.isnan(truth).any(axis=1)
    rows = [r for r, k in zip(rows, keep, strict=True) if k]
    frames = np.arange(truth.shape[1])
    clip = np.array([r["clip"] for r in rows])
    clips = np.random.default_rng(seed).permutation(np.unique(clip))
    return {
        "truth": truth[keep],
        "times": np.array([
            r["t0_growth"] + r["dt_growth"] * (r["start"] + TUBELET * frames)
            for r in rows
        ]),
        "process": np.array([r["process"] for r in rows]),
        "start": np.array([r["start"] for r in rows]),
        "train": np.isin(clip, clips[: len(clips) // 2]),
    }


def context_forecasts(readings: np.ndarray, truth: np.ndarray, train, ctx: int) -> dict:
    """Ridge and kNN forecasts ``(W, 32 - ctx)`` of the mean radius from the context."""
    last = readings[:, ctx - 1 : ctx]
    rel = readings[:, :ctx] / last
    x = np.c_[np.ones(len(rel)), rel, rel**2, np.log(last)]
    target = np.log(truth[:, ctx:] / last)
    w = np.linalg.solve(
        x[train].T @ x[train] + RIDGE_LAMBDA * np.eye(x.shape[1]),
        x[train].T @ target[train],
    )
    dist = ((rel[:, None] - rel[None, train]) ** 2).sum(-1)
    dist[np.flatnonzero(train), np.arange(train.sum())] = (
        np.inf
    )  # never its own neighbour
    nearest = np.argsort(dist, axis=1)[:, :KNN_K]
    return {
        "ridge": last * np.exp(x @ w),
        "knn": last * np.exp(target[train][nearest].mean(axis=1)),
    }


def oracle_forecasts(readings: np.ndarray, traj: dict, ctx: int) -> dict:
    """Last reading x the train-median true ratio per process (and per window start)."""
    truth, train = traj["truth"], traj["train"]
    ratio = truth[:, ctx:] / truth[:, ctx - 1 : ctx]
    keys = {
        "process_given": traj["process"],
        "process_start_given": np.char.add(traj["process"], traj["start"].astype(str)),
    }
    out = {}
    for name, key in keys.items():
        medians = {
            k: np.median(ratio[train & (key == k)], axis=0) for k in np.unique(key)
        }
        out[name] = readings[:, ctx - 1 : ctx] * np.array([medians[k] for k in key])
    return out


def median_exponents(forecast: np.ndarray, traj: dict, ctx: int) -> dict:
    """Median exponent of the forecast over its frames, per process, on test windows."""
    test = ~traj["train"]
    out = {}
    for p in np.unique(traj["process"]).tolist():
        rows = np.flatnonzero(test & (traj["process"] == p))
        out[p] = float(
            np.nanmedian([
                snippet_exponent(traj["times"][i, ctx:], forecast[i]) for i in rows
            ])
        )
    return out


def run_context(traj: dict, ctx: int, rng) -> dict:
    out = {"truth": median_exponents(traj["truth"][:, ctx:], traj, ctx)}
    for sd in READING_SD:
        readings = np.maximum(
            traj["truth"] + rng.normal(0.0, sd, traj["truth"].shape), MIN_READING_PX
        )
        forecasts = context_forecasts(
            readings, traj["truth"], traj["train"], ctx
        ) | oracle_forecasts(readings, traj, ctx)
        out[f"sd_{sd}"] = {
            name: median_exponents(f, traj, ctx) for name, f in forecasts.items()
        }
    return out


def reachable(result: dict) -> dict:
    """The pre-registered reading at context 8, sd 0.8."""
    block = result[f"context_{JUDGED_CONTEXT}"][f"sd_{READING_SD[-1]}"]
    ok = {
        m: all(abs(block[m][p] - EXPONENT[p]) <= EXPONENT_TOL for p in GROWING)
        for m in CONTEXT_ONLY
    }
    return {"by_method": ok, "reachable": any(ok.values())}


def print_context(block: dict, ctx: int) -> None:
    shown = ("diffusion", "reaction", "static")
    print(f"\ncontext {ctx}: median exponent over the forecast  " + "".join(f"{p:>11s}" for p in shown))
    print(f"  {'truth (crop)':30s}" + "".join(f"{block['truth'][p]:11.3f}" for p in shown))
    for sd in READING_SD:
        for name, e in block[f"sd_{sd}"].items():
            print(f"  {name + f' (sd {sd})':30s}" + "".join(f"{e[p]:11.3f}" for p in shown))


def print_report(result: dict) -> None:
    for ctx in CONTEXTS:
        print_context(result[f"context_{ctx}"], ctx)
    print(f"\ntrue: diffusion {EXPONENT['diffusion']:.3f}  reaction {EXPONENT['reaction']:.3f}")
    print(f"bar reachable at context {JUDGED_CONTEXT}, sd {READING_SD[-1]}: {result['reading']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-per-process", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    traj = trajectories(gt_windows(args.n_per_process, args.seed), args.seed)
    rng = np.random.default_rng(args.seed)
    result = {f"context_{ctx}": run_context(traj, ctx, rng) for ctx in CONTEXTS}
    result["reading"] = reachable(result)
    print_report(result)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(result, indent=1))
    print(f"\nwrote {args.out}/metrics.json")


if __name__ == "__main__":
    main()
