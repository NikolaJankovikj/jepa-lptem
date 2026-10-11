"""M0: can a mode forecast (MoP-JEPA style) recover the growth law from the context?

PLAN Step 10, *Mixture of predictors on growth*, fixed 2026-10-11 before it ran.
Data, split, readings and exponent fit are ``exponent_ceiling.py``'s. Instead of one
averaged forecast, a set of heads (possible future curves of log(r(+h) / last
reading)) and a router that weights them from the context:

- ``labelled_top``  -- heads = train-median ratio per process + window start; router =
  majority branch among the 25 nearest train contexts (a reference: it uses labels)
- ``free_top``      -- heads = K-means centroids of the train targets (K = 6); router =
  share of the 25 nearest train contexts in each cluster; the most likely head
- ``free_mean``     -- the router-weighted average of the same heads
- ``ridge``         -- exponent_ceiling's context ridge, the averaged forecast

M1 (candidate C on the pooled state) is built iff at sd 0.8 ``free_top`` has median
exponents within +-0.1 of the truth on diffusion and reaction and of 0 on static.

Runs on the Mac in seconds:
    uv run python scripts/mop_ceiling.py --out experiments/mop-ceiling
"""

from __future__ import annotations

import argparse
import json
import string
from pathlib import Path

import numpy as np
from eval_forecasts import EXPONENT_TOL
from exponent_ceiling import (
    JUDGED_CONTEXT,
    KNN_K,
    MIN_READING_PX,
    context_forecasts,
    median_exponents,
    trajectories,
)
from growth_identifiability import GROWING, READING_SD
from sklearn.cluster import KMeans
from token_persistence import gt_windows

from jepa_lptem.sim.dataset import EXPONENT

N_HEADS = 6
KMEANS_SEED = 0
SHOWN = ("diffusion", "reaction", "static")


def nearest_train(rel: np.ndarray, train: np.ndarray) -> np.ndarray:
    """``(W, KNN_K)`` positions among the train rows of each window's nearest train
    contexts, never the window itself."""
    dist = ((rel[:, None] - rel[None, train]) ** 2).sum(-1)
    dist[np.flatnonzero(train), np.arange(train.sum())] = np.inf
    return np.argsort(dist, axis=1)[:, :KNN_K]


def labelled_modes(readings, traj: dict, nearest, ctx: int) -> tuple[np.ndarray, dict]:
    """Top-branch forecast ``(W, 32 - ctx)`` and router accuracy on test."""
    truth, train = traj["truth"], traj["train"]
    key = np.char.add(traj["process"], traj["start"].astype(str))
    ratio = truth[:, ctx:] / truth[:, ctx - 1 : ctx]
    branches = np.unique(key[train])
    heads = np.array([np.median(ratio[train & (key == b)], axis=0) for b in branches])
    votes = (key[train][nearest][..., None] == branches).sum(axis=1)
    pick = votes.argmax(axis=1)
    test = ~train
    accuracy = {
        "branch": float((branches[pick] == key)[test].mean()),
        "process": float(
            (np.char.rstrip(branches[pick], string.digits) == traj["process"])[
                test
            ].mean()
        ),
    }
    return readings[:, ctx - 1 : ctx] * heads[pick], accuracy


def free_modes(last, target, traj: dict, nearest) -> tuple[dict, dict]:
    """Label-free heads clustered on the train ``target`` (log ratio to ``last``):
    top-head and mixture-mean forecasts, head usage on test."""
    train = traj["train"]
    km = KMeans(N_HEADS, n_init=10, random_state=KMEANS_SEED).fit(target[train])
    weights = np.eye(N_HEADS)[km.labels_][nearest].mean(axis=1)
    top = weights.argmax(axis=1)
    test = ~train
    usage = {
        p: np.bincount(top[test & (traj["process"] == p)], minlength=N_HEADS).tolist()
        for p in np.unique(traj["process"]).tolist()
    }
    forecasts = {
        "free_top": last * np.exp(km.cluster_centers_[top]),
        "free_mean": last * np.exp(weights @ km.cluster_centers_),
    }
    return forecasts, {"head_by_process": usage}


def mae_24(forecast: np.ndarray, traj: dict) -> dict:
    test = ~traj["train"]
    err = np.abs(forecast[:, -1] - traj["truth"][:, -1])
    return {
        p: float(err[test & (traj["process"] == p)].mean())
        for p in np.unique(traj["process"]).tolist()
    }


def score_readings(readings: np.ndarray, traj: dict, ctx: int) -> dict:
    """Every forecaster's exponents and +24 MAE from one set of noisy readings."""
    nearest = nearest_train(
        readings[:, :ctx] / readings[:, ctx - 1 : ctx], traj["train"]
    )
    forecasts = {
        "ridge": context_forecasts(readings, traj["truth"], traj["train"], ctx)["ridge"]
    }
    forecasts["labelled_top"], accuracy = labelled_modes(readings, traj, nearest, ctx)
    last, truth = readings[:, ctx - 1 : ctx], traj["truth"]
    free, usage = free_modes(last, np.log(truth[:, ctx:] / last), traj, nearest)
    forecasts |= free
    # Post hoc, not judged: heads clustered on the true growth ratio, so the reading
    # noise of the last frame cannot drive the clusters.
    true_ratio = np.log(truth[:, ctx:] / truth[:, ctx - 1 : ctx])
    forecasts["posthoc_true_ratio_top"] = free_modes(last, true_ratio, traj, nearest)[
        0
    ]["free_top"]
    return {
        "exponent": {n: median_exponents(f, traj, ctx) for n, f in forecasts.items()},
        "mae_24_px": {n: mae_24(f, traj) for n, f in forecasts.items()},
        "router_accuracy": accuracy,
    } | usage


def run(traj: dict, ctx: int, rng) -> dict:
    out = {"truth": median_exponents(traj["truth"][:, ctx:], traj, ctx)}
    for sd in READING_SD:
        readings = np.maximum(
            traj["truth"] + rng.normal(0.0, sd, traj["truth"].shape), MIN_READING_PX
        )
        out[f"sd_{sd}"] = score_readings(readings, traj, ctx)
    return out


def build_m1(result: dict) -> bool:
    """The pre-registered decision: label-free top head at sd 0.8."""
    e = result[f"sd_{READING_SD[-1]}"]["exponent"]["free_top"]
    return all(abs(e[p] - EXPONENT[p]) <= EXPONENT_TOL for p in (*GROWING, "static"))


def print_block(block: dict, sd: float) -> None:
    for name, e in block["exponent"].items():
        print(f"  {name + f' (sd {sd})':34s}" + "".join(f"{e[p]:11.3f}" for p in SHOWN))
    print(f"  MAE +24 px (sd {sd})")
    for name, m in block["mae_24_px"].items():
        print(f"    {name:32s}" + "".join(f"{m[p]:11.2f}" for p in SHOWN))
    print(f"  router accuracy {block['router_accuracy']}")
    print(f"  label-free head usage by process {block['head_by_process']}")


def print_report(result: dict) -> None:
    print(
        f"{f'context {JUDGED_CONTEXT}: median exponent':36s}"
        + "".join(f"{p:>11s}" for p in SHOWN)
    )
    print(
        f"  {'truth (crop)':34s}"
        + "".join(f"{result['truth'][p]:11.3f}" for p in SHOWN)
    )
    for sd in READING_SD:
        print_block(result[f"sd_{sd}"], sd)
    print(f"\nbuild M1: {result['build_m1']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-per-process", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    traj = trajectories(gt_windows(args.n_per_process, args.seed), args.seed)
    result = run(traj, JUDGED_CONTEXT, np.random.default_rng(args.seed))
    result["build_m1"] = build_m1(result)
    print_report(result)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(result, indent=1))
    print(f"\nwrote {args.out}/metrics.json")


if __name__ == "__main__":
    main()
