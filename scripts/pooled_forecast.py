"""P2: baselines, the linear candidate, and the Gate 2' verdict on the pooled state.

PLAN Step 10, position-free forecasting, fixed 2026-10-11 before any pooled number.
The pooled state is the frame-averaged latent (`mean`, P1). One snippet per window:
context = latent frames 0-7, forecast = 8-31 (+1..+24). Everything is read by the
frozen P1 readout (`experiments/pooled-gate1/readout.npz`):

- ``persistence`` / ``ctxmean``  frame 7, or the average of frames 0-7, held
- ``pooled_trend``   frame 7's readout x the median train ratio of the true mean
                     radius at +h over frame 7 (trend-blind, h-aware)
- ``readout_ridge``  growth_identifiability's ridge on the 8 context readouts: how much
                     the context allows. It reads a label-trained readout, so it is a
                     reference, never a candidate
- ``floor``          the true future pooled state
- ``linear``         candidate A, no labels: top-64 PCA coefficients of the 8 context
                     frames -> coefficients at each +h (one ridge, lambda on val by
                     pooled L1 at +24), projected back

Gate 2' (per candidate), on diffusion and reaction, test physics stratum:
reference = the best of persistence / ctxmean / pooled_trend in mean-radius MAE at +24;
powered iff the floor's paired gain over it has a CI above 0; PASS iff the candidate's
paired gain has a CI above 0 on every powered process, its median exponent over
+1..+24 is within +-0.1 of the true one on both, and on static within +-0.1 of 0.

Fits use all train windows (both strata), as P1 did; the verdict is on test physics.
Runs on the Mac:
    uv run python scripts/pooled_forecast.py \
        --pooled data/features/dynamics-v1-perframe-pooled.npz \
        --gate1 experiments/gate1-perframe --readout experiments/pooled-gate1/readout.npz \
        --out experiments/pooled-forecast
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from eval_forecasts import EXPONENT_TOL, gaussian_w2, moments_from_observables
from gate2_followup import bootstrap_ci
from growth_identifiability import context_features, ridge_fit
from pooled_gate1 import (
    HORIZON,
    JUDGED,
    LAMBDAS,
    RADIUS,
    Readout,
    load_context,
    snippet_exponents,
)

PCA_DIM = 64
REPORT_HORIZONS = (1, 5, 10, 24)
REFERENCES = ("persistence", "ctxmean", "pooled_trend")
MIN_READING_PX = 1.0  # the readout can dip below 0 on a near-empty frame; logs need > 0


def load_readout(path: Path) -> Readout:
    saved = np.load(path)
    readout = Readout()
    for key in ("x_mu", "x_sd", "y_mu", "y_sd", "weights"):
        setattr(readout, key, saved[key])
    readout.lam = float(saved["lam"])
    return readout


def read_z(readout: Readout, z: np.ndarray) -> np.ndarray:
    """Readout of pooled states already z-scored with its own statistics."""
    return z @ readout.weights * readout.y_sd + readout.y_mu


# ------------------------------ forecasts ---------------------------------------


def held_forecasts(z: np.ndarray) -> dict:
    """``(W, ctx + 24, D)`` z-scored pooled states -> persistence / ctxmean / floor
    ``(W, 24, D)``. The context is everything before the last HORIZON frames."""
    return {
        "persistence": np.repeat(z[:, -HORIZON - 1 : -HORIZON], HORIZON, axis=1),
        "ctxmean": np.repeat(z[:, :-HORIZON].mean(axis=1, keepdims=True), HORIZON, axis=1),
        "floor": z[:, -HORIZON:],
    }


def pca_basis(frames: np.ndarray) -> np.ndarray:
    """Top ``PCA_DIM`` principal directions ``(D, PCA_DIM)`` of z-scored frames."""
    evals, evecs = np.linalg.eigh(frames.T @ frames)
    return evecs[:, np.argsort(evals)[::-1][:PCA_DIM]]


def linear_candidate(z: np.ndarray, split: dict) -> tuple[np.ndarray, dict]:
    """Candidate A: ridge from the context's PCA coefficients to every horizon's,
    lambda picked on val by pooled L1 at +24. Returns z-scored ``(W, 24, D)``."""
    basis = pca_basis(z[split["train"]].reshape(-1, z.shape[-1]))
    coeffs = z @ basis
    inputs = np.c_[np.ones(len(z)), coeffs[:, :-HORIZON].reshape(len(z), -1)]
    targets = coeffs[:, -HORIZON:].reshape(len(z), -1)
    tr, va = split["train"], split["val"]
    gram, cross = inputs[tr].T @ inputs[tr], inputs[tr].T @ targets[tr]
    scores = {}
    for lam in LAMBDAS:
        w = np.linalg.solve(gram + lam * np.eye(len(gram)), cross)
        last = (inputs[va] @ w).reshape(len(va), HORIZON, PCA_DIM)[:, -1] @ basis.T
        scores[float(lam)] = float(np.abs(last - z[va, -1]).mean())
    lam = min(scores, key=scores.get)
    w = np.linalg.solve(gram + lam * np.eye(len(gram)), cross)
    forecast = (inputs @ w).reshape(len(z), HORIZON, PCA_DIM) @ basis.T
    return forecast, {"lambda": lam, "val_l1_at_24_by_lambda": scores}


def label_references(readings: np.ndarray, truth: np.ndarray, train) -> dict:
    """Mean-radius forecasts ``(W, 24)`` from the context readouts: pooled trend and
    readout ridge. ``readings`` are the readout's mean radius on the context frames,
    ``truth`` the true mean radius on context and forecast frames."""
    last = np.maximum(readings[:, -1:], MIN_READING_PX)
    with np.errstate(divide="ignore", invalid="ignore"):  # empty last frame: not fitted on
        ratio = truth[:, -HORIZON:] / truth[:, -HORIZON - 1 : -HORIZON]
    fit = train & np.isfinite(ratio).all(axis=1) & (ratio > 0).all(axis=1)
    x = context_features(np.maximum(readings, MIN_READING_PX), readings.shape[1])
    weights = ridge_fit(x[fit], np.log(truth[fit, -HORIZON:] / last[fit]))
    return {
        "pooled_trend": last * np.median(ratio[fit], axis=0),
        "readout_ridge": last * np.exp(x @ weights),
    }


# ------------------------------- scoring ----------------------------------------


def observables(
    readout: Readout, z: np.ndarray, split: dict, truth: np.ndarray
) -> tuple:
    """Every method's readout ``(W, 24, 3)`` (latent methods) and mean radius
    ``(W, 24)`` (all methods), plus the linear candidate's fit info."""
    latents = held_forecasts(z)
    latents["linear"], fit_info = linear_candidate(z, split)
    obs = {name: read_z(readout, lat) for name, lat in latents.items()}
    radius = {name: o[..., RADIUS] for name, o in obs.items()}
    readings = read_z(readout, z[:, :-HORIZON])[..., RADIUS]
    radius |= label_references(
        readings, truth, np.isin(np.arange(len(z)), split["train"])
    )
    return obs, radius, fit_info


def paired_gain(
    radius: dict, truth: np.ndarray, rows: np.ndarray, pair: tuple
) -> np.ndarray:
    """Per snippet, ``|ref - truth| - |method - truth|`` at +24 (positive = method better)."""
    ref, method = pair
    err = {k: np.abs(radius[k][rows, -1] - truth[rows, -1]) for k in pair}
    return err[ref] - err[method]


def exponents(radius: np.ndarray, meta: list[dict], rows: np.ndarray, ctx: int) -> dict:
    """Median exponent over +1..+24 per process, clip-bootstrap CI."""
    full = np.full((len(radius), ctx + HORIZON), np.nan)
    full[:, ctx:] = radius
    out = {}
    for process, pairs in snippet_exponents(full, meta, rows).items():
        values, clips = zip(*pairs, strict=True)
        out[process] = {
            "median": float(np.nanmedian(values)),
            "ci": bootstrap_ci(values, clips, np.nanmedian),
        }
    return out


def mae_table(radius: dict, truth: np.ndarray, rows: np.ndarray) -> dict:
    return {
        name: {
            f"+{h}": float(np.abs(r[rows, h - 1] - truth[rows, h - 1]).mean())
            for h in REPORT_HORIZONS
        }
        for name, r in radius.items()
    }


def w2_table(obs: dict, true_obs: np.ndarray, rows: np.ndarray) -> dict:
    """Gaussian 2-Wasserstein size distance (moments from count, radius, area) per
    latent method; the true moments come from the same three numbers, exactly."""

    def w2(o, t):
        return gaussian_w2(*moments_from_observables(o), *moments_from_observables(t))

    return {
        name: {
            f"+{h}": float(np.mean([w2(o[i, h - 1], true_obs[i, h - 1]) for i in rows]))
            for h in REPORT_HORIZONS
        }
        for name, o in obs.items()
    }


# ------------------------------- verdict ----------------------------------------


def growth_test(
    radius: dict, truth: np.ndarray, rows: np.ndarray, clips: np.ndarray
) -> dict:
    """On one process: the reference, whether the floor is powered over it, and every
    other method's paired gain over it at +24 with a clip CI."""
    mae = {
        k: float(np.abs(radius[k][rows, -1] - truth[rows, -1]).mean())
        for k in REFERENCES
    }
    ref = min(mae, key=mae.get)
    out = {"reference": ref, "reference_mae_24": mae[ref]}
    for method in (m for m in radius if m != ref):
        gain = paired_gain(radius, truth, rows, (ref, method))
        out[method] = {
            "gain": float(gain.mean()),
            "ci": bootstrap_ci(gain, clips[rows]),
        }
    out["powered"] = out["floor"]["ci"][0] > 0
    return out


def gate_result(size: dict, exponent_ok: dict, static_ok: bool) -> str:
    tested = [v for v in size.values() if v is not None]
    if not tested:
        return "untestable"
    return "PASS" if all(tested) and all(exponent_ok.values()) and static_ok else "FAIL"


def verdict(tests: dict, exps: dict, candidate: str) -> dict:
    """Gate 2' for one candidate from the per-process growth tests and exponents."""
    size = {
        p: (tests[p][candidate]["ci"][0] > 0) if tests[p]["powered"] else None
        for p in JUDGED
    }
    exp_ok = {
        p: abs(exps[candidate][p]["median"] - exps["true"][p]) <= EXPONENT_TOL
        for p in JUDGED
    }
    static_ok = abs(exps[candidate]["static"]["median"]) <= EXPONENT_TOL
    return {
        "size_pass": size,
        "exponent_pass": exp_ok,
        "static_exponent_pass": static_ok,
        "gap_closed_to_readout_ridge": {
            p: tests[p][candidate]["gain"] / tests[p]["readout_ridge"]["gain"]
            for p in JUDGED
        },
        "result": gate_result(size, exp_ok, static_ok),
    }


def tables_by_process(obs: dict, radius: dict, targets: np.ndarray, rows_of: dict) -> dict:
    """Reported, not judged: mean-radius MAE (all methods) and Gaussian W2 (latent
    methods) per process and horizon. ``rows_of``: process -> test physics windows."""
    return {
        "mae_px": {p: mae_table(radius, targets[:, -HORIZON:, RADIUS], rows) for p, rows in rows_of.items()},
        "w2_px": {p: w2_table(obs, targets[:, -HORIZON:], rows) for p, rows in rows_of.items()},
    }


def evaluate(readout: Readout, z: np.ndarray, ctx: dict, extra: dict | None = None) -> dict:
    """Forecast, read and judge everything on the test physics windows. ``extra``:
    further candidates' z-scored ``(W, 24, D)`` forecasts (P3), judged alike."""
    meta, targets = ctx["meta"], ctx["targets"]
    obs, radius, fit_info = observables(readout, z, ctx["split"], targets[..., RADIUS])
    for name, forecast in (extra or {}).items():
        obs[name] = read_z(readout, forecast)
        radius[name] = obs[name][..., RADIUS]
    candidates = ("linear", *(extra or {}))
    test = ctx["split"]["test"][ctx["physics"][ctx["split"]["test"]]]
    process = np.array([m["process"] for m in meta])
    rows_of = {p: test[process[test] == p] for p in sorted(set(process))}
    clips = np.array([m["clip"] for m in meta])
    future = targets[:, -HORIZON:, RADIUS]
    tests = {p: growth_test(radius, future, rows_of[p], clips) for p in JUDGED}
    ctx_len = z.shape[1] - HORIZON
    exps = {name: exponents(r, meta, test, ctx_len) for name, r in radius.items()}
    exps["true"] = {p: next(m.get("exponent") for m in meta if m["process"] == p) for p in rows_of}
    return {
        "linear_fit": fit_info,
        "growth_test": tests,
        "verdict": {c: verdict(tests, exps, c) for c in candidates},
        "exponents": exps,
    } | tables_by_process(obs, radius, targets, rows_of)


def print_growth_tests(result: dict) -> None:
    for p, t in result["growth_test"].items():
        print(f"\n{p}: reference {t['reference']} (MAE +24 {t['reference_mae_24']:.2f} px), powered {t['powered']}")
        for m in ("floor", "readout_ridge", *result["verdict"]):
            g = t[m]
            print(f"    gain over reference, {m:14s} {g['gain']:+.2f} [{g['ci'][0]:+.2f}, {g['ci'][1]:+.2f}] px")


def print_exponents(result: dict) -> None:
    shown = ("diffusion", "reaction", "static")
    print("\nmedian exponent over +1..+24   " + "  ".join(f"{p:>11s}" for p in shown))
    for name, e in result["exponents"].items():
        if name != "true":
            print(f"    {name:14s}             " + "  ".join(f"{e[p]['median']:+11.3f}" for p in shown))


def print_report(result: dict) -> None:
    print_growth_tests(result)
    print_exponents(result)
    print("\nmean-radius MAE (px)            " + "".join(f"{'+' + str(h):>8s}" for h in REPORT_HORIZONS))
    for p, table in result["mae_px"].items():
        for i, (name, by_h) in enumerate(table.items()):
            print(f"  {p if i == 0 else '':12s}{name:16s}" + "".join(f"{v:8.2f}" for v in by_h.values()))
    for c, v in result["verdict"].items():
        print(f"\nGate 2' [{c}]: {v['result']}  size {v['size_pass']}  exponent {v['exponent_pass']}"
              f"  static {v['static_exponent_pass']}  gap closed to readout ridge {v['gap_closed_to_readout_ridge']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pooled", type=Path, required=True)
    ap.add_argument("--gate1", type=Path, required=True)
    ap.add_argument("--readout", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pooled, ctx = load_context(args.pooled, args.gate1)
    readout = load_readout(args.readout)
    z = (pooled["mean"].astype(np.float32) - readout.x_mu) / readout.x_sd
    result = evaluate(readout, z, ctx)
    print_report(result)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(result, indent=1))
    print(f"\nwrote {args.out}/metrics.json")


if __name__ == "__main__":
    main()
