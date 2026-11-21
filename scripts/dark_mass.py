"""Dark mass: a position-free, blur-invariant pixel reading of particle volume (PLAN Step 11).

``M = Σ (B − I)`` over the frame, with ``B`` the snippet's background, taken from the
last context frame (see ``background``). A Gaussian blur conserves ``Σ I`` and leaves
``B`` alone, so ``M`` does not change when a forecaster blurs; for spheres of
constant count ``M ∝ Σ r³``, so the exponent of ``M`` over time, divided by 3, is the
radius exponent. ``log=True`` gives the Beer–Lambert form ``Σ log(B / I)``, which stays
proportional to volume when absorption saturates (the pre-registered fallback).

P0b (premise, ground truth only, seconds):
    uv run python scripts/dark_mass.py --clips data/synthetic/pixel-premise --limit 50 \
        --out experiments/pixel-baseline
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from gate2_followup import bootstrap_ci
from scipy.ndimage import gaussian_filter, median_filter

LOG = logging.getLogger(__name__)
MEDIAN_PX = 41
BACKGROUND_SIGMA = 10
VOLUME_TO_RADIUS = 3  # M ∝ r³
MIN_INTENSITY = 1.0  # uint8 zero-electron pixels; log needs > 0
CROP = 256
WINDOW = 64
STARTS = (0, 36)
CTX_FRAMES = 16
TUBELET = 2
SIGMAS = (1, 2, 4, 8)
MAX_JUDGED_SIGMA = 4  # blur invariance is required up to here (P0b ii)
JUDGED = ("diffusion", "reaction")
TRUTH_TOL = 0.1
BLUR_TOL = 0.03
HELD_TOL = 0.03


def background(reference: np.ndarray) -> np.ndarray:
    """Per-snippet background from the last context frame: a 41 px median (particles
    cover < 10 % of the crop, so the median ignores them) smoothed at 10 px. It is
    fixed before the forecast, so blurring a forecast cannot change it."""
    smooth = median_filter(reference.astype(np.float64), size=MEDIAN_PX, mode="reflect")
    return gaussian_filter(smooth, BACKGROUND_SIGMA)


def dark_mass(frame: np.ndarray, bg: np.ndarray, *, log: bool = False) -> float:
    f = frame.astype(np.float64)
    if log:
        return float(np.log(np.maximum(bg, MIN_INTENSITY) / np.maximum(f, MIN_INTENSITY)).sum())
    return float((bg - f).sum())


def latent_masses(frames: np.ndarray, bg: np.ndarray, *, log: bool = False) -> np.ndarray:
    """Dark mass per latent step: the mean over each tubelet's two frames."""
    m = np.array([dark_mass(f, bg, log=log) for f in frames])
    return m.reshape(-1, TUBELET).mean(axis=1)


def radius_exponent(times: np.ndarray, masses: np.ndarray) -> float:
    """Exponent of the radius from the log-log slope of the dark mass; NaN when the
    mass is not positive throughout (nothing to fit)."""
    if (masses <= 0).any():
        return float("nan")
    return float(np.polyfit(np.log(times), np.log(masses), 1)[0] / VOLUME_TO_RADIUS)


# --------------------------------- P0b ------------------------------------------


def snippet_times(z, start: int) -> np.ndarray:
    """Growth time of each forecast latent step (its tubelet's first frame)."""
    frames = start + CTX_FRAMES + TUBELET * np.arange((WINDOW - CTX_FRAMES) // TUBELET)
    return float(z["t0_growth"]) + float(z["dt_growth"]) * frames


def variants(future: np.ndarray, last: np.ndarray) -> dict:
    out = {"true": future, "held": np.repeat(last[None], len(future), 0)}
    for s in SIGMAS:
        out[f"blur_{s}"] = gaussian_filter(future.astype(np.float32), (0, s, s))
    return out


def clip_rows(path: Path) -> list[dict]:
    """Per window: exponent of each variant, linear and log."""
    z = np.load(path)
    frames, size = z["frames"], z["frames"].shape[-1]
    c = (size - CROP) // 2
    rows = []
    for start in STARTS:
        win = frames[start : start + WINDOW, c : c + CROP, c : c + CROP]
        times = snippet_times(z, start)
        row = {"clip": path.stem, "process": str(z["process"])}
        bg = background(win[CTX_FRAMES - 1])
        for name, f in variants(win[CTX_FRAMES:], win[CTX_FRAMES - 1]).items():
            for log in (False, True):
                key = f"{'log' if log else 'linear'}/{name}"
                row[key] = radius_exponent(times, latent_masses(f, bg, log=log))
        rows.append(row)
    return rows


def medians(rows: list[dict], key: str) -> dict:
    out = {}
    for process in sorted({r["process"] for r in rows}):
        sub = [r for r in rows if r["process"] == process]
        values = [r[key] for r in sub]
        out[process] = {
            "median": float(np.nanmedian(values)),
            "ci": bootstrap_ci(values, [r["clip"] for r in sub], np.nanmedian),
        }
    return out


def judge(table: dict, truth: dict) -> dict:
    """The three P0b criteria for one form (linear or log)."""
    true_ok = all(
        abs(table["true"][p]["median"] - truth[p]) <= TRUTH_TOL for p in JUDGED
    )
    blur_ok = all(
        abs(table[f"blur_{s}"][p]["median"] - table["true"][p]["median"]) < BLUR_TOL
        for s in SIGMAS
        if s <= 4
        for p in JUDGED
    )
    held_ok = all(abs(v["median"]) < HELD_TOL for v in table["held"].values())
    return {
        "truth": true_ok,
        "blur": blur_ok,
        "held": held_ok,
        "pass": true_ok and blur_ok and held_ok,
    }


def analyse(rows: list[dict], truth: dict) -> dict:
    out = {}
    for form in ("linear", "log"):
        names = ["true", "held", *(f"blur_{s}" for s in SIGMAS)]
        table = {n: medians(rows, f"{form}/{n}") for n in names}
        out[form] = {"exponents": table, "criteria": judge(table, truth)}
    out["reading"] = next(
        (f for f in ("linear", "log") if out[f]["criteria"]["pass"]),
        "radius_at_true_centres",
    )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    clips = sorted(args.clips.glob("*_physics.npz"))[: args.limit]
    rows = [row for path in clips for row in clip_rows(path)]
    truth = {"diffusion": 1 / 3, "reaction": 1 / 2}
    result = {"n_windows": len(rows)} | analyse(rows, truth)
    (args.out / "premise-dark-mass.json").write_text(json.dumps(result, indent=1))
    LOG.info(
        json.dumps(
            {f: result[f]["criteria"] for f in ("linear", "log")}
            | {"reading": result["reading"]}
        )
    )


if __name__ == "__main__":
    main()
