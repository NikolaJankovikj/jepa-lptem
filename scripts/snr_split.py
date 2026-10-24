"""RQ3: does the pooled forecast degrade at low dose?

PLAN Step 10, *RQ3 -- forecast quality vs dose*, fixed 2026-10-24 before any binned
number. The setting is longer-context design A (stitched clips, forecast frames
starting at latent frame 16, 21, 26), contexts 8 and 16, candidate A (linear) only,
trained on every dose as before and split only for scoring. Bins: beam flux below or
above sqrt(6 * 75) = 21.2, the geometric middle of the physics range (judged), and
thickness variation below or above 35 nm (reported). Per bin and process:

- reading noise of the frozen P1 readout on the true frames (MAE, bias, MAE / 0.80)
- the ground-truth ceiling (as `longer_context.py`) at that bin's reading noise
- mean-radius MAE by horizon, the linear candidate's +24 gain over the pooled trend,
  and the median exponent over +1..+24 of the linear candidate and the floor

Primary reading, context 16: per growing process, the linear candidate's +24 absolute
error, low-dose mean minus high-dose mean, with a two-sample clip CI. Forecasts
**degrade at low dose** iff that CI lies above 0. The floor's difference is the
readout's share.

Runs on the Mac in about a minute:
    uv run python scripts/snr_split.py --out experiments/snr \
        --pooled data/features/dynamics-v1-perframe-pooled.npz \
        --gate1 experiments/gate1-perframe --readout experiments/pooled-gate1/readout.npz
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from exponent_ceiling import trajectories
from gate2_followup import BOOT_SEED, N_BOOT, bootstrap_ci
from longer_context import (
    CEILING_METHODS,
    DESIGNS,
    SHOWN,
    Stitched,
    ceiling,
    cut,
    cut_trajectories,
    gt_clips,
    snippet_values,
    stitch,
)
from pooled_forecast import load_readout, mae_table, observables, paired_gain
from pooled_gate1 import HORIZON, JUDGED, RADIUS, load_context, split_windows
from train_probes import OBS_NAMES

SPLITS = {"beam_flux": float(np.sqrt(6.0 * 75.0)), "thickness_sd_nm": 35.0}
JUDGED_VARIABLE = "beam_flux"
BINS = ("low", "high")
STARTS = DESIGNS["A"]["forecast_starts"]
CONTEXTS = (8, 16)
PRIMARY_CONTEXT = 16
EXPONENT_METHODS = ("linear", "floor")
EFFECT_METHODS = ("linear", "floor")
EFFECT_HORIZONS = (24, 1)
MAE_TO_SD = 0.80  # MAE of zero-mean Gaussian noise is 0.80 sd (growth_identifiability)
COUNT = OBS_NAMES.index("count")


# ------------------------------- helpers ----------------------------------------


def diff_ci(a, clips_a, b, clips_b) -> list[float]:
    """95 % CI of mean(a) - mean(b), resampling whole clips within each group."""
    groups = []
    for values, clips in ((a, clips_a), (b, clips_b)):
        values = np.asarray(values, dtype=float)
        _, inverse = np.unique(np.asarray(clips), return_inverse=True)
        groups.append([values[inverse == k] for k in range(inverse.max() + 1)])
    rng = np.random.default_rng(BOOT_SEED)

    def draw(members):
        picks = rng.integers(0, len(members), len(members))
        return np.concatenate([members[k] for k in picks]).mean()

    stats = [draw(groups[0]) - draw(groups[1]) for _ in range(N_BOOT)]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def reading_noise(pred: np.ndarray, truth: np.ndarray, count: np.ndarray) -> dict:
    """The readout's mean-radius error on true frames that show a particle."""
    seen = count > 0
    err = pred[seen] - truth[seen]
    mae = float(np.abs(err).mean())
    return {
        "mae": mae,
        "bias": float(err.mean()),
        "sd_equivalent": mae / MAE_TO_SD,
        "n_frames": int(seen.sum()),
    }


def bins(meta: list[dict], rows: np.ndarray, key: str) -> dict:
    value = np.array([meta[i][key] for i in rows])
    return {"low": rows[value < SPLITS[key]], "high": rows[value >= SPLITS[key]]}


def by_process(meta: list[dict], rows: np.ndarray) -> dict:
    process = np.array([meta[i]["process"] for i in rows])
    return {p: rows[process == p] for p in SHOWN}


def test_physics(meta: list[dict], gate1: Path) -> np.ndarray:
    test = split_windows(meta, gate1)["test"]
    return test[[meta[i]["stratum"] == "physics" for i in test]]


# ---------------------------- reading noise -------------------------------------


def noise_by_bin(clips: Stitched, readout) -> dict:
    """Reading noise per variable, bin and process (and all processes pooled), on
    every frame of the test physics clips."""
    pred = readout.predict(clips.seq.astype(np.float32))[..., RADIUS]
    truth, count = clips.targets[..., RADIUS], clips.targets[..., COUNT]
    rows = test_physics(clips.meta, clips.gate1)
    out = {}
    for key in SPLITS:
        for b, in_bin in bins(clips.meta, rows, key).items():
            groups = by_process(clips.meta, in_bin) | {"all": in_bin}
            out.setdefault(key, {})[b] = {
                g: reading_noise(pred[r], truth[r], count[r]) for g, r in groups.items()
            }
    return out


def ceiling_by_bin(noise: dict, n_per_process: int, seed: int) -> dict:
    """The ground-truth ceiling at each dose bin's reading noise, plus noise-free."""
    traj = trajectories(gt_clips(n_per_process, seed), seed)
    sds = {"noise_free": 0.0} | {
        b: noise[JUDGED_VARIABLE][b]["all"]["sd_equivalent"] for b in BINS
    }
    out = {}
    for ctx in CONTEXTS:
        snip = cut_trajectories(traj, ctx, STARTS)
        block = {}
        for name, sd in sds.items():
            result = ceiling(snip, ctx, seed, sds=(sd,))
            block["truth"] = result["truth"]
            block[name] = result[f"sd_{sd}"] | {"sd": sd, "best": result["best_at_sd"]}
        out[f"context_{ctx}"] = block
    return out


# ------------------------------ forecasts ---------------------------------------


@dataclass
class Scored:
    """One snippet set's mean-radius forecasts ``(S, 24)`` per method, the truth, the
    per-snippet exponents (NaN outside the test physics rows), clips and meta."""

    radius: dict
    truth: np.ndarray
    exps: dict
    clip: np.ndarray
    meta: list[dict]


def score_snippets(clips: Stitched, readout, ctx: int) -> tuple[Scored, np.ndarray]:
    """Forecast and read every snippet of one context; returns the test physics rows."""
    raw, targets, meta = cut(clips, ctx, STARTS)
    z = (raw.astype(np.float32) - readout.x_mu) / readout.x_sd
    _, radius, _ = observables(
        readout, z, split_windows(meta, clips.gate1), targets[..., RADIUS]
    )
    test = test_physics(meta, clips.gate1)
    exps = {}
    for m in EXPONENT_METHODS:
        exps[m] = np.full(len(meta), np.nan)
        exps[m][test] = snippet_values(radius[m], meta, test, ctx)
    clip = np.array([m["clip"] for m in meta])
    return Scored(radius, targets[:, -HORIZON:, RADIUS], exps, clip, meta), test


def bin_entry(s: Scored, rows: np.ndarray) -> dict:
    gain = paired_gain(s.radius, s.truth, rows, ("pooled_trend", "linear"))
    return {
        "n_clips": len(set(s.clip[rows])),
        "n_snippets": len(rows),
        "mean_true_radius": float(s.truth[rows].mean()),
        "mae_px": mae_table(s.radius, s.truth, rows),
        "gain_linear_over_trend_24": {
            "gain": float(gain.mean()),
            "ci": bootstrap_ci(gain, s.clip[rows]),
        },
        "exponent": {
            m: {
                "median": float(np.nanmedian(s.exps[m][rows])),
                "ci": bootstrap_ci(s.exps[m][rows], s.clip[rows], np.nanmedian),
            }
            for m in EXPONENT_METHODS
        },
    }


def dose_effect(s: Scored, low: np.ndarray, high: np.ndarray) -> dict:
    """Low-bin minus high-bin absolute error at each EFFECT_HORIZONS, two-sample clip
    CI. +24 is the pre-registered one; +1 was added after the run (post hoc)."""
    out = {}
    for m in EFFECT_METHODS:
        for h in EFFECT_HORIZONS:
            err = np.abs(s.radius[m][:, h - 1] - s.truth[:, h - 1])
            out.setdefault(m, {})[f"+{h}"] = {
                "delta": float(err[low].mean() - err[high].mean()),
                "ci": diff_ci(err[low], s.clip[low], err[high], s.clip[high]),
            }
    return out


def entries(s: Scored, rows: np.ndarray) -> dict:
    return {p: bin_entry(s, r) for p, r in by_process(s.meta, rows).items()}


def score_context(clips: Stitched, readout, ctx: int) -> dict:
    s, test = score_snippets(clips, readout, ctx)
    out = {"all": entries(s, test)}
    for key in SPLITS:
        split_rows = bins(s.meta, test, key)
        out[key] = {b: entries(s, r) for b, r in split_rows.items()}
        low, high = (by_process(s.meta, split_rows[b]) for b in BINS)
        out[key]["effect"] = {p: dose_effect(s, low[p], high[p]) for p in SHOWN}
    return out


def reading(forecasts: dict) -> dict:
    effect = forecasts[f"context_{PRIMARY_CONTEXT}"][JUDGED_VARIABLE]["effect"]
    return {p: effect[p]["linear"]["+24"]["ci"][0] > 0 for p in JUDGED}


def run(args: argparse.Namespace) -> dict:
    pooled, ctx = load_context(args.pooled, args.gate1)
    clips = Stitched(*stitch(pooled["mean"], ctx["targets"], ctx["meta"]), args.gate1)
    readout = load_readout(args.readout)
    noise = noise_by_bin(clips, readout)
    forecasts = {f"context_{c}": score_context(clips, readout, c) for c in CONTEXTS}
    return {
        "splits": SPLITS,
        "reading_noise": noise,
        "ceiling": ceiling_by_bin(noise, args.n_per_process, args.seed),
        "ceiling_methods": CEILING_METHODS,
        "forecasts": forecasts,
        "degrades_at_low_dose": reading(forecasts),
    }


# ------------------------------- report -----------------------------------------


def fmt(v: dict) -> str:
    return f"{v['median' if 'median' in v else 'gain']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]"


def print_noise(noise: dict) -> None:
    print(
        "\nreading noise of the P1 readout on true frames (MAE / bias / sd-equivalent, px)"
    )
    for key, by_bin in noise.items():
        for b, groups in by_bin.items():
            print(
                f"  {key:16s} {b:5s} "
                + "  ".join(
                    f"{g} {n['mae']:.2f}/{n['bias']:+.2f}/{n['sd_equivalent']:.2f}"
                    for g, n in groups.items()
                )
            )


def print_ceiling(ceil: dict) -> None:
    print(
        "\nceiling at each bin's reading noise, best of ridge / kNN (median exponent)"
    )
    for ctx, block in ceil.items():
        print(
            f"  {ctx:11s} truth "
            + " ".join(f"{p} {block['truth'][p]:.3f}" for p in SHOWN)
        )
        for name in ("noise_free", *BINS):
            b = block[name]
            print(
                f"  {'':11s} {name:10s} sd {b['sd']:.2f}  "
                + " ".join(f"{p} {b['best'][p]:.3f}" for p in SHOWN)
            )


def print_bin(label: str, by_proc: dict) -> None:
    for p, e in by_proc.items():
        mae = e["mae_px"]
        print(
            f"  {label:5s} {p:10s} clips {e['n_clips']:3d}  MAE+24 linear {mae['linear']['+24']:.2f}"
            f" trend {mae['pooled_trend']['+24']:.2f} floor {mae['floor']['+24']:.2f}"
            f"  gain {fmt(e['gain_linear_over_trend_24'])}  exp linear {fmt(e['exponent']['linear'])}"
            f" floor {e['exponent']['floor']['median']:+.3f}"
        )


def print_split(block: dict, key: str) -> None:
    print(f"\n  split by {key} at {SPLITS[key]:.1f}")
    for b in BINS:
        print_bin(b, block[key][b])
    for p, eff in block[key]["effect"].items():
        print(
            f"  low - high {p:10s} "
            + "  ".join(
                f"{m} {h} {v['delta']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]"
                for m, by_h in eff.items()
                for h, v in by_h.items()
            )
        )


def print_report(result: dict) -> None:
    print_noise(result["reading_noise"])
    print_ceiling(result["ceiling"])
    for ctx, block in result["forecasts"].items():
        print(f"\n{ctx}")
        print_bin("all", block["all"])
        for key in SPLITS:
            print_split(block, key)
    print(
        f"\nprimary (context {PRIMARY_CONTEXT}): forecasts degrade at low dose {result['degrades_at_low_dose']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pooled", type=Path, required=True)
    ap.add_argument("--gate1", type=Path, required=True)
    ap.add_argument("--readout", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-per-process", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    result = run(args)
    print_report(result)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}/metrics.json")


if __name__ == "__main__":
    main()
