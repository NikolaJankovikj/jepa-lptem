"""Longer context: do the pooled forecasts' growth exponents rise when they see more?

PLAN Step 10, *Longer context*, fixed 2026-10-11 before it ran. At context 8 the
label-free candidates' exponents (0.18-0.27) sit inside the context-only ceiling
(`experiments/exponent-ceiling/`), so the shortfall is either the data (8 noisy
frames) or the model. More context separates the two: the ceiling rises with it, and
the question is whether the candidates rise with it.

Per-frame features encode each tubelet alone, so a clip's two cached windows (video
frames 0-63 and 36-99) stitch into one 50-latent-frame sequence; the 14 shared frames
agree exactly. Snippets are cut from it so that every context of a design forecasts
the *same* frames, which makes the contexts a paired comparison:

    A  contexts 8, 12, 16   forecast begins at latent frame 16, 21, 26   (primary)
    B  contexts 8, 16, 24   forecast begins at latent frame 24, 26       (reported)

On each snippet set:

- ``ceiling``  simulator ground truth only (as `exponent_ceiling.py`): ridge and kNN
               on the context's mean radius plus N(0, sd) noise, sd 0 and 0.8 px
- candidates A (``linear``) and B (``transformer``), unchanged from P2/P3 apart from
  the context length, judged by ``pooled_forecast.evaluate`` (Gate 2' criteria)

Primary reading, design A: a candidate **tracks the context** iff the per-snippet
paired difference of its exponent over +1..+24, context 16 minus context 8, has a
median with a clip-bootstrap CI above 0 on both diffusion and reaction.

Runs on the Mac (the ceiling in seconds; the candidates in ~20 min, mostly P3 training):
    uv run python scripts/longer_context.py --out experiments/longer-context --ceiling-only
    uv run python scripts/longer_context.py --out experiments/longer-context \
        --pooled data/features/dynamics-v1-perframe-pooled.npz \
        --gate1 experiments/gate1-perframe --readout experiments/pooled-gate1/readout.npz
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path

import numpy as np
from eval_forecasts import snippet_exponent
from exponent_ceiling import context_forecasts, median_exponents, trajectories
from extract_features import crop_and_align_gt
from gate2_followup import bootstrap_ci
from pooled_forecast import evaluate, linear_candidate, load_readout, read_z
from pooled_gate1 import HORIZON, JUDGED, RADIUS, load_context, split_windows
from pooled_transformer import forecast_all, train
from token_persistence import CROP, FULL_SIZE, N_FRAMES, TUBELET

from jepa_lptem.sim.dataset import PROCESSES, build_clip

DESIGNS = {
    "A": {"contexts": (8, 12, 16), "forecast_starts": (16, 21, 26)},
    "B": {"contexts": (8, 16, 24), "forecast_starts": (24, 26)},
}
PRIMARY = ("A", 8, 16)
SECONDARY = ("B", 8, 24)
SECOND_START = 36  # video frame of each clip's second cached window
CEILING_SD = (0.0, 0.8)
CEILING_METHODS = ("ridge", "knn")
CANDIDATES = ("linear", "transformer")
SHOWN = ("diffusion", "reaction", "static")


@dataclass
class Stitched:
    """Whole-clip pooled sequences ``(C, 50, D)``, their targets ``(C, 50, 3)``, the
    clips' meta, and the Gate-1 run whose by-clip split every snippet inherits."""

    seq: np.ndarray
    targets: np.ndarray
    meta: list[dict]
    gate1: Path


# ------------------------------ snippets ----------------------------------------


def stitch(features: np.ndarray, targets: np.ndarray, meta: list[dict]) -> tuple:
    """One ``(C, 50, ...)`` sequence per clip from its windows at video frames 0 and
    36, and the clips' meta (in sorted clip order)."""
    index = {(m["clip"], m["start"]): i for i, m in enumerate(meta)}
    clips = sorted({m["clip"] for m in meta})
    first = [index[c, 0] for c in clips]
    second = [index[c, SECOND_START] for c in clips]
    new = (
        features.shape[1] - SECOND_START // TUBELET
    )  # second window's first unseen frame
    return (
        np.concatenate([features[first], features[second][:, new:]], axis=1),
        np.concatenate([targets[first], targets[second][:, new:]], axis=1),
        [meta[i] for i in first],
    )


def snippet_frames(n_seq: int, ctx: int, starts) -> tuple[np.ndarray, np.ndarray]:
    """Row ``(S, 1)`` and frame ``(S, ctx + HORIZON)`` indices of every snippet whose
    forecast begins at a latent frame in ``starts``, sequence-major, so every context
    of a design lines up row by row."""
    seq = np.repeat(np.arange(n_seq), len(starts))[:, None]
    first = np.tile(np.asarray(starts) - ctx, n_seq)
    return seq, first[:, None] + np.arange(ctx + HORIZON)


def cut(clips: Stitched, ctx: int, starts) -> tuple:
    """Pooled snippets, their targets and meta; ``start`` is the snippet's first video
    frame, as in the window cache."""
    seq, frames = snippet_frames(len(clips.seq), ctx, starts)
    meta = [
        clips.meta[c] | {"start": int(f[0]) * clips.meta[c]["frames_per_latent"]}
        for c, f in zip(seq[:, 0], frames, strict=True)
    ]
    return clips.seq[seq, frames], clips.targets[seq, frames], meta


# ------------------------------ ceiling -----------------------------------------


def gt_clips(n_per_process: int, seed: int) -> list[dict]:
    """Physics-stratum ground truth over whole clips, crop-aligned at latent times."""
    rng = np.random.default_rng(seed)
    rows = []
    for clip, process in enumerate(p for p in PROCESSES for _ in range(n_per_process)):
        _, _, meta, pos, rad = build_clip(
            process, N_FRAMES, (FULL_SIZE, FULL_SIZE), rng
        )
        p, r = crop_and_align_gt(pos, rad, 0, N_FRAMES, FULL_SIZE, CROP)
        rows.append({
            "process": process,
            "clip": clip,
            "start": 0,
            "pos": p,
            "rad": r,
            "t0_growth": meta["t0_growth"],
            "dt_growth": meta["dt_growth"],
        })
    return rows


def cut_trajectories(traj: dict, ctx: int, starts) -> dict:
    """The ground-truth trajectories cut into the same snippets as the pooled cache."""
    seq, frames = snippet_frames(len(traj["truth"]), ctx, starts)
    return {
        "truth": traj["truth"][seq, frames],
        "times": traj["times"][seq, frames],
        "process": traj["process"][seq[:, 0]],
        "train": traj["train"][seq[:, 0]],
    }


def ceiling(snip: dict, ctx: int, seed: int) -> dict:
    """Context-only extrapolators' median exponents on one snippet set; ``best_at_sd``
    is, per process, whichever method lands closer to the truth at the probe's noise."""
    rng = np.random.default_rng(seed)
    out = {"truth": median_exponents(snip["truth"][:, ctx:], snip, ctx)}
    for sd in CEILING_SD:
        noisy = snip["truth"] + rng.normal(0.0, sd, snip["truth"].shape)
        forecasts = context_forecasts(
            np.maximum(noisy, 1.0), snip["truth"], snip["train"], ctx
        )
        out[f"sd_{sd}"] = {
            m: median_exponents(forecasts[m], snip, ctx) for m in CEILING_METHODS
        }
    at_sd = out[f"sd_{CEILING_SD[-1]}"]
    out["best_at_sd"] = {
        p: min((at_sd[m][p] for m in CEILING_METHODS), key=lambda e: abs(e - true))
        for p, true in out["truth"].items()
    }
    return out


def run_ceiling(n_per_process: int, seed: int) -> dict:
    traj = trajectories(gt_clips(n_per_process, seed), seed)
    return {
        name: {
            f"context_{c}": ceiling(
                cut_trajectories(traj, c, d["forecast_starts"]), c, seed
            )
            for c in d["contexts"]
        }
        for name, d in DESIGNS.items()
    }


# ------------------------------ candidates --------------------------------------


def snippet_values(
    radius: np.ndarray, meta: list[dict], rows: np.ndarray, ctx: int
) -> np.ndarray:
    """Per snippet in ``rows``, the exponent of its ``(24,)`` forecast over +1..+24."""
    frames = np.arange(ctx, ctx + HORIZON)
    out = []
    for i in rows:
        m = meta[i]
        times = m["t0_growth"] + m["dt_growth"] * (
            m["start"] + m["frames_per_latent"] * frames
        )
        out.append(snippet_exponent(times, radius[i]))
    return np.array(out)


def fit_candidates(z: np.ndarray, split: dict) -> tuple[dict, int]:
    """Candidates A and B's z-scored ``(W, 24, D)`` forecasts, and B's selected epoch."""
    model, history = train(z, split)
    best = min(history, key=itemgetter("val_l1_24"))["epoch"]
    forecasts = {
        "linear": linear_candidate(z, split)[0],
        "transformer": forecast_all(model, z),
    }
    return forecasts, best


def run_context(clips: Stitched, readout, ctx: int, starts) -> tuple[dict, dict]:
    """Train and judge both candidates on one snippet set. Returns the Gate 2' result
    and the per-snippet exponents of the test physics snippets, for the paired test."""
    raw, targets, meta = cut(clips, ctx, starts)
    z = (raw.astype(np.float32) - readout.x_mu) / readout.x_sd
    split = split_windows(meta, clips.gate1)
    physics = np.array([m["stratum"] == "physics" for m in meta])
    forecasts, best = fit_candidates(z, split)
    context = {"targets": targets, "split": split, "meta": meta, "physics": physics}
    result = evaluate(
        readout, z, context, extra={"transformer": forecasts["transformer"]}
    )
    rows = split["test"][physics[split["test"]]]
    values = {
        "process": np.array([meta[i]["process"] for i in rows]),
        "clips": np.array([meta[i]["clip"] for i in rows]),
    } | {
        n: snippet_values(read_z(readout, f)[..., RADIUS], meta, rows, ctx)
        for n, f in forecasts.items()
    }
    return result | {"n_snippets": len(z), "transformer_best_epoch": best}, values


def tracks(short: dict, long: dict) -> dict:
    """Per candidate and process, the median per-snippet exponent rise (long context
    minus short, same forecast frames) with a clip CI; ``tracks`` iff its CI lies above
    0 on every judged process."""
    if not np.array_equal(short["clips"], long["clips"]):
        msg = "contexts are not aligned snippet by snippet"
        raise ValueError(msg)
    out = {}
    for name in CANDIDATES:
        rise = long[name] - short[name]
        per = {
            p: {
                "median": float(np.nanmedian(rise[short["process"] == p])),
                "ci": bootstrap_ci(
                    rise[short["process"] == p],
                    short["clips"][short["process"] == p],
                    np.nanmedian,
                ),
            }
            for p in SHOWN
        }
        out[name] = per | {"tracks": all(per[p]["ci"][0] > 0 for p in JUDGED)}
    return out


def run_candidates(clips: Stitched, readout) -> dict:
    results, values = {}, {}
    for name, design in DESIGNS.items():
        for c in design["contexts"]:
            print(f"\n=== design {name}, context {c}", flush=True)
            results.setdefault(name, {})[f"context_{c}"], values[name, c] = run_context(
                clips, readout, c, design["forecast_starts"]
            )
            print_candidates(results[name][f"context_{c}"])
    readings = {}
    for design, short, long in (PRIMARY, SECONDARY):
        readings[f"{design}_{short}_to_{long}"] = tracks(
            values[design, short], values[design, long]
        )
    return {"designs": results, "tracks": readings}


# ------------------------------- report -----------------------------------------


def print_ceiling(ceil: dict) -> None:
    print(
        "\nceiling: median exponent over +1..+24     "
        + "".join(f"{p:>11s}" for p in SHOWN)
    )
    for name, by_ctx in ceil.items():
        for key, block in by_ctx.items():
            rows = {"truth": block["truth"]} | {
                f"{m} sd {sd}": block[f"sd_{sd}"][m]
                for sd in CEILING_SD
                for m in CEILING_METHODS
            }
            for label, e in rows.items():
                print(
                    f"  {name} {key:11s} {label:26s}"
                    + "".join(f"{e[p]:11.3f}" for p in SHOWN)
                )


def print_candidates(result: dict) -> None:
    for c in CANDIDATES:
        e, v = result["exponents"][c], result["verdict"][c]
        gains = {p: result["growth_test"][p][c] for p in JUDGED}
        print(
            f"  {c:12s} exponent "
            + " ".join(f"{p} {e[p]['median']:+.3f}" for p in SHOWN)
            + "  gain +24 "
            + " ".join(
                f"{p} {g['gain']:+.2f} [{g['ci'][0]:+.2f}, {g['ci'][1]:+.2f}]"
                for p, g in gains.items()
            )
            + f"  -> Gate 2' criteria {v['result']}"
        )


def print_tracks(readings: dict) -> None:
    for key, by_candidate in readings.items():
        print(f"\nexponent rise, design {key} (paired, median [clip CI])")
        for c, per in by_candidate.items():
            print(
                f"  {c:12s} "
                + "  ".join(
                    f"{p} {per[p]['median']:+.3f} [{per[p]['ci'][0]:+.3f}, {per[p]['ci'][1]:+.3f}]"
                    for p in SHOWN
                )
                + f"  -> tracks: {per['tracks']}"
            )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ceiling-only", action="store_true")
    ap.add_argument("--pooled", type=Path)
    ap.add_argument("--gate1", type=Path)
    ap.add_argument("--readout", type=Path)
    ap.add_argument("--n-per-process", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ceil = run_ceiling(args.n_per_process, args.seed)
    print_ceiling(ceil)
    (args.out / "ceiling.json").write_text(json.dumps(ceil, indent=1))
    if args.ceiling_only:
        return
    pooled, ctx = load_context(args.pooled, args.gate1)
    clips = Stitched(*stitch(pooled["mean"], ctx["targets"], ctx["meta"]), args.gate1)
    result = run_candidates(clips, load_readout(args.readout))
    print_tracks(result["tracks"])
    (args.out / "metrics.json").write_text(json.dumps(result, indent=1))
    print(f"\nwrote {args.out}/ceiling.json, metrics.json")


if __name__ == "__main__":
    main()
