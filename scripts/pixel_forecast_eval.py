"""Step 11 P3: judge the ConvLSTM pixel baseline against candidate A (PLAN Step 11).

Test clips, physics stratum, the snippets of ``pooled-forecast`` (context = latent
frames 0-7 of each cached window, forecast = 8-31, i.e. video frames s+16..s+63).

- **Judged** (re-stated verdict): per snippet, the growth-exponent error
  ``|alpha - alpha_true|``. The ConvLSTM is read with the radial-profile radius at the
  true future centres (8-frame blocks; P0b v2 sent the reading there), candidate A with
  its pooled readout. Latent beats pixel iff the median of (ConvLSTM error - A error)
  has a clip-bootstrap CI above 0 on diffusion and reaction; pixel beats latent iff
  below 0 on both; otherwise tied.
- **Reported:** the same pixel reading on the true future frames (what the instrument
  can give); dark mass; the pooled path (``pooled_forecast.evaluate`` unchanged, with
  the ConvLSTM as an extra candidate) and the attentive readout, both under P0's
  warning that V-JEPA cannot fairly read a smooth forecast; pixel MSE against
  persistence; context 16 (unpaired, against ``longer-context`` design A).

    uv run python scripts/pixel_forecast_eval.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from dark_mass import background, latent_masses, radius_exponent
from eval_forecasts import snippet_exponent
from gate2_followup import bootstrap_ci
from pixel_premise import PIXEL_BLOCK, pixel_reading
from pooled_forecast import evaluate, load_readout, observables
from pooled_gate1 import JUDGED, RADIUS, load_context

LOG = logging.getLogger(__name__)
STARTS = (0, 36)
CTX_FRAMES = 16
FORECAST = 48
TUBELET = 2
LONG_CTX_END = 52  # ctx16 forecasts video frames 52..99
MSE_FRAMES = (1, 10, 20, 48)  # forecast frames at which pixel MSE is reported
PIXEL_MAX = 255.0


def load_encoded(run: Path) -> dict:
    """The encode shards (one per GPU) merged; ``encoded.npz`` if there are none."""
    shards = sorted(run.glob("encoded-*of*.npz")) or [run / "encoded.npz"]
    out = {}
    for path in shards:
        z = np.load(path)
        out |= {k: z[k] for k in z.files}
    return out


def load_inputs(args) -> dict:
    tube_meta = [
        json.loads(line)["clip"] for line in (args.tubelets / "meta.jsonl").open()
    ]
    forecasts = np.load(args.run / "forecasts.npz")
    test_clips = sorted({k.rsplit("/", 1)[0] for k in forecasts.files})
    pooled, ctx = load_context(args.pooled, args.gate1)
    readout = load_readout(args.readout)
    return {
        "forecasts": forecasts,
        "encoded": load_encoded(args.run),
        "gt": np.load(args.tubelets / "gt.npz"),
        "gt_row": {c: i for i, c in enumerate(tube_meta)},
        # test_frames.npy: the true frames of the test clips, in sorted clip order
        "frames": np.load(args.tubelets / "test_frames.npy", mmap_mode="r"),
        "frame_row": {c: i for i, c in enumerate(test_clips)},
        "ctx": ctx,
        "readout": readout,
        "z": (pooled["mean"].astype(np.float32) - readout.x_mu) / readout.x_sd,
    }


def growth_time(m: dict, frames: np.ndarray) -> np.ndarray:
    return m["t0_growth"] + m["dt_growth"] * np.asarray(frames, dtype=float)


def pixel_columns(frames: dict, gt: tuple, m: dict, first: int) -> dict:
    """Per stand-in (``conv``, ``true``): the radius at the true centres per 8-frame
    block, and its exponent, each block timed at its middle frame. ``first``: video
    frame of the first forecast frame."""
    out = {}
    for name, label in (("conv", "convlstm"), ("true", "true")):
        radius = pixel_reading(frames[name], *gt)
        mid = first + np.arange(len(radius)) * PIXEL_BLOCK + (PIXEL_BLOCK - 1) / 2
        out[f"radius/{label}"] = radius
        out[f"pixel/{label}"] = snippet_exponent(growth_time(m, mid), radius)
    return out


def dark_exponent(frames: np.ndarray, ref: np.ndarray, m: dict, first: int) -> float:
    latent_first = first + TUBELET * np.arange(len(frames) // TUBELET)
    return radius_exponent(
        growth_time(m, latent_first), latent_masses(frames, background(ref))
    )


def future_gt(inputs: dict, clip: str, first: int) -> tuple:
    """True ``(positions, radii)`` at the latent times of video frames first..first+47."""
    g = inputs["gt_row"][clip]
    lat = slice(first // TUBELET, (first + FORECAST) // TUBELET)
    return inputs["gt"]["positions"][g][:, lat], inputs["gt"]["radii"][g][:, lat]


def pixel_mse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    d = (pred.astype(np.float32) - true.astype(np.float32)) / PIXEL_MAX
    return (d**2).mean(axis=(1, 2))


def snippet_frames(inputs: dict, m: dict) -> dict:
    """One ctx8 snippet's frames: the ConvLSTM forecast, the true future, the last
    true context frame, and the true future centres."""
    clip, first = m["clip"], m["start"] + CTX_FRAMES
    clip_frames = inputs["frames"][inputs["frame_row"][clip]]
    return {
        "conv": inputs["forecasts"][f"{clip}/ctx8"][STARTS.index(m["start"])],
        "true": np.asarray(clip_frames[first : first + FORECAST]),
        "ref": np.asarray(clip_frames[first - 1]),
        "gt": future_gt(inputs, clip, first),
    }


def snippet_row(inputs: dict, w: int, linear_radius: np.ndarray) -> dict:
    """Every exponent of one ctx8 snippet (window ``w`` of the pooled meta)."""
    m = inputs["ctx"]["meta"][w]
    f, first = snippet_frames(inputs, m), m["start"] + CTX_FRAMES
    latent_t = growth_time(m, first + TUBELET * np.arange(FORECAST // TUBELET))
    attentive = inputs["encoded"][f"{m['clip']}/ctx8/attentive"][
        STARTS.index(m["start"])
    ]
    return {
        "clip": m["clip"],
        "process": m["process"],
        "truth": m.get("exponent", np.nan),  # only growth processes have one
        **pixel_columns(f, f["gt"], m, first),
        "dark/convlstm": dark_exponent(f["conv"], f["ref"], m, first),
        "dark/true": dark_exponent(f["true"], f["ref"], m, first),
        "attentive/convlstm": snippet_exponent(latent_t, attentive[:, RADIUS]),
        "pooled/linear": snippet_exponent(latent_t, linear_radius[w]),
        "mse/convlstm": pixel_mse(f["conv"], f["true"]),
        "mse/persistence": pixel_mse(f["ref"][None], f["true"]),
    }


def long_rows(inputs: dict, test: np.ndarray) -> list[dict]:
    """The ctx16 snippet of every test physics clip."""
    by_clip = {inputs["ctx"]["meta"][w]["clip"]: inputs["ctx"]["meta"][w] for w in test}
    return [long_row(inputs, clip, m) for clip, m in by_clip.items()]


def long_row(inputs: dict, clip: str, meta: dict) -> dict:
    """The ctx16 snippet (frames 20..51 -> 52..99) of one test clip, pixel reading."""
    frames = {
        "conv": inputs["forecasts"][f"{clip}/ctx16"],
        "true": np.asarray(inputs["frames"][inputs["frame_row"][clip], LONG_CTX_END:]),
    }
    gt = future_gt(inputs, clip, LONG_CTX_END)
    return {
        "clip": clip,
        "process": meta["process"],
        **pixel_columns(frames, gt, meta, LONG_CTX_END),
    }


def median_ci(values, clips) -> dict:
    return {
        "median": float(np.nanmedian(values)),
        "ci": bootstrap_ci(values, clips, np.nanmedian),
        "n": int(np.isfinite(values).sum()),
    }


def exponent_table(rows: list[dict], keys: list[str]) -> dict:
    out = {}
    for p in sorted({r["process"] for r in rows}):
        sub = [r for r in rows if r["process"] == p]
        clips = [r["clip"] for r in sub]
        out[p] = {k: median_ci(np.array([r[k] for r in sub]), clips) for k in keys}
    return out


def judged(rows: list[dict]) -> dict:
    """Paired (ConvLSTM error - A error) per process, and the verdict."""
    out = {}
    for p in JUDGED:
        sub = [r for r in rows if r["process"] == p]
        diff = np.array(
            [
                abs(r["pixel/convlstm"] - r["truth"])
                - abs(r["pooled/linear"] - r["truth"])
                for r in sub
            ]
        )
        out[p] = median_ci(diff, [r["clip"] for r in sub])
    above = all(out[p]["ci"][0] > 0 for p in JUDGED)
    below = all(out[p]["ci"][1] < 0 for p in JUDGED)
    out["verdict"] = (
        "latent beats pixel" if above else "pixel beats latent" if below else "tied"
    )
    return out


def radius_table(rows: list[dict]) -> dict:
    """Median radius at the true centres per 8-frame block, per process: a growing
    reading on static clips is blur, not growth."""
    return {
        p: {
            k: np.nanmedian(
                [r[f"radius/{k}"] for r in rows if r["process"] == p], axis=0
            ).tolist()
            for k in ("convlstm", "true")
        }
        for p in sorted({r["process"] for r in rows})
    }


def mse_table(rows: list[dict]) -> dict:
    return {
        name: {
            f"frame +{f}": float(np.mean([r[f"mse/{name}"][f - 1] for r in rows]))
            for f in MSE_FRAMES
        }
        for name in ("convlstm", "persistence")
    }


def pooled_path(inputs: dict, test: np.ndarray) -> dict:
    """``pooled_forecast.evaluate`` with the ConvLSTM's encoded forecasts as a candidate."""
    readout, meta = inputs["readout"], inputs["ctx"]["meta"]
    extra = np.zeros(
        (len(meta), FORECAST // TUBELET, inputs["z"].shape[-1]), np.float32
    )
    for w in test:
        m = meta[w]
        pooled = inputs["encoded"][f"{m['clip']}/ctx8/pooled"][STARTS.index(m["start"])]
        extra[w] = (pooled - readout.x_mu) / readout.x_sd
    result = evaluate(readout, inputs["z"], inputs["ctx"], {"convlstm": extra})
    return {k: result[k] for k in ("growth_test", "verdict", "exponents", "mae_px")}


def test_physics(ctx: dict) -> np.ndarray:
    test = ctx["split"]["test"]
    return test[ctx["physics"][test]]


def evaluate_all(inputs: dict) -> dict:
    ctx = inputs["ctx"]
    test = test_physics(ctx)
    _, radius, _ = observables(
        inputs["readout"], inputs["z"], ctx["split"], ctx["targets"][..., RADIUS]
    )
    rows = [snippet_row(inputs, w, radius["linear"]) for w in test]
    LOG.info("ctx8: %d snippets", len(rows))
    keys = [
        "pixel/convlstm",
        "pixel/true",
        "pooled/linear",
        "dark/convlstm",
        "dark/true",
        "attentive/convlstm",
    ]
    return {
        "n_snippets": len(rows),
        "judged": judged(rows),
        "exponents_ctx8": exponent_table(rows, keys),
        "radius_px_by_block_ctx8": radius_table(rows),
        "pixel_mse_ctx8": mse_table(rows),
        "exponents_ctx16": exponent_table(
            long_rows(inputs, test), ["pixel/convlstm", "pixel/true"]
        ),
        "pooled_path": pooled_path(inputs, test),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, default=Path("experiments/pixel-baseline"))
    ap.add_argument("--tubelets", type=Path, default=Path("data/tubelets/dynamics-v1"))
    ap.add_argument(
        "--pooled",
        type=Path,
        default=Path("data/features/dynamics-v1-perframe-pooled.npz"),
    )
    ap.add_argument("--gate1", type=Path, default=Path("experiments/gate1-perframe"))
    ap.add_argument(
        "--readout", type=Path, default=Path("experiments/pooled-gate1/readout.npz")
    )
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    result = evaluate_all(load_inputs(args))
    (args.run / "metrics.json").write_text(json.dumps(result, indent=1))
    LOG.info(
        json.dumps(
            {"judged": result["judged"], "pixel_mse": result["pixel_mse_ctx8"]},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
