"""Step 11 P0: premise check for the pixel-space baseline (PLAN Step 11), on the Mac.

Fresh physics-stratum clips, encoded per frame. Each window's future frames (16-63)
are replaced by a pixel-space stand-in and read three ways:

- ``pooled``  frame mean of the tokens -> the frozen ridge readout (pooled-gate1)
- ``token``   the Gate-1 attentive probe on the tokens (gate1-perframe)
- ``pixel``   the radial-profile radius at the true particle centres, 8-frame blocks

P0.1: holding the last context frame in pixels must read like holding its latent.
P0.2: what Gaussian blur of the true future does to each reading (the failure mode of
an MSE forecaster), with the stop rule on the pooled mean radius at sigma 2.

    uv run python scripts/generate_dynamics_dataset.py --out data/synthetic/pixel-premise \
        --n-videos 100 --n-frames 100 --seed 7
    uv run python scripts/pixel_premise.py --clips data/synthetic/pixel-premise \
        --out experiments/pixel-baseline
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from compare_to_real import apparent_radius_px
from extract_features import TUBELET, crop_and_align_gt, make_encoder, pick_device
from gate2_followup import bootstrap_ci
from pooled_forecast import load_readout
from train_probes import attentive_targets, load_probes

LOG = logging.getLogger(__name__)
CROP = 256
WINDOW = 64
STARTS = (0, 36)
CTX_FRAMES = 16
CTX_LAT = CTX_FRAMES // TUBELET
HORIZONS = (1, 5, 10, 24)
SIGMAS = (1, 2, 4, 8)
PIXEL_BLOCK = 8  # frames per pixel-path reading: one radial profile pools k frames
RADIUS = 1  # [count, mean_radius, total_area]
P01_TOL_PX = 0.1
STOP_SIGMA = 2
STOP_SHIFT_PX = 0.42  # candidate A's +24 gain on diffusion (pooled-forecast)


def stand_ins(future: np.ndarray, last: np.ndarray) -> dict:
    """Pixel-space forecasts of the future frames: held last frame, and blurred truth."""
    from scipy.ndimage import gaussian_filter

    out = {"true": future, "pixel_persistence": np.repeat(last[None], len(future), 0)}
    for s in SIGMAS:
        blurred = gaussian_filter(future.astype(np.float32), sigma=(0, s, s))
        out[f"blur_{s}"] = np.clip(np.rint(blurred), 0, 255).astype(np.uint8)
    return out


def pixel_reading(frames: np.ndarray, pos: np.ndarray, rad: np.ndarray) -> np.ndarray:
    """Radial-profile radius per block of frames, at the true centres of each frame's
    latent time (crop coordinates, ``(row, col)``)."""

    def centres(_frame, i):
        lat = i // TUBELET
        return pos[rad[:, lat] > 0, lat]

    blocks = range(0, len(frames), PIXEL_BLOCK)
    return np.array(
        [
            apparent_radius_px(
                frames[b:], lambda f, i, b=b: centres(f, b + i), k=PIXEL_BLOCK
            )
            for b in blocks
        ]
    )


class Readers:
    """The two latent readings, frozen: pooled ridge and attentive probe."""

    def __init__(self, readout: Path, probes: Path, device: str):
        self.readout = load_readout(readout)
        self.att, _, (self.mu, self.sd) = load_probes(probes, device)
        self.device = device

    def __call__(self, tokens: np.ndarray) -> dict:
        x = torch.from_numpy(tokens.astype(np.float32)).to(self.device)
        with torch.no_grad():
            token = self.att(x).cpu().numpy() * self.sd + self.mu
        return {
            "pooled": self.readout.predict(tokens.astype(np.float32).mean(1)),
            "token": token,
        }


def window_record(win: np.ndarray, gt: tuple, encode, read) -> dict:
    """Every reading of one window: true context and future, and each stand-in."""
    pos, rad = gt
    future_pos, future_rad = pos[:, CTX_LAT:], rad[:, CTX_LAT:]
    true_read = read(encode(win))
    rec = {
        "truth": attentive_targets(rad),
        "context": {k: v[:CTX_LAT] for k, v in true_read.items()},
    }
    for name, frames in stand_ins(win[CTX_FRAMES:], win[CTX_FRAMES - 1]).items():
        latent = true_read if name == "true" else read(encode(frames))
        rec[name] = {k: v[-len(frames) // TUBELET :] for k, v in latent.items()}
        rec[name]["pixel"] = pixel_reading(frames, future_pos, future_rad)
    return rec


def clip_windows(path: Path):
    """``(window frames, (positions, radii) at latent times)`` for each window."""
    z = np.load(path)
    frames, size = z["frames"], z["frames"].shape[-1]
    c = (size - CROP) // 2
    for start in STARTS:
        win = frames[start : start + WINDOW, c : c + CROP, c : c + CROP]
        yield (
            win,
            crop_and_align_gt(z["positions"], z["radii"], start, WINDOW, size, CROP),
        )


def collect(clips: list[Path], encode, read) -> list[dict]:
    records = []
    for i, path in enumerate(clips):
        process = path.stem.split("_")[2]
        for win, gt in clip_windows(path):
            records.append(
                {
                    "clip": path.stem,
                    "process": process,
                    **window_record(win, gt, encode, read),
                }
            )
        LOG.info("[%d/%d] %s", i + 1, len(clips), path.stem)
    return records


def radius_errors(records: list[dict], method: str) -> np.ndarray:
    """``(W, 24)`` absolute mean-radius error of the pooled reading."""
    truth = np.stack([r["truth"][CTX_LAT:, RADIUS] for r in records])
    if method == "latent_persistence":
        held = np.stack([r["context"]["pooled"][-1, RADIUS] for r in records])
        return np.abs(held[:, None] - truth)
    return np.abs(np.stack([r[method]["pooled"][:, RADIUS] for r in records]) - truth)


def persistence_check(records: list[dict]) -> dict:
    """P0.1: pixel-held minus latent-held pooled mean-radius MAE, paired, per horizon."""
    clips = [r["clip"] for r in records]
    diff = radius_errors(records, "pixel_persistence") - radius_errors(
        records, "latent_persistence"
    )
    out = {
        f"+{h}": {
            "diff_px": float(diff[:, h - 1].mean()),
            "ci": bootstrap_ci(diff[:, h - 1], clips),
        }
        for h in HORIZONS
    }
    out["pass"] = all(abs(v["diff_px"]) <= P01_TOL_PX for v in out.values())
    return out


def shift(records: list[dict], variant: str, path: str, col: int | None) -> dict:
    """Mean over windows of (variant - true) reading, over all forecast frames."""

    def per_window(r):
        a, b = r[variant][path], r["true"][path]
        a, b = (a, b) if col is None else (a[:, col], b[:, col])
        return float(np.nanmean(a - b))

    values = [per_window(r) for r in records]
    return {
        "mean": float(np.nanmean(values)),
        "ci": bootstrap_ci(values, [r["clip"] for r in records]),
    }


def blur_response(records: list[dict]) -> dict:
    """P0.2: reading shift per blur level and path."""
    out = {}
    for s in SIGMAS:
        v = f"blur_{s}"
        out[v] = {
            f"{p}_{name}": shift(records, v, p, j)
            for p in ("pooled", "token")
            for j, name in enumerate(("count", "mean_radius", "area"))
        }
        out[v]["pixel_mean_radius"] = shift(records, v, "pixel", None)
    return out


def analyse(records: list[dict]) -> dict:
    blur = blur_response(records)
    stop = abs(blur[f"blur_{STOP_SIGMA}"]["pooled_mean_radius"]["mean"]) > STOP_SHIFT_PX
    return {
        "n_windows": len(records),
        "n_clips": len({r["clip"] for r in records}),
        "p0_1_persistence": persistence_check(records),
        "p0_2_blur": blur,
        "stop_rule_fired": stop,
        "judged_path": "pixel" if stop else "pooled",
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument(
        "--readout", type=Path, default=Path("experiments/pooled-gate1/readout.npz")
    )
    ap.add_argument(
        "--probes", type=Path, default=Path("experiments/gate1-perframe/probes.pt")
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    device = pick_device()
    clips = sorted(args.clips.glob("*_physics.npz"))[: args.limit]
    records = collect(
        clips,
        make_encoder(device, per_frame=True),
        Readers(args.readout, args.probes, device),
    )
    result = analyse(records)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "premise.json").write_text(json.dumps(result, indent=1))
    LOG.info(
        json.dumps(
            {k: result[k] for k in ("p0_1_persistence", "stop_rule_fired")}, indent=1
        )
    )


if __name__ == "__main__":
    main()
