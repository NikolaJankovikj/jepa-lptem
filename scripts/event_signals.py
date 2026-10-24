"""RQ2, the PC part: per-token signals around merges, for event_anticipation.py.

PLAN Step 12, fixed 2026-10-24. For every coalescence clip of the chosen split parts,
per latent frame t and token (16 x 16, row-major):

- ``model``        predictor-perframe teacher-forced on frames t-8..t-1: L1 between its
                   forecast of t and the true frame, mean over channels (standardized)
- ``persistence``  |x_t - x_(t-1)|, the same L1
- ``ctxmean``      |x_t - mean(x_(t-8..t-1))|
- ``pixel``        |change| of each token's mean of P_t, the tubelet's two frames averaged

NaN where a frame has no context. Also the clip's ground truth in crop coordinates
(centres, radii zeroed outside the crop, merges). Reads the cache and the clips; trains
nothing. Writes one file (~10 MB) to rsync back. On the PC:
    uv run python -u scripts/event_signals.py --clips data/dynamics-v1 \
        --features data/features/dynamics-v1-perframe \
        --predictor experiments/predictor-perframe/predictor.pt \
        --gate1 experiments/gate1-perframe --out data/events/signals-v1.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from eval_forecasts import load_predictor
from train_predictor import pick_device, standardize

TUBELET = 2
GRID = CELL = 16
BATCH = 8


def select_clips(clips_dir: Path, gate1: Path, parts) -> list[Path]:
    """Coalescence clip files whose clip is in the Gate-1 split ``parts``."""
    split = json.loads((gate1 / "metrics.json").read_text())["split"]
    keep = {c for p in parts for c in split[p]}
    return sorted(
        f for f in clips_dir.glob("clip_*_coalescence_*.npz") if f.stem in keep
    )


def latent_sequence(features: Path, stem: str, n_lat: int) -> np.ndarray:
    """``(n_lat, 256, D)`` features of one clip from its cached windows; a frame
    two windows share is identical in a per-frame cache, so the first one is kept."""
    seq, filled = None, np.zeros(n_lat, bool)
    for path in sorted(features.glob(f"{stem}__w*.npz")):
        first = int(path.stem.split("__w")[1]) // TUBELET
        with np.load(path) as z:
            feats = z["features"]
        if seq is None:
            seq = np.zeros((n_lat, *feats.shape[1:]), np.float32)
        frames = np.arange(first, first + len(feats))
        new = ~filled[frames]
        seq[frames[new]] = feats[new]
        filled[frames] = True
    if seq is None or not filled.all():
        msg = f"{stem}: cached windows do not cover all {n_lat} latent frames"
        raise SystemExit(msg)
    return seq


def pixel_signal(frames: np.ndarray) -> np.ndarray:
    """``(T_video, crop, crop)`` uint8 -> ``(T_lat, 256)`` |change| of each token's mean.

    The token mean comes first: a mean of per-pixel |change| is all shot noise and
    cannot even tell particles from background (PLAN Step 12, premise v1)."""
    t_lat = len(frames) // TUBELET
    tube = frames[: t_lat * TUBELET].astype(np.float32)
    tube = tube.reshape(t_lat, TUBELET, *frames.shape[1:]).mean(axis=1)
    side = frames.shape[1] // GRID
    tokens = (
        tube.reshape(t_lat, GRID, side, GRID, side).mean(axis=(2, 4)).reshape(t_lat, -1)
    )
    change = np.full_like(tokens, np.nan)
    change[1:] = np.abs(tokens[1:] - tokens[:-1])
    return change


def model_signals(model, z: np.ndarray, ctx: int, device: str) -> dict:
    """Teacher-forced +1 error and the two held baselines, ``(T, 256)`` each, on a
    standardized sequence ``z`` ``(T, 256, D)``."""
    t_lat = len(z)
    out = {
        k: np.full((t_lat, z.shape[1]), np.nan, np.float32)
        for k in ("model", "persistence", "ctxmean")
    }
    targets = np.arange(ctx, t_lat)
    for chunk in np.array_split(targets, max(1, len(targets) // BATCH)):
        context = torch.from_numpy(np.stack([z[t - ctx : t] for t in chunk])).to(device)
        with torch.no_grad():
            forecast = model(context).float().cpu().numpy()
        out["model"][chunk] = np.abs(forecast - z[chunk]).mean(-1)
    out["persistence"][1:] = np.abs(z[1:] - z[:-1]).mean(-1)
    for t in targets:
        out["ctxmean"][t] = np.abs(z[t] - z[t - ctx : t].mean(0)).mean(-1)
    return out


def crop_gt(positions: np.ndarray, radii: np.ndarray, full: int, crop: int) -> tuple:
    """Centres in crop coordinates and radii zeroed where the centre is outside."""
    pos = positions - (full - crop) / 2.0
    rad = np.where(((pos < 0) | (pos >= crop)).any(-1), 0.0, radii)
    return pos.astype(np.float32), rad.astype(np.float32)


def clip_record(path: Path, args, model_and_stats) -> dict:
    model, (mean, std), ctx = model_and_stats
    with np.load(path) as clip:
        frames, positions, radii = clip["frames"], clip["positions"], clip["radii"]
        merges = np.asarray(clip["merge_events"], dtype=int).reshape(-1, 3)
        meta = {k: clip[k].item() for k in ("stratum", "beam_flux") if k in clip.files}
    full, off = frames.shape[1], (frames.shape[1] - args.crop) // 2
    n_lat = len(frames) // TUBELET
    z = standardize(latent_sequence(args.features, path.stem, n_lat), mean, std)
    record = model_signals(model, z, ctx, args.device)
    record["pixel"] = pixel_signal(
        frames[:, off : off + args.crop, off : off + args.crop]
    )
    record["pos"], record["rad"] = crop_gt(positions, radii, full, args.crop)
    return record | {"merges": merges, "meta": {"clip": path.stem} | meta}


def pad(arrays: list, fill) -> np.ndarray:
    """Stack arrays that differ in their first axis, padding it with ``fill``."""
    n = max(len(a) for a in arrays)
    out = np.full((len(arrays), n, *arrays[0].shape[1:]), fill, arrays[0].dtype)
    for i, a in enumerate(arrays):
        out[i, : len(a)] = a
    return out


def save(records: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        **{
            k: np.stack([r[k] for r in records]).astype(np.float16)
            for k in ("model", "persistence", "ctxmean", "pixel")
        },
        pos=pad([r["pos"] for r in records], 0.0),
        rad=pad([r["rad"] for r in records], 0.0),
        merges=pad([r["merges"] for r in records], -1),
        meta=json.dumps([r["meta"] for r in records]),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--predictor", type=Path, required=True)
    ap.add_argument("--gate1", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parts", nargs="+", default=["val", "test"])
    ap.add_argument("--crop", type=int, default=256)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.device = pick_device()
    model_and_stats = load_predictor(args.predictor, args.device)
    paths = select_clips(args.clips, args.gate1, args.parts)
    print(
        f"{len(paths)} coalescence clips in {args.parts}, device {args.device}",
        flush=True,
    )
    records = []
    for i, path in enumerate(paths):
        records.append(clip_record(path, args, model_and_stats))
        print(
            f"  [{i + 1}/{len(paths)}] {path.stem}: {len(records[-1]['merges'])} merges",
            flush=True,
        )
    save(records, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
