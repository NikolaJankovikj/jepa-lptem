"""Step 6: extract frozen V-JEPA 2 features from dataset clips into a cache.

One forward pass per window, saved once, trained on many times: every probe and
predictor downstream reads this cache instead of re-running the 326M-param
encoder. Each output ``.npz`` holds the features *and* the ground truth aligned
to them, so Step 7 never has to re-derive the correspondence:

    features   fp16 (T_lat, 256, 1024)  one row of 16x16 tokens per latent frame
    positions  float (N, T_lat, 2)      crop-frame (row, col) at the sampled frames
    radii      float (N, T_lat)         0 = absent (outside crop or dissolved)
    + every scalar from the source clip (process, D, drift, msd_max_lag, ...)

Alignment choices, made once here:

- **Crop, never resize** (Step 4g): a center ``--crop`` window is cut from each
  frame, and the ground truth is shifted into crop coordinates. A particle whose
  centre leaves the crop gets radius 0 at that frame -- same convention as the
  visibility floor, so probes are never graded on a target that is not in view.
- **Latent frame t = input frame 2t** (tubelet depth 2). GT is sampled at the
  even frame of each tubelet pair. Per-latent-frame drift is therefore
  ``2 * drift``; consumers get ``frames_per_latent`` in the meta rather than
  hardcoding it.
- Windowing default 64/36 per the plan (100-frame videos -> 2 windows covering
  frames 0-63 and 36-99); both are CLI args so the 48-frame smoke clips work
  locally with ``--window 48``.

Usage (local smoke, MPS ~2 min; remote runs the same command on the full set):
    uv run python scripts/extract_features.py --data data/synthetic/smoke_v3 \
        --out data/features/smoke_v3 --window 48 --stride 48
Remote, one shard per GPU into the same --out (re-run the same command to resume):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/extract_features.py ... --shard 0/2
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/extract_features.py ... --shard 1/2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
TUBELET = 2  # input frames per latent frame


def window_starts(t_total: int, window: int, stride: int) -> list[int]:
    """Start indices of complete windows; empty if the clip is too short."""
    if t_total < window:
        return []
    return list(range(0, t_total - window + 1, stride))


def crop_and_align_gt(
    positions: np.ndarray,
    radii: np.ndarray,
    start: int,
    window: int,
    full_size: int,
    crop: int,
):
    """Ground truth for one window, in crop coordinates at latent-frame times.

    Samples every ``TUBELET``-th frame of ``[start, start+window)``, shifts
    positions by the center-crop offset, and zeroes the radius of any particle
    whose centre is outside the crop at that frame (absent-by-convention, like
    the visibility floor -- never a target that is not in the image).
    """
    t_idx = np.arange(start, start + window, TUBELET)
    off = (full_size - crop) / 2.0
    pos = positions[:, t_idx, :] - off
    rad = radii[:, t_idx].copy()
    outside = ((pos < 0) | (pos >= crop)).any(axis=-1)
    rad[outside] = 0.0
    return pos, rad


def load_processor():
    """The V-JEPA 2 processor with its resize and centre-crop switched off.

    Its defaults resize the shortest edge to 292 (256 * 256/224) and centre-crop
    back to 256, so a 256 px crop reaches the encoder zoomed 1.14x and trimmed by
    18 px per side. That breaks both halves of crop-never-resize (Step 4g):
    apparent particle size is rescaled, and the ground truth -- in crop
    coordinates -- sits up to 18 px (more than one 16 px token cell) from where
    the features put the particle. Only rescale + ImageNet normalization remain.
    """
    from transformers import VJEPA2VideoProcessor

    return VJEPA2VideoProcessor.from_pretrained(
        MODEL_NAME, do_resize=False, do_center_crop=False
    )


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def window_meta(z, clip: str, start: int, crop: int) -> dict:
    """Clip-level scalars plus where this window sits in its clip."""
    meta = {
        k: z[k].item()
        for k in z.files
        if k not in ("frames", "positions", "radii", "drift", "merge_events")
        and z[k].ndim == 0
    }
    meta.update(
        clip=clip,
        start=start,
        frames_per_latent=TUBELET,
        crop=crop,
        drift=[float(d) for d in z["drift"]],
    )
    return meta


def save_atomic(path: Path, **arrays) -> None:
    """Write via a temp file, so a crash never leaves a truncated window that a
    resumed run would then skip as done."""
    partial = path.with_name(path.name + ".tmp")
    with partial.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(partial, path)


def make_encoder(device: str):
    """The frozen encoder as a function: ``(T, crop, crop)`` uint8 frames -> fp16
    features ``(T / TUBELET, 256, D)``."""
    import torch
    from transformers import VJEPA2Model

    model = VJEPA2Model.from_pretrained(MODEL_NAME).eval().to(device)
    processor = load_processor()

    def encode(win: np.ndarray) -> np.ndarray:
        rgb = np.repeat(win[..., None], 3, axis=-1)
        inp = processor(list(rgb), return_tensors="pt").to(device)
        with torch.no_grad():
            h = model(**inp).last_hidden_state  # (1, T_lat*256, D)
        return h.reshape(len(win) // TUBELET, 256, -1).half().cpu().numpy()

    return encode


def extract_clip(path: Path, args, encode) -> int:
    """Encode every not-yet-cached window of one clip; returns windows written.

    Resumable: a window whose ``.npz`` exists is skipped, so a crashed multi-hour
    run restarts where it stopped. ``encode`` maps a ``(T, crop, crop)`` uint8
    window to fp16 features ``(T_lat, 256, D)``.
    """
    stem = path.stem
    z = np.load(path)
    t_total = z["radii"].shape[1]  # never decompress the frames of a done clip
    starts = window_starts(t_total, args.window, args.stride)
    todo = [s for s in starts if not (args.out / f"{stem}__w{s:03d}.npz").exists()]
    print(
        f"{stem}: {len(todo)} to encode, {len(starts) - len(todo)} cached"
        + ("" if starts else f" (< window {args.window}, skipped)")
    )
    if not todo:
        return 0
    frames = z["frames"]  # (T, H, W) uint8
    full_size = frames.shape[1]
    off = (full_size - args.crop) // 2
    for start in todo:
        feats = encode(
            frames[
                start : start + args.window,
                off : off + args.crop,
                off : off + args.crop,
            ]
        )
        pos, rad = crop_and_align_gt(
            z["positions"], z["radii"], start, args.window, full_size, args.crop
        )
        save_atomic(
            args.out / f"{stem}__w{start:03d}.npz",
            features=feats,
            positions=pos,
            radii=rad,
            meta=json.dumps(window_meta(z, stem, start, args.crop)),
        )
    return len(todo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=36)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument(
        "--shard",
        default="0/1",
        help="K/N: take every N-th clip starting at K -- one shard per GPU",
    )
    args = ap.parse_args()
    k, n = map(int, args.shard.split("/"))

    encode = make_encoder(pick_device())
    clips = sorted(glob.glob(str(args.data / "clip_*.npz")))[k::n]
    args.out.mkdir(parents=True, exist_ok=True)
    n_windows = sum(extract_clip(Path(f), args, encode) for f in clips)
    print(
        f"\nWrote {n_windows} windows from {len(clips)} clips to {args.out} "
        f"({n_windows * (args.window // TUBELET)} latent frames)"
    )


if __name__ == "__main__":
    main()
