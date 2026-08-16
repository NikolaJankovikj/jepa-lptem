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
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
TUBELET = 2  # input frames per latent frame


def window_starts(t_total: int, window: int, stride: int) -> list[int]:
    """Start indices of complete windows; empty if the clip is too short."""
    if t_total < window:
        return []
    return list(range(0, t_total - window + 1, stride))


def crop_and_align_gt(positions: np.ndarray, radii: np.ndarray, start: int,
                      window: int, full_size: int, crop: int):
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


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=36)
    ap.add_argument("--crop", type=int, default=256)
    args = ap.parse_args()

    import torch
    from transformers import VJEPA2Model, VJEPA2VideoProcessor

    device = pick_device()
    model = VJEPA2Model.from_pretrained(MODEL_NAME).eval().to(device)
    processor = VJEPA2VideoProcessor.from_pretrained(MODEL_NAME)

    clips = sorted(glob.glob(str(args.data / "clip_*.npz")))
    args.out.mkdir(parents=True, exist_ok=True)
    n_windows = 0
    with (args.out / "manifest.jsonl").open("w") as mf:
        for ci, f in enumerate(clips):
            z = np.load(f)
            frames = z["frames"]  # (T, H, W) uint8
            t_total, full_size = frames.shape[0], frames.shape[1]
            starts = window_starts(t_total, args.window, args.stride)
            if not starts:
                print(f"[skip] {Path(f).name}: {t_total} frames < window {args.window}")
                continue
            off = (full_size - args.crop) // 2
            for start in starts:
                win = frames[start:start + args.window,
                             off:off + args.crop, off:off + args.crop]
                rgb = np.repeat(win[..., None], 3, axis=-1)
                inp = processor(list(rgb), return_tensors="pt").to(device)
                with torch.no_grad():
                    h = model(**inp).last_hidden_state  # (1, T_lat*256, D)
                t_lat = args.window // TUBELET
                feats = h.reshape(t_lat, 256, -1).half().cpu().numpy()
                pos, rad = crop_and_align_gt(z["positions"], z["radii"], start,
                                             args.window, full_size, args.crop)
                meta = {k: z[k].item() for k in z.files
                        if k not in ("frames", "positions", "radii", "drift",
                                     "merge_events")
                        and z[k].ndim == 0}
                meta.update(clip=Path(f).stem, start=start,
                            frames_per_latent=TUBELET, crop=args.crop,
                            drift=[float(d) for d in z["drift"]])
                name = f"{Path(f).stem}__w{start:03d}.npz"
                np.savez_compressed(args.out / name, features=feats,
                                    positions=pos, radii=rad,
                                    meta=json.dumps(meta))
                mf.write(json.dumps({"file": name, **meta}) + "\n")
                n_windows += 1
            print(f"[{ci + 1}/{len(clips)}] {Path(f).stem}: {len(starts)} window(s)")

    print(f"\nWrote {n_windows} windows to {args.out} "
          f"({n_windows * (args.window // TUBELET)} latent frames)")


if __name__ == "__main__":
    main()
