"""Step 15 M1: raw-frame tubelets for the models that train on pixels (CNN, LoRA).

The feature cache holds V-JEPA outputs, which a CNN or a LoRA run cannot use. This
writes the centre crop of every clip once, uncompressed, so training reads tubelets
by memory map: Gate 1 spent ~90 % of its time decompressing windows.

    frames.npy    uint8 (C, T, crop, crop)   C clips, T input frames
    gt.npz        positions (C, N, T_lat, 2), radii (C, N, T_lat), 0 = absent,
                  padded to the largest N; attentive (C, T_lat, 3);
                  occupancy (C, T_lat, G, G); offsets (C, T_lat, G, G, 2)
    meta.jsonl    one line per clip, as the feature cache's window meta with start 0

Latent frame t is input frames ``2t, 2t+1``, the tubelet the per-frame cache
encodes; the ground truth is ``crop_and_align_gt`` over the whole clip, so cache
window ``w{s}`` latent j is latent ``s/2 + j`` here.

Usage:
    uv run python scripts/tubelet_cache.py --data data/synthetic/dynamics-v1 \
        --out data/tubelets/dynamics-v1
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from extract_features import TUBELET, crop_and_align_gt, window_meta
from train_probes import attentive_targets, centroid_targets


def clip_targets(z, crop: int) -> dict:
    """Ground truth of one whole clip at latent-frame times, crop coordinates."""
    t_total = z["radii"].shape[1]
    pos, rad = crop_and_align_gt(
        z["positions"], z["radii"], 0, t_total, z["frames"].shape[1], crop
    )
    occ, off, _ = centroid_targets(pos, rad, crop)
    return {
        "positions": pos,
        "radii": rad,
        "attentive": attentive_targets(rad),
        "occupancy": occ,
        "offsets": off,
    }


def pad_particles(arrays: list[np.ndarray]) -> np.ndarray:
    """Stack per-clip ``(N_i, ...)`` arrays into ``(C, max N, ...)``, zero-padded:
    a zero radius already means absent, so padding needs no mask."""
    n_max = max(a.shape[0] for a in arrays)
    out = np.zeros((len(arrays), n_max, *arrays[0].shape[1:]), arrays[0].dtype)
    for i, a in enumerate(arrays):
        out[i, : a.shape[0]] = a
    return out


def save_gt(out: Path, gts: list[dict]) -> None:
    np.savez(
        out / "gt.npz",
        positions=pad_particles([g["positions"] for g in gts]),
        radii=pad_particles([g["radii"] for g in gts]),
        **{
            k: np.stack([g[k] for g in gts])
            for k in ("attentive", "occupancy", "offsets")
        },
    )


def list_clips(clips_dir: Path) -> tuple[list[str], int, int]:
    """Clip paths, frames per clip and full frame size, read off the first clip."""
    paths = sorted(glob.glob(str(clips_dir / "clip_*.npz")))
    if not paths:
        msg = f"no clips in {clips_dir}"
        raise SystemExit(msg)
    with np.load(paths[0]) as z:
        t_total, full = z["frames"].shape[:2]
    return paths, t_total, full


def build(clips_dir: Path, out: Path, crop: int) -> int:
    paths, t_total, full = list_clips(clips_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = open_memmap(
        out / "frames.npy", "w+", np.uint8, (len(paths), t_total, crop, crop)
    )
    gts, metas = [], []
    off = (full - crop) // 2
    for i, p in enumerate(paths):
        with np.load(p) as z:
            if z["frames"].shape[:2] != (t_total, full):
                msg = f"{p}: shape differs from the first clip"
                raise SystemExit(msg)
            frames[i] = z["frames"][:, off : off + crop, off : off + crop]
            gts.append(clip_targets(z, crop))
            metas.append(window_meta(z, Path(p).stem, 0, crop))
    frames.flush()
    save_gt(out, gts)
    (out / "meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metas))
    return len(paths)


class TubeletSet:
    """Tubelets of the given clips as a torch-style dataset: item ``k`` is
    ``(frames (TUBELET, crop, crop) uint8, attentive (3,), occupancy (G, G),
    offsets (G, G, 2))``. Frames stay uint8; each model normalizes its own input."""

    def __init__(self, root: Path, clips: set[str] | None = None):
        self.frames = np.load(root / "frames.npy", mmap_mode="r")
        gt = np.load(root / "gt.npz")
        self.attentive, self.occupancy, self.offsets = (
            gt[k] for k in ("attentive", "occupancy", "offsets")
        )
        self.meta = [json.loads(line) for line in (root / "meta.jsonl").open()]
        rows = [
            i for i, m in enumerate(self.meta) if clips is None or m["clip"] in clips
        ]
        t_lat = self.attentive.shape[1]
        self.index = [(c, t) for c in rows for t in range(t_lat)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int):
        c, t = self.index[k]
        return (
            np.array(self.frames[c, TUBELET * t : TUBELET * (t + 1)]),
            self.attentive[c, t],
            self.occupancy[c, t],
            self.offsets[c, t],
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--crop", type=int, default=256)
    args = ap.parse_args()
    n = build(args.data, args.out, args.crop)
    print(f"wrote {n} clips to {args.out}")


if __name__ == "__main__":
    main()
