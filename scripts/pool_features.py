"""Pool the per-frame feature cache into position-free summaries (PLAN Step 10,
position-free forecasting, stage P0).

Half the particles leave their 16 px token within one latent step
(``experiments/token-persistence/``), so a deterministic forecast of the token grid
erases them. Sizes, counts and growth do not depend on where particles are. This job
averages each latent frame's 16x16 tokens away, with no labels, into three summaries
of increasing spatial detail:

    mean   (W, T, D)     average over all 256 tokens
    std    (W, T, D)     per-channel spread over the tokens (kept beside the mean)
    grid2  (W, T, 4, D)  average over each 8x8-token quadrant (row-major)

stored as fp16, plus the ground truth the later stages grade against:

    targets  (W, T, 3)  [count, mean radius, total area] of the crop-visible particles
    meta     JSON list, the cache's per-window meta (clip, start, process, stratum,
             t0_growth, dt_growth, ...), in the same order as the arrays

Disk-bound, no GPU. The output is ~1.6 GB against the cache's ~68 GB, small enough to
copy to the Mac, where every later stage runs.

Usage (remote, CPU):
    uv run python -u scripts/pool_features.py --features data/features/dynamics-v1-perframe \
        --out data/features/dynamics-v1-perframe-pooled.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from train_probes import WindowCache, attentive_targets

from jepa_lptem.eval.centroid_eval import GRID


def pool(feats: np.ndarray) -> dict:
    """``(T, GRID*GRID, D)`` tokens -> ``mean (T, D)``, ``std (T, D)``, ``grid2 (T, 4, D)``."""
    t, _, d = feats.shape
    half = GRID // 2
    quads = feats.reshape(t, 2, half, 2, half, d).mean(axis=(2, 4))
    return {
        "mean": feats.mean(axis=1),
        "std": feats.std(axis=1),
        "grid2": quads.reshape(t, 4, d),
    }


def pool_cache(cache: WindowCache) -> dict:
    """Every window's pooled features (fp16) and GT targets, in cache order."""
    feats, _, _, _ = cache.load(0)
    t, _, d = feats.shape
    n = len(cache.files)
    out = {
        "mean": np.empty((n, t, d), np.float16),
        "std": np.empty((n, t, d), np.float16),
        "grid2": np.empty((n, t, 4, d), np.float16),
        "targets": np.empty((n, t, 3), np.float32),
    }
    start = time.time()
    for i in range(n):
        feats, _, rad, _ = cache.load(i)
        for key, value in pool(feats).items():
            out[key][i] = value
        out["targets"][i] = attentive_targets(rad)
        if (i + 1) % 250 == 0:
            print(f"{i + 1}/{n} windows, {time.time() - start:.0f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cache, out = WindowCache(args.features), args.out
    pooled = pool_cache(cache)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **pooled, meta=json.dumps(cache.meta))
    print(f"wrote {out}: {len(cache.files)} windows")


if __name__ == "__main__":
    main()
