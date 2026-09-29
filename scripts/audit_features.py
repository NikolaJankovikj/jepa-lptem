"""Audit a Step-6 feature cache before any probe is trained on it.

Three levels, cheapest first:

1. **Integrity** -- every window loads, has the expected shape, is finite and
   not constant, and every visible particle lies inside the crop.
2. **Temporal alignment** -- the two windows of a clip overlap in time
   (frames 36-63), so the tokens they give for the *same* frames must agree far
   better than tokens one latent step apart. Catches an off-by-one in windowing.
3. **Spatial alignment** -- a linear classifier reads "this patch holds a
   particle centre" off single tokens, trained and tested on disjoint clips.
   The control repeats it with the labels rolled 5 cells away; it must collapse
   toward chance. A high real score with a collapsed control means the ground
   truth sits on the tokens that actually see the particle -- the property the
   processor resize bug broke.

Usage (remote, after extraction; ~10 min for 4000 windows):
    uv run python scripts/audit_features.py data/features/dynamics-v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from train_probes import GRID, WindowCache, centroid_targets, split_by_clip

CONTROL_ROLL = 5  # cells; far enough that a rolled label never covers its particle
FRAMES_PER_WINDOW = 8  # latent frames sampled per window for the linear probe


def check_integrity(cache: WindowCache) -> None:
    shapes, bad, out_of_crop, in_view = set(), [], 0, []
    for i, f in enumerate(cache.files):
        feats, pos, rad, meta = cache.load(i)
        shapes.add(feats.shape)
        if not np.isfinite(feats).all() or feats.std() == 0:
            bad.append(Path(f).name)
        visible = rad > 0
        out_of_crop += int(((pos[visible] < 0) | (pos[visible] >= meta["crop"])).sum())
        in_view.append(visible.sum(axis=0).mean())
    print(f"[integrity] windows {len(cache.files)}  shapes {shapes}")
    print(f"[integrity] non-finite or constant windows: {len(bad)} {bad[:3]}")
    print(f"[integrity] visible particles outside the crop: {out_of_crop} (want 0)")
    print(f"[integrity] particles in view per frame: mean {np.mean(in_view):.1f}")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return float((a * b).sum(axis=-1).mean())


def window_pairs(cache: WindowCache, n_clips: int) -> list[list[int]]:
    """Up to ``n_clips`` clips' window indices, ordered by start, for clips with
    more than one window."""
    by_clip: dict[str, list[int]] = {}
    for i, m in enumerate(cache.meta):
        by_clip.setdefault(m["clip"], []).append(i)
    return [sorted(v, key=lambda i: cache.meta[i]["start"])
            for v in by_clip.values() if len(v) > 1][:n_clips]


def check_temporal(cache: WindowCache, n_clips: int) -> None:
    pairs = window_pairs(cache, n_clips)
    if not pairs:
        print("[temporal] no clip has two windows -- skipped")
        return
    same, shifted = [], []
    for first, second in (p[:2] for p in pairs):
        a, b = cache.load(first)[0], cache.load(second)[0]
        m = cache.meta[first]
        lag = (cache.meta[second]["start"] - m["start"]) // m["frames_per_latent"]
        overlap = len(a) - lag
        same.append(cosine(a[lag:], b[:overlap]))
        shifted.append(cosine(a[lag + 1 :], b[: overlap - 1]))
    print(
        f"[temporal] same frames, two windows: cos {np.mean(same):.3f}   "
        f"one latent step apart: cos {np.mean(shifted):.3f}   (want same > shifted)"
    )


def token_samples(cache, idx, roll: int, rng):
    """Balanced (token, label) pairs: every particle cell plus as many empty cells."""
    xs, ys = [], []
    for i in idx:
        feats, pos, rad, meta = cache.load(i)
        occ = np.roll(centroid_targets(pos, rad, meta["crop"])[0], roll, axis=(1, 2))
        for t in rng.choice(
            len(feats), min(FRAMES_PER_WINDOW, len(feats)), replace=False
        ):
            flat = occ[t].reshape(-1)
            pos_cells = np.flatnonzero(flat)
            neg_cells = rng.choice(
                np.flatnonzero(flat == 0), len(pos_cells), replace=False
            )
            cells = np.concatenate([pos_cells, neg_cells])
            xs.append(feats[t][cells])
            ys.append(flat[cells])
    return np.concatenate(xs), np.concatenate(ys)


def split_windows(cache: WindowCache, n_windows: int, rng, seed: int):
    """A random sample of windows, split by clip exactly as train_probes splits
    them -- the probe is scored on clips it never saw."""
    split = split_by_clip([m["clip"] for m in cache.meta], seed=seed)
    test_clips = split["val"] | split["test"]
    idx = rng.permutation(len(cache.files))[:n_windows]
    train = [i for i in idx if cache.meta[i]["clip"] not in test_clips]
    test = [i for i in idx if cache.meta[i]["clip"] in test_clips]
    return train, test


def check_spatial(cache: WindowCache, n_windows: int, seed: int = 0) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    train, test = split_windows(cache, n_windows, rng, seed)
    for name, roll in (
        ("real labels", 0),
        (f"control, rolled {CONTROL_ROLL}", CONTROL_ROLL),
    ):
        x_tr, y_tr = token_samples(cache, train, roll, rng)
        x_te, y_te = token_samples(cache, test, roll, rng)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        auc = roc_auc_score(y_te, clf.fit(x_tr, y_tr).predict_proba(x_te)[:, 1])
        print(
            f"[spatial] {name:20s} held-out AUC {auc:.3f}  "
            f"({len(y_tr)} train / {len(y_te)} test tokens)"
        )
    print(
        f"[spatial] want: real well above the control; control near 0.5 ({GRID}x{GRID} grid)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("features", type=Path)
    ap.add_argument("--probe-windows", type=int, default=400)
    ap.add_argument("--overlap-clips", type=int, default=100)
    args = ap.parse_args()
    cache = WindowCache(args.features)
    check_integrity(cache)
    check_temporal(cache, args.overlap_clips)
    check_spatial(cache, args.probe_windows)


if __name__ == "__main__":
    main()
