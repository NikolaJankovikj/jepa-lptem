"""Measure the sim-to-real gap where it actually matters: in V-JEPA's latent space.

Every pixel statistic in `compare_to_real.py` is a hand-picked proxy. The model
never sees pixels the way those statistics do -- it sees the frozen encoder's
features. So the direct question is: **do real clips land inside the synthetic
feature distribution?** If they do, appearance questions are settled regardless
of which pixel panels look off; if they don't, we know before spending GPU-days
that encoder adaptation (Step 8) is the main path, not a comparison run.

Two numbers, each with a same-pipeline calibration reference so it is
interpretable on its own:

1. **Containment ratio** -- median cosine distance from each real sample to its
   nearest synthetic neighbour, over the median nearest-neighbour distance
   *between* synthetic clips. ~1 means real sits on the synthetic feature
   manifold (as close to synthetic data as synthetic data is to itself);
   >> 1 means real is off-manifold. This is the latent-space version of the
   Step-4g containment criterion, and the number that matters for Gate 1.
   Reference: the same ratio for held-out synthetic clips (~1 by construction).

2. **Linear separability (AUC)** -- can a logistic regression tell real from
   synthetic features? Group-aware CV (grouped by clip / video) so it cannot
   cheat off same-clip near-duplicates. Reference: the same AUC for
   physics-vs-experiment *synthetic* strata. If two strata of our own generator
   separate at ~1.0, then real-vs-synthetic at ~1.0 is fine-grained
   distinguishability, not evidence of a harmful gap -- separable-but-adjacent
   is the expected outcome. The pair (AUC, ratio) disambiguates: harmful is
   high AUC *with* high ratio.

Caveat, stated once: "real" is effectively one video (Au nanorods). The
aggregation video is listed as a source but its frames are 200x220 -- below the
256 px crop, and resizing is the one preprocessing operation ruled out (it
rescales apparent particle size, Step 4g), so it contributes nothing here. One
video cannot represent all real LPTEM; Step 13 remains the arbiter of transfer.

Usage (Mac, ~6 min on MPS):
    uv run python scripts/latent_gap.py --syn data/synthetic/smoke_v3
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

REAL_SOURCES = [  # (zip, member predicate) -- one group each for the classifier CV
    ("data/sam-em/experimentalAuNanorods.zip", lambda x: x.endswith(".jpeg")),
    ("data/sam-em/experimentalAggregation.zip", lambda x: x.endswith(".jpeg")),
]
N_FRAMES = 16   # per sample; tubelet 2 -> 8 latent frames, well-formed for V-JEPA 2
TILE = 256      # the model's native resolution; crop, never resize (Step 4g)


def tile_windows(frames: np.ndarray, n_frames: int = N_FRAMES, tile: int = TILE):
    """Cut a (T, H, W) video into non-overlapping (n_frames, tile, tile) samples.

    Returns ``(samples, positions)``, positions as ``"r{row}c{col}"`` strings --
    the spatial tile a sample came from. With a single real video, spatial
    position is the only honest CV grouping available: holding out a region
    keeps the classifier from scoring by memorizing that region's static window
    texture. (Temporal correlation across windows of one video remains and is
    part of the stated caveat.)
    """
    t_max, h, w = frames.shape
    out, where = [], []
    for t0 in range(0, t_max - n_frames + 1, n_frames):
        for r0 in range(0, h - tile + 1, tile):
            for c0 in range(0, w - tile + 1, tile):
                out.append(frames[t0:t0 + n_frames, r0:r0 + tile, c0:c0 + tile])
                where.append(f"r{r0}c{c0}")
    return out, where


def subsample(samples: list, groups: list, cap: int, rng: np.random.Generator):
    """Cap the sample count, spreading evenly across groups (round-robin)."""
    order: dict = {}
    for i, g in enumerate(groups):
        order.setdefault(g, []).append(i)
    for idxs in order.values():
        rng.shuffle(idxs)
    keep = []
    while len(keep) < cap and any(order.values()):
        for g in list(order):
            if order[g] and len(keep) < cap:
                keep.append(order[g].pop())
    return [samples[i] for i in keep], [groups[i] for i in keep]


def cosine_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance between rows of a (n, d) and b (m, d)."""
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return 1.0 - a @ b.T


def containment_ratio(real_f: np.ndarray, syn_f: np.ndarray, syn_groups: list):
    """(median real->syn NN distance) / (median cross-clip syn->syn NN distance).

    The baseline excludes same-clip neighbours: frames of one clip are
    near-duplicates in feature space, and letting them count would deflate the
    denominator and overstate the gap.
    """
    d_real = cosine_dist(real_f, syn_f).min(axis=1)
    d_syn_all = cosine_dist(syn_f, syn_f)
    g = np.asarray(syn_groups)
    d_syn = np.array([row[g != g[i]].min() for i, row in enumerate(d_syn_all)])
    return float(np.median(d_real) / np.median(d_syn)), d_real, d_syn


def separability_auc(f_a: np.ndarray, f_b: np.ndarray,
                     groups_a: list, groups_b: list) -> float:
    """Group-aware cross-validated AUC of a linear real-vs-synthetic classifier."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    X = np.concatenate([f_a, f_b])
    y = np.concatenate([np.zeros(len(f_a)), np.ones(len(f_b))])
    groups = np.array([f"a:{g}" for g in groups_a] + [f"b:{g}" for g in groups_b])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    n_splits = min(5, len(set(groups_a)) + len(set(groups_b)))
    p = cross_val_predict(clf, X, y, groups=groups, method="predict_proba",
                          cv=GroupKFold(n_splits=n_splits))[:, 1]
    return float(roc_auc_score(y, p))


def encode(samples: list, device: str, model, processor) -> np.ndarray:
    """Mean-pooled last_hidden_state per sample -> (n, 1024)."""
    import torch
    feats = []
    for i, s in enumerate(samples):
        rgb = np.repeat(np.asarray(s, dtype=np.uint8)[..., None], 3, axis=-1)
        inp = processor(list(rgb), return_tensors="pt").to(device)
        with torch.no_grad():
            h = model(**inp).last_hidden_state
        feats.append(h.mean(dim=1).squeeze(0).float().cpu().numpy())
        if (i + 1) % 16 == 0:
            print(f"  encoded {i + 1}/{len(samples)}")
    return np.stack(feats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn", type=Path, default=Path("data/synthetic/smoke_v3"))
    ap.add_argument("--n-per-side", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    from compare_to_real import load_sequence  # scripts/ is on sys.path when run

    real_samples, real_groups = [], []
    for zpath, pred in REAL_SOURCES:
        frames = load_sequence(zpath, pred, limit=250)
        s, where = tile_windows(frames.astype(np.uint8))
        if not s:
            print(f"[skip] {zpath}: frames {frames.shape[1:]} are below the "
                  f"{TILE} px crop (resizing would rescale apparent size)")
            continue
        real_samples += s
        real_groups += [f"{Path(zpath).stem}:{w}" for w in where]
    real_samples, real_groups = subsample(real_samples, real_groups,
                                          args.n_per_side, rng)

    syn_samples, syn_groups, syn_strata = [], [], {}
    for f in sorted(glob.glob(str(args.syn / "clip_*.npz"))):
        clip = Path(f).stem
        syn_strata[clip] = "experiment" if clip.endswith("_experiment") else "physics"
        for s in tile_windows(np.load(f)["frames"])[0]:
            syn_samples.append(s)
            syn_groups.append(clip)  # synthetic groups by clip; position is moot
    syn_samples, syn_groups = subsample(syn_samples, syn_groups,
                                        args.n_per_side, rng)

    n_videos = len({g.split(":")[0] for g in real_groups})
    print(f"real: {len(real_samples)} samples from {n_videos} video(s), "
          f"{len(set(real_groups))} spatial groups; "
          f"synthetic: {len(syn_samples)} samples from {len(set(syn_groups))} clips")

    import torch
    from transformers import VJEPA2Model, VJEPA2VideoProcessor
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    name = "facebook/vjepa2-vitl-fpc64-256"
    model = VJEPA2Model.from_pretrained(name).eval().to(device)
    processor = VJEPA2VideoProcessor.from_pretrained(name)
    print(f"encoding on {device}...")
    real_f = encode(real_samples, device, model, processor)
    syn_f = encode(syn_samples, device, model, processor)
    np.savez_compressed(args.syn / "latent_gap_features.npz",
                        real=real_f, syn=syn_f,
                        real_groups=np.array(real_groups),
                        syn_groups=np.array(syn_groups))

    # -- containment, with a held-out synthetic reference ----------------------
    clips = sorted(set(syn_groups))
    held = set(clips[::4])  # every 4th clip plays the role of "real"
    is_held = np.array([g in held for g in syn_groups])
    ratio, d_real, _ = containment_ratio(real_f, syn_f, syn_groups)
    ref_ratio, _, _ = containment_ratio(
        syn_f[is_held], syn_f[~is_held],
        [g for g, h in zip(syn_groups, is_held) if not h])
    nn_idx = cosine_dist(real_f, syn_f).argmin(axis=1)
    exp_frac = float(np.mean([syn_strata[syn_groups[i]] == "experiment"
                              for i in nn_idx]))
    exp_base = float(np.mean([syn_strata[g] == "experiment" for g in syn_groups]))

    # -- separability, with a stratum-vs-stratum reference ---------------------
    auc = separability_auc(real_f, syn_f, real_groups, syn_groups)
    is_exp = np.array([syn_strata[g] == "experiment" for g in syn_groups])
    ref_auc = separability_auc(
        syn_f[is_exp], syn_f[~is_exp],
        [g for g, e in zip(syn_groups, is_exp) if e],
        [g for g, e in zip(syn_groups, is_exp) if not e])

    print(f"""
=== latent-space sim-to-real gap (frozen V-JEPA 2 ViT-L) ===

containment ratio        {ratio:5.2f}   (real->syn NN dist / cross-clip syn->syn)
  reference (held-out syn) {ref_ratio:5.2f}   (~1 by construction)
  real NN dist spread    {np.percentile(d_real, 5):.3f}-{np.percentile(d_real, 95):.3f} (p5-p95)
  NN in experiment stratum {exp_frac:.0%}   (base rate {exp_base:.0%})

linear separability AUC  {auc:5.2f}   (real vs synthetic, group-CV)
  reference (phys vs exp)  {ref_auc:5.2f}   (two strata of our own generator)

reading: harmful gap = high AUC *with* high ratio. High AUC alone is
fine-grained distinguishability (our own strata separate too); ratio near the
held-out-synthetic reference means real sits on the synthetic feature manifold.
""")


if __name__ == "__main__":
    main()
