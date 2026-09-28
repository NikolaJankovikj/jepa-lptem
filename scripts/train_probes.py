"""Step 7 / Gate 1: train attentive + centroid probes on cached frozen features.

The gate question: **can physical observables be read off frozen V-JEPA latents
at all?** PASS (held-out R^2 >= 0.8 for mean radius and count on the physics
stratum; decoded-track MSD within 20 % of the analytic 4Dt, coverage reported)
means the frozen encoder suffices and LoRA shrinks to a comparison run. FAIL
makes Step 8 the main path. Either way it is a thesis figure.

Data flow: one training sample = one *window* (all its latent frames as a
batch). Windows load lazily from the Step-6 cache -- the full remote cache
(~70 GB) can never sit in RAM, and one window (~16 MB) decompressed per step is
the natural unit on both machines. Targets are derived from the cache's aligned
ground truth, so this script never touches the raw clips:

- ``AttentiveProbe`` per latent frame -> ``[count, mean_radius, total_area]``,
  z-scored on train statistics (R^2 is invariant; optimization is not).
- ``CentroidProbe`` -> occupancy (16x16) + sub-cell offsets, BCE (pos-weighted:
  occupancy is ~5 % sparse) + L2 on occupied cells only.

Evaluation on held-out clips (split 80/10/10 **by source clip**, so no clip
leaks between splits): R^2 per observable (overall + physics stratum), and the
full centroid -> trackpy -> MSD pipeline pooled across windows -- per-window MSD
at these particle counts is noise (Step 4e), so windows pool with n_full
weights, drift removed exactly, lags capped by each clip's ``msd_max_lag``.
The occupancy threshold is picked on val (max detection F1) and applied to
test; the same MSD pipeline also runs on ground-truth centroids, giving the
perfect-probe ceiling the probe's MSD and coverage are read against.

Usage (local smoke; remote is the same command on the full cache):
    uv run python scripts/train_probes.py --features data/features/smoke_v3 \
        --out experiments/gate1-smoke --epochs 50
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from jepa_lptem.eval.centroid_eval import GRID, eval_centroid_msd, pick_threshold
from jepa_lptem.eval.probes import AttentiveProbe, CentroidProbe

OBS_NAMES = ("count", "mean_radius", "total_area")


# --------------------------- targets (pure numpy) -----------------------------


def attentive_targets(radii: np.ndarray) -> np.ndarray:
    """(N, T_lat) radii -> (T_lat, 3) [count, mean_radius, total_area]; 0 = absent."""
    present = radii > 0
    count = present.sum(axis=0).astype(float)
    with np.errstate(invalid="ignore"):
        mean_r = np.where(count > 0, radii.sum(axis=0) / np.maximum(count, 1), 0.0)
    area = np.pi * (radii**2).sum(axis=0)
    return np.stack([count, mean_r, area], axis=1)


def centroid_targets(positions: np.ndarray, radii: np.ndarray, crop: int):
    """GT -> per-latent-frame occupancy (T,G,G) and offsets (T,G,G,2).

    Inverse of ``decode_centroids``: a particle at ``pos`` lands in cell
    ``floor(pos / cell)`` with offset ``pos/cell - (cell_idx + 0.5)`` in
    ``[-0.5, 0.5)``. Two particles in one cell keep the first (the collision
    count is reported by the caller -- at our coverage caps it is rare, and a
    16x16 grid cannot represent both anyway).
    """
    cell = crop / GRID
    n, t = radii.shape
    occ = np.zeros((t, GRID, GRID), dtype=np.float32)
    off = np.zeros((t, GRID, GRID, 2), dtype=np.float32)
    collisions = 0
    for ti in range(t):
        for pi in range(n):
            if radii[pi, ti] <= 0:
                continue
            r, c = positions[pi, ti] / cell
            i, j = int(r), int(c)
            if occ[ti, i, j]:
                collisions += 1
                continue
            occ[ti, i, j] = 1.0
            off[ti, i, j] = (r - i - 0.5, c - j - 0.5)
    return occ, off, collisions


def split_by_clip(clip_names: list[str], seed: int = 0):
    """Deterministic 80/10/10 split of unique clips -> {'train','val','test'} sets."""
    clips = sorted(set(clip_names))
    order = np.random.default_rng(seed).permutation(len(clips))
    n_val = max(1, len(clips) // 10)
    n_test = max(1, len(clips) // 10)
    val = {clips[i] for i in order[:n_val]}
    test = {clips[i] for i in order[n_val : n_val + n_test]}
    train = set(clips) - val - test
    return {"train": train, "val": val, "test": test}


def r_squared(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ------------------------------- lazy windows ---------------------------------


class WindowCache:
    """Lazy view over the Step-6 cache: meta in memory, features on demand."""

    def __init__(self, features_dir: Path):
        self.files = sorted(glob.glob(str(features_dir / "*__w*.npz")))
        if not self.files:
            raise SystemExit(f"no feature windows in {features_dir}")
        self.meta = []
        for f in self.files:
            with np.load(f) as z:
                self.meta.append(json.loads(str(z["meta"])))

    def load(self, i: int):
        with np.load(self.files[i]) as z:
            return (
                z["features"].astype(np.float32),
                z["positions"],
                z["radii"],
                self.meta[i],
            )


# --------------------------------- training -----------------------------------


def train_probes(cache, idx_train, device, epochs, lr=1e-3, seed=0):
    import torch

    torch.manual_seed(seed)
    # Target statistics + BCE pos-weight from the train split only.
    t_all = np.concatenate([attentive_targets(cache.load(i)[2]) for i in idx_train])
    mu, sd = t_all.mean(axis=0), t_all.std(axis=0) + 1e-9
    occ_frac = np.mean([
        centroid_targets(cache.load(i)[1], cache.load(i)[2], cache.meta[i]["crop"])[
            0
        ].mean()
        for i in idx_train[:4]
    ])
    pos_weight = torch.tensor(
        (1.0 - occ_frac) / max(occ_frac, 1e-4), dtype=torch.float32, device=device
    )

    att = AttentiveProbe(out_dim=3).to(device)
    cen = CentroidProbe(grid=GRID).to(device)
    opt = torch.optim.AdamW(list(att.parameters()) + list(cen.parameters()), lr=lr)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        att.train()
        cen.train()
        losses = []
        for i in rng.permutation(idx_train):
            feats, pos, rad, meta = cache.load(int(i))
            occ_t, off_t, _ = centroid_targets(pos, rad, meta["crop"])
            x = torch.from_numpy(feats).to(device)  # (T, 256, D)
            y = torch.from_numpy(
                ((attentive_targets(rad) - mu) / sd).astype(np.float32)
            ).to(device)
            occ_y = torch.from_numpy(occ_t).to(device)
            off_y = torch.from_numpy(off_t).to(device)

            pred = att(x)
            occ_p, off_p = cen(x)
            mask = occ_y.unsqueeze(-1)
            loss = (
                torch.nn.functional.mse_loss(pred, y)
                + bce(occ_p, occ_y)
                + ((off_p - off_y) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if (ep + 1) % max(1, epochs // 10) == 0:
            print(f"  epoch {ep + 1:3d}/{epochs}  train loss {np.mean(losses):.4f}")
    return att, cen, (mu, sd)


# -------------------------------- evaluation ----------------------------------


def eval_attentive(att, cache, idx, norm, device):
    import torch

    mu, sd = norm
    preds, trues, strata = [], [], []
    att.eval()
    for i in idx:
        feats, _, rad, meta = cache.load(i)
        with torch.no_grad():
            p = att(torch.from_numpy(feats).to(device)).cpu().numpy()
        preds.append(p * sd + mu)
        trues.append(attentive_targets(rad))
        strata += [meta["stratum"]] * len(feats)
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    is_phys = np.array(strata) == "physics"
    out = {}
    for j, name in enumerate(OBS_NAMES):
        out[f"r2_{name}"] = r_squared(preds[:, j], trues[:, j])
        if is_phys.any():
            out[f"r2_{name}_physics"] = r_squared(preds[is_phys, j], trues[is_phys, j])
    return out


def probe_windows(cen, cache, idx, device):
    """Yield ``(occ_prob (T,G,G), offsets (T,G,G,2), GT centroids per frame, meta)``."""
    import torch

    cen.eval()
    for i in idx:
        feats, pos, rad, meta = cache.load(i)
        with torch.no_grad():
            occ, off = cen(torch.from_numpy(feats).to(device))
        gt = [pos[rad[:, t] > 0, t] for t in range(rad.shape[1])]
        yield torch.sigmoid(occ).cpu().numpy(), off.cpu().numpy(), gt, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    cache = WindowCache(args.features)
    clip_of = [m["clip"] for m in cache.meta]
    split = split_by_clip(clip_of, seed=args.seed)
    idx = {k: [i for i, c in enumerate(clip_of) if c in v] for k, v in split.items()}
    n_coll = sum(
        centroid_targets(cache.load(i)[1], cache.load(i)[2], cache.meta[i]["crop"])[2]
        for i in idx["train"]
    )
    print(
        f"windows: {len(cache.files)}  split "
        f"{len(idx['train'])}/{len(idx['val'])}/{len(idx['test'])} "
        f"(by clip)  cell collisions in train targets: {n_coll}  device: {device}"
    )

    att, cen, norm = train_probes(
        cache, idx["train"], device, args.epochs, seed=args.seed
    )

    metrics = {
        "n_windows": len(cache.files),
        "epochs": args.epochs,
        "split": {k: sorted(v) for k, v in split.items()},
    }
    threshold = pick_threshold(probe_windows(cen, cache, idx["val"], device))
    for part in ("val", "test"):
        m = eval_attentive(att, cache, idx[part], norm, device)
        m.update(
            eval_centroid_msd(probe_windows(cen, cache, idx[part], device), threshold)
        )
        metrics[part] = m
        print(
            f"\n[{part}] "
            + "  ".join(f"{k}={v:.3f}" for k, v in m.items() if isinstance(v, float))
        )

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"attentive": att.state_dict(), "centroid": cen.state_dict(), "norm": norm},
        args.out / "probes.pt",
    )
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nwrote {args.out}/probes.pt and metrics.json")
    print(
        "Gate 1 criteria (judge on the full remote run, not the smoke): "
        "r2_mean_radius_physics >= 0.8, r2_count_physics >= 0.8, "
        "msd_ratio_pooled in [0.8, 1.2], read against msd_ratio_pooled_gt "
        "(the perfect-probe ceiling on this crop) with coverage reported."
    )


if __name__ == "__main__":
    main()
