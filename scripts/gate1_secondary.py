"""Gate 1 secondary analyses, from the saved probes -- no retraining.

Reported next to the pre-registered Gate-1 numbers, never in place of them
(experiments/gate1/README.md). One question per limitation the headline run
exposed:

1. **Threshold.** The val search stopped at its 0.95 grid edge. Re-pick on val
   over a grid reaching 0.99 and report test detection + MSD at both.
2. **Coverage.** A track had to be detected in all 32 latent frames. Re-link with
   gap closing (memory 1, 2): a missed frame no longer breaks a track, the gap
   stays NaN and MSD averages observed pairs only. Probe and ceiling get the
   same linking, so the comparison stays fair.
3. **Count.** Is the error concentrated where a particle straddles the crop edge
   -- where "centre inside the crop" disagrees with what is visible? Count R^2
   and MAE on frames with vs without a particle centre within one token cell of
   the border, and R^2 per process.

Runs on CPU on purpose: the probes are tiny, the job is disk-bound, and the GPUs
stay free. Val + test are read once (~800 windows); ~15-30 min on the PC.

Usage:
    uv run python -u scripts/gate1_secondary.py --features data/features/dynamics-v1 \
        --gate1 experiments/gate1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from train_probes import WindowCache, attentive_targets, r_squared

from jepa_lptem.eval.centroid_eval import GRID, eval_centroid_msd, pick_threshold
from jepa_lptem.eval.probes import AttentiveProbe, CentroidProbe

FINE_THRESHOLDS = np.round(
    np.concatenate([np.arange(0.30, 0.90, 0.05), np.arange(0.90, 0.995, 0.01)]), 2
)
MEMORIES = (0, 1, 2)
EDGE_MARGIN_PX = 16.0  # one token cell
EDGE_BUCKETS = 4  # 0, 1, 2, 3+ straddling particles


def edge_counts(positions: np.ndarray, crop: int, margin: float = EDGE_MARGIN_PX):
    """``(N, T, 2)`` crop-frame positions -> ``(T,)`` number of particle centres
    within ``margin`` of the crop border, on either side of it.

    Step 6 keeps the positions of particles outside the crop (only their radius is
    zeroed), so both ambiguous cases count: a labelled particle mostly cut off, and
    an unlabelled one partly in view.
    """
    near_line = (np.abs(positions) < margin) | (np.abs(positions - crop) < margin)
    in_band = (positions > -margin) & (positions < crop + margin)
    return (near_line.any(axis=-1) & in_band.all(axis=-1)).sum(axis=0)


def load_probes(path: Path):
    import torch

    # our own checkpoint; weights_only=False because norm is a tuple of arrays
    state = torch.load(path, map_location="cpu", weights_only=False)
    att, cen = AttentiveProbe(out_dim=3), CentroidProbe(grid=GRID)
    att.load_state_dict(state["attentive"])
    cen.load_state_dict(state["centroid"])
    return att.eval(), cen.eval(), state["norm"]


def collect(cache: WindowCache, idx, probes) -> list[dict]:
    """One pass over the windows: probe outputs and targets; features are dropped."""
    import torch

    att, cen, (mu, sd) = probes
    records = []
    for i in idx:
        feats, pos, rad, meta = cache.load(i)
        x = torch.from_numpy(feats)
        with torch.no_grad():
            pred = att(x).numpy() * sd + mu
            occ, off = cen(x)
        gt = [pos[rad[:, t] > 0, t] for t in range(rad.shape[1])]
        records.append({
            "window": (torch.sigmoid(occ).numpy(), off.numpy(), gt, meta),
            "pred": pred,
            "true": attentive_targets(rad),
            "edge": edge_counts(pos, meta["crop"]),
        })
    return records


def detection_and_msd(val_records, test_records, headline_threshold: float) -> dict:
    picked = pick_threshold((r["window"] for r in val_records), FINE_THRESHOLDS)
    out = {"val_picked_threshold": picked}
    for label, th in (("preregistered", headline_threshold), ("fine_grid", picked)):
        for memory in MEMORIES:
            m = eval_centroid_msd((r["window"] for r in test_records), th, memory)
            out[f"{label}_memory{memory}"] = m
            print(f"[test] threshold {th:.2f} memory {memory}: "
                  f"precision {m['det_precision']:.3f} recall {m['det_recall']:.3f}  "
                  f"MSD {m['msd_ratio_pooled']:.3f} (ceiling {m['msd_ratio_pooled_gt']:.3f})  "
                  f"coverage {m['track_coverage_mean']:.3f} "
                  f"(ceiling {m['track_coverage_mean_gt']:.3f})  "
                  f"windows {m['msd_windows_used']}/{m['msd_windows_used_gt']}")
    return out


def count_by_edge(test_records) -> dict:
    """Physics-stratum count error against how many particles straddle the crop
    edge. If the centre-in-crop label is what makes count hard, the error grows
    with that number and is small on frames with none."""
    phys = [r for r in test_records if r["window"][3]["stratum"] == "physics"]
    pred = np.concatenate([r["pred"][:, 0] for r in phys])
    true = np.concatenate([r["true"][:, 0] for r in phys])
    n_edge = np.concatenate([r["edge"] for r in phys])
    clean = n_edge == 0
    out = {"r2_count_no_edge_particle": r_squared(pred[clean], true[clean]),
           "r2_count_with_edge_particle": r_squared(pred[~clean], true[~clean])}
    print(f"[count, physics, margin {EDGE_MARGIN_PX:.0f} px] R2 on frames with no edge "
          f"particle {out['r2_count_no_edge_particle']:.3f}, with >= 1 "
          f"{out['r2_count_with_edge_particle']:.3f}")
    for k in range(EDGE_BUCKETS):
        mask = n_edge >= k if k == EDGE_BUCKETS - 1 else n_edge == k
        err = pred[mask] - true[mask]
        label = f"{k}+" if k == EDGE_BUCKETS - 1 else str(k)
        out[f"edge_{label}"] = {"frames": int(mask.sum()),
                                "mae": float(np.abs(err).mean()) if mask.any() else None,
                                "bias": float(err.mean()) if mask.any() else None}
        print(f"  {label:2s} edge particles: {mask.sum():6d} frames  "
              f"MAE {np.abs(err).mean():.2f}  bias {err.mean():+.2f}")
    return out


def r2_by_process(test_records) -> dict:
    """Count and mean-radius R^2 per physical process (all strata)."""
    out = {}
    for process in sorted({r["window"][3]["process"] for r in test_records}):
        rows = [r for r in test_records if r["window"][3]["process"] == process]
        pred = np.concatenate([r["pred"] for r in rows])
        true = np.concatenate([r["true"] for r in rows])
        out[f"r2_count_{process}"] = r_squared(pred[:, 0], true[:, 0])
        out[f"r2_mean_radius_{process}"] = r_squared(pred[:, 1], true[:, 1])
        print(f"  {process:12s} R2 count {out[f'r2_count_{process}']:.3f}  "
              f"mean radius {out[f'r2_mean_radius_{process}']:.3f}")
    return out


def split_indices(cache: WindowCache, metrics_path: Path) -> dict:
    """The headline run's own val/test windows, from the split it recorded."""
    split = json.loads(metrics_path.read_text())["split"]
    return {part: [i for i, m in enumerate(cache.meta) if m["clip"] in set(split[part])]
            for part in ("val", "test")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--gate1", required=True, type=Path,
                    help="the headline run's dir: probes.pt + metrics.json (its split)")
    args = ap.parse_args()

    cache = WindowCache(args.features)
    idx = split_indices(cache, args.gate1 / "metrics.json")
    probes = load_probes(args.gate1 / "probes.pt")
    print(f"val {len(idx['val'])} / test {len(idx['test'])} windows -- reading once")
    val_records = collect(cache, idx["val"], probes)
    test_records = collect(cache, idx["test"], probes)

    # the threshold the headline run picked on val, not a constant: each cache picks its own
    headline = json.loads((args.gate1 / "metrics.json").read_text())["val"]["threshold"]
    results = {"detection_msd": detection_and_msd(val_records, test_records, headline),
               "count_by_edge": count_by_edge(test_records),
               "r2_by_process": r2_by_process(test_records)}
    out = args.gate1 / "secondary.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
