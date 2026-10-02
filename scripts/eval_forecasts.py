"""Step 10 / Gate 2: do forecast latents decode to the right physics?

For every held-out test window (the Gate-1 test clips; the predictor never trained
on them), three fixed snippets of 8 context + 10 future latent frames. Four ways
to get the 10 future latents, all read by the **frozen Gate-1 probes**:

- ``model``       -- the Step-9 forecaster rolled out 10 steps on its own forecasts
- ``persistence`` -- the last context frame, held
- ``ctxmean``     -- the mean of the 8 context frames, held (removes per-frame noise
                     without any dynamics: the bar for "learned dynamics")
- ``floor``       -- the true encoded future frames: what the probes can do at best,
                     which separates "wrong physics" from "off-manifold latents"

Scored against the ground truth at the same future frame, at +1, +5, +10:

(a) **size distribution** -- the probes read count, mean radius and total area, not
    individual radii, so the distribution is taken by its first two moments
    (mean, sd; sd from E[r^2] = area / (pi count)) and compared by the 2-Wasserstein
    distance between Gaussians, sqrt(dmean^2 + dsd^2). Fixed in PLAN Step 10 before
    any Gate-2 number existed.
(b) **growth exponent** -- log-log slope of probed mean radius against the clip's
    physical time over the 10 forecast frames, median per process.
(c) **positions** -- decoded centroids (the Gate-1 run's val-picked threshold) matched to GT: F1 and
    mean matched distance. Brownian position is a martingale, so the best possible
    forecast is the current position plus drift: parity with persistence is the
    expected, correct result. Drift-removed MSD/4Dt of tracks linked with gap closing
    ``memory=2`` (fixed before Gate 2) is reported for every method, but it measures
    diffusion only for the floor and the ceiling: a deterministic forecast predicts
    the *expected* position, which does not diffuse at all.

The verdict is computed on the physics stratum, per PLAN: PASS = on diffusion- and
reaction-limited growth and ripening at +5, size-distribution distance at least 30 %
below persistence, and the probed exponent within 0.1 of the true one where one
exists. The context-mean comparison is reported next to it.

Usage (remote, ~10-20 min):
    uv run python -u scripts/eval_forecasts.py --features data/features/dynamics-v1 \
        --gate1 experiments/gate1 --predictor experiments/predictor-v1/predictor.pt \
        --out experiments/core
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gate1_secondary import load_probes, split_indices
from train_predictor import eval_starts, pick_device, standardize, unstandardize
from train_probes import WindowCache, attentive_targets

from jepa_lptem.eval.centroid_eval import GRID, match_distances, pool_msd, window_msd
from jepa_lptem.eval.observables import fit_growth_exponent
from jepa_lptem.eval.probes import decode_centroids
from jepa_lptem.models.predictor import LatentForecaster

HORIZON = 10
REPORT_HORIZONS = (1, 5, 10)
SNIPPETS_PER_WINDOW = 3
METHODS = ("model", "persistence", "ctxmean", "floor")
LINK_MEMORY = 2  # fixed in PLAN Step 10 before any Gate-2 number
VERDICT_HORIZON = 5
VERDICT_PROCESSES = ("diffusion", "reaction", "ripening")
W2_MARGIN = 0.30  # size-distribution distance at least 30 % below persistence
EXPONENT_TOL = 0.1


# ------------------------------ pure pieces -----------------------------------


def gaussian_w2(mean_a: float, sd_a: float, mean_b: float, sd_b: float) -> float:
    """2-Wasserstein distance between two 1-D Gaussians."""
    return float(np.hypot(mean_a - mean_b, sd_a - sd_b))


def moments_from_observables(obs: np.ndarray) -> tuple[float, float]:
    """Probed ``[count, mean_radius, total_area]`` -> ``(mean, sd)`` of radius."""
    count, mean, area = (float(v) for v in obs)
    if count <= 0 or mean <= 0:
        return 0.0, 0.0
    second = area / (np.pi * count)
    return mean, float(np.sqrt(max(second - mean**2, 0.0)))


def gt_moments(radii: np.ndarray) -> tuple[float, float]:
    """``(mean, sd)`` of the visible (``> 0``) radii in one frame."""
    visible = radii[radii > 0]
    if visible.size == 0:
        return 0.0, 0.0
    return float(visible.mean()), float(visible.std())


def snippet_exponent(times: np.ndarray, mean_radius: np.ndarray) -> float:
    try:
        return fit_growth_exponent(times, mean_radius)
    except ValueError:
        return float("nan")


def forecasts(
    model, feats: np.ndarray, start: int, ctx: int, horizon: int, stats
) -> dict:
    """The four ``(horizon, S, D)`` raw-feature futures for one snippet."""
    mean, std = stats
    context = feats[start : start + ctx]
    z = torch.from_numpy(standardize(context, mean, std))[None]
    with torch.no_grad():
        rolled = model.rollout(z.to(next(model.parameters()).device), horizon)[0]
    return {
        "model": unstandardize(rolled.cpu().numpy(), mean, std),
        "persistence": np.repeat(context[-1:], horizon, axis=0),
        "ctxmean": np.repeat(context.mean(axis=0, keepdims=True), horizon, axis=0),
        "floor": feats[start + ctx : start + ctx + horizon],
    }


# ------------------------------- scoring ---------------------------------------


def load_gate1(gate1_dir: Path, device: str):
    """The frozen Gate-1 probes plus the occupancy threshold that run picked on val.

    The threshold is read, never hardcoded: a Gate-1 run on a different cache
    (e.g. ``--per-frame``) picks its own, and decoding with another run's cut
    would grade its probes at the wrong operating point."""
    att, cen, norm = load_probes(gate1_dir / "probes.pt")
    threshold = json.loads((gate1_dir / "metrics.json").read_text())["val"]["threshold"]
    return att.to(device), cen.to(device), norm, threshold


def read_probes(probes, latents: np.ndarray, device: str):
    """Frozen Gate-1 probes on ``(H, S, D)`` latents -> observables, occupancy, offsets."""
    att, cen, (mu, sd), _ = probes
    x = torch.from_numpy(latents.astype(np.float32)).to(device)
    with torch.no_grad():
        obs = att(x).cpu().numpy() * sd + mu
        occ, off = cen(x)
    return obs, torch.sigmoid(occ).cpu().numpy(), off.cpu().numpy()


def score_method(probes, latents, truth: dict, device: str) -> dict:
    """One method's 10 future latents against the ground truth at the same frames."""
    obs, occ, off = read_probes(probes, latents, device)
    cell = truth["cell"]
    cents = [
        decode_centroids(occ[h], off[h], threshold=probes[3], cell_size=cell)
        for h in range(len(obs))
    ]
    dists = [
        match_distances(c, g, cell / 2)
        for c, g in zip(cents, truth["pos"], strict=True)
    ]
    return {
        "obs": obs,
        "peak_occ": occ.reshape(len(occ), -1).max(axis=1),
        "count_err": np.abs(obs[:, 0] - truth["obs"][:, 0]),
        "meanr_err": np.abs(obs[:, 1] - truth["obs"][:, 1]),
        "w2": np.array([
            gaussian_w2(*moments_from_observables(o), *g)
            for o, g in zip(obs, truth["moments"], strict=True)
        ]),
        "tp": np.array([len(d) for d in dists]),
        "n_pred": np.array([len(c) for c in cents]),
        "dist_sum": np.array([d.sum() for d in dists]),
        "exponent": snippet_exponent(truth["times"], obs[:, 1]),
        "centroids": cents,
    }


def snippet_truth(pos, rad, meta, frames: np.ndarray) -> dict:
    """Ground truth at the forecast frames, plus the clip's physical time axis."""
    video_frames = meta["start"] + meta["frames_per_latent"] * frames
    obs = attentive_targets(rad[:, frames])
    times = meta["t0_growth"] + meta["dt_growth"] * video_frames
    return {
        "obs": obs,
        "times": times,
        "cell": meta["crop"] / GRID,
        "moments": [gt_moments(rad[:, f]) for f in frames],
        "pos": [pos[rad[:, f] > 0, f] for f in frames],
        "exponent_crop": snippet_exponent(times, obs[:, 1]),
    }


def score_window(cache, i, model, probes, stats, ctx, device) -> list[dict]:
    feats, pos, rad, meta = cache.load(i)
    records = []
    for start in eval_starts(len(feats), ctx + HORIZON, SNIPPETS_PER_WINDOW):
        truth = snippet_truth(
            pos, rad, meta, np.arange(start + ctx, start + ctx + HORIZON)
        )
        futures = forecasts(model, feats, int(start), ctx, HORIZON, stats)
        records.append({
            "meta": meta,
            "truth": truth,
            "methods": {
                m: score_method(probes, lat, truth, device)
                for m, lat in futures.items()
            },
        })
    return records


# ------------------------------ aggregation ------------------------------------


def horizon_summary(rows: list[dict], method: str, h: int) -> dict:
    i = h - 1
    parts = [r["methods"][method] for r in rows]
    tp = sum(p["tp"][i] for p in parts)
    n_pred = sum(p["n_pred"][i] for p in parts)
    n_gt = sum(len(r["truth"]["pos"][i]) for r in rows)
    return {
        "w2": float(np.mean([p["w2"][i] for p in parts])),
        "count_mae": float(np.mean([p["count_err"][i] for p in parts])),
        "mean_radius_mae": float(np.mean([p["meanr_err"][i] for p in parts])),
        "det_f1": 2 * tp / max(n_pred + n_gt, 1),
        # nan, not 0, when nothing matched: no detections is not a perfect position
        "pos_err_px": float(sum(p["dist_sum"][i] for p in parts) / tp)
        if tp
        else float("nan"),
    }


def exponent_summary(values) -> dict:
    v = np.asarray([x for x in values if np.isfinite(x)])
    if v.size == 0:
        return {"median": None, "iqr": None, "n": 0}
    return {
        "median": float(np.median(v)),
        "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))],
        "n": int(v.size),
    }


def summarize(records: list[dict]) -> dict:
    """Per process: every method at every report horizon, and growth exponents."""
    out = {}
    for proc in sorted({r["meta"]["process"] for r in records}):
        rows = [r for r in records if r["meta"]["process"] == proc]
        entry = {
            "n_snippets": len(rows),
            "true_exponent": rows[0]["meta"].get("exponent"),
            "gt_crop_exponent": exponent_summary(
                r["truth"]["exponent_crop"] for r in rows
            ),
        }
        for m in METHODS:
            entry[m] = {f"+{h}": horizon_summary(rows, m, h) for h in REPORT_HORIZONS}
            entry[m]["exponent"] = exponent_summary(
                r["methods"][m]["exponent"] for r in rows
            )
        out[proc] = entry
    return out


def pooled_msd(frames_list, records, memory: int) -> dict:
    """One source's pooled MSD/4Dt; a window the linker cannot resolve is dropped
    and shows up as ``msd_windows_used`` falling short."""
    from trackpy.linking.utils import SubnetOversizeException

    results = []
    for frames, r in zip(frames_list, records, strict=True):
        try:
            results.append(window_msd(frames, r["meta"], memory))
        except SubnetOversizeException:
            continue
    return pool_msd(results)


def msd_summary(records: list[dict]) -> dict:
    """Pooled MSD/4Dt per method over the forecast frames; GT centroids as ceiling."""
    sources = {m: [r["methods"][m]["centroids"] for r in records] for m in METHODS}
    sources["ceiling"] = [r["truth"]["pos"] for r in records]
    return {
        f"{name}/memory{memory}": pooled_msd(frames_list, records, memory)
        for memory in (LINK_MEMORY, 0)
        for name, frames_list in sources.items()
    }


def process_verdict(entry: dict) -> dict:
    """One process at the verdict horizon: size distribution vs persistence (and
    vs context mean, reported), and the exponent where the process has one."""
    w2 = {m: entry[m][f"+{VERDICT_HORIZON}"]["w2"] for m in METHODS}
    true_exp, model_exp = entry["true_exponent"], entry["model"]["exponent"]["median"]
    exp_pass = (
        None
        if true_exp is None
        else model_exp is not None and abs(model_exp - true_exp) <= EXPONENT_TOL
    )
    return {
        "w2": w2,
        "w2_vs_persistence": 1 - w2["model"] / w2["persistence"],
        "w2_vs_ctxmean": 1 - w2["model"] / w2["ctxmean"],
        "w2_pass": w2["model"] <= (1 - W2_MARGIN) * w2["persistence"],
        "true_exponent": true_exp,
        "model_exponent": model_exp,
        "floor_exponent": entry["floor"]["exponent"]["median"],
        "exponent_pass": exp_pass,
    }


def verdict(summary: dict) -> dict:
    """The pre-registered Gate-2 test at +5 on the physics stratum. ``PASS`` is None
    when a verdict process has no test snippets (only in smoke runs)."""
    out = {
        proc: process_verdict(summary[proc])
        for proc in VERDICT_PROCESSES
        if proc in summary
    }
    if len(out) < len(VERDICT_PROCESSES):
        out["PASS"] = None
        return out
    out["PASS"] = all(
        v["w2_pass"] and v["exponent_pass"] is not False for v in out.values()
    )
    return out


# -------------------------------- reporting ------------------------------------


def print_metric(summary: dict, title: str, key: str, fmt: str = ".2f") -> None:
    """One table: per process, every method at every report horizon."""
    print(f"\n{title}, physics stratum ({' / '.join(METHODS)}):")
    for proc, e in summary.items():
        print(
            f"  {proc:12s} "
            + "   ".join(
                f"+{h}: " + " / ".join(format(e[m][f"+{h}"][key], fmt) for m in METHODS)
                for h in REPORT_HORIZONS
            )
        )


def print_report(summary: dict, msd: dict, gate: dict) -> None:
    print_metric(summary, "size-distribution distance (px)", "w2")
    print_metric(summary, "decoded-centroid F1 at the Gate-1 threshold", "det_f1")
    print_metric(summary, "matched position error (px)", "pos_err_px", ".1f")
    print("\ngrowth exponent (median over snippets): model / floor / GT-in-crop / true")
    for proc, e in summary.items():
        vals = [
            e["model"]["exponent"]["median"],
            e["floor"]["exponent"]["median"],
            e["gt_crop_exponent"]["median"],
            e["true_exponent"],
        ]
        print(
            f"  {proc:12s} "
            + " / ".join("-" if v is None else f"{v:.3f}" for v in vals)
        )
    print(
        f"\nMSD/4Dt pooled (memory {LINK_MEMORY}): "
        + "  ".join(
            f"{m} {msd[f'{m}/memory{LINK_MEMORY}']['msd_ratio_pooled']:.3f}"
            for m in (*METHODS, "ceiling")
        )
    )
    print(
        "  (meaningful for floor and ceiling only: a deterministic forecast predicts the\n"
        "   expected position, which does not diffuse; held frames minus the known drift\n"
        "   move ballistically, and the lag-averaged ratio of that can land near 1)"
    )
    print_verdict(gate)


def print_verdict(gate: dict) -> None:
    label = {
        True: "PASS",
        False: "FAIL",
        None: "not judgeable (a verdict process is missing)",
    }
    print(f"\nGATE 2 (+{VERDICT_HORIZON}, physics): {label[gate['PASS']]}")
    for proc in VERDICT_PROCESSES:
        g = gate.get(proc)
        if g is None:
            print(f"  {proc:12s} no test snippets")
            continue
        print(
            f"  {proc:12s} size dist {g['w2_vs_persistence']:+.1%} vs persistence "
            f"({'pass' if g['w2_pass'] else 'fail'}), {g['w2_vs_ctxmean']:+.1%} vs ctx-mean; "
            f"exponent {g['exponent_pass']}"
        )


def load_predictor(path: Path, device: str):
    ckpt = torch.load(
        path, map_location="cpu", weights_only=False
    )  # our own checkpoint
    c = ckpt["config"]
    model = LatentForecaster(
        enc_dim=c["enc_dim"],
        dim=c["dim"],
        depth=c["depth"],
        heads=c["heads"],
        n_spatial=c["n_spatial"],
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), (ckpt["mean"], ckpt["std"]), c["ctx"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument(
        "--gate1", required=True, type=Path, help="probes.pt + metrics.json (split)"
    )
    ap.add_argument("--predictor", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    device = pick_device()
    cache = WindowCache(args.features)
    test = split_indices(cache, args.gate1 / "metrics.json")["test"]
    probes = load_gate1(args.gate1, device)
    model, stats, ctx = load_predictor(args.predictor, device)
    print(
        f"test windows {len(test)}  device {device}  ctx {ctx}  horizon {HORIZON}"
        f"  threshold {probes[3]}"
    )

    records = [
        rec
        for i in test
        for rec in score_window(cache, i, model, probes, stats, ctx, device)
    ]
    physics = [r for r in records if r["meta"]["stratum"] == "physics"]
    summary, msd = summarize(physics), msd_summary(physics)
    gate = verdict(summary)
    print_report(summary, msd, gate)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(
        json.dumps(
            {
                "n_snippets": len(records),
                "n_physics": len(physics),
                "summary": summary,
                "msd": msd,
                "verdict": gate,
            },
            indent=2,
            default=float,
        )
    )
    print(f"\nwrote {args.out}/metrics.json")


if __name__ == "__main__":
    main()
