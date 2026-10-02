"""Gate-2 follow-up -- POST HOC, criterion fixed in PLAN Step 10 before this ran.

Gate 2's size test had no power at +5: even the true future barely beat persistence,
because radii change less over 5 latent steps than the probes' reading error. Here
the forecast runs to +24 (context = latent frames 0-7, forecast = 8-31, one snippet
per test window), and every comparison carries a 95 % bootstrap CI over test clips.

(A) size at +24 -- paired gains over persistence, per snippet. A process is
    *powered* when the floor's gain is significantly positive; on powered verdict
    processes the model passes when its own gain is.
(B) growth exponent over the 24 forecast frames, +-0.1 of the true one.
(C) blur vs growth, diagnostic -- probed count, radius and peak occupancy per
    horizon, model vs floor; and each process's exponent minus the static one.

Usage (remote, about as long as the Gate-2 run):
    uv run python -u scripts/gate2_followup.py --features data/features/dynamics-v1 \
        --gate1 experiments/gate1 --predictor experiments/predictor-v1/predictor.pt \
        --out experiments/core
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from eval_forecasts import (
    EXPONENT_TOL,
    METHODS,
    VERDICT_PROCESSES,
    forecasts,
    horizon_summary,
    load_gate1,
    load_predictor,
    score_method,
    snippet_truth,
)
from gate1_secondary import split_indices
from train_predictor import pick_device
from train_probes import WindowCache

HORIZON = 24
REPORT_HORIZONS = (1, 5, 10, 24)
N_BOOT = 2000
BOOT_SEED = 0


def bootstrap_ci(values, clips, stat=np.mean) -> list[float]:
    """95 % percentile CI of ``stat(values)``, resampling whole clips: snippets of
    one clip are not independent."""
    values = np.asarray(values, dtype=float)
    _, inverse = np.unique(np.asarray(clips), return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(inverse.max() + 1)]
    rng = np.random.default_rng(BOOT_SEED)
    draws = (rng.integers(0, len(members), len(members)) for _ in range(N_BOOT))
    stats = [stat(values[np.concatenate([members[k] for k in d])]) for d in draws]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def size_test(w2_persist, w2_model, w2_floor, clips) -> dict:
    """(A): paired gains over persistence at one horizon; ``pass`` is None when the
    floor itself does not measurably beat persistence (the test has no power)."""
    gain_floor = np.asarray(w2_persist) - np.asarray(w2_floor)
    gain_model = np.asarray(w2_persist) - np.asarray(w2_model)
    ci_floor = bootstrap_ci(gain_floor, clips)
    ci_model = bootstrap_ci(gain_model, clips)
    powered = ci_floor[0] > 0
    return {
        "gain_floor": float(gain_floor.mean()),
        "gain_floor_ci": ci_floor,
        "gain_model": float(gain_model.mean()),
        "gain_model_ci": ci_model,
        "gap_closed": float(gain_model.mean() / gain_floor.mean()),
        "powered": bool(powered),
        "pass": bool(ci_model[0] > 0) if powered else None,
    }


def collect(cache, test, predictor, probes, contexts=None) -> list[dict]:
    """Score every physics-stratum test window: one snippet, 8 context + 24 future.
    ``contexts`` (window index -> context latents) replaces the cached context."""
    model, stats, ctx = predictor
    device = str(next(model.parameters()).device)
    records = []
    for i in test:
        feats, pos, rad, meta = cache.load(i)
        if meta["stratum"] != "physics":
            continue
        if contexts is not None:
            feats = feats.copy()
            feats[:ctx] = contexts[i]
        truth = snippet_truth(pos, rad, meta, np.arange(ctx, ctx + HORIZON))
        futures = forecasts(model, feats, 0, ctx, HORIZON, stats)
        methods = {
            m: score_method(probes, lat, truth, device) for m, lat in futures.items()
        }
        records.append({"meta": meta, "truth": truth, "methods": methods})
    return records


# ------------------------------ per process -------------------------------------


def per_horizon(rows, method: str, key: str, col: int | None = None) -> dict:
    """Mean of one per-frame score over snippets, at each report horizon."""
    series = np.array([r["methods"][method][key] for r in rows])
    if col is not None:
        series = series[..., col]
    return {f"+{h}": float(series[:, h - 1].mean()) for h in REPORT_HORIZONS}


def exponent_entry(rows, method: str) -> dict:
    vals = [r["methods"][method]["exponent"] for r in rows]
    clips = [r["meta"]["clip"] for r in rows]
    return {
        "median": float(np.nanmedian(vals)),
        "ci": bootstrap_ci(vals, clips, np.nanmedian),
    }


def exponent_block(rows) -> dict:
    """(B): model and floor exponents with CIs, against the truth where defined."""
    true_exp = rows[0]["meta"].get("exponent")
    model = exponent_entry(rows, "model")
    return {
        "model": model,
        "floor": exponent_entry(rows, "floor"),
        "gt_crop": float(np.nanmedian([r["truth"]["exponent_crop"] for r in rows])),
        "true": true_exp,
        "pass": None
        if true_exp is None
        else abs(model["median"] - true_exp) <= EXPONENT_TOL,
    }


def blur_block(rows, method: str) -> dict:
    """(C): blur predicts falling count and peak occupancy with rising radius."""
    return {
        "count": per_horizon(rows, method, "obs", 0),
        "mean_radius": per_horizon(rows, method, "obs", 1),
        "peak_occ": per_horizon(rows, method, "peak_occ"),
    }


def process_entry(rows) -> dict:
    clips = [r["meta"]["clip"] for r in rows]
    w2_final = {m: [r["methods"][m]["w2"][HORIZON - 1] for r in rows] for m in METHODS}
    return {
        "n_snippets": len(rows),
        "w2": {m: per_horizon(rows, m, "w2") for m in METHODS},
        "size_test": size_test(
            w2_final["persistence"], w2_final["model"], w2_final["floor"], clips
        ),
        "det_f1": {
            m: {f"+{h}": horizon_summary(rows, m, h)["det_f1"] for h in REPORT_HORIZONS}
            for m in METHODS
        },
        "exponent": exponent_block(rows),
        "blur": {m: blur_block(rows, m) for m in ("model", "floor")},
    }


def analyse(records: list[dict]) -> dict:
    by_process = {
        proc: process_entry([r for r in records if r["meta"]["process"] == proc])
        for proc in sorted({r["meta"]["process"] for r in records})
    }
    if "static" in by_process:  # blur would add the same spurious growth everywhere
        static = by_process["static"]["exponent"]["model"]["median"]
        for e in by_process.values():
            e["exponent"]["model_minus_static"] = (
                e["exponent"]["model"]["median"] - static
            )
    return by_process


# -------------------------------- reporting -------------------------------------


def fmt_ci(ci) -> str:
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def print_size(by_process: dict) -> None:
    print(
        f"\n(A) size distance at +{HORIZON} (px), model / persistence / ctxmean / floor;"
        " mean gain over persistence [95 % CI]:"
    )
    for proc, e in by_process.items():
        s = e["size_test"]
        verdict = {True: "PASS", False: "FAIL", None: "untestable"}[s["pass"]]
        print(
            f"  {proc:12s} "
            + " / ".join(f"{e['w2'][m][f'+{HORIZON}']:.2f}" for m in METHODS)
            + f"   floor {s['gain_floor']:+.3f} {fmt_ci(s['gain_floor_ci'])}"
            + f"   model {s['gain_model']:+.3f} {fmt_ci(s['gain_model_ci'])}"
            + f"   gap closed {s['gap_closed']:.2f}"
            + (f"  -> {verdict}" if proc in VERDICT_PROCESSES else "")
        )
    print_by_horizon(by_process, "size distance (px) by horizon", "w2")


def print_by_horizon(by_process: dict, title: str, key: str) -> None:
    shown = ("model", "persistence", "floor")
    print(f"\n{title}, {' / '.join(shown)}:")
    for proc, e in by_process.items():
        print(
            f"  {proc:12s} "
            + "   ".join(
                f"{h}: " + " / ".join(f"{e[key][m][h]:.2f}" for m in shown)
                for h in e[key]["model"]
            )
        )


def print_exponents(by_process: dict) -> None:
    print(
        f"\n(B) growth exponent over {HORIZON} forecast frames:"
        " model [95 % CI] / floor / GT-in-crop / true / model minus static"
    )
    for proc, e in by_process.items():
        x = e["exponent"]
        true = "-" if x["true"] is None else f"{x['true']:.3f}"
        tag = {True: "  pass", False: "  FAIL", None: ""}[x["pass"]]
        print(
            f"  {proc:12s} {x['model']['median']:.3f} {fmt_ci(x['model']['ci'])} / "
            f"{x['floor']['median']:.3f} / {x['gt_crop']:.3f} / {true} / "
            f"{x.get('model_minus_static', float('nan')):+.3f}{tag}"
        )


def print_blur(by_process: dict) -> None:
    print(
        "\n(C) blur check, model/floor: probed count | mean radius (px) | peak occupancy"
    )
    for proc, e in by_process.items():
        b = e["blur"]
        print(
            f"  {proc:12s} "
            + "   ".join(
                f"{h}: "
                + " | ".join(
                    f"{b['model'][k][h]:.1f}/{b['floor'][k][h]:.1f}"
                    if k != "peak_occ"
                    else f"{b['model'][k][h]:.2f}/{b['floor'][k][h]:.2f}"
                    for k in ("count", "mean_radius", "peak_occ")
                )
                for h in b["model"]["count"]
            )
        )


def print_report(by_process: dict) -> None:
    print_size(by_process)
    print_exponents(by_process)
    print_blur(by_process)
    print_by_horizon(
        by_process, "decoded-centroid F1 (Gate-1 threshold) by horizon", "det_f1"
    )


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
    predictor = load_predictor(args.predictor, device)
    print(
        f"test windows {len(test)}  device {device}  ctx {predictor[2]}  horizon {HORIZON}"
        f"  threshold {probes[3]}  (post hoc)"
    )

    by_process = analyse(collect(cache, test, predictor, probes))
    print_report(by_process)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "followup.json").write_text(
        json.dumps(by_process, indent=2, default=float)
    )
    print(f"\nwrote {args.out}/followup.json")


if __name__ == "__main__":
    main()
