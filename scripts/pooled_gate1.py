"""P1: can physics be read off the pooled (position-free) latent state?

PLAN Step 10, position-free forecasting, fixed 2026-10-11 before any pooled number.
For each candidate state, in this order:

    mean       the 256 tokens averaged                         (D)
    mean+std   that average and the per-channel spread         (2D)
    grid2      the average of each 8x8-token quadrant          (4D)

a ridge readout maps one frame's pooled vector (z-scored per channel on train
frames) to the z-scored ``[count, mean radius, total area]``. Lambda is picked on val
by mean-radius R^2. A state is **usable** iff test R^2 of mean radius >= 0.8
(physics stratum) **and** the readout, applied to the true future frames 8-31 of the
physics test windows, gives a median growth exponent within +-0.1 of the true one
on diffusion and on reaction. The first usable state is *the pooled state*, and its
readout is saved, frozen, for P2-P3. Count R^2 is reported, not judged.

Labels are used here only to fit the readout, an evaluation instrument, as in Gate 1.
Runs on the Mac, on the P0 output:
    uv run python scripts/pooled_gate1.py \
        --pooled data/features/dynamics-v1-perframe-pooled.npz \
        --gate1 experiments/gate1-perframe --out experiments/pooled-gate1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from eval_forecasts import EXPONENT_TOL, snippet_exponent
from gate2_followup import bootstrap_ci
from train_probes import OBS_NAMES, r_squared

STATES = ("mean", "mean+std", "grid2")
LAMBDAS = 10.0 ** np.arange(-2, 5)
R2_BAR = 0.8
CTX, HORIZON = 8, 24
JUDGED = ("diffusion", "reaction")
RADIUS = OBS_NAMES.index("mean_radius")


def state_features(z, name: str) -> np.ndarray:
    """``(W, T, D')`` float32 pooled vectors for one candidate state."""
    if name == "mean":
        return z["mean"].astype(np.float32)
    if name == "mean+std":
        return np.concatenate([z["mean"], z["std"]], axis=-1).astype(np.float32)
    w, t = z["grid2"].shape[:2]
    return z["grid2"].reshape(w, t, -1).astype(np.float32)


def split_windows(meta: list[dict], gate1_dir: Path) -> dict:
    """Window indices per part of the Gate-1 per-frame split (by clip)."""
    split = json.loads((gate1_dir / "metrics.json").read_text())["split"]
    clip = np.array([m["clip"] for m in meta])
    return {part: np.flatnonzero(np.isin(clip, clips)) for part, clips in split.items()}


class Readout:
    """Ridge from a z-scored pooled frame to z-scored observables; lambda picked on
    val. ``fit`` returns the per-lambda val R^2 of mean radius."""

    def fit(self, x, y, train_windows, val_windows) -> dict:
        frames = x[train_windows].reshape(-1, x.shape[-1])
        targets = y[train_windows].reshape(-1, y.shape[-1])
        self.x_mu, self.x_sd = frames.mean(0), frames.std(0) + 1e-6
        self.y_mu, self.y_sd = targets.mean(0), targets.std(0)
        a = (frames - self.x_mu) / self.x_sd
        evals, evecs = np.linalg.eigh(a.T @ a)
        proj = evecs.T @ (a.T @ ((targets - self.y_mu) / self.y_sd))
        scores = {}
        for lam in LAMBDAS:
            self.weights = evecs @ (proj / (evals + lam)[:, None])
            scores[float(lam)] = r_squared(
                self.predict(x[val_windows])[..., RADIUS], y[val_windows][..., RADIUS]
            )
        self.lam = max(scores, key=scores.get)
        self.weights = evecs @ (proj / (evals + self.lam)[:, None])
        return scores

    def predict(self, x: np.ndarray) -> np.ndarray:
        """``(..., D')`` pooled vectors -> ``(..., 3)`` observables in physical units."""
        return ((x - self.x_mu) / self.x_sd) @ self.weights * self.y_sd + self.y_mu


def r2_table(pred: np.ndarray, y: np.ndarray, physics: np.ndarray) -> dict:
    out = {}
    for j, name in enumerate(OBS_NAMES):
        out[f"r2_{name}"] = r_squared(pred[..., j], y[..., j])
        out[f"r2_{name}_physics"] = r_squared(pred[physics][..., j], y[physics][..., j])
    return out


def snippet_exponents(radius: np.ndarray, meta: list[dict], windows: np.ndarray) -> dict:
    """Per process, ``(exponent, clip)`` of the readout's mean radius over the last
    HORIZON frames (the forecast frames, 8-31 of a 32-frame window) of each physics
    window."""
    frames = np.arange(radius.shape[1] - HORIZON, radius.shape[1])
    rows = {}
    for i in windows:
        m = meta[i]
        if m["stratum"] != "physics":
            continue
        times = m["t0_growth"] + m["dt_growth"] * (
            m["start"] + m["frames_per_latent"] * frames
        )
        rows.setdefault(m["process"], []).append((
            snippet_exponent(times, radius[i, frames]),
            m["clip"],
        ))
    return rows


def floor_exponents(radius: np.ndarray, meta: list[dict], windows: np.ndarray) -> dict:
    """Median exponent per process over the physics test windows, clip-bootstrap CI."""
    out = {}
    for process, pairs in sorted(snippet_exponents(radius, meta, windows).items()):
        values, clips = zip(*pairs, strict=True)
        true = next(m.get("exponent") for m in meta if m["process"] == process)
        out[process] = {
            "median": float(np.nanmedian(values)),
            "ci": bootstrap_ci(values, clips, np.nanmedian),
            "true": true,
        }
    return out


def judge(r2: dict, exponents: dict) -> dict:
    exp_ok = {
        p: abs(exponents[p]["median"] - exponents[p]["true"]) <= EXPONENT_TOL
        for p in JUDGED
    }
    return {
        "radius_r2_ok": r2["r2_mean_radius_physics"] >= R2_BAR,
        "exponent_ok": exp_ok,
        "usable": r2["r2_mean_radius_physics"] >= R2_BAR and all(exp_ok.values()),
    }


def evaluate_state(z, name: str, ctx: dict) -> tuple[dict, Readout]:
    """Fit, score and judge one candidate state. ``ctx``: targets, split, meta, physics mask."""
    x = state_features(z, name)
    split, y = ctx["split"], ctx["targets"]
    readout = Readout()
    val_scores = readout.fit(x, y, split["train"], split["val"])
    pred = readout.predict(x)
    test = split["test"]
    r2 = r2_table(pred[test], y[test], ctx["physics"][test])
    exponents = floor_exponents(pred[..., RADIUS], ctx["meta"], test)
    entry = {
        "lambda": readout.lam,
        "val_r2_mean_radius_by_lambda": val_scores,
        "test": r2,
        "floor_exponent": exponents,
    }
    return entry | judge(r2, exponents), readout


def print_entry(name: str, e: dict) -> None:
    t = e["test"]
    print(
        f"\n[{name}] lambda {e['lambda']:g}  test physics R2: count {t['r2_count_physics']:.3f}"
        f"  mean radius {t['r2_mean_radius_physics']:.3f}  area {t['r2_total_area_physics']:.3f}"
    )
    for p, v in e["floor_exponent"].items():
        true = "-" if v["true"] is None else f"{v['true']:.3f}"
        print(
            f"    floor exponent {p:12s} {v['median']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]  true {true}"
        )
    print(
        f"    radius R2 >= {R2_BAR}: {e['radius_r2_ok']}  exponents within {EXPONENT_TOL}: {e['exponent_ok']}"
        f"  -> usable: {e['usable']}"
    )


def save_readout(out: Path, name: str, readout: Readout) -> None:
    np.savez(
        out / "readout.npz",
        state=name,
        lam=readout.lam,
        x_mu=readout.x_mu,
        x_sd=readout.x_sd,
        y_mu=readout.y_mu,
        y_sd=readout.y_sd,
        weights=readout.weights,
    )


def load_context(pooled: Path, gate1_dir: Path) -> tuple:
    z = np.load(pooled)
    meta = json.loads(str(z["meta"]))
    return z, {
        "targets": z["targets"],
        "split": split_windows(meta, gate1_dir),
        "meta": meta,
        "physics": np.array([m["stratum"] == "physics" for m in meta]),
    }


def select_state(z, ctx: dict, out: Path) -> dict:
    """Judge the states in order; the first usable one's readout is saved, frozen."""
    results, chosen = {}, None
    for name in STATES:
        results[name], readout = evaluate_state(z, name, ctx)
        print_entry(name, results[name])
        if results[name]["usable"] and chosen is None:
            chosen = name
            save_readout(out, name, readout)
    return {"states": results, "pooled_state": chosen}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pooled", type=Path, required=True)
    ap.add_argument("--gate1", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    z, ctx = load_context(args.pooled, args.gate1)
    args.out.mkdir(parents=True, exist_ok=True)
    result = select_state(z, ctx, args.out)
    (args.out / "metrics.json").write_text(json.dumps(result, indent=1))
    chosen = result["pooled_state"]
    print(f"\npooled state: {chosen}" + ("" if chosen else " -- none usable: stop (PLAN Step 10)"))


if __name__ == "__main__":
    main()
