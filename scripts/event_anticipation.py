"""RQ2: does the forecaster's error rise at a merge, or before it?

PLAN Step 12, fixed 2026-10-24 before any number. Coalescence clips of the val and
test splits; each clip's latent frames 0..T-1 (two cached windows stitched). A merge
at video frame f is an event at latent frame t_m = f // 2, kept iff both particles'
centres are inside the crop at f - 1 and 10 <= t_m <= T - 2.

- token signals (``data/events/signals-v1.npz``, from `event_signals.py` on the PC):
  ``model`` (predictor-perframe's teacher-forced +1 error), ``persistence``,
  ``ctxmean`` and ``pixel``, each per latent frame and token. Scored on the 3x3-token
  patch around the survivor, against patches around particles far from any merge.
- position-free signals (pooled cache, here): ``pooled_linear`` (+1 error of a
  one-step label-free linear forecaster) and ``pooled_change``. Scored per frame.

AUC per signal and lag (-2, -1, 0, +1), clip-bootstrap CIs, paired differences on
the same draws. Readings R1-R3 and P1-P2 as in PLAN. Runs on the Mac:
    uv run python scripts/event_anticipation.py --signals data/events/signals-v1.npz \
        --pooled data/features/dynamics-v1-perframe-pooled.npz \
        --gate1 experiments/gate1-perframe --readout experiments/pooled-gate1/readout.npz \
        --out experiments/events
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gate2_followup import BOOT_SEED, N_BOOT
from longer_context import stitch
from pooled_forecast import load_readout, pca_basis
from pooled_gate1 import LAMBDAS, load_context
from scipy.stats import rankdata

GRID = CELL = 16
LAGS = (-2, -1, 0, 1)
LEAD_LAGS = (-4, -3, -2, -1, 0)
T_MIN = 10
NEAR_FRAMES = 3  # a negative is at least this many latent frames from any merge...
NEAR_PX = 48.0  # ...or this far (3 tokens) from it
LEAD_PERCENTILE = 95
TOKEN_SIGNALS = ("model", "persistence", "ctxmean", "pixel")
TOKEN_BASELINES = ("persistence", "ctxmean", "pixel")
POOLED_SIGNALS = ("pooled_linear", "pooled_change")


@dataclass
class ClipGT:
    """One clip's ground truth: centres ``(N, T_video, 2)`` in crop coordinates, radii
    ``(N, T_video)`` (0 = absent or centre outside the crop), merges ``(M, 3)`` as
    ``[f, survivor, absorbed]``, and its number of latent frames."""

    pos: np.ndarray
    rad: np.ndarray
    merges: np.ndarray
    n_lat: int


# ------------------------------ events ------------------------------------------


def token_of(p: np.ndarray) -> tuple[int, int]:
    i, j = (np.clip(p // CELL, 0, GRID - 1)).astype(int)
    return int(i), int(j)


def events(gt: ClipGT) -> list[dict]:
    """The scored merges: both centres in the crop just before, t_m in range."""
    out = []
    for f, s, a in gt.merges:
        t = int(f) // 2
        if f < 1 or not T_MIN <= t <= gt.n_lat - 2:
            continue
        if gt.rad[s, f - 1] > 0 and gt.rad[a, f - 1] > 0:
            out.append({"t": t, "token": token_of(gt.pos[s, f])})
    return out


def near_merge(gt: ClipGT, t: int, p: np.ndarray, i: int) -> bool:
    """Whether particle ``i`` at ``p`` is in, or within NEAR_PX of, a merge within
    NEAR_FRAMES latent frames of ``t``."""
    for f, s, a in gt.merges:
        if abs(int(f) // 2 - t) > NEAR_FRAMES:
            continue
        if i in (s, a) or np.hypot(*(p - gt.pos[s, f])) < NEAR_PX:
            return True
    return False


def token_negatives(gt: ClipGT) -> list[dict]:
    out = []
    for t in range(T_MIN, gt.n_lat - 1):
        f = 2 * t
        for i in np.flatnonzero(gt.rad[:, f] > 0):
            if not near_merge(gt, t, gt.pos[i, f], i):
                out.append({"t": t, "token": token_of(gt.pos[i, f])})
    return out


def frame_negatives(gt: ClipGT) -> list[int]:
    merge_t = np.asarray(gt.merges[:, 0], dtype=int) // 2
    return [
        t
        for t in range(T_MIN, gt.n_lat - 1)
        if not (np.abs(merge_t - t) <= NEAR_FRAMES).any()
    ]


def patch_mean(signal: np.ndarray, t: int, token: tuple[int, int]) -> float:
    """Mean of a ``(T, 256)`` signal over the 3x3 tokens around ``token`` at ``t``."""
    grid = signal[t].reshape(GRID, GRID)
    i, j = token
    return float(grid[max(i - 1, 0) : i + 2, max(j - 1, 0) : j + 2].mean())


# ------------------------------ scoring -----------------------------------------


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC: P(a positive scores above a negative), ties count half."""
    ranks = rankdata(np.r_[pos, neg])
    n = len(pos)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * len(neg)))


@dataclass
class Scores:
    """Per signal, positive values ``{lag: (E,)}`` and negative values ``(K,)``, with
    each value's clip. Every signal is scored on the same events and negatives."""

    pos: dict
    neg: dict
    pos_clip: np.ndarray
    neg_clip: np.ndarray


def draws(scores: Scores) -> list[tuple[np.ndarray, np.ndarray]]:
    """N_BOOT clip resamples as (positive rows, negative rows)."""
    clips = np.unique(np.r_[scores.pos_clip, scores.neg_clip])
    pos_of = {c: np.flatnonzero(scores.pos_clip == c) for c in clips}
    neg_of = {c: np.flatnonzero(scores.neg_clip == c) for c in clips}
    rng = np.random.default_rng(BOOT_SEED)
    out = []
    for _ in range(N_BOOT):
        pick = rng.choice(clips, len(clips))
        out.append(
            (
                np.concatenate([pos_of[c] for c in pick]),
                np.concatenate([neg_of[c] for c in pick]),
            )
        )
    return out


def ci(values) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def auc_table(scores: Scores, pairs: tuple, lags=LAGS) -> dict:
    """AUC with CI per signal and lag, and each ``(a, b)`` pair's paired difference."""
    boot = draws(scores)
    out = {"auc": {}, "diff": {}}
    for lag in lags:
        point, dist = {}, {}
        for name, by_lag in scores.pos.items():
            pos, neg = by_lag[lag], scores.neg[name]
            point[name] = auc(pos, neg)
            dist[name] = np.array([auc(pos[p], neg[n]) for p, n in boot])
            out["auc"].setdefault(name, {})[lag] = {
                "auc": point[name],
                "ci": ci(dist[name]),
            }
        for a, b in pairs:
            out["diff"].setdefault(f"{a}-{b}", {})[lag] = {
                "diff": point[a] - point[b],
                "ci": ci(dist[a] - dist[b]),
            }
    return out


def lead_times(values: dict, neg: np.ndarray, neg_clip, pos_clip) -> list:
    """Per event, the first lag in LEAD_LAGS whose value exceeds its clip's negatives'
    LEAD_PERCENTILE, or None; events without every lead lag are left out.
    ``values``: lag -> (E,), NaN where the lag is out of range."""
    out = []
    for e, clip in enumerate(pos_clip):
        if np.isnan([values[lag][e] for lag in LEAD_LAGS]).any():
            continue  # too close to the clip start for every lead lag
        threshold = np.percentile(neg[neg_clip == clip], LEAD_PERCENTILE)
        hit = [lag for lag in LEAD_LAGS if values[lag][e] > threshold]
        out.append(hit[0] if hit else None)
    return out


# ------------------------------ sampling ----------------------------------------


@dataclass
class Points:
    """Where every signal is read: per event ``(clip, t_m, token)`` and per negative
    ``(clip, t, token)``. Position-free signals ignore the token."""

    events: list
    negatives: list


def clip_gts(z) -> list[ClipGT]:
    """The ground truth of every clip in a signals file (merges unpadded)."""
    n_lat = z["model"].shape[1]
    merges = [m[m[:, 0] >= 0] for m in z["merges"]]
    return [
        ClipGT(z["pos"][c], z["rad"][c], merges[c], n_lat) for c in range(len(merges))
    ]


def sample_points(gts: list[ClipGT], rows, per_token: bool) -> Points:
    ev, neg = [], []
    for c in rows:
        ev += [(c, e["t"], e["token"] if per_token else None) for e in events(gts[c])]
        if per_token:
            neg += [(c, n["t"], n["token"]) for n in token_negatives(gts[c])]
        else:
            neg += [(c, t, None) for t in frame_negatives(gts[c])]
    return Points(ev, neg)


def read_value(signal: np.ndarray, t: int, token) -> float:
    """One signal ``(T, 256)`` or ``(T,)`` of one clip at frame ``t``, NaN if out of range."""
    if not 0 <= t < len(signal):
        return np.nan
    return float(signal[t]) if token is None else patch_mean(signal, t, token)


def score(points: Points, signals: dict, lags=LAGS) -> Scores:
    """``signals``: name -> per-clip arrays. Every signal is read at the same points."""
    pos = {
        name: {
            lag: np.array([read_value(sig[c], t + lag, k) for c, t, k in points.events])
            for lag in lags
        }
        for name, sig in signals.items()
    }
    neg = {
        name: np.array([read_value(sig[c], t, k) for c, t, k in points.negatives])
        for name, sig in signals.items()
    }
    return Scores(
        pos,
        neg,
        np.array([c for c, _, _ in points.events]),
        np.array([c for c, _, _ in points.negatives]),
    )


# --------------------------- position-free --------------------------------------

CTX = 8


def lagged_coefficients(coeffs: np.ndarray) -> np.ndarray:
    """``(C, T, K)`` -> ``(C, T - CTX, 1 + CTX*K)``: row i holds frames i..i+CTX-1."""
    c, t, _ = coeffs.shape
    lagged = np.concatenate([coeffs[:, k : t - CTX + k] for k in range(CTX)], axis=-1)
    return np.concatenate([np.ones((c, t - CTX, 1)), lagged], axis=-1)


def one_step_error(z: np.ndarray, split: dict) -> np.ndarray:
    """``pooled_linear``: +1 L1 error ``(C, T)`` of a label-free ridge from the 8 context
    frames' top-64 PCA coefficients to the next frame's, lambda on val by that error."""
    basis = pca_basis(z[split["train"]].reshape(-1, z.shape[-1]))
    x, y = lagged_coefficients(z @ basis), (z @ basis)[:, CTX:]
    tr = (
        x[split["train"]].reshape(-1, x.shape[-1]),
        y[split["train"]].reshape(-1, y.shape[-1]),
    )
    gram, cross = tr[0].T @ tr[0], tr[0].T @ tr[1]

    def error(lam, rows):
        w = np.linalg.solve(gram + lam * np.eye(len(gram)), cross)
        return np.abs((x[rows] @ w) @ basis.T - z[rows, CTX:]).mean(-1)

    lam = min(LAMBDAS, key=lambda lam: error(lam, split["val"]).mean())
    out = np.full(z.shape[:2], np.nan)
    out[:, CTX:] = error(lam, np.arange(len(z)))
    return out


def pooled_signals(args, names: list[str]) -> dict:
    """Both position-free signals ``(T,)`` per dumped clip, in the dump's order."""
    pooled, ctx = load_context(args.pooled, args.gate1)
    readout = load_readout(args.readout)
    seq, _, meta = stitch(pooled["mean"], ctx["targets"], ctx["meta"])
    z = (seq.astype(np.float32) - readout.x_mu) / readout.x_sd
    index = {m["clip"]: i for i, m in enumerate(meta)}
    split = clip_split(args.gate1, [m["clip"] for m in meta])
    change = np.full(z.shape[:2], np.nan)
    change[:, 1:] = np.abs(z[:, 1:] - z[:, :-1]).mean(-1)
    signals = {"pooled_linear": one_step_error(z, split), "pooled_change": change}
    rows = [index[n] for n in names]
    return {k: v[rows] for k, v in signals.items()}


def clip_split(gate1: Path, names: list[str]) -> dict:
    """Row indices of ``names`` per part of the Gate-1 per-frame split."""
    split = json.loads((gate1 / "metrics.json").read_text())["split"]
    names = np.array(names)
    return {
        part: np.flatnonzero(np.isin(names, clips)) for part, clips in split.items()
    }


# ------------------------------- readings ---------------------------------------


def above(entry: dict) -> bool:
    """An AUC whose CI lies above chance."""
    return entry["ci"][0] > 0.5


def positive(entry: dict) -> bool:
    """A paired AUC difference whose CI lies above 0."""
    return entry["ci"][0] > 0


def judge(token: dict, pooled: dict) -> dict:
    """R1-R3 and P1-P2 on one part's tables (PLAN Step 12)."""
    return {
        "R1_detection": above(token["auc"]["model"][0]),
        "R2_beyond_raw_change": all(
            positive(token["diff"][f"model-{b}"][0]) for b in TOKEN_BASELINES
        ),
        "R3_anticipation": above(token["auc"]["model"][-1])
        and positive(token["diff"]["model-pixel"][-1]),
        "P1_pooled_detection": above(pooled["auc"]["pooled_linear"][0]),
        "P2_pooled_beyond_change": positive(
            pooled["diff"]["pooled_linear-pooled_change"][0]
        ),
    }


def lead_summary(gts, rows, signals: dict) -> dict:
    """Per token signal, how many events first cross their clip's bar at each lead lag."""
    scores = score(sample_points(gts, rows, per_token=True), signals, LEAD_LAGS)
    out = {}
    for name in signals:
        leads = lead_times(
            scores.pos[name], scores.neg[name], scores.neg_clip, scores.pos_clip
        )
        out[name] = {str(lag): leads.count(lag) for lag in (*LEAD_LAGS, None)}
    return out


def analyse_part(gts, rows, token_sig: dict, pooled_sig: dict) -> dict:
    token_pts = sample_points(gts, rows, per_token=True)
    frame_pts = sample_points(gts, rows, per_token=False)
    token = auc_table(
        score(token_pts, token_sig), tuple(("model", b) for b in TOKEN_BASELINES)
    )
    pooled = auc_table(score(frame_pts, pooled_sig), (POOLED_SIGNALS,))
    return {
        "n_clips": len(rows),
        "n_events": len(token_pts.events),
        "n_token_negatives": len(token_pts.negatives),
        "n_frame_negatives": len(frame_pts.negatives),
        "token": token,
        "pooled": pooled,
        "lead_times": lead_summary(gts, rows, token_sig),
        "reading": judge(token, pooled),
    }


# -------------------------------- report ----------------------------------------


def print_part(name: str, part: dict) -> None:
    print(
        f"\n[{name}] {part['n_clips']} clips, {part['n_events']} events, "
        f"{part['n_token_negatives']} patch / {part['n_frame_negatives']} frame negatives"
    )
    for block in ("token", "pooled"):
        for sig, by_lag in part[block]["auc"].items():
            print(
                f"  AUC {sig:14s} "
                + "  ".join(
                    f"{lag:+d} {v['auc']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]"
                    for lag, v in by_lag.items()
                )
            )
        for pair, by_lag in part[block]["diff"].items():
            print(
                f"  diff {pair:24s} "
                + "  ".join(
                    f"{lag:+d} {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]"
                    for lag, v in by_lag.items()
                )
            )
    print(
        f"  lead times (first lag over the clip's {LEAD_PERCENTILE}th percentile): {part['lead_times']}"
    )
    print(f"  reading: {part['reading']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for flag in ("--signals", "--pooled", "--gate1", "--readout", "--out"):
        ap.add_argument(flag, type=Path, required=True)
    args = ap.parse_args()
    z = np.load(args.signals)
    names = [m["clip"] for m in json.loads(str(z["meta"]))]
    gts = clip_gts(z)
    token_sig = {s: z[s].astype(np.float32) for s in TOKEN_SIGNALS}
    pooled_sig = pooled_signals(args, names)
    parts = clip_split(args.gate1, names)
    result = {
        p: analyse_part(gts, parts[p], token_sig, pooled_sig)
        for p in ("test", "val")
        if len(parts[p])
    }
    for p, part in result.items():
        print_part(p, part)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(result, indent=1))
    print(f"\nwrote {args.out}/metrics.json")


if __name__ == "__main__":
    main()
