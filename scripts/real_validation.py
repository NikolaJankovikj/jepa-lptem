"""Step 13 on published movies: does what we built on simulation transfer to real?

Fixed in `PLAN.md` Step 13 (revised 2026-11-16) before any real movie was encoded.
The movies are the ones that passed `real_video_qc.py` and the look
(`real_video_qc.REAL_MOVIES`). Four analyses:

- A  containment: `latent_gap.py`'s protocol, per movie, with a held-out-synthetic
     reference and an HAADF movie as the positive control
- B  readout transfer: the frozen P1 readout's mean radius against a validated
     pixel estimator, compared with the same pair on fresh simulated clips
- C  growth direction: do readout and pixel estimator agree on the sign of growth?
- D  forecast: candidate A, fitted on the cache, against persistence and the
     context mean on real snippets, in the readout's z-scored pooled space

Runs on the Mac (MPS), ~15 min:
    uv run python scripts/real_validation.py --out experiments/real
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from compare_to_real import apparent_radius_px
from gate2_followup import bootstrap_ci
from latent_gap import containment_ratio, subsample, tile_windows
from pooled_forecast import fit_linear, forecast_linear, load_readout
from pooled_gate1 import CTX, HORIZON, RADIUS, load_context
from real_features import TUBELET, load_or_encode
from real_video_qc import REAL_MOVIES, load_movie
from scipy.stats import spearmanr
from verify_clips import detect_dark_blobs

BLOCK = 16  # video frames per pixel-estimator reading
SNIPPET, SNIPPET_STRIDE = CTX + HORIZON, 4  # latent frames
SYN_SAMPLES, REAL_SAMPLES_PER_MOVIE = 64, 32  # containment, as latent_gap.py
REPORT_HORIZONS = (1, 5, 10, 24)
BASELINES = ("persistence", "ctxmean")
CONTROL = "cm2c03226_si_004.avi"
PATHS = {
    "pooled": Path("data/features/dynamics-v1-perframe-pooled.npz"),
    "gate1": Path("experiments/gate1-perframe"),
    "readout": Path("experiments/pooled-gate1/readout.npz"),
    "smoke": Path("data/synthetic/smoke_v3"),
}


# ------------------------------ A: containment ----------------------------------


def containment_samples(rng) -> tuple:
    """Synthetic 16-frame tiles (grouped by clip) and each movie's (tiled by place)."""
    syn, syn_groups = [], []
    for f in sorted(glob.glob(str(PATHS["smoke"] / "clip_*.npz"))):
        for s in tile_windows(np.load(f)["frames"])[0]:
            syn.append(s)
            syn_groups.append(Path(f).stem)
    real = {}
    for name in REAL_MOVIES:
        s, where = tile_windows(load_movie(name))
        real[name] = subsample(s, where, REAL_SAMPLES_PER_MOVIE, rng)
    return subsample(syn, syn_groups, SYN_SAMPLES, rng), real


def window_encoder(device: str = "mps"):
    """``latent_gap.encode`` bound to the frozen encoder: whole-window, mean-pooled."""
    from extract_features import MODEL_NAME, load_processor
    from latent_gap import encode
    from transformers import VJEPA2Model

    model = VJEPA2Model.from_pretrained(MODEL_NAME).eval().to(device)
    processor = load_processor()
    return lambda samples: encode(samples, device, model, processor)


def containment() -> dict:
    """A: per movie, the containment ratio against held-out synthetic clips'."""
    (syn, syn_groups), real = containment_samples(np.random.default_rng(0))
    encode = window_encoder()
    syn_f = encode(syn)
    held = held_out_ratios(syn_f, syn_groups)
    limit = max(held.values())
    movies = {
        name: movie_containment(encode(s), (syn_f, syn_groups, limit))
        for name, (s, _) in real.items()
    }
    return {
        "held_out_ratios": held,
        "limit": limit,
        "movies": movies,
        "control_detected": not movies[CONTROL]["contained"],
    }


def held_out_ratios(syn_f: np.ndarray, syn_groups: list) -> dict:
    """Each synthetic clip's containment ratio against all the other clips."""
    groups = np.asarray(syn_groups)
    return {
        c: containment_ratio(
            syn_f[groups == c], syn_f[groups != c], list(groups[groups != c])
        )[0]
        for c in sorted(set(syn_groups))
    }


def movie_containment(f, reference: tuple) -> dict:
    """``reference``: synthetic features, their clip groups, the containment limit."""
    syn_f, syn_groups, limit = reference
    ratio = containment_ratio(f, syn_f, syn_groups)[0]
    return {"ratio": ratio, "contained": ratio <= limit, "n_samples": len(f)}


# ------------------------------ B, C: readout -----------------------------------


COUNT = 0  # column of the readout's observables


def block_pairs(z: np.ndarray, frames: np.ndarray, readout) -> np.ndarray:
    """Per 16-frame block: ``(readout mean radius, pixel-estimator radius, block,
    readout count, detector count)``. Readout values are medians over the block's 8
    latent frames; the detector count is the mean over its 16 frames."""
    obs = readout.predict(z)
    per = BLOCK // TUBELET
    rows = []
    for b in range(len(frames) // BLOCK):
        block = frames[b * BLOCK : (b + 1) * BLOCK].astype(float)
        lat = obs[b * per : (b + 1) * per]
        rows.append((
            float(np.median(lat[:, RADIUS])),
            apparent_radius_px(block, k=BLOCK),
            b,
            float(np.median(lat[:, COUNT])),
            float(np.mean([len(detect_dark_blobs(f)[0]) for f in block])),
        ))
    return np.array(rows)


def measured(rows: np.ndarray) -> np.ndarray:
    """Blocks where the pixel estimator found particles."""
    return np.isfinite(rows[:, 1]) & (rows[:, 1] > 0)


def block_truth(clip: dict) -> np.ndarray:
    """The true mean radius per 16-frame block (median over its latent frames)."""
    per = BLOCK // TUBELET
    return np.array([
        np.median(clip["truth"][b * per : (b + 1) * per])
        for b in range(len(clip["px"]) // BLOCK)
    ])


def sim_reference(sim: dict, readout) -> dict:
    """The simulated readout / pixel ratio band, and each side's error against truth."""
    rows = np.concatenate([block_pairs(c["z"], c["px"], readout) for c in sim.values()])
    truth = np.concatenate([block_truth(c) for c in sim.values()])
    ok = measured(rows)
    ratio = rows[ok, 0] / rows[ok, 1]
    return {
        "band_p5_p95": [
            float(np.percentile(ratio, 5)),
            float(np.percentile(ratio, 95)),
        ],
        "median_ratio": float(np.median(ratio)),
        "readout_mae_px": float(np.mean(np.abs(rows[:, 0] - truth))),
        "pixel_mae_px": float(np.mean(np.abs(rows[ok, 1] - truth[ok]))),
        "n_blocks": int(ok.sum()),
        "count_readout": float(np.median(rows[:, 3])),
        "count_detector": float(np.median(rows[:, 4])),
    }


def movie_readout(rows: np.ndarray, band: list) -> dict:
    """B and C for one movie's blocks."""
    ok = measured(rows)
    med = float(np.median(rows[ok, 0] / rows[ok, 1]))
    return {
        "median_ratio": med,
        "in_range": band[0] <= med <= band[1],
        "readout_px": float(np.median(rows[:, 0])),
        "pixel_px": float(np.median(rows[ok, 1])),
        "n_blocks": int(ok.sum()),
        "growth_rho_readout": float(spearmanr(rows[:, 2], rows[:, 0])[0]),
        "growth_rho_pixel": float(spearmanr(rows[ok, 2], rows[ok, 1])[0]),
        "count_readout": float(np.median(rows[:, 3])),
        "count_detector": float(np.median(rows[:, 4])),
    }


def by_movie(real: dict) -> dict:
    """``{movie: [tile entries]}``."""
    out: dict = {}
    for key, v in real.items():
        out.setdefault(key.split("|")[0], []).append((key, v))
    return out


def readout_transfer(real: dict, sim: dict, readout) -> dict:
    ref = sim_reference(sim, readout)
    movies = {
        name: movie_readout(
            np.concatenate([block_pairs(v["z"], v["px"], readout) for _, v in parts]),
            ref["band_p5_p95"],
        )
        for name, parts in by_movie(real).items()
    }
    return {"sim": ref, "movies": movies}


# ------------------------------ D: forecast -------------------------------------


def snippets(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every ``SNIPPET``-frame window at stride ``SNIPPET_STRIDE``, and its start."""
    starts = np.arange(0, len(z) - SNIPPET + 1, SNIPPET_STRIDE)
    if not len(starts):
        return np.zeros((0, SNIPPET, z.shape[-1])), starts
    return np.stack([z[s : s + SNIPPET] for s in starts]), starts


def errors(z: np.ndarray, basis, w) -> dict:
    """Mean |forecast - truth| per snippet and horizon ``(S, 24)`` for each method."""
    truth = z[:, CTX:]
    held = {
        "linear": forecast_linear(z, basis, w),
        "persistence": np.repeat(z[:, CTX - 1 : CTX], HORIZON, axis=1),
        "ctxmean": np.repeat(z[:, :CTX].mean(axis=1, keepdims=True), HORIZON, axis=1),
    }
    return {m: np.abs(f - truth).mean(axis=-1) for m, f in held.items()}


def gain_entry(err: dict, clusters: np.ndarray) -> dict:
    """Errors at the reported horizons, and the paired gain at +24 over the better
    baseline (the one with the lower mean error at +24 on these snippets)."""
    best = min(BASELINES, key=lambda m: err[m][:, -1].mean())
    gain = err[best][:, -1] - err["linear"][:, -1]
    return {
        "error": {
            m: {h: float(e[:, h - 1].mean()) for h in REPORT_HORIZONS}
            for m, e in err.items()
        },
        "baseline": best,
        "gain_24": float(gain.mean()),
        "gain_24_ci": bootstrap_ci(gain, clusters),
        "n_snippets": len(gain),
        "n_clusters": len(set(clusters.tolist())),
    }


def tile_errors(key: str, z: np.ndarray, basis, w) -> tuple[dict, np.ndarray]:
    """One tile's errors and clusters: (tile, non-overlapping 32-frame block)."""
    snip, starts = snippets(z)
    return errors(snip, basis, w), np.array([f"{key}|{s // SNIPPET}" for s in starts])


def merge(parts: list) -> tuple[dict, np.ndarray]:
    err = {m: np.concatenate([p[0][m] for p in parts]) for m in parts[0][0]}
    return err, np.concatenate([p[1] for p in parts])


def fit_on_cache(readout) -> tuple:
    """Candidate A fitted on the cache's train split, and its scores on the test split."""
    pooled, ctx = load_context(PATHS["pooled"], PATHS["gate1"])
    z_sim = (pooled["mean"].astype(np.float32) - readout.x_mu) / readout.x_sd
    basis, w, _ = fit_linear(z_sim, ctx["split"])
    test = ctx["split"]["test"]
    clips = np.array([ctx["meta"][i]["clip"] for i in test])
    return basis, w, gain_entry(errors(z_sim[test], basis, w), clips)


def forecast(real: dict, readout) -> dict:
    """D: candidate A fitted on the cache's train split, scored on its test split and
    on every real tile long enough for a snippet."""
    basis, w, sim_test = fit_on_cache(readout)
    out = {"sim_test": sim_test}
    every = []
    for name, parts in by_movie(real).items():
        scored = [
            tile_errors(k, (v["z"] - readout.x_mu) / readout.x_sd, basis, w)
            for k, v in parts
        ]
        scored = [s for s in scored if len(s[1])]
        if scored:
            out[name] = gain_entry(*merge(scored))
            every += scored
    out["real_pooled"] = gain_entry(*merge(every))
    return out


# ------------------------------ report ------------------------------------------


def print_containment(a: dict) -> None:
    print(
        f"\nA  containment (limit = largest held-out synthetic ratio {a['limit']:.2f})"
    )
    for n, m in a["movies"].items():
        print(
            f"   {n:34s} ratio {m['ratio']:5.2f}  "
            f"{'contained' if m['contained'] else 'NOT contained'}"
        )
    print(f"   positive control detected: {a['control_detected']}")


def print_readout(b: dict) -> None:
    print(f"\nB  readout / pixel radius, simulated band {b['sim']['band_p5_p95']}")
    for n, m in b["movies"].items():
        print(
            f"   {n:34s} ratio {m['median_ratio']:.2f} ({m['readout_px']:.1f} / "
            f"{m['pixel_px']:.1f} px) {'in range' if m['in_range'] else 'OUT'}   "
            f"C: rho readout {m['growth_rho_readout']:+.2f} "
            f"pixel {m['growth_rho_pixel']:+.2f}"
        )


def print_forecast(d: dict) -> None:
    print("\nD  forecast, gain at +24 over the better baseline (z-scored L1)")
    for n, m in d.items():
        lo, hi = m["gain_24_ci"]
        print(
            f"   {n:34s} vs {m['baseline']:11s} {m['gain_24']:+.4f} "
            f"[{lo:+.4f}, {hi:+.4f}]  n={m['n_snippets']} clusters={m['n_clusters']}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    out: Path = ap.parse_args().out
    readout = load_readout(PATHS["readout"])
    real, sim = load_or_encode()
    result = {
        "containment": containment(),
        "readout": readout_transfer(real, sim, readout),
        "forecast": forecast(real, readout),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=1, default=float))
    print_containment(result["containment"])
    print_readout(result["readout"])
    print_forecast(result["forecast"])


if __name__ == "__main__":
    main()
