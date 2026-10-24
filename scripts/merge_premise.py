"""RQ2 premise: is a merge visible at all, in the pixels and in position-free numbers?

PLAN Step 12, fixed 2026-10-24 before it ran. Ground truth only: fresh coalescence
physics clips (100 frames), rendered exactly as dynamics-v1 and centre-cropped to 256.
The events, patches, negatives and lags are event_anticipation's, scored with:

- ``pixel``          |change| of each token's mean (event_signals.pixel_signal; v2)
- ``obs_noisy``      sum over [count, mean radius, total area] of |change| / sd, with
                     Gaussian noise at the P1 readout's error on true frames added
- ``obs_clean``      the same without the noise

Gate, then stop rule (v2): the alignment check (particle patches vs background) must
have a CI above 0.5, else ``pixel`` is blind and the premise is untestable; then, if
``pixel``'s AUC at lag 0 has a CI not above 0.5, merges are not visible at this SNR
and there is no PC run. Runs on the Mac in ~5 min (rendering):
    uv run python scripts/merge_premise.py --out experiments/events \
        --pooled data/features/dynamics-v1-perframe-pooled.npz \
        --gate1 experiments/gate1-perframe --readout experiments/pooled-gate1/readout.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from event_anticipation import (
    CELL,
    GRID,
    NEAR_PX,
    T_MIN,
    ClipGT,
    Points,
    above,
    auc_table,
    sample_points,
    score,
    token_negatives,
)
from event_signals import TUBELET, crop_gt, pixel_signal
from generate_dynamics_dataset import load_mtf
from pooled_forecast import load_readout
from pooled_gate1 import load_context
from train_probes import attentive_targets

from jepa_lptem.sim.dataset import build_clip
from jepa_lptem.sim.render_bridge import render_video, to_uint8

FULL, CROP, N_FRAMES = 512, 256, 100
MTF_PATH = Path("code/LPTEMsimulator/MTFmatrix512.mat")
BACKGROUND_SAMPLE = 20000


def readout_sd(args) -> np.ndarray:
    """The P1 readout's error sd per observable on true test physics frames that
    show a particle."""
    pooled, ctx = load_context(args.pooled, args.gate1)
    readout = load_readout(args.readout)
    rows = ctx["split"]["test"][ctx["physics"][ctx["split"]["test"]]]
    pred = readout.predict(pooled["mean"][rows].astype(np.float32))
    truth = ctx["targets"][rows]
    seen = truth[..., 0] > 0
    return (pred[seen] - truth[seen]).std(axis=0)


def render_clip(rng, mtf) -> dict:
    """One rendered coalescence clip: cropped frames and crop-frame ground truth."""
    pos_t, rad_t, meta, pos, rad = build_clip(
        "coalescence", N_FRAMES, (FULL, FULL), rng
    )
    video = render_video(
        pos_t,
        rad_t,
        (FULL, FULL),
        mtf=mtf,
        rng=rng,
        pixel_size_nm=meta["pixel_size_nm"],
        beam_flux=meta["beam_flux"],
        thickness_sd_nm=meta["thickness_sd_nm"],
    )
    frames = np.stack([to_uint8(f, beam_flux=meta["beam_flux"]) for f in video])
    off = (FULL - CROP) // 2
    merges = np.asarray(meta["merge_events"], dtype=int).reshape(-1, 3)
    return {
        "frames": frames[:, off : off + CROP, off : off + CROP],
        "gt": ClipGT(*crop_gt(pos, rad, FULL, CROP), merges, N_FRAMES // TUBELET),
    }


def simulate(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    mtf = load_mtf(MTF_PATH, FULL)
    return [render_clip(rng, mtf) for _ in range(n)]


def observable_change(gt: ClipGT, sd: np.ndarray, rng) -> np.ndarray:
    """``(T_lat,)`` sum of |change| / sd of the crop observables, noise ``sd`` added
    (pass ``rng=None`` for noise-free)."""
    obs = attentive_targets(gt.rad[:, ::TUBELET])
    if rng is not None:
        obs = obs + rng.normal(0.0, sd, obs.shape)
    out = np.full(len(obs), np.nan)
    out[1:] = (np.abs(np.diff(obs, axis=0)) / sd).sum(axis=1)
    return out


def background_patches(gt: ClipGT) -> list[tuple[int, tuple[int, int]]]:
    """Tokens whose centre is over NEAR_PX from every visible particle."""
    centres = (np.indices((GRID, GRID)).reshape(2, -1).T + 0.5) * CELL
    out = []
    for t in range(T_MIN, gt.n_lat - 1):
        seen = gt.pos[gt.rad[:, 2 * t] > 0, 2 * t]
        dist = np.hypot(*(centres[:, None] - seen[None]).transpose(2, 0, 1))
        far = (dist > NEAR_PX).all(axis=1) if len(seen) else np.ones(len(centres), bool)
        out += [(t, (int(k) // GRID, int(k) % GRID)) for k in np.flatnonzero(far)]
    return out


def alignment_check(gts: list[ClipGT], pixel: list, seed: int) -> dict:
    """Post hoc sanity check of the patch coordinates: pixel change around particles
    against empty background (a random BACKGROUND_SAMPLE of its patches)."""
    particles = [
        (c, n["t"], n["token"]) for c, g in enumerate(gts) for n in token_negatives(g)
    ]
    background = [
        (c, t, k) for c, g in enumerate(gts) for t, k in background_patches(g)
    ]
    keep = np.random.default_rng(seed).choice(
        len(background), BACKGROUND_SAMPLE, replace=False
    )
    points = Points(particles, [background[i] for i in sorted(keep)])
    table = auc_table(score(points, {"pixel": pixel}, lags=(0,)), (), lags=(0,))
    return table["auc"]["pixel"][0]


def pixel_table(clips: list[dict], seed: int) -> dict:
    gts = [c["gt"] for c in clips]
    points = sample_points(gts, np.arange(len(clips)), per_token=True)
    signal = {"pixel": [pixel_signal(c["frames"]) for c in clips]}
    return {
        "n_events": len(points.events),
        "alignment_particle_vs_background": alignment_check(gts, signal["pixel"], seed),
    } | auc_table(score(points, signal), ())


def observable_table(gts: list[ClipGT], sd: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    signal = {
        "obs_noisy": [observable_change(g, sd, rng) for g in gts],
        "obs_clean": [observable_change(g, sd, None) for g in gts],
    }
    points = sample_points(gts, np.arange(len(gts)), per_token=False)
    return auc_table(score(points, signal), (("obs_noisy", "obs_clean"),))


def run(args) -> dict:
    sd = readout_sd(args)
    clips = simulate(args.n_clips, args.seed)
    token = pixel_table(clips, args.seed)
    alignment = token.pop("alignment_particle_vs_background")
    frame = observable_table([c["gt"] for c in clips], sd, args.seed)
    return {
        "readout_sd": dict(
            zip(("count", "mean_radius", "total_area"), sd.tolist(), strict=True)
        ),
        "n_clips": len(clips),
        "n_events": token.pop("n_events"),
        "alignment_particle_vs_background": alignment,
        "token": token,
        "frame": frame,
        "pixel_sees_particles": above(alignment),
        "pixel_visible": above(token["auc"]["pixel"][0]),
        "position_free_visible": above(frame["auc"]["obs_noisy"][0]),
    }


def print_report(result: dict) -> None:
    print(f"readout error sd on true frames: {result['readout_sd']}")
    print(f"{result['n_clips']} clips, {result['n_events']} events")
    for block in ("token", "frame"):
        for sig, by_lag in result[block]["auc"].items():
            print(
                f"  AUC {sig:10s} "
                + "  ".join(
                    f"{lag:+d} {v['auc']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]"
                    for lag, v in by_lag.items()
                )
            )
    alignment = result["alignment_particle_vs_background"]
    print(
        f"alignment gate: pixel AUC particle vs background {alignment['auc']:.3f} [{alignment['ci'][0]:.3f}, {alignment['ci'][1]:.3f}]"
    )
    print(
        f"pixel sees particles: {result['pixel_sees_particles']}  (False -> premise untestable)"
    )
    print(f"pixel visible at lag 0: {result['pixel_visible']}  (False -> no PC run)")
    print(f"position-free visible at lag 0: {result['position_free_visible']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for flag in ("--out", "--pooled", "--gate1", "--readout"):
        ap.add_argument(flag, type=Path, required=True)
    ap.add_argument("--n-clips", type=int, default=100)
    ap.add_argument("--seed", type=int, default=3)  # v1 ran on seed 2
    args = ap.parse_args()
    result = run(args)
    print_report(result)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "premise.json").write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}/premise.json")


if __name__ == "__main__":
    main()
