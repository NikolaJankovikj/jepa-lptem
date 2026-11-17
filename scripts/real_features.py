"""Encode the screened real movies and a fresh simulated reference set, once.

Both sides go through the pipeline exactly as the pooled cache did: 256 crops
(never resized), each 2-frame tubelet encoded alone by frozen V-JEPA 2, the 256
tokens averaged per latent frame. The pixels are kept beside the latents so that
`real_validation.py` can run its pixel estimator on the same frames. Cached in
`data/real-figshare/` (not in git); `real_validation.py` calls `load_or_encode`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from extract_features import crop_and_align_gt, make_encoder
from real_video_qc import REAL_MOVIES, load_movie
from train_probes import attentive_targets

TILE, TUBELET = 256, 2
N_SIM, SIM_FRAMES, SIM_FULL, SIM_SEED = 40, 64, 512, 13
READOUT_TIER = 2  # PLAN Step 13: 1 containment only, 2 readout, 3 forecast
MEAN_RADIUS = 1  # column of attentive_targets
MTF_PATH = Path("code/LPTEMsimulator/MTFmatrix512.mat")
REAL_CACHE = Path("data/real-figshare/real_pooled.npz")
SIM_CACHE = Path("data/real-figshare/sim_pooled.npz")



def tiles(frames: np.ndarray) -> dict:
    """Non-overlapping 256 crops of a ``(T, H, W)`` movie, even ``T``, by position."""
    t = len(frames) // TUBELET * TUBELET
    _, h, w = frames.shape
    return {
        f"r{r}c{c}": frames[:t, r : r + TILE, c : c + TILE]
        for r in range(0, h - TILE + 1, TILE)
        for c in range(0, w - TILE + 1, TILE)
    }


def pooled_sequence(encode, frames: np.ndarray, chunk: int = 64) -> np.ndarray:
    """Per-frame encoding, tokens averaged: ``(T, 256, 256)`` -> ``(T / 2, 1024)``."""
    return np.concatenate([
        encode(frames[i : i + chunk]).astype(np.float32).mean(axis=1)
        for i in range(0, len(frames), chunk)
    ])


def encode_real(encode) -> dict:
    """``{movie|tile: {z, px}}`` for every tile of the readout-tier movies."""
    out = {}
    for name, spec in REAL_MOVIES.items():
        if spec["tier"] < READOUT_TIER:
            continue
        for where, frames in tiles(load_movie(name)).items():
            out[f"{name}|{where}"] = {
                "z": pooled_sequence(encode, frames),
                "px": frames,
            }
            print(f"  encoded {name} {where}: {len(frames)} frames")
    return out


def process_and_stratum(i: int) -> tuple[str, str]:
    """Clip ``i``'s process and stratum: the cycle and diagonal of
    `generate_dynamics_dataset.py`, so 1 clip in 5 is experiment-grade on every
    process."""
    from jepa_lptem.sim.dataset import EXPERIMENT_EVERY, PROCESSES

    proc, round_ = i % len(PROCESSES), i // len(PROCESSES)
    stratum = "experiment" if (round_ + proc) % EXPERIMENT_EVERY == 0 else "physics"
    return PROCESSES[proc], stratum


def render_clip(i: int, rng, mtf) -> dict:
    """Clip ``i`` of the fresh set, rendered as dynamics-v1, centre-cropped, with the
    true mean radius per latent frame."""
    from jepa_lptem.sim.dataset import build_clip
    from jepa_lptem.sim.render_bridge import render_video, to_uint8

    process, stratum = process_and_stratum(i)
    size = (SIM_FULL, SIM_FULL)
    pos_t, rad_t, meta, pos, rad = build_clip(process, SIM_FRAMES, size, rng, stratum)
    video = render_video(
        pos_t,
        rad_t,
        size,
        mtf=mtf,
        rng=rng,
        pixel_size_nm=meta["pixel_size_nm"],
        beam_flux=meta["beam_flux"],
        thickness_sd_nm=meta["thickness_sd_nm"],
    )
    off = (SIM_FULL - TILE) // 2
    frames = np.stack([to_uint8(f, beam_flux=meta["beam_flux"]) for f in video])
    _, r = crop_and_align_gt(pos, rad, 0, SIM_FRAMES, SIM_FULL, TILE)
    return {
        "px": frames[:, off : off + TILE, off : off + TILE],
        "truth": attentive_targets(r)[:, MEAN_RADIUS],
    }


def encode_sim(encode) -> dict:
    from generate_dynamics_dataset import load_mtf

    rng = np.random.default_rng(SIM_SEED)
    mtf = load_mtf(MTF_PATH, SIM_FULL)
    clips = {str(i): render_clip(i, rng, mtf) for i in range(N_SIM)}
    for c in clips.values():
        c["z"] = pooled_sequence(encode, c["px"])
    return clips


def save(path: Path, items: dict) -> None:
    np.savez(
        path, **{f"{k}#{part}": v for k, d in items.items() for part, v in d.items()}
    )


def load(path: Path) -> dict:
    z, out = np.load(path), {}
    for key in z.files:
        name, part = key.rsplit("#", 1)
        out.setdefault(name, {})[part] = z[key]
    return out


def load_or_encode() -> tuple[dict, dict]:
    """Real tiles and fresh simulated clips, encoded once and cached."""
    if not REAL_CACHE.exists():
        encode = make_encoder("mps", per_frame=True)
        save(REAL_CACHE, encode_real(encode))
        save(SIM_CACHE, encode_sim(encode))
    return load(REAL_CACHE), load(SIM_CACHE)
