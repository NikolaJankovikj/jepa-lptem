# dynamics-v1 — the synthetic LPTEM dataset

**Date:** 2026-09-29 · **Script:** `scripts/generate_dynamics_dataset.py` (commit `78d73ae`) · **Audit:** `scripts/audit_dataset.py` · **Compute:** work PC, one CPU core, ~3 s/clip · **Location:** `data/dynamics-v1/` on the PC (not in git; ~20 MB/clip, ~40 GB).

```bash
uv run python scripts/generate_dynamics_dataset.py --out data/dynamics-v1 \
    --n-videos 2000 --n-frames 100          # --size 512, --seed 0 (defaults)
```

## What it is

2000 simulated liquid-phase TEM videos of nanoparticles with an exact answer key: the
known physics (Step 3 dynamics) pushed through our re-implementation of the LPTEMsimulator
image formation (`sim/render_bridge.py`: Beer–Lambert through gold/water/SiNx, Poisson dose,
calibrated detector MTF, bulging-cell thickness field, stage drift).

| | |
|---|---|
| Processes | diffusion-limited growth (r ∝ t^⅓), reaction-limited growth (r ∝ t^½), static, Ostwald ripening (mass-conserving LSW), coalescence |
| Strata | `physics` (8–22 px particles, 6–16 per clip, dose 6–75) — where gates are decided; `experiment` (real-video imaging conditions, ≤2 % coverage, dose 8–12) — puts real data inside the training distribution |
| Per clip | `frames` uint8 (100, 512, 512); `positions` (N, 100, 2); `radii` (N, 100), 0 = absent; `merge_events` (M, 3); scalars: process, exponent, D, drift, msd_max_lag, dose, pixel size, thickness sd, … |
| Manifest | `manifest.jsonl`, one line per clip with every scalar |

Imaging conditions are *randomised, not fitted* (Step 4g): the criterion is that real
experimental video lies inside the generated range on every statistic that survives the
published frames' unknown contrast stretch — apparent radius, |dI|/mean, and the
temporal/spatial noise ratio all passed (Step 4j). Latent-space containment: ratio 0.92
against 1.00 for held-out synthetic (`experiments/latent-gap/`).

## Audit of the full run

| check | result |
|---|---|
| clips on disk / manifest lines | 2000 / 2000 |
| unreadable files | 0 |
| frame shape | (100, 512, 512) uint8 for all 2000 |
| process × stratum | exactly 320 physics + 80 experiment for each of the 5 processes |
| coalescence merge events | 1035 (~2.6 per coalescence clip) — the Step 12 event set |
| clips with an empty frame | 0 |
| black/saturated pixels | mean 0.27 %, worst clip 2.89 % |

The black-pixel tail is physics, not clipping: at the lowest doses (flux ~6) behind the
thickest liquid a pixel receives ~4 electrons on average, so a few percent receive none, and
a counting detector reads 0 there too. White saturation needs 4× the background and does
not occur. The worst clip of 2000 sits above the ~1.2 % seen in 20–25-clip smoke runs
because it is further into the same tail (Step 4h measured 2.7 % at the old dose floor).

## Regression guards

`tests/test_dataset_physics.py` grades the answer key itself on freshly generated clips
(15 test functions, 41 parametrised cases): growth exponents to ±0.001, MSD = 4Dt pooled across clips (per-clip MSD at these
particle counts is noise, Step 4e) with the lag it is valid over, peak coverage under each
stratum's cap, nothing labelled below the visibility floor, no brightness→size leak,
background structure bracketing real video, bigger particles diffusing slower, ripening
actually coarsening, no size-distribution collapse, drift as a coherent component, and
timed volume-conserving merges.

## Open

- Physics round trip on a sample of the full set (`scripts/verify_clips.py`, one clip per
  process) — run on the smoke sets throughout Step 4, not yet on dynamics-v1.
- The ~4.5 particles in view per 256² window (Step 6) are set by the 9 % coverage cap, not by
  the crop: rendering at 256² would move the reflecting walls into the image without raising
  the count.
