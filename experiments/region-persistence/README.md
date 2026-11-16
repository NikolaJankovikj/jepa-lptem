# Region persistence: does forecasting at a coarser scale keep the particles?

**Date:** 2026-11-16 · **Scripts:** `scripts/token_persistence.py --cell` (persistence),
`scripts/region_predictability.py` (predictability) · **Tests:**
`tests/test_token_persistence.py`, `tests/test_region_predictability.py` · **Data:**
simulator ground truth only, 200 physics-stratum clips per process (seed 1), 2 windows
each, centre crop 256² · **Compute:** Mac, seconds.

## Question

Token-level forecasts fade because half the particles leave their 16 px token in one
latent step (`token-persistence/`). Pooling over coarser regions keeps them longer,
but a regional forecaster is only worth building if a region's future departs from
the frame average in a way the region's own context predicts. Otherwise it is the
pooled forecaster (`pooled-forecast/`) with extra noise.

## Pre-registered before the predictability run

The persistence table (below) was measured first; the predictability part was written
down before `region_predictability.py` ran.

- **Measure:** per region, the deviation of its particle count and total area from the
  frame mean at +h. R² of a ridge on frame + region context features, two-fold by clip,
  against a ridge on frame features alone. Clip-bootstrap 95 % CI. Noise-free readings,
  so an upper bound.
- **Prediction:** the frame-only fit is ~0 by construction. With ~5 particles per crop,
  a 64 px region holds ~0.3 particles, so the regional signal is coarse particle
  location. Region R² roughly follows the persistence table: ~0.6 at +1, ~0.3 at +10,
  ~0.1–0.2 at +24 at 64 px; higher at 128 px.
- **Decision rule:** a regional forecaster goes to a PC run only if, at 64 px, area R²
  at +10 is ≥ 0.2 with the CI above 0, on all processes pooled and on ripening or
  coalescence (where local growth could differ from the frame). Otherwise stop: the
  pooled forecaster already takes what the context offers.

## Results

### Persistence: fraction of particles whose centre is still in the same region

All processes, ~10,000 particles per horizon (`persistence-<cell>.json`); every
process is within ±0.02.

| region | tokens | +1 | +2 | +3 | +5 | +10 | +24 |
|---|---|---|---|---|---|---|---|
| 16 px | 1×1 | 0.49 | 0.37 | 0.30 | 0.23 | 0.15 | 0.06 |
| 32 px | 2×2 | 0.73 | 0.63 | 0.55 | 0.47 | 0.34 | 0.20 |
| 64 px | 4×4 | 0.88 | 0.83 | 0.78 | 0.72 | 0.61 | 0.45 |
| 128 px | 8×8 | 0.96 | 0.94 | 0.93 | 0.90 | 0.86 | 0.78 |

### Predictability: R² of the region's deviation from the frame mean (`predictability.json`)

Frame-only fit: 0.00 everywhere. Region fit, area (count within ±0.04 except where noted):

| 64 px (0.33 particles / region) | +1 | +5 | +10 | +24 |
|---|---|---|---|---|
| all | 0.74 | 0.47 | **0.31** [0.28, 0.33] | 0.11 |
| ripening | 0.73 | 0.51 | **0.38** [0.32, 0.43] | 0.14 |
| static | 0.77 | 0.50 | 0.36 | 0.12 |
| diffusion / reaction | 0.72 / 0.73 | 0.42 / 0.43 | 0.27 / 0.30 | 0.12 / 0.13 |
| coalescence | 0.67 | 0.38 | 0.17 [0.11, 0.22] | 0.06 |

| 128 px (1.33 particles / region) | +1 | +5 | +10 | +24 |
|---|---|---|---|---|
| all | 0.83 | 0.66 | 0.52 | 0.29 |
| ripening | 0.83 | 0.69 | 0.58 | 0.32 |
| coalescence | 0.73 | 0.52 | 0.34 | 0.11 (count 0.24) |

## Reading

1. **The decision rule passes.** At 64 px and +10, area R² is 0.31 on all processes and
   0.38 on ripening, both with the CI above 0.2. The prediction held to within ~0.05.
2. **What passes is coarse location, not local physics.** A 64 px region holds a third
   of a particle on average, so its "statistics" are whether a particle is there and
   how big it is. Static clips score as high as ripening (0.36 vs 0.38), so the signal
   is the same particle still being there, not growth that differs between places.
   Coalescence is lowest: particles disappear into merges.
3. **The pooled forecaster cannot see any of this** (frame-only R² 0.00), so a regional
   forecaster would add something new: where the particles are, at 64 px, up to ~+10.
   At +24 that is mostly gone (0.11).

## Caveats

- Noise-free ground truth. A forecaster has to read region count and area off the
  latent, and Gate 1 already reads frame-level count at only R² 0.73; a region with
  under one particle will read worse. The real R² will be lower than these.
- Centre-assigned regions; a particle on a border is split by its centre only.
- Physics stratum only; real video has ~9.5 px/frame steps, about as fast as the
  faster half of this stratum.
