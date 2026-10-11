# P2: forecasting the pooled state — baselines, linear candidate, Gate 2′

**Date:** 2026-10-11 · **Script:** `scripts/pooled_forecast.py` (test: `tests/test_pooled_forecast.py`) · **Data:** the pooled `mean` state (`experiments/pooled-gate1/`), Gate-1 per-frame split, one snippet per window (context = latent frames 0–7, forecast = 8–31) · **Readout:** frozen `experiments/pooled-gate1/readout.npz` · **Compute:** Mac, 3 s · **Protocol:** `PLAN.md` Step 10, *Position-free forecasting*, fixed 2026-10-11 before any pooled number. `metrics.json` is here; the run log is `logs/pooled-forecast.log` (not in git).

## Question

Does a forecaster trained without labels on the frame-averaged latent carry growth
physics into the future, beyond trend-blind extrapolation of the present?

## Methods

All are read by the same frozen P1 readout; verdict on test, physics stratum.

| | what | labels? |
|---|---|---|
| persistence / context mean | frame 7 / the average of frames 0–7, held | no |
| pooled trend | frame 7's readout × the median train ratio of the true mean radius at +h (one curve for all processes) | ratio from train labels |
| readout ridge | `growth_identifiability`'s ridge on the 8 context readouts — how much the context allows; a reference, never a candidate | yes |
| floor | the true future pooled state | – |
| **linear (candidate A)** | top-64 PCA coefficients of the 8 context frames → coefficients at +1..+24 (ridge, λ on val), projected back | **no** |

## Results

### Growth test at +24 (paired gain in mean-radius error over the reference, 95 % clip-bootstrap CI)

| | reference (MAE +24) | floor | readout ridge | **linear** |
|---|---|---|---|---|
| diffusion | pooled trend (2.01 px) | +1.34 [+1.04, +1.63] | +0.73 [+0.52, +0.95] | **+0.42 [+0.15, +0.68]** |
| reaction | pooled trend (2.87 px) | +2.13 [+1.76, +2.56] | +0.96 [+0.75, +1.18] | **+0.66 [+0.35, +0.98]** |

Both processes are powered. The linear candidate closes 57 % (diffusion) and 68 %
(reaction) of the readout ridge's gain.

### Growth exponent over +1..+24 (median [CI])

| | diffusion | reaction | static |
|---|---|---|---|
| **linear** | **0.208** [0.170, 0.246] | **0.269** [0.239, 0.314] | **0.008** [−0.017, 0.038] |
| readout ridge | 0.272 [0.240, 0.292] | 0.337 [0.315, 0.363] | 0.041 |
| pooled trend | 0.168 | 0.194 | 0.157 |
| persistence / context mean | 0.000 | 0.000 | 0.000 |
| floor | 0.311 | 0.473 | 0.009 |
| true | 0.333 | 0.500 | 0 |

### Mean-radius MAE (px), +1 / +5 / +10 / +24

| | persistence | context mean | pooled trend | readout ridge | linear | floor |
|---|---|---|---|---|---|---|
| diffusion | 0.73 / 0.95 / 1.62 / 3.12 | 1.09 / 1.67 / 2.38 / 3.93 | 0.73 / 0.80 / 1.14 / 2.01 | 0.58 / 0.66 / 0.85 / 1.28 | 0.68 / 0.75 / 1.01 / 1.59 | 0.80 / 0.66 / 0.62 / 0.67 |
| reaction | 0.73 / 1.13 / 1.86 / 3.98 | 1.00 / 1.70 / 2.57 / 4.68 | 0.73 / 0.91 / 1.37 / 2.87 | 0.60 / 0.74 / 0.99 / 1.91 | 0.73 / 0.87 / 1.21 / 2.22 | 0.68 / 0.77 / 0.73 / 0.75 |
| static | 0.81 / 0.88 / 0.85 / 0.87 | 0.65 / 0.72 / 0.68 / 0.70 | 0.81 / 0.96 / 1.18 / 1.90 | 0.71 / 0.84 / 0.91 / 1.20 | 0.94 / 1.03 / 1.09 / 1.38 | 0.73 / 0.88 / 0.82 / 0.83 |

Ripening and coalescence are in `metrics.json`. On both, no method beats persistence or
the context mean by much, and the linear candidate is the worst (1.57 / 1.60 at +24).

Gaussian-W2 size distance at +24 (reported; count is weak): linear 2.62 / 3.30 px against
persistence 3.81 / 4.37 and floor 2.11 / 2.24 on diffusion / reaction.

**Verdict (candidate A): FAIL, as pre-registered.** The size criterion passes on both
powered processes, and static stays at 0. But the exponents (0.208, 0.269) are not
within ±0.1 of 1/3 and 1/2.

## Reading

1. **For the first time, a label-free forecast carries growth.** The linear map on the
   frame-averaged latent beats the best trend-blind baseline at +24 on both growth
   processes (CIs above 0). It does not invent growth on static clips (0.008), where one
   pooled trend for all processes predicts 0.157. So it tells growing from non-growing
   clips and extrapolates each clip's own trend. Every token-level forecaster
   (`experiments/core-perframe/`, `forecast-probe/`) failed exactly this.
2. **The growth it carries is damped.** The exponents are roughly 60 % of the true
   values, ordered correctly (reaction > diffusion > static ≈ 0). That is the familiar
   shrinkage of a regularised linear forecaster toward "no change". λ was picked at the
   edge of the pre-registered grid (10⁴, val L1 still falling), so the fit wanted even
   more regularisation.
3. **The exponent criterion may be out of reach from this context.** This is a post hoc
   observation, not a change to the gate. The readout ridge reads the context directly
   with labels, and it reaches only 0.272 / 0.337, also outside ±0.1 on reaction. Eight
   latent frames (16 video frames) pin down the size trend well enough to beat
   persistence, but not well enough to fix the exponent over a 48-frame extrapolation.
   That echoes the original Gate 2's size criterion, set before its floor was measured
   (`PLAN.md`, protocol errors).
4. **Where the remaining error comes from (post hoc diagnostic).** Projecting the
   *true* future onto the 64 PCA components already raises the readout's MAE by
   0.15–0.2 px (diffusion 0.64 → 0.81, static 0.81 → 0.98). Candidate A cannot see
   anything outside those components, which explains part of its excess error on static,
   ripening and coalescence.

## What it means

The position-free reframing works where the token level could not: a forecaster that
never saw a label carries clip-specific growth into the future. It does not yet carry
the growth *law* with the required precision. Next, as pre-registered: P3 (candidate B,
the small causal transformer), judged by the same Gate 2′. Whether the exponent
criterion can be met from 8 frames at all deserves its own measured floor before any
decision is drawn from it.

---

# P3: candidate B, the small causal transformer (2026-10-11)

**Script:** `scripts/pooled_transformer.py` (test: `tests/test_pooled_transformer.py`) · same data, readout and Gate 2′ as P2 (`pooled_forecast.evaluate`) · **Compute:** Mac CPU, 127 s · `metrics_p3.json` (results + training history) is here; `transformer.pt` stays out of git; the run log is `logs/pooled-transformer.log`.

Trained as pre-registered: 100 epochs, best val L1 at +24 at epoch 38 (0.484; 0.560 at
epoch 1, 0.494 at epoch 100). The model is a change from the last context frame,
zero-initialised, so it starts as persistence.

| | linear (A) | **transformer (B)** | readout ridge | floor |
|---|---|---|---|---|
| gain over pooled trend at +24, diffusion | +0.42 [+0.15, +0.68] | **+0.54 [+0.30, +0.77]** | +0.73 | +1.34 |
| gain over pooled trend at +24, reaction | +0.66 [+0.35, +0.98] | **+0.81 [+0.59, +1.03]** | +0.96 | +2.13 |
| gap closed to readout ridge | 57 % / 68 % | **73 % / 84 %** | – | – |
| exponent, diffusion / reaction | 0.208 / 0.269 | **0.180 / 0.245** | 0.272 / 0.337 | 0.311 / 0.473 |
| exponent, static | 0.008 | **0.110** [0.100, 0.125] | 0.041 | 0.009 |
| mean-radius MAE at +1, diffusion / reaction | 0.68 / 0.73 | 1.08 / 1.03 | 0.58 / 0.60 | 0.80 / 0.68 |

Transformer minus linear at +24 (paired, reported): diffusion +0.12 [−0.18, +0.41],
reaction +0.16 [−0.12, +0.43], static +0.02 [−0.32, +0.36]. No significant difference.

**Verdict (candidate B): FAIL, as pre-registered.** The size criterion passes on both
processes, but the exponents (0.180 / 0.245) miss ±0.1. The exponent criterion is
unpowered at context 8 (`experiments/exponent-ceiling/`). B also invents growth on
static clips (0.110 > 0.1).

## Reading

1. **The transformer carries a little more size at +24 than the linear map, but not
   significantly.** It closes 73–84 % of the readout ridge's gain against A's 57–68 %,
   and its CIs are tighter. The paired difference to A spans 0 on every process.
2. **It buys that partly with a population prior.** Its static exponent is 0.110 and
   its exponents on the growing processes are flatter than A's. That is the signature
   of the pooled trend: some average growth applied to every clip. It has learned more
   of "clips usually grow" and less of "this clip grows like this" than the linear map.
3. **It is worse at short range.** At +1 it is 0.3–0.4 px worse than persistence,
   because one loss averaged over all 24 horizons is dominated by the long horizons,
   where change is large.
4. **Taken together with A and the ceiling:** two label-free forecasters of the pooled
   state, one linear and one non-linear, both carry clip-specific growth beyond any
   trend-blind baseline (powered, CIs above 0). Both land where 8 noisy context frames
   allow (`exponent-ceiling`). The bottleneck is now the information in the context,
   not the forecaster.
