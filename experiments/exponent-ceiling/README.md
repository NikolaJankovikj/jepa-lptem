# Exponent ceiling: can a context-only forecaster meet Gate 2′'s exponent bar?

**Date:** 2026-10-11 · **Script:** `scripts/exponent_ceiling.py` (test: `tests/test_exponent_ceiling.py`) · **Data:** simulator ground truth only, the `token-persistence` windows (200 physics clips per process, seed 1; empty-frame windows dropped), split by clip · **Compute:** Mac, seconds · **Protocol:** `PLAN.md` Step 10, *Exponent ceiling from the context*, fixed and committed (`f851731`) before it ran. `metrics.json` is here.

## Question

Gate 2′ requires a forecast's growth exponent over +1..+24 to be within ±0.1 of the
true one. Candidate A reached 0.208 / 0.269 (`experiments/pooled-forecast/`); even the
label-reading readout ridge reached only 0.272 / 0.337. Is the bar reachable at all
for a forecaster that sees only 8 context frames read at probe quality?

## Results: median exponent over the forecast (test windows)

| context 8 | diffusion | reaction | static |
|---|---|---|---|
| truth in the crop | 0.334 | 0.499 | 0.000 |
| ridge, noise-free context | 0.304 | 0.342 | 0.077 |
| kNN, noise-free context | 0.339 | 0.494 | 0.024 |
| **ridge, sd 0.8 px** | **0.263** | **0.313** | 0.090 |
| **kNN, sd 0.8 px** | **0.194** | **0.222** | 0.062 |
| process given (reference) | 0.265 | 0.580 | 0.000 |
| process + window start given (reference) | 0.333 | 0.500 | 0.000 |

| context 16 (forecast +1..+16; reported, not judged) | diffusion | reaction | static |
|---|---|---|---|
| ridge, sd 0.8 px | 0.300 | 0.357 | 0.062 |
| kNN, sd 0.8 px | 0.299 | 0.357 | 0.035 |
| kNN, noise-free context | 0.343 | 0.489 | 0.013 |

**Reading, as pre-registered: the bar is not reachable at context 8.** At probe-quality
noise neither context-only extrapolator comes within ±0.1 on both processes. Gate 2′'s
exponent criterion is therefore recorded as **unpowered at context 8**, as the original
Gate 2 size criterion was. Candidate A's recorded FAIL stands, and P3 is judged as
pre-registered.

## Reading

1. **The information is in the trajectory, the noise hides it.** With a noise-free
   context, kNN recovers the exponents almost exactly (0.339 / 0.494). With the
   probe's reading noise (sd 0.8 px) every context-only method falls to 0.19–0.31.
2. **The shortfall is shrinkage, not a modelling bug.** A point forecast under
   uncertainty pulls each clip toward the average of the clips its context resembles.
   That damps growing clips and lifts static ones (ridge's static 0.09). Candidate A's
   0.208 / 0.269 sits in this range, so its damped exponents are about what 8 noisy
   frames allow.
3. **What a forecaster would need to know.** The two references show the bottleneck.
   Knowing the process alone is not enough (reaction 0.580: the two window starts sit
   at different points of the growth curve). Knowing process *and* where on the curve
   the clip is gives the law exactly. From 8 noisy frames, neither can be inferred well
   enough.
4. **A longer context helps but does not close it.** At context 16 diffusion comes
   within the bar (0.300), but reaction stays at 0.357, still 0.14 short. Against
   that, the forecast window shrinks from 24 to 16 steps.

## Consequence

- The exponent bar, as written, measures the context's noise more than the
  forecaster. Applied to a point forecast from 8 frames, it cannot separate a good
  forecaster from a bad one. A future gate needs a bar measured against this ceiling,
  for example "within ±0.1 of the context-only ridge at the same noise" or
  "exponent ordered correctly and static ≈ 0". That has to be pre-registered
  separately; it does not change Gate 2′.
- The size criterion, which candidate A passed, is powered and remains the
  informative test.

## Caveats

- "Not reachable" is shown for two reasonable estimators, not proven for every
  estimator. Any point forecast faces the same shrinkage, but a cleverer one could
  close part of the gap.
- Readings are truth plus independent Gaussian noise of 0.8 px. The pooled readout's
  error on the true future is 0.64–0.78 px MAE on the growth processes, i.e. sd about
  0.8–1.0 (`experiments/pooled-forecast/`, post hoc diagnostic). So 0.8 is, if
  anything, generous.
