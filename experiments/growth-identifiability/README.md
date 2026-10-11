# Growth identifiability: is a clip's growth predictable from its context?

**Date:** 2026-10-11 · **Script:** `scripts/growth_identifiability.py` (test: `tests/test_growth_identifiability.py`) · **Data:** simulator ground truth only, the same 2000 windows as `experiments/token-persistence/` (1950 kept; 50 have a frame with no visible particle), split 50/50 by clip · **Compute:** Mac, seconds. `metrics.json` is here.

## Question

The forecast-trained readout found no growth in the per-frame forecasts
(`experiments/forecast-probe/`). Before redesigning the forecaster: is a clip's growth
recoverable at all from 8 context frames read at probe quality? And how much can a
forecaster gain at most from the context, as opposed to the true future (the floor)?

## Protocol

Mean radius of the crop-visible particles per latent frame. The context is frames 0–7;
the forecast at +h is scored against the true value at 7+h. "Readings" are the true
context values plus Gaussian noise. sd 0.8 px matches the probe's floor MAE of
0.61–0.70 px in `forecast-probe/` (MAE = 0.8 sd); sd 0 is the noise-free bound.
Extrapolators see only the readings: `persistence`, `pooled_trend` (one growth curve
for all processes), `line_fit`, `ridge` (20 terms, all processes pooled, no labels),
and `process_trend` (true process given: an oracle reference).

## Results: mean-radius MAE (px) on test clips, reading sd 0.8 px

| | persistence | pooled trend | line fit | **ridge** | process given (oracle) |
|---|---|---|---|---|---|
| diffusion +24 | 3.26 | 2.26 | 4.33 | **1.31** | 1.60 |
| reaction +24 | 3.90 | 2.92 | 3.13 | **1.86** | 1.79 |
| static +24 | 0.82 | 1.52 | 3.23 | 1.35 | 0.82 |
| ripening +24 | 1.72 | 1.66 | 4.78 | 1.64 | 1.60 |
| coalescence +24 | 1.14 | 1.32 | 3.19 | 1.44 | 1.09 |

Growing (diffusion, reaction) vs not, from the context slope alone: accuracy **0.76**
(chance 0.56). Noise-free: 0.84; ridge at +24 then 0.89 / 1.57 px on diffusion / reaction.
All horizons and both noise levels are in `metrics.json`.

## Reading

1. **Growth is identifiable from the context at probe quality.** On the growing
   processes ridge cuts the +24 error by 2.5× against persistence (3.26 → 1.31,
   3.90 → 1.86), and by 1.6–1.7× against one pooled growth curve.
2. **The gain comes from recognising the regime, not from extending the line.** A
   straight-line fit is worse than doing nothing on every process, because the curve
   bends and the readings are noisy. Ridge matches or beats the oracle that is given
   the process, so it also reads where on the growth curve the clip is.
3. **The pooled extrapolator pays on non-growing clips.** Ridge adds spurious growth
   on static (1.35 vs persistence 0.82). Any bar must be judged per process, and a
   forecaster must not grow static clips.
4. **This re-calibrates the forecast-probe test, which was probably underpowered.**
   - Its reference was the h-aware persistence probe. That probe reached 1.51 / 2.15 px
     at +24 (diffusion / reaction), better than `pooled_trend` here (2.26 / 2.92). So
     it already reads part of the regime from a single frame.
   - If these numbers transfer, the context leaves about 0.2–0.3 px for a forecaster
     to gain over it (ridge 1.31 / 1.86). The test's 95 % CIs are ±0.25–0.3 px wide.
   - The predictor's gains (−0.23 [−0.50, +0.03] and +0.04 [−0.21, +0.30]) are
     therefore below that on diffusion, but inconclusive on reaction. "No growth in
     the forecasts" stands as recorded; it was a weak test.
   - The floor's gain (+0.91 / +1.45 px) mostly reflects information that is not in
     the context at all.
5. **Use as a bar.** Any new forecaster of growth should beat `ridge` per process, not
   only persistence or the context mean.

## Caveats

- Readings are truth plus independent Gaussian noise. A real probe's errors may be
  correlated over time or biased per clip.
- The mean radius moves when particles enter or leave the crop. That part is
  unpredictable and is included in every number here, as it is in Gate 2.
