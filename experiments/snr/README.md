# RQ3: does the pooled forecast degrade at low dose?

**Date:** 2026-10-24 · **Script:** `scripts/snr_split.py` (test: `tests/test_snr_split.py`) · **Data:** the pooled `mean` state (`data/features/dynamics-v1-perframe-pooled.npz`), stitched into 50-frame clips as in `longer-context`; frozen P1 readout; Gate-1 per-frame split; simulator ground truth for the ceiling (200 clips per process, seed 1) · **Compute:** Mac CPU, 40 s · **Protocol:** `PLAN.md` Step 10, *RQ3 — forecast quality vs dose*, fixed and committed (`5d02778`) with predictions before any binned number existed. `metrics.json` is here; the run log is `logs/snr.log` (not in git).

## Question

RQ3 asks how forecast quality degrades with prediction horizon and with SNR. The
horizon half is in every pooled table (+1, +5, +10, +24). This is the SNR half. The
exponent ceiling found that reading noise, not missing information, limits the
context-8 exponents. If that holds for the model, high-dose clips should forecast
better.

## Design

- **Variable:** beam flux (electrons per pixel per frame), drawn log-uniform on 6–75
  in the physics stratum. Two bins split at √(6·75) = 21.2, the geometric middle of
  the range. Median flux: 12.3 (low) and 39.3 (high). Mean water thickness is 150 nm in
  every clip, so dose is the only shot-noise knob in dynamics-v1. Thickness
  *variation* (15–55 nm, split at 35) is reported the same way.
- **Setting:** longer-context design A (forecast frames start at latent frame 16, 21,
  26), contexts 8 and 16, candidate A (linear) only, trained on every dose and split
  only for scoring. The unsplit numbers reproduce `experiments/longer-context/` exactly.
- **Primary reading (context 16):** per growing process, Δ = the linear candidate's +24
  mean-radius absolute error, low-dose mean minus high-dose mean, with a two-sample
  clip-bootstrap CI. Forecasts **degrade at low dose** iff the CI lies above 0.

## Results (test, physics stratum)

### Reading noise: the P1 readout on true frames (px)

| beam flux | diffusion MAE | reaction MAE | static MAE | all: MAE / bias / sd-equivalent |
|---|---|---|---|---|
| low (< 21.2) | 0.76 | 1.05 | 1.06 | 0.97 / −0.23 / **1.21** |
| high (≥ 21.2) | 0.58 | 0.64 | 0.58 | 0.70 / +0.01 / **0.88** |

At low dose the readout errs about 40 % more and on average reads particles small
(bias −0.23 px; −0.63 on static, +0.03 on diffusion). The previous analyses used one sd of 0.8 px for all doses.

### Ceiling at each bin's reading noise (best of ridge / kNN, median exponent)

| | diffusion | reaction | static |
|---|---|---|---|
| truth | 0.333 | 0.500 | 0 |
| context 8, noise-free | 0.344 | 0.467 | 0.034 |
| context 8, low dose (sd 1.21) | 0.212 | 0.259 | 0.116 |
| context 8, high dose (sd 0.88) | 0.219 | 0.270 | 0.110 |
| context 16, noise-free | 0.347 | 0.488 | 0.021 |
| context 16, low dose (sd 1.21) | 0.285 | 0.337 | 0.044 |
| context 16, high dose (sd 0.88) | 0.308 | 0.371 | 0.035 |

### Forecasts, context 16, split by dose

| | clips | MAE +24 linear / trend / floor | gain over trend +24 | exponent linear [CI] | exponent floor |
|---|---|---|---|---|---|
| low, diffusion | 17 | 1.23 / 1.81 / 0.74 | +0.59 [+0.23, +0.93] | 0.273 [0.254, 0.314] | 0.314 |
| high, diffusion | 14 | 0.91 / 1.41 / 0.59 | +0.51 [+0.19, +0.78] | 0.294 [0.256, 0.333] | 0.294 |
| low, reaction | 10 | 1.89 / 2.36 / 0.87 | +0.47 [+0.09, +0.84] | 0.311 [0.248, 0.343] | 0.423 |
| high, reaction | 22 | 1.61 / 2.67 / 0.87 | +1.06 [+0.82, +1.32] | 0.365 [0.317, 0.419] | 0.502 |
| low, static | 18 | 1.52 / 1.55 / 1.15 | +0.02 [−0.53, +0.59] | −0.019 | 0.012 |
| high, static | 21 | 1.14 / 1.76 / 0.69 | +0.62 [+0.26, +0.99] | +0.026 | 0.003 |

Linear candidate's mean-radius MAE by horizon, context 16 (low / high dose):

| | +1 | +5 | +10 | +24 |
|---|---|---|---|---|
| diffusion | 0.84 / 0.43 | 0.82 / 0.45 | 0.93 / 0.57 | 1.23 / 0.91 |
| reaction | 1.15 / 0.51 | 1.28 / 0.66 | 1.40 / 0.88 | 1.89 / 1.61 |
| static | 1.10 / 0.60 | 1.21 / 0.62 | 1.21 / 0.80 | 1.52 / 1.14 |

### Primary reading: low minus high dose, absolute error (px, 95 % two-sample clip CI)

| | linear +24 (**judged**) | floor +24 | linear +1 (post hoc) | floor +1 (post hoc) |
|---|---|---|---|---|
| diffusion | +0.32 [−0.20, +1.03] | +0.15 [−0.06, +0.37] | +0.41 [+0.10, +0.77] | +0.08 [−0.17, +0.36] |
| reaction | +0.28 [−0.61, +1.27] | +0.00 [−0.54, +0.57] | +0.64 [+0.07, +1.47] | +0.73 [+0.16, +1.48] |
| static | +0.39 [−0.10, +0.95] | +0.46 [+0.09, +0.86] | +0.49 [+0.06, +0.96] | +0.52 [+0.21, +0.86] |

**Primary: not shown on either growing process** (CIs include 0). Context 8 is the
same: +0.22 / +0.34 / +0.35 at +24, every CI across 0. The +1 column was added after
the run, when the horizon table showed the gap was largest there.

### Thickness variation (reported)

At context 16 the linear candidate's +24 Δ (thin minus thick variation) is +0.14
[−0.40, +0.92], +0.60 [−0.15, +1.46] and −0.38 [−0.87, +0.09]. The floor's Δ is
significant on reaction (+0.49) and on static (−0.47), with opposite signs. That
pattern follows which ~15 clips land in each bin, not thickness.

## Predictions vs outcome

| written before the run | outcome |
|---|---|
| Reading noise higher at low dose, sd-equivalent ~0.9 vs ~0.65 | Higher, as predicted, but larger at both doses: 1.21 vs 0.88 |
| Ceiling: high dose a few hundredths closer to the truth, bins within 0.05 | As predicted: 0.007–0.034 apart |
| Primary: Δ > 0 on both, +0.2 to +0.5 px, CI above 0 on at most one | As predicted: +0.32 / +0.28, CI above 0 on neither |
| The floor's Δ about half the linear candidate's | Diffusion yes (0.15 vs 0.32); reaction no (0.00 vs 0.28) |
| Exponents: high dose closer by 0.02–0.05, CIs overlap; static within ±0.05 | As predicted: +0.021 / +0.054, CIs overlap; static −0.019 / +0.026 |
| Thickness variation: no effect | No consistent effect, but two floor differences reach significance in opposite directions |

## Reading

1. **Dose acts through the reading.** At low dose the readout's error is about 40 %
   larger (sd-equivalent 1.21 vs 0.88 px), and on average it reads particles small. That is the
   measurement the forecasts start from.
2. **The low-dose penalty does not grow with horizon.** It is 0.3–0.6 px at every
   horizon, while the high-dose forecast error itself grows from 0.4–0.6 px at +1 to
   0.9–1.6 px at +24. So dose
   matters most at short horizons, where the error is mostly reading error. At +24 the
   growth extrapolation dominates, and the penalty falls inside the noise of a ~15-clip
   split. That is why the pre-registered +24 test is not significant while the post hoc
   +1 test is, on all three processes.
3. **Context length matters more than dose for exponents.** Over this dose range
   (median flux 12 vs 39) the ceiling moves by at most 0.034. Doubling the context
   (8 → 16) moves it by 0.07–0.10. The model behaves the same way: its exponents move
   by 0.02–0.05 between the dose bins, against about 0.10 between contexts
   (`experiments/longer-context/`). Within 6–75 e/px, more frames help growth recovery
   more than brighter frames.
4. **Part of the low-dose exponent loss is the readout's, not the forecast's.** On
   reaction the floor (readout on the *true* future) reads 0.423 at low dose against
   0.502 at high dose. The readout flattens true growth at low dose before any
   forecast is made.

## Consequence

- RQ3 answer for the thesis: forecast quality degrades strongly with horizon and
  mildly with dose. Dose costs about 0.3–0.6 px of mean-radius error at every horizon,
  through the readout. The growth exponent is limited by how many frames the context
  has, more than by dose, over 6–75 e/px.
- The "liquid thickness" half of RQ3 cannot be answered with dynamics-v1: mean water
  thickness is fixed at 150 nm. Answering it needs clips that vary mean thickness.
- The single reading sd of 0.8 px used by `growth-identifiability`, `exponent-ceiling`
  and `longer-context` is about right on average (0.70 MAE at high dose, 0.97 at low)
  but hides a 1.4× spread across doses.

## Caveats

- Bins are unbalanced on reaction (10 vs 22 clips) and differ in mean true radius
  there (11.9 vs 10.5 px). Dose was drawn independently of everything else, so this is
  chance, but with ~15 clips per cell one or two clips move a bin's mean.
- The ceiling models the reading error as iid Gaussian with the bin's sd-equivalent.
  The real error has a dose-dependent bias, which the ceiling leaves out.
- The +1 effect was computed after the run; it is reported as post hoc.
- "More frames beat brighter frames" compares changes in the ceiling, not two designs
  at an equal electron budget. Testing that would need clips rendered at matched total
  dose.
