# Forecast-trained readout: does growth survive under the blur?

**Date:** 2026-10-04 · **Script:** `scripts/forecast_probe.py` · **Predictor:** `experiments/predictor-perframe/` (frozen) · **Data:** per-frame cache, Gate-1 split (3200 train / 400 val / 400 test windows), context 8, horizon 24 · **Compute:** 1× RTX 3090. `metrics.json` and `logs/forecast-probe.log` are here. **Post hoc**, protocol fixed in `PLAN.md` Step 10 on 2026-10-03, before it ran.

## Question

With the frozen Gate-1 probes the per-frame forecasts fade below persistence
(`experiments/core-perframe/`). Is the growth signal still in the forecast latents,
just unreadable by probes trained on sharp frames? Each source of the latent at +h
gets its own probe, trained to read the true size at +h:

- **model:** the predictor's rollout;
- **persistence:** context frame 7, held;
- **context mean:** the 8 context frames averaged, held;
- **floor:** the true future latent.

Every probe knows h. So the two baseline probes are learned extrapolators from the
context, and they are the bar the forecast must clear.

## Results

### Mean-radius error (px), model / persistence / context mean / floor

| | +1 | +5 | +10 | +24 |
|---|---|---|---|---|
| coalescence | 0.91 / 0.88 / 0.84 / 0.65 | 1.03 / 0.92 / 0.85 / 0.72 | 1.13 / 1.06 / 1.03 / 0.71 | 1.62 / 1.40 / 1.46 / 0.72 |
| diffusion | 0.81 / 0.83 / 0.79 / 0.71 | 0.94 / 0.83 / 0.97 / 0.60 | 1.15 / 1.01 / 1.26 / 0.60 | 1.75 / **1.51** / 2.07 / 0.61 |
| reaction | 0.71 / 0.82 / 0.84 / 0.82 | 0.84 / 0.80 / 1.04 / 0.71 | 1.17 / 1.04 / 1.36 / 0.61 | 2.12 / **2.15** / 2.60 / 0.70 |
| ripening | 0.89 / 1.10 / 0.95 / 0.96 | 1.33 / 1.39 / 1.24 / 0.93 | 1.24 / 1.40 / 1.07 / 0.87 | 1.39 / 1.59 / 1.32 / 0.92 |
| static | 0.84 / 1.01 / 1.00 / 0.59 | 1.01 / 1.06 / 1.06 / 0.60 | 1.24 / 1.13 / 1.10 / 0.69 | 1.59 / 1.72 / 1.46 / 0.63 |

Bold: the reference baseline at +24 (persistence on both processes).

### Growth test at +24 (gain in mean-radius error over the reference, 95 % clip-bootstrap CI)

| | floor gain | model gain | powered | growth in the forecasts |
|---|---|---|---|---|
| diffusion | +0.91 [+0.58, +1.25] | −0.23 [−0.50, +0.03] | yes | **no** |
| reaction | +1.45 [+1.04, +1.95] | +0.04 [−0.21, +0.30] | yes | **no** |

### Growth exponent over +1..+24, median [CI] (reported, not judged)

| | model | persistence | context mean | floor | true |
|---|---|---|---|---|---|
| coalescence | 0.144 [0.11, 0.18] | 0.153 [0.14, 0.20] | 0.121 [0.11, 0.13] | 0.036 | – |
| diffusion | 0.183 [0.14, 0.23] | 0.204 [0.18, 0.23] | 0.166 [0.15, 0.19] | 0.318 | 0.333 |
| reaction | 0.305 [0.26, 0.34] | 0.240 [0.22, 0.27] | 0.211 [0.17, 0.25] | 0.464 | 0.500 |
| ripening | 0.088 [0.07, 0.12] | 0.138 [0.12, 0.20] | 0.099 [0.08, 0.13] | 0.079 | – |
| static | 0.075 [0.05, 0.11] | 0.128 [0.11, 0.14] | 0.094 [0.09, 0.10] | 0.012 | 0.000 |

**Verdict: no, as pre-registered.** Both processes are powered: read from the true
future, the probe beats the best extrapolator by 0.9–1.5 px. Read from the forecast, it
does not: the gain is −0.23 px on diffusion and +0.04 px on reaction, both CIs spanning 0.

## Reading

1. **The forecast carries no size dynamics beyond the last context frame.** The best
   extrapolator from frame 7 matches it at +24 on every process to within ±0.25 px.
   The floor shows 0.9–1.5 px is there to be gained.
2. **The fade in Gate 2 was mostly the frozen probes.**
   - With the frozen Gate-1 probes the forecast was 4.9 px (diffusion) and 4.5 px
     (reaction) worse than persistence at +24. With probes trained on forecasts it is
     within 0.25 px.
   - So the blurred latents still hold the context's sizes. Probes trained on sharp
     frames read blur as small, faint particles.
   - Gate 2 is defined with the frozen probes, so its FAIL stands. But "particles
     dissolve" (`core-perframe` reading 2) describes what those probes see, not lost
     information.
3. **The extrapolator probes learn a pooled growth prior.**
   - Persistence and context mean predict growth on static clips (0.13, 0.09) and too
     little on diffusion and reaction. They apply roughly one average trend to every
     process.
   - The model's exponents spread further by process: reaction − static is 0.23,
     against 0.11 (persistence), 0.12 (context mean) and 0.45 (floor).
   - So the forecast may carry some process identity. But it does not lower the size
     error, so it is a hint, not evidence.
4. **The model's probe fits more and generalises no better.** Train loss falls to 0.24
   against persistence's 0.32, but val stays at the same level (0.28–0.34, noisy by
   ±0.03 per epoch). The forecast varies more over h, and none of that variation carries
   information about the future.
5. **Single cells are noisy.** At +1 on reaction the floor (0.82) is no better than the
   model (0.71). Readout noise per cell is about 0.1–0.2 px, so only the +24 test with
   its CIs is read.

## What it means

The deterministic per-token L1 objective did not teach the predictor growth, not even as
an average trend. Per-token loss is dominated by positions and per-frame noise, which are
unpredictable. The size trend is a small part of it. As pre-registered, the next step is
**to redesign the objective so it targets the predictable quantities**, not a
sharper output for the same objective.

## Open

- The redesign is a decision for `PLAN.md` Step 10, recorded before it runs.
