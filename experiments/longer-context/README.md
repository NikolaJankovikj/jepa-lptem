# Longer context: is the exponent shortfall the data's or the model's?

**Date:** 2026-10-11 · **Script:** `scripts/longer_context.py` (test: `tests/test_longer_context.py`) · **Data:** the pooled `mean` state (`data/features/dynamics-v1-perframe-pooled.npz`), each clip's two windows stitched into 50 latent frames; frozen P1 readout; Gate-1 per-frame split · **Compute:** Mac CPU, 22 min (mostly transformer training) · **Protocol:** `PLAN.md` Step 10, *Longer context*, fixed and committed (`5af9306`) with the ceiling and predictions before any candidate ran. `ceiling.json` and `metrics.json` are here; the run log is `logs/longer-context.log` (not in git).

## Question

At context 8 both label-free candidates' exponents (A 0.208 / 0.269, B 0.180 / 0.245)
sit inside the context-only ceiling (`experiments/exponent-ceiling/`). That fits two
stories: the data allows no more, or the model takes less than the data allows. The
ceiling rises with context. Do the candidates rise with it?

## Design

Per-frame features encode each tubelet alone, so a clip's windows at video frames
0–63 and 36–99 stitch into one 50-latent-frame sequence (shared frames agree at
cosine 1.000, targets identical). Snippets are cut so that every context of a design
forecasts the **same frames** (+1..+24), which makes contexts a paired comparison:

| design | contexts | forecast begins at latent frame | snippets |
|---|---|---|---|
| **A (primary)** | 8, 12, 16 | 16, 21, 26 | 6000 |
| B (reported) | 8, 16, 24 | 24, 26 | 4000 |

Design B's two nearby forecast positions tell the forecaster roughly where on the
growth curve it is, at every context alike. Candidates, readout, references and
Gate 2′ criteria are unchanged from P2/P3 (`pooled_forecast.evaluate`); each candidate
is trained afresh per snippet set.

## Results (test, physics stratum; true exponents 0.333 / 0.500 / 0)

### Median exponent over +1..+24 (diffusion / reaction / static)

| design A | ceiling, best at sd 0.8 | readout ridge (labels) | **linear (A)** | transformer (B) |
|---|---|---|---|---|
| context 8 | 0.228 / 0.279 | 0.213 / 0.284 / 0.099 | **0.192 / 0.235 / 0.014** | 0.180 / 0.221 / 0.126 |
| context 12 | 0.268 / 0.335 | 0.261 / 0.342 / 0.055 | **0.274 / 0.308 / 0.010** | 0.182 / 0.225 / 0.116 |
| context 16 | 0.315 / 0.385 | 0.305 / 0.378 / 0.038 | **0.287 / 0.340 / 0.004** | 0.190 / 0.248 / 0.106 |

| design B | ceiling, best at sd 0.8 | readout ridge (labels) | **linear (A)** | transformer (B) |
|---|---|---|---|---|
| context 8 | 0.211 / 0.266 | 0.206 / 0.266 / 0.130 | **0.164 / 0.193 / 0.041** | 0.170 / 0.193 / 0.129 |
| context 16 | 0.315 / 0.381 | 0.275 / 0.359 / 0.064 | **0.271 / 0.332 / −0.007** | 0.164 / 0.195 / 0.110 |
| context 24 | 0.353 / 0.425 | 0.364 / 0.419 / 0.025 | **0.319 / 0.363 / 0.009** | 0.179 / 0.210 / 0.107 |

The floor (readout on the true future) reads 0.309 / 0.479 (A) and 0.311 / 0.467 (B).

### Primary reading: paired exponent rise (median per snippet, 95 % clip CI)

| | diffusion | reaction | static | tracks | rise ÷ ceiling's rise |
|---|---|---|---|---|---|
| **A 8→16, linear** | **+0.097** [+0.079, +0.107] | **+0.104** [+0.083, +0.125] | −0.013 [−0.038, +0.002] | **yes** | 1.11 / 0.98 |
| A 8→16, transformer | +0.014 [+0.007, +0.020] | +0.027 [+0.017, +0.031] | −0.019 [−0.024, −0.013] | yes (formally) | 0.16 / 0.25 |
| B 8→24, linear | +0.161 [+0.137, +0.182] | +0.163 [+0.147, +0.193] | −0.038 [−0.075, −0.017] | yes | 1.13 / 1.03 |
| B 8→24, transformer | +0.003 [−0.002, +0.011] | +0.009 [+0.007, +0.018] | −0.021 [−0.028, −0.014] | no | 0.02 / 0.06 |

"Rise ÷ ceiling's rise" divides a median of paired differences by a difference of
medians; read it as approximate.

### Size at +24 (paired gain over the pooled trend, px, 95 % clip CI)

| design A | linear diffusion | linear reaction | transformer diffusion | transformer reaction |
|---|---|---|---|---|
| context 8 | +0.22 [+0.01, +0.42] | +0.29 [+0.04, +0.52] | +0.28 [+0.08, +0.47] | +0.29 [+0.08, +0.50] |
| context 12 | +0.55 [+0.36, +0.75] | +0.64 [+0.42, +0.84] | +0.52 [+0.31, +0.71] | +0.69 [+0.51, +0.87] |
| context 16 | +0.55 [+0.32, +0.80] | +0.88 [+0.65, +1.09] | +0.56 [+0.36, +0.75] | +0.86 [+0.65, +1.07] |

The pooled trend is the reference on every process and context; the floor is powered
everywhere (gain +0.96 / +1.70 on A).

### Gate 2′ criteria per context (reported, not a re-run of the gate)

Not met at any context by either candidate. The linear candidate meets the size
criterion at every context of design A and from context 16 on design B, the diffusion
exponent from context 12 on, and static at every context. It misses only the reaction exponent: best 0.363 at B 24,
0.137 short of 0.5. The ceiling itself misses reaction on design A (0.385 at context 16)
and reaches 0.425 only on design B at 24. The transformer misses every exponent and
static at every context. **Gate 2′'s FAIL at context 8 stands.**

## Predictions vs outcome

| written before the run | outcome |
|---|---|
| Both candidates track, rising 50–100 % of the ceiling's rise | **Linear: yes, ~100 %. Transformer: no** — CI above 0 on A, but 16–25 % of the ceiling's rise, and nothing on B |
| Linear near 0.27 / 0.33 at A 16 | 0.287 / 0.340 |
| Transformer a little lower than linear | Far lower: 0.190 / 0.248 |
| Linear static below 0.1; transformer's spurious static growth shrinks | Linear 0.004. Transformer 0.126 → 0.106, still failing static |
| A 16: size passes, diffusion exponent within ±0.1, reaction not | As predicted |
| B 24: reaction short of ±0.1 by a few hundredths | 0.363, 0.037 short of the bar |

## Reading

1. **For the linear candidate, the exponent shortfall is the data's.** Its exponents
   rise one-for-one with the ceiling (paired rise ≈ the ceiling's rise, CIs well
   above 0 on both designs). The gap to the ceiling does not widen: 0.00–0.07 below
   it at every context. On design A they match the label-trained readout ridge within
   0.04 from context 12 on. A label-free linear
   map on the frame-averaged V-JEPA latent takes what the context offers.
2. **Growth forecasting improves with context.** On design A, from context 8 to 16 the
   linear candidate's +24 size gain over the pooled trend grows from +0.22 to +0.55 px
   (diffusion) and +0.29 to +0.88 px (reaction). At context 16 its diffusion exponent is
   0.287 against a true 0.333.
3. **The transformer is model-limited.** It matches the linear candidate's +24 size gain
   at every context, but its exponents barely move with context, and it keeps inventing
   growth on static clips (~0.11). Its +1 mean-radius error is worse than the linear
   candidate's (A 16: 0.80 vs 0.65 px diffusion, 0.93 vs 0.71 reaction), while +24 is
   equal. So the shortfall looks like the *shape* of its forecast curve, not what it
   extracts from the context. Its checkpoint is selected on val L1 at +24 only, which
   does not constrain the early horizons. That is a hypothesis; it is not tested here.
4. **Reaction's exponent is bounded by context and noise, not by the latent.** Even the
   ceiling reaches the ±0.1 bar on reaction only at design B context 24, where the
   forecaster nearly knows its position on the curve.

## Consequence

- The pooled result for the thesis is candidate A against the ceiling curve. It carries
  clip-specific growth, and its exponent error is what 8–24 noisy context frames allow.
  The context-8 Gate 2′ FAIL is reported with this as its explanation, not replaced by it.
- The transformer adds nothing over the linear map at any context. It stays a reported
  negative and is not tuned further here; selecting it on all horizons would test
  reading 3.

## Caveats

- One seed per candidate per context; the transformer's checkpoint epoch varies (26–48).
- Design B's absolute exponents are inflated by its near-fixed forecast position. Its
  paired rise is the informative number.
- The ceiling uses fresh simulator clips (seed 1), not the cached dataset's clips, with
  Gaussian reading noise of 0.8 px standing in for the readout's error (0.64–0.78 px
  MAE on the true future, `experiments/pooled-forecast/`).
