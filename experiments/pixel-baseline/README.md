# Step 11: the ConvLSTM pixel-space baseline against candidate A

**Date:** 2026-11-21 · **Scripts:** `scripts/train_convlstm.py` (PC), `scripts/pixel_forecast_eval.py` (Mac, 9 min; test: `tests/test_pixel_forecast_eval.py`) · **Model:** `src/jepa_lptem/models/convlstm.py` · **Protocol:** `PLAN.md` Step 11, fixed before any number; verdict re-stated after P0 and its reading fixed by P0b v2, both before training. `metrics.json` is here; `convlstm.pt`, `forecasts.npz` (1.9 GB) and `encoded*.npz` are not in git.

## Question

Does a pixel-space forecaster trained on the same clips without labels carry growth
as well as candidate A, the label-free linear forecaster on the frame-averaged
V-JEPA latent?

## Setup

- **ConvLSTM** (2.6 MB of weights): encoder 256² → 64², 2 ConvLSTM layers of 64, decoder back; the output is the last frame plus an update whose head starts at zero, so the untrained model is persistence. Pixel MSE over a 48-frame rollout from 16 context frames, teacher forcing ramped 1 → 0 over epochs 1–15, 30 epochs on one 3090, checkpoint at the lowest val rollout MSE (epoch 30).
- **Snippets:** the 316 test physics snippets of `pooled-forecast/` (context = video frames s..s+15, forecast s+16..s+63, s ∈ {0, 36}); context 16 = frames 20–51 → 52–99, one per clip, reported.
- **Judged reading** (P0b v2): the radial-profile radius at the **true** future centres, 8-frame blocks, each timed at its middle frame; exponent = log-log slope over the 6 blocks. Candidate A keeps its pooled readout.
- **Verdict:** per snippet, `|α_ConvLSTM − α_true| − |α_A − α_true|`; latent beats pixel iff the median's clip-bootstrap CI is above 0 on diffusion and reaction.

## Results

### Verdict: **latent beats pixel**

| | median (ConvLSTM error − A error) | 95 % CI | snippets |
|---|---|---|---|
| diffusion | +0.249 | [+0.124, +0.546] | 52 |
| reaction | +0.660 | [+0.399, +1.386] | 54 |

### But the ConvLSTM's growth is blur, not physics

The ConvLSTM's exponent is too **high**, not too low, and it is as high on static
clips, where nothing grows. The radius at the true centres, median per 8-frame block
of the forecast (px):

| | block 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| static, true | 11.0 | 11.0 | 12.0 | 11.5 | 11.0 | 11.0 |
| static, ConvLSTM | 14.0 | 19.0 | 21.0 | 22.0 | 22.0 | 22.0 |
| diffusion, true | 8.0 | 8.5 | 9.0 | 10.0 | 10.5 | 11.0 |
| diffusion, ConvLSTM | 14.0 | 19.0 | 21.0 | 20.0 | 21.5 | 22.0 |
| reaction, true | 7.0 | 8.0 | 8.0 | 9.0 | 10.0 | 10.0 |
| reaction, ConvLSTM | 13.0 | 19.0 | 21.0 | 22.0 | 22.5 | 24.0 |

The ConvLSTM spreads every particle to about twice its width within 16 frames, the
same way in every process. The reading measures that spreading as growth. This is
the failure P0 predicted for an MSE pixel forecaster (blur), and the reading's limit
(P0: σ = 8 blur raises it by 2.5 px) is exceeded.

### Exponents over the forecast, median [CI], context 8

| reading | diffusion | reaction | static | ripening | coalescence |
|---|---|---|---|---|---|
| **ConvLSTM, radius at true centres (judged)** | **0.467** [0.28, 0.70] | **0.806** [0.43, 1.09] | **0.566** [0.43, 0.63] | 0.594 | 0.428 |
| true frames, same reading | 0.393 [0.32, 0.43] | 0.508 [0.46, 0.54] | −0.012 | 0.000 | 0.066 |
| **candidate A, pooled readout** | **0.208** [0.17, 0.25] | **0.269** [0.24, 0.31] | **0.008** [−0.02, 0.04] | 0.034 | 0.035 |
| ConvLSTM, dark mass (reported) | 0.191 [0.16, 0.24] | 0.279 [0.20, 0.35] | 0.082 [0.06, 0.13] | 0.136 | 0.169 |
| true frames, dark mass | 0.269 [0.21, 0.32] | 0.386 [0.31, 0.43] | −0.032 | −0.004 | 0.018 |
| ConvLSTM, attentive probe (P0 warning) | −0.421 | −0.538 | +0.029 | −0.722 | −0.441 |
| true | 0.333 | 0.500 | 0 | – | – |

The judged reading works on true frames (0.393 / 0.508 / −0.012), so the failure is
the forecast's, not the instrument's. Dark mass, which blur cannot change, gives the
ConvLSTM exponents close to A's (0.191 / 0.279 against 0.208 / 0.269), about 70 % of
what dark mass reads on the true frames, but also spurious growth where there is none
(static 0.082, ripening 0.136, coalescence 0.169, against A's 0.008 / 0.034 / 0.035).
Dark mass failed its premise check (P0b v2) and is not judged.

### Pixel MSE, mean over test snippets

| forecast frame | +1 | +10 | +20 | +48 |
|---|---|---|---|---|
| ConvLSTM | 0.0073 | 0.0074 | 0.0074 | 0.0075 |
| persistence (last frame held) | 0.0144 | 0.0146 | 0.0147 | 0.0148 |

The ConvLSTM halves persistence's error from the first frame on, and its error barely
rises with the horizon. Shot noise dominates the MSE: a forecast of the noise-free
mean halves it whatever the dynamics, which is what the ConvLSTM learned (val MSE
flat from epoch 3).

### Pooled path, reported under P0's warning

Through V-JEPA and the frozen ridge readout, the ConvLSTM fails Gate 2′ on every
criterion: +24 gain over pooled trend −5.17 [−6.47, −3.83] px (diffusion) and −4.36
[−5.60, −3.20] (reaction); mean-radius MAE at +1 2.19 / 1.89 px against persistence's
0.73 / 0.73; exponents −0.42 / −0.33 / static −0.32. P0 showed this path reads a
σ = 1 px blur as a 5.7 px smaller radius, so these numbers measure the readout's
response to smooth frames more than the forecast.

### Context 16 (reported, unpaired)

ConvLSTM, radius at the true centres: diffusion 0.758 [0.45, 1.00], reaction 1.062
[0.81, 1.83], static 0.786 [0.53, 0.98]; the same reading on true frames 0.404 / 0.439
/ −0.000. Design A at context 16 (`longer-context/`): 0.287 / 0.340 / 0.004. The longer
rollout from a later start blurs further.

## Predictions (written before P0)

1. P0.1 passes; blur leaves total area unchanged and biases radius up at σ ≥ 4. **Wrong** (P0: the stop rule fired).
2. The ConvLSTM fades: detection F1 at +5 below persistence, peak occupancy below 0.6 by +10. **Not tested**: the encode step saved the attentive readings, not the occupancy map. The attentive probe reads the ConvLSTM's particles as shrinking (exponents −0.42 / −0.54), which fits, under P0's warning.
3. It carries some growth (gain over the reference at +24 above 0 on reaction), but latent beats pixel. **First half wrong** (−4.36 px on the pooled path), **second half right**.
4. Its exponents are below A's and its static exponent is within ±0.1 of 0; Gate 2′ FAIL. **Wrong on the judged reading** (above A's, static 0.566), **mixed on dark mass** (0.191 below 0.208, 0.279 above 0.269, static 0.082 within ±0.1); **Gate 2′ FAIL right**.
5. At +1 its mean-radius MAE is within 0.1 px of persistence. **Wrong** (2.19 / 1.89 against 0.73 / 0.73, pooled path).

## Conclusion

Latent beats pixel, as pre-registered, on both processes. The reason is not that
the ConvLSTM under-forecasts growth: it trades the particles' shape for a lower pixel
MSE, spreading them at the same rate on every process, so its "growth" is the same
on static clips as on growing ones. A blur-proof reading recovers some real growth
from it, comparable to A's, but with spurious growth on the processes that have none.
Candidate A's growth is weak (0.208 / 0.269) but specific to the processes that grow.
