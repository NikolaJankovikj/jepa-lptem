# Gate 2 on per-frame features: the first forecast evaluation without look-ahead

**Date:** 2026-10-03 · **Scripts:** `scripts/eval_forecasts.py`, `scripts/gate2_followup.py`, unchanged apart from reading the Gate-1 threshold (`load_gate1`) · **Predictor:** `experiments/predictor-perframe/` (v1's defaults, trained from scratch on the per-frame cache) · **Probes:** `experiments/gate1-perframe/probes.pt`, frozen, threshold 0.95 · **Data:** the 400 Gate-1 test windows · **Compute:** 1× RTX 3090.

## Question

v1's Gate-2 numbers were read off contexts that had seen the future (`experiments/leak-check/`).
Here every latent sees only its own 2 frames. Read by the same frozen probes, do the
forecasts carry the physics: sizes growing at the right rate, particles in the right place?

## Protocol

Unchanged from Gate 2 and its follow-up (`PLAN.md` Step 10, fixed before the first run),
and from the per-frame run's entry, which was fixed before training.

## Results

### Gate 2: 10-frame forecasts, verdict at +5

| | model | persistence | context mean | floor | model vs persistence | floor vs persistence |
|---|---|---|---|---|---|---|
| coalescence | 4.41 | 2.53 | 3.01 | 2.98 | −74 % | −18 % |
| diffusion | 3.41 | 1.94 | 2.73 | 1.75 | **−76 %** | +10 % |
| reaction | 3.72 | 2.07 | 2.86 | 2.08 | **−80 %** | −1 % |
| ripening | 4.66 | 2.57 | 2.78 | 3.03 | **−82 %** | −18 % |
| static | 3.51 | 2.84 | 2.64 | 2.68 | −24 % | +6 % |

Size distance in px; positive means closer to the truth than persistence. The verdict
rows are diffusion, reaction and ripening.

**Growth exponent** (median, interquartile range in brackets):

| | model | floor | true value |
|---|---|---|---|
| coalescence | −1.22 [−2.01, −0.31] | 0.055 | – |
| diffusion | **−0.83** [−1.74, −0.16] | 0.348 | 0.333 |
| reaction | **−1.25** [−2.24, −0.25] | 0.498 | 0.500 |
| ripening | −0.91 [−1.51, −0.51] | 0.037 | – |
| static | −0.43 [−1.35, −0.10] | 0.002 | 0.000 |

**Detection F1** for model / persistence / context mean / floor:

| | +1 | +5 | +10 |
|---|---|---|---|
| diffusion | 0.61 / 0.58 / 0.33 / 0.86 | 0.14 / 0.22 / 0.16 / 0.86 | 0.02 / 0.14 / 0.11 / 0.87 |
| static | 0.63 / 0.58 / 0.31 / 0.86 | 0.16 / 0.24 / 0.18 / 0.85 | 0.04 / 0.13 / 0.11 / 0.86 |

The other processes look the same. At +1 the model is 0.02–0.06 above persistence
everywhere. By +5 it is below persistence everywhere.

**Verdict: FAIL as pre-registered.** At +5 the model's size distance is 76–82 % *worse*
than persistence. Its exponents are negative where they should be +0.33 and +0.50.

### Follow-up: +24 rollout, 95 % clip-bootstrap CIs

| | floor gain [CI] | model gain [CI] | result |
|---|---|---|---|
| diffusion | +1.51 [+1.08, +1.91] | −4.91 [−5.64, −4.23] | **FAIL** |
| reaction | +2.09 [+1.60, +2.47] | −4.46 [−5.08, −3.91] | **FAIL** |
| ripening | −0.94 [−2.51, +0.21] | −4.73 [−5.88, −3.54] | untestable |

On the per-frame cache **reaction is powered** (it was not on the whole-window cache), so
both growth modes are tested now, and both fail.

Exponent over 24 frames:
- diffusion −1.22 [−1.36, −0.96] (floor 0.350), FAIL;
- reaction −1.12 [−1.30, −1.02] (floor 0.520), FAIL;
- static −1.30, where it should be 0.

**What the forecasts look like** (static clips, model / floor):

| | +1 | +5 | +10 | +24 |
|---|---|---|---|---|
| probed count | 5.6 / 6.0 | 5.5 / 5.9 | 4.9 / 5.7 | 4.1 / 5.6 |
| mean radius (px) | 12.6 / 12.6 | 12.1 / 12.4 | 10.1 / 12.7 | **5.0** / 12.6 |
| peak occupancy | 0.98 / 1.00 | 0.75 / 1.00 | 0.47 / 1.00 | **0.29** / 1.00 |

By +24 **every process** reads alike: count about 4, radius 4.2–5.0 px, peak occupancy
0.29–0.44. The floor stays sharp all the way out (F1 0.83–0.88, occupancy ≥ 0.99).

## Reading

1. **The honest predictor has learned no usable dynamics in probe space.**
   - At +1 it barely beats persistence on positions (+0.02–0.06 F1, no CI). At +1 its size
     distance is already worse than persistence on diffusion and reaction.
   - From then on it is worse on every metric. This agrees with the leak check, where v1
     on causal contexts fell to persistence level.
   - v1's apparent skill was the look-ahead.
2. **The failure is fading, not wrong physics.**
   - Peak occupancy falls from 0.98 to 0.29: particles dissolve into the background.
   - The radius probe reads a dissolving particle as a shrinking one. The negative
     "exponents" are fading measured on a log scale, not predicted shrinkage.
   - Every process converges to the same generic state. It looks like the predictor's
     unconditional average of a per-frame latent.
3. **Why per-frame fades faster than v1.**
   - A deterministic L1 forecaster predicts the per-channel median of the futures it
     cannot tell apart.
   - Per-frame latents carry what is unpredictable: Brownian displacement and per-frame
     shot noise, which the whole-window encoder averaged away (`experiments/gate1-perframe/`,
     secondary analyses).
   - So the median of the possible futures smears particles out within a few steps. v1's
     targets had seen their own context, which made them easier to predict.
   - This is the standard regression-to-the-mean failure of deterministic forecasting.
     V-JEPA 2-AC trains the same way (L1, rollout of 2), but it never decodes its
     forecasts. It only ranks them by distance to a goal.
4. **The encoder and probes are not the bottleneck.**
   - On the true future (floor), the probes recover the exponents (0.348 / 0.498 against
     0.333 / 0.500) and keep F1 ≥ 0.83 out to +24.
   - The physics is readable from per-frame latents. It is the forecast that loses it.
5. **A weak residue of process information, not tested.** Offset against static, the
   +24 exponents order reaction (+0.18) > diffusion (+0.08) > static (0), as the truth
   does. But coalescence (+0.21) sits above both, so this is not evidence of growth
   physics.

## Training check (2026-10-03)

`experiments/predictor-perframe/README.md`:
- The predictor converged (val L1 flat from epoch 30).
- It beats the context mean by 4.2 % in feature space at +1, against 9.3 % for v1, the
  same for every process.

So the test set out above resolves it: **better in L1, worse in probe space. The cause is
the objective, not the training.**

## Open

- What comes next is a decision for `PLAN.md` Step 10, recorded before the next run.
