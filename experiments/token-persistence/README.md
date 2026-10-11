# Token persistence: how long does a particle stay in its token?

**Date:** 2026-10-11 · **Script:** `scripts/token_persistence.py` (test: `tests/test_token_persistence.py`) · **Data:** simulator ground truth only, no rendering or encoder: 200 physics-stratum clips per process (seed 1), 2 windows each (video frames 0 and 36), center crop 256², 16 px token cells · **Compute:** Mac, seconds. `metrics.json` is here.

## Question

The per-frame forecasts fade (`experiments/core-perframe/`). The L1-optimal
deterministic forecast is the per-token median, so a particle survives in a token only
while most possible futures still put it there. How fast does that stop being true?
Measured from the last context frame (latent frame 7), as in the Gate-2 follow-up.

## Results (all processes; 7,900–10,200 particles per horizon)

| | +1 | +2 | +3 | +5 | +10 | +24 |
|---|---|---|---|---|---|---|
| centre still in the same 16 px token | **0.49** | 0.37 | 0.30 | 0.23 | 0.15 | 0.06 |
| old centre still inside the particle | 0.63 | 0.49 | 0.40 | 0.30 | 0.19 | 0.08 |
| median displacement (px) | 6.9 | 9.7 | 11.9 | 15.3 | 21.3 | 33.1 |

Per process the "same token" fraction is 0.47–0.51 at +1 and 0.06–0.07 at +24.
Ripening and static keep the old centre inside the particle longer (0.70–0.74 at +1),
because their particles are larger; the growth clips are lowest (0.54–0.56).

## Reading

1. **Half the particles leave their token after one latent step (2 video frames).**
   Any deterministic per-token forecast, whatever the loss or the rollout length,
   has to hedge from +1 on. Under L1 the hedge is background, so particles are
   erased, not just blurred.
2. **This matches the measured fade.** Peak occupancy in the per-frame forecasts falls
   0.96 → 0.69 → 0.44 at +1/+5/+10 (`core-perframe/followup.json`, diffusion), and
   persistence's detection F1 falls from 0.58 to 0.19 between +1 and +5.
3. **It is a property of the data, not a bug.** The step size is calibrated to real
   Au-nanorod video (~1.5 radii per frame, `sim/dataset.py`), so real LPTEM has it too.
4. **Consequence:** the frozen-probe Gate 2 cannot be passed by any deterministic
   token-level forecaster beyond the first step. What stays predictable is
   position-free: sizes, counts, growth (`experiments/growth-identifiability/`).

## Caveats

- Ground truth only. A particle covers several tokens, so the encoder's tokens may
  hold it a little longer than the centre-cell fraction suggests. The
  "old centre inside" row is the looser bound.
- Stage drift is included in the positions and is not subtracted. It is 0–12 % of the
  Brownian step, too small to change the reading.
