# Gate 2 — probing forecast latents (Step 10, the core experiment)

**Date:** 2026-10-01 · **Script:** `scripts/eval_forecasts.py` (commit `5c3674e`) · **Predictor:** `experiments/predictor-v1` (best epoch 49) · **Probes:** `experiments/gate1/probes.pt`, frozen, threshold 0.95 · **Data:** the 400 Gate-1 **test** windows (200 clips, never seen by probes or predictor) · **Compute:** 1× RTX 3090.

## Question

When the latent predictor rolls the future forward, do the frozen Gate-1 probes read the
right physics off its forecasts — sizes that grow at the right rate, particles in the right
place?

## Protocol (fixed before the run, `PLAN.md` Step 10)

Per window 3 snippets of 8 context + 10 future latent frames (starts 0, 7, 14); physics
stratum. Four futures, all read by the same probes:

- **model**: 10-step rollout on its own forecasts;
- **persistence**: the last context frame, held;
- **context mean**: the mean of the 8 context frames, held;
- **floor**: the true encoded future, i.e. the best the probes can do.

Metrics at +1/+5/+10 latent steps (one step = 2 frames):

- (a) **size distribution**: Gaussian 2-Wasserstein between the probed and GT radius
  distributions, via the first two moments the probes give;
- (b) **growth exponent**: the log-log slope of probed mean radius vs physical time over
  the 10 forecast frames;
- (c) **positions**: decoded-centroid F1 and matched distance.

**Pre-registered verdict:** at +5 on diffusion, reaction and ripening, the size distance
must be **≥ 30 % below persistence**, AND the exponent must be **within ±0.1 of the true
one** where defined (ripening has none).

## Results

**Size distance (px) at +5**, with the change against persistence for the model and the floor
(positive = closer to the truth):

| | model | persistence | ctx mean | floor | model vs persist. | floor vs persist. |
|---|---|---|---|---|---|---|
| coalescence | 2.79 | 2.18 | 2.50 | 2.26 | −28 % | −4 % |
| diffusion | 1.78 | 1.81 | 2.45 | 1.71 | +2 % | +6 % |
| reaction | 2.07 | 2.06 | 2.88 | 2.00 | −0 % | +3 % |
| ripening | 3.21 | 2.50 | 2.92 | 2.85 | −28 % | −14 % |
| static | 3.11 | 2.59 | 2.85 | 2.54 | −20 % | +2 % |

The model's size distance grows with horizon (from +1 to +10): static goes 2.43 → 4.77,
ripening 2.70 → 4.65 and coalescence 2.39 → 3.57. Persistence stays within 2.1–2.7 px. At
+1 the model and persistence are both within ~0.3 px of the floor.

**Growth exponent** (median over snippets):

| | model | floor | GT in crop | true |
|---|---|---|---|---|
| coalescence | 0.040 | 0.050 | 0.007 | – |
| diffusion | **0.296** | 0.319 | 0.333 | 0.333 |
| reaction | **0.371** | 0.501 | 0.500 | 0.500 |
| ripening | 0.110 | 0.063 | 0.024 | – |
| static | 0.104 | 0.019 | 0.000 | 0.000 |

**Positions:** decoded-centroid F1, with the matched error in brackets. The table shows
diffusion; the other processes are within ±0.07 F1.

| | +1 | +5 | +10 |
|---|---|---|---|
| model | **0.82** (3.5 px) | **0.46** (4.5 px) | **0.25** (5.0 px) |
| persistence | 0.56 (4.5 px) | 0.21 (5.3 px) | 0.14 (5.6 px) |
| ctx mean | 0.33 (4.8 px) | 0.16 (5.2 px) | 0.09 (5.2 px) |
| floor | 0.87 (2.7 px) | 0.87 (2.7 px) | 0.87 (2.7 px) |

Pooled MSD/4Dt, gap closing with memory 2: floor 0.996, ceiling 0.959. Both agree with
Gate 1. The model scored 0.464 and persistence 0.051, but neither number is interpretable
(see the protocol).

## Verdict: **FAIL** as pre-registered

- **Size distance vs persistence at +5:**
  - diffusion +1.7 %, reaction −0.3 % and ripening −28.4 %, all far from the required
    +30 %;
  - against the context mean, diffusion and reaction are +27 % and +28 %, but that is not
    the test.
- **Exponent:**
  - diffusion passes (0.296 vs 0.333);
  - reaction fails (0.371 vs 0.500, off by 0.13).

## Reading

1. **The size criterion had no power at +5, and this was a flaw in the pre-registration.**
   - Even the true future (floor) beats persistence by only 6 % (diffusion) and 3 %
     (reaction), and loses on ripening (−14 %). It "fails" the 30 % bar on every process.
   - The reason: over 5 latent steps (10 frames) the radii change by much less than the
     probes' own reading error, which is the ~1.6–2.9 px floor.
   - The 30 % margin was set without first checking the floor. This is reported as a
     protocol error, not explained away. Any replacement criterion is post hoc and must be
     labelled so.
2. **The exponent test does have power, and it shows partial physics.**
   - On the true future the probes recover the exponents almost exactly (0.319 / 0.501
     vs 0.333 / 0.500).
   - The model's exponents are **ordered by process** (reaction > diffusion > ripening ≈
     static > coalescence), so it infers something about the growth mode from 8 context
     frames.
   - But they are **shrunk toward a common value**: reaction is too slow (0.37), and
     static, which should not grow, gets 0.10.
   - There are two candidate causes, not yet separated:
     - regression to the mean of a deterministic forecaster that is unsure of the
       process;
     - the rollout blurring, so that blurred particles read as larger. This would show
       up as an offset of about +0.1 shared by every process.
3. **The rollout degrades sizes.**
   - By +10 the model's size distance is worse than persistence on every process except
     diffusion (tie), worst on static.
   - Holding the frame keeps sizes sharp; rolling forward smears them.
   - At +1 there is no problem: the first step is within ~0.3 px of the floor.
4. **Positions are a clear win over both baselines.**
   - F1 is about 1.5–2× persistence at every horizon, and +1 recovers most of the floor
     (0.82 vs 0.87).
   - Brownian motion is unpredictable, but stage drift is deterministic. The model learned
     to carry it forward, which probably explains most of the 9 % feature-space gain over
     the context mean seen in Step 9.
   - Detections fade with horizon (F1 0.25 at +10): the forecast's occupancy blurs below
     the 0.95 threshold.
5. **The context mean is a poor size forecaster.** Averaging 8 frames of moving particles
   smears them, so its size distance is the worst of the four at every horizon on
   diffusion and reaction, even though its feature-space L1 beats persistence.
6. **No confidence intervals yet.** Differences of a few percent (e.g. diffusion model vs
   persistence) should not be read as real until bootstrap intervals over clips are added.

## Next

These follow the FAIL branch of `PLAN.md` (systematic debugging) and are not done yet:

- **Give the size test power.** Roll out to +24 (the window allows 8 + 24 latent frames),
  so that size change exceeds probe error. Then compare model and persistence *relative to
  the floor*. Fix that criterion in `PLAN.md` before running it, labelled as post hoc.
- **Separate blur from physics in the exponent.** Subtract each snippet's static-clip
  offset, or check whether probed count drops while radius rises on static forecasts.
- **Add bootstrap CIs over test clips** to the headline numbers.
- **The money plots** (size distribution at +10, growth curve against r ∝ t^⅓) are still
  open.
