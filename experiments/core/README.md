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

## Follow-up: +24 rollout (POST HOC)

**Script:** `scripts/gate2_followup.py` (commit `b7d4458`). The criterion was fixed in
`PLAN.md` (commit `8110715`) before the run. Output: `followup.json`.

**Setup.**
- One snippet per test window: context = latent frames 0–7, forecast = 8–31.
- Physics stratum, 95 % bootstrap CIs over test clips.
- The predictor was trained on +1/+2 only, so +24 is far outside its training rollout.

**(A) Size at +24.** "Gain" is the size-distance improvement over persistence; positive
means closer to the truth.

| | model | persistence | floor | floor gain [CI] | model gain [CI] | result |
|---|---|---|---|---|---|---|
| diffusion | 6.68 | 3.65 | 1.79 | +1.86 [+1.57, +2.16] | −3.04 [−5.45, −1.14] | **FAIL** |
| reaction | 8.15 | 4.45 | 2.83 | +1.62 [−0.30, +2.80] | −3.70 [−9.76, +0.02] | untestable |
| ripening | 10.87 | 2.86 | 3.34 | −0.48 [−0.88, −0.15] | −8.02 [−11.80, −4.78] | untestable |
| static | 17.11 | 2.51 | 2.68 | −0.16 [−0.73, +0.39] | −14.59 [−27.03, −7.19] | (not judged) |

**(B) Growth exponent, fitted over the 24 forecast frames:**

| | model [CI] | floor | GT in crop | true | result |
|---|---|---|---|---|---|
| coalescence | −0.108 [−0.259, +0.088] | 0.057 | 0.043 | – | |
| diffusion | 0.218 [+0.153, +0.275] | 0.321 | 0.333 | 0.333 | **FAIL** |
| reaction | 0.414 [+0.318, +0.469] | 0.497 | 0.500 | 0.500 | pass |
| ripening | 0.012 [−0.064, +0.162] | 0.065 | 0.085 | – | |
| static | 0.006 [−0.028, +0.071] | 0.013 | −0.004 | 0.000 | (pass) |

**(C) Blur check: static clips, model / floor.**

| | +1 | +5 | +10 | +24 |
|---|---|---|---|---|
| probed count | 5.9 / 6.0 | 5.5 / 6.0 | 4.9 / 5.7 | **1.9** / 5.5 |
| mean radius (px) | 12.6 / 12.4 | 12.7 / 12.5 | 13.1 / 12.5 | 12.0 / 12.7 |
| peak occupancy | 1.00 / 1.00 | 0.99 / 1.00 | 0.97 / 1.00 | 0.89 / 1.00 |

All processes show the same pattern: by +24 the model loses 43–74 % of the count while the
floor keeps it. F1 at +24: model 0.02, persistence 0.03–0.06, floor 0.85–0.88.

### Reading

1. **The follow-up does not rescue Gate 2.**
   - **Size:** the one testable process, diffusion, fails. The model is *significantly
     worse* than holding the frame at +24.
   - **Exponent:** reaction now passes (0.414) and diffusion fails (0.218). This is the
     reverse of the 10-frame Gate-2 run.
   - So neither run passes both growth modes, and the per-process exponents are not stable
     across horizons.
2. **The failure is the rollout, not the encoder or the probes.**
   - The floor stays sharp to +24 (F1 ≥ 0.85, size distance 1.8–3.3 px), so the frozen
     features and probes still carry the physics that far out.
   - The model's forecasts **fade**: probed count and peak occupancy fall steadily, and
     particles drop out.
   - This is what a deterministic L1-trained forecaster does when it averages uncertain
     futures, made worse by training on only 2 rollout steps.
   - The huge +24 size distances (up to 17 px) are mostly this fading, not wrong sizes.
     With few particles left, the area/count ratio stops being consistent, and the spread
     (sd) derived from it blows up. The probed mean radius stays within ~2 px of the floor.
3. **The +10 spurious static growth is blur, as suspected.**
   - On static clips the probed radius rises above the floor up to +10 (13.1 vs 12.5 px)
     while count and peak occupancy fall. That is the blur signature.
   - It then falls again by +24 as particles vanish, which is why the 24-frame static
     exponent is ≈ 0.
   - So neither exponent window measures growth cleanly: rollout artefacts dominate both.
4. **There is a real process-dependent signal underneath.**
   - Over 24 frames the model ranks reaction (0.41) > diffusion (0.22) > static (0.01), as
     the truth does (0.50 > 0.33 > 0), and gets static right.
   - The predictor tells growth modes apart from 8 context frames. It cannot keep its
     forecasts on the feature manifold long enough for the probes to read them reliably.
5. **Exploratory only, not a test.** With these early-clip snippets, diffusion at +10 is
   closer to the floor (model 1.78, persistence 2.32, floor 1.72), because sizes change
   fastest early in a clip. Reaction at +10 is not (2.71 / 2.50 / 1.64).

**Status:** Gate 2 stays **FAIL**. The diagnosis is that the predictor's forecasts fade
beyond ~+5, which in PLAN's terms is "how error grows with horizon". The latent space and
the probes are not the cause.

## Next

- **predictor-v2: train on a longer rollout** (`--rollout 8`, already a flag of
  `train_predictor.py`), so the model is penalised for fading. Then rerun Gate 2 and the
  follow-up unchanged. A result for v2 is a new attempt and must be reported as such next
  to v1's FAIL.
- **The money plots** (size distribution at +10, growth curve against r ∝ t^⅓) are still
  open.
