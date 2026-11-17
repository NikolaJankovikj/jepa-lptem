# Where the project stands (2026-11-16, real-data validation added)

The findings so far, by research question. Every number is from an experiment README
and its `metrics.json`, and each section names its folder under `experiments/`. Each
rule was fixed in `PLAN.md` before its run. This replaces the 2026-10-02 snapshot,
which is in git history (`b303d07`).

## 1. The answer so far

| Question (`thesis_brief.md`) | Answer | Where |
|---|---|---|
| **RQ1.** Do predicted latents decode to correct future physics? | **Partly.** Token-level forecasts fail for a structural reason. A label-free forecaster on the frame-averaged latent carries each clip's growth (size criterion passes), but its exponents are damped. The damping comes from the context's noise, not the model. | §2 |
| **RQ2.** Does prediction error anticipate coalescence? | **Detects, does not anticipate.** The forecaster's error rises at a merge (AUC 0.71), but no earlier than the pixels show it, and a context average or a brightness change detects merges as well. | §4 |
| **RQ3.** How does quality degrade with horizon and SNR? | **Strongly with horizon, mildly with dose.** Low dose costs 0.3–0.6 px of mean-radius error at every horizon, through the readout. Not significant at the pre-registered +24, significant post hoc at +1. Context length matters more than dose for exponents. | §3 |
| **Real data.** Does any of it transfer to real LPTEM video? | **No.** No published real movie lies inside the simulated feature distribution, the readout reads real particles small, and the forecaster loses to the context mean on every real movie from the first step. Claims are scoped to simulation. | §5 |

## 2. RQ1: forecasts that decode to physics

### 2a. The frozen encoder reads the present (Gate 1)

Frozen V-JEPA 2 ViT-L, each 2-frame tubelet encoded alone (`gate1-perframe/`, test,
physics stratum):

| | value | bar |
|---|---|---|
| R² mean radius | 0.902 | ≥ 0.8 ✔ |
| R² count | **0.726** | ≥ 0.8 ✘ |
| R² total area | 0.928 | – |
| MSD / 4Dt | 0.986 (perfect-detector ceiling 0.955) | within 20 % ✔ |

**FAIL on count only.** The count loss is uniform across crop-edge buckets and largest
on reaction and ripening, the processes with many small particles. It looks like shot
noise on small particles in a single frame, not merging or edge effects.

### 2b. Token-level forecasting fails, and the reason is structural

- **Gate 2 FAIL** (`core-perframe/`): at +5 the forecasts are 76–82 % *worse* than
  holding the last frame in size distance. They fade: peak occupancy 0.96 → 0.69 → 0.44
  at +1 / +5 / +10.
- **No growth under the blur** (`forecast-probe/`): readouts trained on the forecasts
  read them about as well as the last context frame, but find no growth beyond it
  (gain −0.23 [−0.50, +0.03] / +0.04 [−0.21, +0.30] px on diffusion / reaction).
- **Why** (`token-persistence/`, ground truth only): **half the particles leave their
  16 px token within one latent step** (0.49 at +1, 0.23 at +5, 0.06 at +24). The
  L1-optimal deterministic forecast per token is then background, so particles are
  erased, whatever the loss or rollout. This is the conditional-mean collapse the
  stochastic-JEPA papers (MoP-JEPA, Variational JEPA) describe, here on a scientific
  instrument and measured.
- **What stays predictable** (`growth-identifiability/`): from 8 noisy readings of mean
  radius, a 20-term ridge with no labels cuts the +24 error from 3.26 / 3.90 px
  (persistence) to 1.31 / 1.86 px. Growth is predictable; positions are not.

### 2c. Position-free forecasting carries growth (Gate 2′)

The state is the 256 tokens averaged per frame (`pooled-gate1/`: R² mean radius 0.877,
area 0.942, count 0.723; exponents read off the true future 0.311 / 0.473 against
1/3 and 1/2).

| context 8, +24 (`pooled-forecast/`) | gain over the pooled trend (px) | exponent diffusion / reaction / static |
|---|---|---|
| A: linear (PCA 64 → ridge), no labels | +0.42 [+0.15, +0.68] / +0.66 [+0.35, +0.98] | 0.208 / 0.269 / 0.008 |
| B: causal transformer, no labels | +0.54 [+0.30, +0.77] / +0.81 [+0.59, +1.03] | 0.180 / 0.245 / 0.110 |
| B minus A | +0.12 [−0.18, +0.41] / +0.16 [−0.12, +0.43] | – |

**Gate 2′ FAIL for both, as pre-registered.** The size criterion passes on both
processes, and the exponent criterion fails (±0.1 bar). A bigger model is not
significantly better.

### 2d. The exponent shortfall is the context's, not the latent's

- **Exponent ceiling** (`exponent-ceiling/`, ground truth): with noise-free context,
  kNN recovers 0.339 / 0.494. At the probe's reading noise (sd 0.8 px) the best
  context-only method reaches 0.263 / 0.313. **The bar is not reachable from 8 noisy
  frames by any forecaster.** It is recorded as unpowered; no verdict changed.
- **Longer context** (`longer-context/`): from context 8 to 16, the linear candidate's
  exponents rise +0.097 / +0.104 (paired, CIs above 0). That is about the ceiling's own
  rise. They reach 0.287 / 0.340, and the gap to the ceiling stays 0.00–0.07. The
  transformer rises only +0.014 / +0.027 and keeps inventing growth on static clips
  (~0.11): it is model-limited.
- **Mixture of predictors** (`mop-ceiling/`, ground truth): mode forecasts undo the
  damping only when the context identifies the process. At sd 0.8 the router picks the
  right process 33 % of the time (chance 20 %), and the top head falls to
  0.191 / 0.120, below the averaged ridge. Not built.

**Reading for RQ1:** a label-free linear map on the frame-averaged V-JEPA latent takes
what the context offers. The remaining exponent error is what 8–16 noisy frames allow.

## 3. RQ3: horizon and dose (`snr/`)

Linear candidate, longer-context design A, test clips split at beam flux 21.2 (median
12.3 vs 39.3 e/px):

- **The readout is noisier at low dose:** sd-equivalent 1.21 vs 0.88 px (MAE 0.97 vs
  0.70), and it reads particles slightly small (bias −0.23 px).
- **Pre-registered test, context 16, +24:** low minus high dose +0.32 [−0.20, +1.03] px
  (diffusion) and +0.28 [−0.61, +1.27] (reaction). **Not shown.** With ~15 clips per
  bin and process, it was close to unpowered, as predicted.
- **Post hoc, +1:** +0.41 / +0.64 / +0.49 px (diffusion / reaction / static), all CIs
  above 0. The penalty stays 0.3–0.6 px at every horizon while the forecast error grows,
  so at +24 the growth extrapolation swamps it.
- **Exponents:** between dose bins the ceiling moves by at most 0.034, against 0.07–0.10
  from context 8 → 16. Over 6–75 e/px, more frames help growth recovery more than
  brighter frames.
- **Not answerable here:** mean liquid thickness is fixed at 150 nm in dynamics-v1, so
  the thickness half of RQ3 needs new clips.

## 4. RQ2: merges (`events/`)

Coalescence clips of the test split (48 merges with both particles in the crop; val,
38, replicates every reading). Signals on the 3 × 3-token patch around the merge,
against patches around particles far from any merge:

| AUC at the merge | test | one step before |
|---|---|---|
| forecaster's +1 error (`model`) | **0.710 [0.618, 0.793]** | 0.592 |
| raw latent change | 0.635 | 0.550 |
| distance from the context average | 0.738 | 0.661 |
| token brightness change (pixels) | 0.704 | 0.657 |
| frame-averaged forecaster's +1 error | 0.598 [0.524, 0.679] | 0.569 |

- **Detection: yes.** The forecaster's error is larger where two particles merge than
  where one just moves.
- **Specific to the forecaster: no.** It beats only the raw latent change (+0.075
  [+0.033, +0.112], by denoising) and ties the context average and the pixels.
- **Anticipation: no.** One step before contact everything is above chance, pixels
  included, because a pre-merge patch holds two particles; the forecaster adds nothing
  to that (−0.065 against pixels).
- **It adapts in one frame.** One step after the merge the forecaster's AUC falls to
  0.57 while the context average and pixels stay near 0.70: it has seen the merged
  particle and forecasts it. A merge is a one-frame surprise, as expected of a
  deterministic forecaster facing random encounters (§2b).
- **Protocol error, caught by a sanity check.** The first premise check used a pixel
  signal (mean per-pixel change) that could not even tell particles from background
  (AUC 0.513). It was recorded, replaced by the token-mean change (0.736) and re-run on
  fresh clips before the PC run.

## 5. Real data (`real/`)

Published supplementary movies, screened before use: 132 movies from 34 LPTEM papers
(ACS, via figshare). A display copy breaks two premises of the pipeline (one pixel is
one detector pixel, one frame is one time step), so each movie had to show white shot
noise on its pixel grid, grey levels, ≥ 16 clean frames and ≥ 256 px, and an
overlay-free 256² region. **10 of 132 pass**; most were rescaled for display. 8 movies
from 7 papers were used, plus the SAM-EM Au nanorods.

| check | simulation | real |
|---|---|---|
| A. containment ratio (limit 1.50, the largest held-out synthetic clip) | median 0.93 | nanorods **1.51**, six other bright-field movies **3.2–8.6**; HAADF control 2.58 |
| B. readout ÷ pixel estimator, mean radius | median 1.11, band 0.79–1.74 | 0.46–1.00; 2 of 5 movies in the band |
| C. sign of growth, readout vs pixels | – | disagree on `am1c` (ρ −0.87 vs +0.79, 35 blocks) |
| D. forecaster gain over the context mean at +24 | **+0.061 [+0.045, +0.078]** | **−0.418 [−0.511, −0.286]**; negative on every movie, already at +1 |

- **The simulator's appearance range does not cover real LPTEM video.** Only the video
  it was calibrated on comes near, and that one sits at the limit.
- **Correction to `latent-gap/`.** Its containment of 0.92 (Step 4k) was measured with
  the processor's default 256 → 292 resize, which the pipeline dropped on 2026-09-28
  (`78d73ae`). It was never re-measured. With the pipeline's preprocessing, the same
  video reads 1.51 (0.83 with the old processor on the same samples).
- **Nothing downstream transfers.** The forecaster fails at the input, not through
  extrapolation: real pooled latents change less per step than simulated ones (context
  mean +1 error 0.30 vs 0.40), but its one-step error doubles (0.65 vs 0.40).
- Predictions: A wrong (bright-field movies were expected inside), B mostly wrong (the
  small-particle movies read low, not high), C mixed, D right.

## 6. Methodological findings

1. **The look-ahead leak** (`leak-check/`). Encoding whole 64-frame windows leaks the
   future into a latent forecaster's context. The context latents agreed with honest
   re-encodings at only cosine 0.74–0.84, and v1's one clear win (detection F1 0.80 vs
   persistence 0.56 at +1) fell to 0.54 with honest contexts. Fixed by per-frame
   encoding (shared frames now agree at cosine 1.000). Everything after it was redone.
2. **Measure the floor or ceiling before setting a bar, and test every baseline on a
   known answer.** Twice a pre-registered
   criterion had no power: Gate 2's 30 % size rule (even the true future beats
   persistence by only 3–6 %), and Gate 2′'s exponent bar (§2d). A third time a baseline
   was blind (RQ2's first pixel signal, §4). All are recorded as protocol errors; no
   verdict was changed after the fact.
3. **Check the premise on ground truth first, on the Mac.** Token persistence, growth
   identifiability, the exponent ceiling and the MoP check each took seconds. Each
   decided whether a GPU run was worth doing.
4. **Validate every estimator against known answers.** Three simulator statistics were
   nearly tuned to artefacts of their own estimators (`PLAN.md` Step 4). Each was caught
   by pointing the estimator at clips whose answer was known.
5. **Re-measure a premise when the pipeline under it changes.** The latent containment
   that justified training on simulation was measured with a preprocessing step the
   pipeline later dropped. Under the actual preprocessing it fails (§5).

## 7. What the thesis can say now

1. A frozen video model never shown a microscope reads nanoparticle size, area and
   diffusion from single frames at R² ≈ 0.9 and the perfect-detector MSD.
2. Deterministic token-level latent forecasting cannot work on LPTEM. Half the particles
   leave their token every step, so the optimal single guess erases them.
3. Forecasting a position-free state works without labels for growth: it beats every
   trend-blind baseline at +24, and its exponent error tracks the information limit of
   the context.
4. Forecast quality degrades strongly with horizon and mildly with dose, by a roughly
   constant offset that enters through the readout.
5. The forecaster's error flags merges as they happen but does not anticipate them, and
   simple baselines flag them as well: a deterministic latent forecaster's surprise is not
   a learned event detector here.
6. The look-ahead leak is a general warning for latent world models on video encoders.
7. None of this transfers to published real LPTEM video with a frozen encoder. The one
   experiment the simulator was calibrated on sits at the edge of its range (1.51
   against a limit of 1.50); no other real movie comes near. Most published movies are not usable as data without screening.

## 8. Open items

- [ ] RQ3 thickness half: needs clips that vary mean water thickness, or is stated as out of scope.
- [x] Real-data validation (`PLAN.md` Step 13): done on 8 screened published movies plus the SAM-EM nanorods (§5). Negative.
- [ ] Closing the sim-to-real gap: wider appearance randomisation, or LoRA on real video (Step 8). Future Work unless there is time.
- [ ] RQ2 on real clips: still proposed to skip (needs the token predictor on the PC; §5 shows the inputs are off-manifold anyway).
- [ ] Pixel ConvLSTM baseline (Step 11): proposed to skip unless asked for.
- [ ] Branch `worktree-objective-redesign`: an uncommitted, never-run draft (`predictor-obs`), superseded by §2c. Keep or delete.
- [ ] Delete the leaking whole-window cache on the PC once nothing reads it.
- [ ] Thesis text: Results draft from the experiment READMEs.
