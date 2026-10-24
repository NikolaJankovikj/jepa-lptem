# RQ2: does the forecaster's error rise at a merge, or before it?

**Date:** 2026-10-24 · **Scripts:** `scripts/merge_premise.py` (ground truth, Mac), `scripts/event_signals.py` (PC), `scripts/event_anticipation.py` (Mac); tests `tests/test_event_signals.py`, `tests/test_event_anticipation.py` · **Data:** the 84 coalescence clips of the val and test splits of dynamics-v1 (241 merges), the per-frame cache, the predictor-perframe checkpoint, the pooled cache and P1 readout · **Compute:** premise 4 min on the Mac; signals ~minutes on one 3090; analysis 30 s on the Mac · **Protocol:** `PLAN.md` Step 12, fixed and committed (`491d75d`) with predictions before any number; premise v2 committed (`1ad9045`) before it ran. `premise-v1.json`, `premise.json` and `metrics.json` are here; `data/events/signals-v1.npz` (7 MB) is not in git; logs in `logs/merge-premise*.log` and `logs/events-analysis.log`.

## Question

RQ2 asks whether the model's prediction error anticipates coalescence before it is
visually identifiable. A rise *at* a merge is a surprise signal: encounters are random,
so even a perfect deterministic model cannot time them. A rise *before* a merge, beyond
what the pixels already show, would be anticipation.

## Design

- **Events:** merges whose two particles are both inside the 256 crop just before
  contact, at latent frame t_m = f // 2 with 10 ≤ t_m ≤ 48. Each clip's two cached
  windows stitched into 50 latent frames. 48 events in 46 test clips (judged), 38 in 38
  val clips (replication).
- **Signals, per latent frame and token:**
  - `model`: the predictor-perframe checkpoint, teacher-forced on frames t−8…t−1, L1
    between its forecast of t and the true frame (standardized). Its mean over all
    tokens, 0.709, matches its recorded val L1 at +1 (0.717), so the checkpoint and
    standardization are the trained ones;
  - `persistence`: |x_t − x_{t−1}|, the raw latent change;
  - `ctxmean`: |x_t − mean of the 8 context frames|;
  - `pixel`: the change of each token's mean brightness (premise v2);
  - position-free, per frame: `pooled_linear` (+1 error of a one-step label-free linear
    forecaster on the frame-averaged latent) and `pooled_change`.
- **Scoring:** token signals on the 3 × 3-token patch around the survivor against
  patches around particles at least 3 latent frames and 48 px from any merge;
  position-free signals per frame against frames with no merge within ±3. ROC AUC per
  lag (−2, −1, 0, +1), 95 % clip-bootstrap CIs, paired differences on the same draws.
- **Known confound:** before contact a positive patch holds two particles and a negative
  one, so every signal's AUC before the merge is inflated alike. The judged claims are
  therefore paired comparisons between signals.

## Premise check (simulator ground truth, 100 fresh clips)

| | v1 (per-pixel change) | **v2 (token-mean change)** |
|---|---|---|
| events | 92 | 124 |
| particle patches vs empty background | 0.513 [0.479, 0.548] | **0.736 [0.700, 0.774]** |
| `pixel` AUC at the merge | 0.512 [0.454, 0.578] | **0.781 [0.733, 0.824]** |
| `pixel` AUC one step before | 0.478 | 0.681 [0.638, 0.722] |
| position-free stand-in, with readout noise | 0.511 | 0.527 [0.477, 0.575] |
| same, noise-free | 0.757 | 0.656 |

v1's stop rule fired, but its pixel signal could not see particles at all: a mean of
per-pixel |change| is shot noise. That was our protocol error, recorded in `PLAN.md`.
v2 averages first, sees particles and merges, and the PC run went ahead. The
position-free stand-in (the three readout observables plus the readout's noise) is
blind to merges, so P1 / P2 were expected to be null.

## Results (AUC [95 % clip CI])

### Token signals, test split

| lag | model | persistence | ctxmean | pixel |
|---|---|---|---|---|
| −2 | 0.595 [0.506, 0.674] | 0.542 | 0.608 | 0.533 |
| −1 | 0.592 [0.509, 0.669] | 0.550 | 0.661 [0.598, 0.722] | 0.657 [0.579, 0.721] |
| **0** | **0.710 [0.618, 0.793]** | 0.635 [0.544, 0.716] | 0.738 [0.646, 0.821] | 0.704 [0.623, 0.779] |
| +1 | 0.574 [0.494, 0.645] | 0.512 | 0.698 [0.631, 0.763] | 0.695 [0.613, 0.766] |

### Paired differences, model minus baseline, test split

| lag | − persistence | − ctxmean | − pixel |
|---|---|---|---|
| −1 | +0.042 [+0.006, +0.081] | −0.069 [−0.126, −0.019] | −0.065 [−0.149, +0.021] |
| **0** | **+0.075 [+0.033, +0.112]** | **−0.028 [−0.093, +0.023]** | **+0.006 [−0.071, +0.073]** |
| +1 | +0.062 [+0.004, +0.127] | −0.124 [−0.202, −0.051] | −0.122 [−0.225, −0.030] |

### Position-free, test split

| lag | pooled_linear | pooled_change | difference |
|---|---|---|---|
| −1 | 0.569 [0.481, 0.658] | 0.481 | +0.089 [+0.023, +0.149] |
| **0** | **0.598 [0.524, 0.679]** | 0.588 [0.520, 0.658] | **+0.010 [−0.053, +0.071]** |
| +1 | 0.597 | 0.515 | +0.082 [+0.030, +0.141] |

### Readings

| | test | val (replication) |
|---|---|---|
| **R1** detection: model AUC at lag 0 above 0.5 | **yes** (0.710) | yes (0.667 [0.572, 0.760]) |
| **R2** beyond raw change: model beats persistence, ctxmean *and* pixel at lag 0 | **no**: beats persistence only | no: beats persistence only (+0.054 [+0.004, +0.100]) |
| **R3** anticipation: at lag −1, above 0.5 and beats pixel | **no** (−0.065 vs pixel) | no |
| **P1** position-free detection | **yes** (0.598) | yes, barely (0.574 [0.502, 0.647]) |
| **P2** position-free beyond change | **no** (+0.010) | no |

Lead times (reported): the model's patch signal crosses its clip's 95th percentile of
negatives within lags −4…0 for 19 of 44 test events, most often at lag 0 (8); 25 never
cross. The pixel signal crosses for 10, 7 of them at lag 0.

## Predictions vs outcome

| written before the run | outcome |
|---|---|
| R1 passes, model AUC 0.65–0.75 at lag 0 | 0.710 test, 0.667 val |
| R2 fails: the model does not beat persistence or pixel | Fails, but partly wrong: it **does** beat persistence (+0.075), and ties pixel and ctxmean |
| R3 fails | Fails |
| P1 passes or is borderline; P2 fails | P1 passes (0.598), borderline on val; P2 fails |
| Lead times cluster at lag 0 | Among events that cross, lag 0 is the most common; most never cross |
| (premise) position-free stand-in blind → P1 expected-null | The pooled latent does better than its three readout observables: 0.598 against 0.527 |

## Reading

1. **The forecaster's error does flag merges** (R1, both splits): a patch where two
   particles merge has a larger +1 error than one where a particle just moves (AUC 0.71).
2. **But the signal is not the forecaster's.** Two baselines that need no trained model
   do as well: the distance from the context average (0.738) and the change in mean
   brightness (0.704). The model beats only the raw frame-to-frame latent change (+0.075),
   which is noisier because it compares two single noisy frames. The predictor's one
   advantage over that is denoising, which the context average gives for free. This is
   the same picture as `experiments/predictor-perframe/`, where the predictor sat 4 %
   below the context mean in L1.
3. **No anticipation.** One step before contact every signal is already above chance,
   the pixels included (0.657), because a pre-merge patch holds two particles. The model
   does not add to that: it is 0.065 below the pixel signal at lag −1.
4. **After the merge the model adapts, the baselines do not.** At +1 the model's AUC
   falls back to 0.57 while ctxmean and pixel stay near 0.70. The model has seen one
   post-merge frame and forecasts the merged particle; the context average still holds
   the two old ones. So the model registers a merge as a one-frame surprise, as a
   deterministic forecaster of a random encounter should.
5. **The frame-averaged latent carries a weak merge signal** (P1, 0.598), more than its
   three readout observables would give under the readout's noise (0.527 in the
   premise). But a one-step linear forecast of it is no better than its raw change at
   the merge (P2).

## Consequence

- RQ2 answer for the thesis: the forecaster's prediction error detects merges as they
  happen (AUC ≈ 0.7) but does not anticipate them, and the detection is not specific to
  the learned model: the context average and the brightness change detect them as well.
  This is the expected behaviour of a deterministic forecaster facing random encounters
  (`experiments/token-persistence/`), now measured.
- The qualitative repeat on real aggregation clips was conditional on R1 and is still
  open. With R2 failing it would show a surprise signal that a brightness change already
  gives, so it adds little.

## Caveats

- 48 test events: CIs are about ±0.08 on an AUC. A model advantage over ctxmean or pixel
  smaller than ~0.07 would not be detected.
- Premise v2's pixel definition was chosen after a 12-clip exploratory look (`PLAN.md`),
  then confirmed on fresh clips; the main run used it unchanged.
- The negatives control for "a particle moves here" but not for "two particles are close
  here". That confound inflates every signal before contact alike, and is why the
  readings are paired.
- One predictor (per-token L1, ctx 8, from `predictor-perframe`); a stochastic or
  position-aware predictor could behave differently.
