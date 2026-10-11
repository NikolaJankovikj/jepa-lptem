# MoP ceiling (M0): can a mode forecast recover the growth law from the context?

**Date:** 2026-10-11 · **Script:** `scripts/mop_ceiling.py` (test: `tests/test_mop_ceiling.py`) · **Data:** simulator ground truth only, the `exponent-ceiling` windows (200 physics clips per process, seed 1), split by clip · **Compute:** Mac, 5 s · **Protocol:** `PLAN.md` Step 10, *Mixture of predictors on growth*, fixed and committed (`a3f9582`) before it ran. `metrics.json` is here.

## Question

The exponent ceiling showed that a point forecast from 8 noisy frames shrinks growth
toward the average clip, and that process + window start gives the law exactly. MoP-JEPA
(arXiv 2607.05238) avoids that shrinkage by keeping several heads and forecasting with
the most likely one. Does that recover the exponents here, before any model is trained?

## Results: median exponent over +1..+24 (test windows, context 8)

| | diffusion | reaction | static |
|---|---|---|---|
| truth in the crop | 0.334 | 0.499 | 0.000 |
| *noise-free context* | | | |
| ridge (averaged forecast) | 0.304 | 0.342 | 0.077 |
| labelled top head (reference) | 0.333 | 0.500 | 0.000 |
| label-free top head, K = 6 | 0.328 | 0.424 | −0.002 |
| label-free mixture mean | 0.337 | 0.437 | 0.030 |
| *sd 0.8 px (probe quality)* | | | |
| ridge (averaged forecast) | 0.263 | 0.313 | 0.090 |
| labelled top head (reference) | 0.333 | 0.371 | 0.000 |
| **label-free top head, K = 6 (judged)** | **0.191** | **0.120** | **0.043** |
| label-free mixture mean | 0.191 | 0.208 | 0.090 |
| post hoc: heads on the true growth ratio | 0.194 | 0.240 | −0.002 |

Router accuracy on test (labelled, 10 branches / 5 processes): 0.46 / 0.55 noise-free,
**0.22 / 0.33 at sd 0.8** (chance 0.10 / 0.20). At sd 0.8 the label-free heads no longer
split by process: every process uses the same heads in similar shares
(`head_by_process` in `metrics.json`).

Mean-radius MAE at +24 at sd 0.8 (px, diffusion / reaction / static): ridge 1.34 / 1.94 /
1.33; label-free top head 1.93 / 2.71 / 0.87; mixture mean 1.72 / 2.57 / 1.19.

**Reading, as pre-registered: do not build M1.** The label-free top head at sd 0.8 is far
outside ±0.1 on diffusion and reaction (0.191 / 0.120).

## Reading

1. **MoP works when the context carries the information.** Noise-free, the mode forecasts
   undo the shrinkage: the labelled top head is exact, and the label-free one beats the
   ridge on both processes (0.328 / 0.424 against 0.304 / 0.342).
2. **At probe noise, the router is the bottleneck, not the averaging.** From 8 frames at
   sd 0.8 the router picks the right process a third of the time. Committing to one head
   then often commits to the wrong law, so on diffusion and reaction the top head is
   *worse* than the average in both exponent and +24 MAE. The exponent ceiling said the same: noise, not missing
   information, limits every context-only method.
3. **The label-free clusters follow the noise.** The pre-registered target is the log
   ratio to the *noisy* last reading, so the frame-7 reading error (about 15 % of the
   radius) dominates the clusters. Clustering on the true growth ratio instead (post hoc)
   lifts reaction from 0.120 to 0.240, still below the ridge. The router remains the limit.
4. **The labelled reference's diffusion 0.333 is not skill.** With router accuracy 0.33,
   most diffusion windows pick another process's head; a median over picks spread across
   heads can still land on the middle law. This is the
   caveat fixed in the protocol: a median of mode picks rewards ordering, not correct picks.

## Consequence

- A mixture of predictors cannot recover the growth law from 8 frames read at probe
  quality, so candidate C (MoP on the pooled state) is not built.
- For the thesis: MoP-JEPA's fix addresses shrinkage *when the modes are identifiable from
  the context*. Here they are not, which is the same limit the exponent ceiling found.
  Lower reading noise or a longer context (context 16 brought diffusion within the bar)
  would have to come first.
