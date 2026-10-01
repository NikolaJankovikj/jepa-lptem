# predictor-v1 — block-causal latent forecaster (Step 9)

**Date:** 2026-09-30/10-01 · **Script:** `scripts/train_predictor.py` (commit `87c960a`) · **Model:** `src/jepa_lptem/models/predictor.py` · **Data:** frozen V-JEPA 2 cache of `dynamics-v1`, Gate-1 split (3200 train / 400 val windows) · **Compute:** 1× RTX 3090, 6.4 h (7.7 min/epoch, GPU-bound).

## What was trained

`LatentForecaster`, 26.4 M parameters: 8 latent frames × 256 tokens in, a block-causal
transformer (dim 512, depth 8, 8 heads), the next frame's 256 × 1024 tokens out. It
predicts the *change* from the last context frame with a zero-initialised head, so it starts
as the persistence baseline. Loss: L1 on per-channel z-scored features, teacher forcing over
the context plus one autoregressive rollout step. AdamW 1e-4, cosine with warmup, bf16,
50 epochs, 4 random 10-frame snippets per window load, 8 loader workers. The checkpoint
(`predictor.pt`, not in git) carries the z-score statistics so forecasts map back to raw
feature space for the Gate-1 probes.

## Baselines

- **Persistence** — the next frame equals the last context frame.
- **Context mean** — the next frame equals the average of the 8 context frames. Added after
  the PC trial: every latent frame carries fresh shot noise, so averaging alone beats
  persistence by ~23 % — on static clips too. **Beating persistence is not evidence of
  dynamics; beating the context mean is.**

## Results (val, 400 windows × 3 fixed snippets)

| | +1 step | +2 steps (own forecast fed back) |
|---|---|---|
| model | **0.672** | **0.681** |
| persistence | 0.957 (model −29.8 %) | 0.962 (−29.2 %) |
| context mean | 0.741 (model −9.3 %) | 0.740 (−7.9 %) |

Gain over the context mean at +1, by process: coalescence 9.2 %, diffusion 9.7 %, reaction
9.4 %, ripening 8.9 %, static 9.4 %. At +2: 7.9 / 8.4 / 8.1 / 7.5 / 7.9 %.

**Learning curve.** Epoch 1 lands on the context mean (−0.2 to −1.0 %): averaging the noise
away is learned within one epoch. The gain beyond it then grows quickly (3.2 % at epoch 2,
6.6 % at 5, 8.0 % at 10) and saturates (9.0 % at 30, 9.4 % at 50). Best epoch 49; the last
five epochs moved by ≤ 5e-5, so the run converged — more epochs on this schedule would not
help. Train loss (teacher + rollout, 1.346) ≈ the sum of the val terms (0.672 + 0.681): no
overfitting gap.

## Reading

- **The model learned more than denoising**: ~9 % below the context mean at +1 and ~8 % at
  +2, consistently from epoch 2 on.
- **The gain is the same for every process, static included.** The processes differ only in
  how particle *size* changes, and per latent frame that change is tiny next to what all
  processes share: per-frame noise, Brownian motion and stage drift (a "static" clip has
  constant radii, not still particles). Feature-space L1 is dominated by the shared part, so
  the 9 % most plausibly comes from a better-than-uniform denoising filter plus drift and the
  typical spread of motion. **This metric cannot show whether growth physics was learned.**
- That question is Gate 2's (Step 10): the frozen Gate-1 probes read sizes, counts and
  tracks off the forecasts over +1/+5/+10 steps, where growth and static must diverge.
- The gain shrinks only a little from +1 to +2 (9.3 → 7.9 %) with the model feeding on its
  own forecast — no sign yet of rollout collapse; +5 and +10 are measured in Gate 2.

## Not done here

No test-split numbers: test is reserved for Gate 2.
