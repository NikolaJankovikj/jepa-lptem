# predictor-perframe: the block-causal forecaster on look-ahead-free features

**Date:** 2026-10-03 · **Script:** `scripts/train_predictor.py`, v1's defaults, from scratch · **Data:** the per-frame cache `data/features/dynamics-v1-perframe` (audited: shared frames cos 1.000), Gate-1 split (3200 train / 400 val windows) · **Compute:** 1× RTX 3090, 6.4 h (7.5 min/epoch). `predictor.pt` stays on the PC (not in git); `history.json` and `logs/pf-predictor.log` are here.

## Results (val, 400 windows × 3 fixed snippets)

| | +1 step | +2 steps (own forecast fed back) |
|---|---|---|
| model | **0.717** | **0.722** |
| persistence | 0.972 (model −26.3 %) | 0.980 |
| context mean | 0.749 (model **−4.2 %**) | 0.752 (−4.0 %) |
| *v1 on the whole-window cache, gain over the context mean* | *9.3 %* | *7.9 %* |

- **Gain over the context mean, by process at +1:** coalescence 4.3 %, diffusion 4.3 %,
  reaction 4.2 %, ripening 4.1 %, static 4.3 %.
- **Learning curve:**
  - Epoch 1 lands on the context mean (−1.1 %). The gain then rises to 3.3 % at epoch 5
    and 4.1 % at epoch 20, and plateaus at 4.2 % from epoch 30.
  - Val L1 is constant to the fourth decimal over the last 20 epochs: converged.
  - Train loss (1.428) is close to the sum of the val terms (0.717 + 0.722): no
    overfitting.

## Reading

- **Training worked; the objective is what fails.** The model converged, and it beats both
  baselines in feature space. Yet read through the probes it falls below persistence
  within 5 steps (`experiments/core-perframe/`).
  - A lower L1 bought blur. The L1-optimal forecast of an uncertain future is the
    per-channel median, and in this cache the median smears particles out.
- **Feature-space L1 is dominated by noise.** Averaging the 8 context frames alone cuts the
  error by 23 % against persistence, more than the model adds on top (4 %).
  - The context mean is exactly the kind of latent the probes cannot read: F1 0.25–0.33
    at +1, against 0.58 for persistence.
  - The model sits next to it, so the metric it was trained on rewards the wrong thing
    for physics read-out.
- **Half of v1's "learned" gain was the look-ahead.** The gain is 9.3 % on the leaking
  cache and 4.2 % without it.
  - The remainder is the same for every process, static included, so it is not growth
    physics. It is most plausibly a better denoising filter plus stage drift (v1's
    reading).
