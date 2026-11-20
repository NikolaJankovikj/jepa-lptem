# Encoder comparison: frozen V-JEPA 2 vs LoRA vs CNN (PLAN Step 15)

Same input (one 2-frame tubelet, 256² centre crop of `dynamics-v1`), same heads
(`AttentiveProbe` + `CentroidProbe`), same loss, same split (`split_by_clip` seed 0,
1600 / 200 / 200 clips); only the encoder changes. Each model trains on 10 / 30 / 100 %
of the train clips (nested subsets, identical across models) with the same sample
budget (epochs × 1/frac).

| | encoder | trainable | epochs 100 / 30 / 10 % |
|---|---|---|---|
| **F** frozen | V-JEPA 2 ViT-L, frozen | probes only | 50 / 167 / 500 (fixed) |
| **L** LoRA | V-JEPA 2 ViT-L + LoRA r 16 on q/k/v/proj | 7.62 M | 6 / 20 / 60 (best val) |
| **C** CNN | 4 conv blocks, stride 16, from scratch | 1.51 M | 20 / 67 / 200 (best val) |

Runs on the PC, 2026-11-18/19 (one RTX 3090 each): LoRA 1 h 42 min / 1 h 55 min /
2 h 41 min, CNN 12–17 min each, frozen ~3.8 h (disk-bound). Analysis:
`scripts/encoder_compare_report.py` → `metrics.json` (seconds on the Mac).

Scoring: test clips, **physics stratum (158 clips)**, latent frames as Gate 1 (window
overlaps counted twice). 95 % CIs are a clip bootstrap (2000 draws, seed 0), the same
draws for every run, so differences are paired. The script reproduces every run's own
R² exactly. Mean-radius **reading sd** = MAE / 0.80 on frames with a particle
(`snr_split.reading_noise`), dose split at beam flux 21.2.

## Results

### R² (test, physics, 95 % CI)

| data | model | count | mean radius | total area |
|---|---|---|---|---|
| 100 % | F | 0.726 [0.665, 0.774] | 0.902 [0.876, 0.922] | 0.928 [0.914, 0.941] |
| | C | 0.838 [0.802, 0.865] | 0.950 [0.936, 0.961] | **0.978** [0.973, 0.982] |
| | L | **0.843** [0.795, 0.877] | **0.951** [0.936, 0.961] | 0.976 [0.969, 0.981] |
| 30 % | F | 0.646 [0.562, 0.709] | 0.872 [0.842, 0.894] | 0.910 [0.893, 0.928] |
| | C | 0.757 [0.671, 0.821] | 0.918 [0.898, 0.933] | 0.943 [0.929, 0.955] |
| | L | **0.781** [0.731, 0.821] | **0.941** [0.923, 0.955] | **0.946** [0.927, 0.957] |
| 10 % | F | 0.587 [0.482, 0.667] | 0.819 [0.769, 0.857] | 0.859 [0.829, 0.890] |
| | C | 0.661 [0.552, 0.745] | 0.864 [0.824, 0.894] | 0.910 [0.877, 0.941] |
| | L | **0.726** [0.652, 0.782] | **0.922** [0.899, 0.938] | **0.931** [0.906, 0.954] |

### Mean-radius reading sd (px, sd-equivalent)

| data | model | all | low dose | high dose | low − high |
|---|---|---|---|---|---|
| 100 % | F | 0.93 [0.87, 1.00] | 1.06 | 0.83 | 0.23 [0.11, 0.37] |
| | C | 0.63 [0.58, 0.69] | 0.76 | 0.52 | 0.24 [0.13, 0.36] |
| | L | **0.62** [0.57, 0.67] | 0.72 | 0.54 | 0.17 [0.07, 0.29] |
| 30 % | F | 1.06 [0.98, 1.14] | 1.23 | 0.92 | 0.31 [0.15, 0.49] |
| | C | 0.89 [0.83, 0.96] | 0.99 | 0.81 | 0.18 [0.05, 0.32] |
| | L | **0.69** [0.63, 0.75] | 0.79 | 0.61 | 0.19 [0.07, 0.31] |
| 10 % | F | 1.25 [1.14, 1.37] | 1.45 | 1.09 | 0.35 [0.13, 0.60] |
| | C | 1.09 [0.98, 1.20] | 1.20 | 1.00 | 0.20 [−0.02, 0.42] |
| | L | **0.83** [0.76, 0.89] | 0.93 | 0.74 | 0.19 [0.07, 0.33] |

### Detection and diffusion (test, all strata, no CI)

| data | model | precision | recall | F1 | MSD/4Dt (ceiling 0.955) |
|---|---|---|---|---|---|
| 100 % | F / C / L | 0.870 / 0.920 / 0.923 | 0.797 / 0.860 / 0.866 | 0.832 / 0.889 / 0.893 | 0.986 / 0.971 / 0.981 |
| 30 % | F / C / L | 0.863 / 0.922 / 0.924 | 0.805 / 0.848 / 0.851 | 0.833 / 0.883 / 0.886 | 0.991 / 0.967 / 0.993 |
| 10 % | F / C / L | 0.874 / 0.906 / 0.923 | 0.786 / 0.829 / 0.842 | 0.827 / 0.866 / 0.881 | 1.007 / 0.966 / 1.000 |

## Predictions (fixed before any M3 number)

| # | prediction | verdict |
|---|---|---|
| 1 | C ≥ F on count at 100 % | **right**: C − F = +0.112 [0.075, 0.154] |
| 2 | F > C at 10 % on count and mean radius | **wrong**: F − C = −0.075 [−0.121, −0.031] (count), −0.045 [−0.076, −0.020] (radius). Pretraining without adaptation does not pay even with 160 clips. |
| 3 | L ≥ F and L ≥ C on every R² at 100 % | **wrong by the letter, tied in substance**: L beats F on all three (+0.05 to +0.12, CIs above 0); L vs C is +0.005 / +0.001 / −0.002, every CI spans 0. |
| 4 | L's reading sd below F's by less than half; the low-dose penalty shrinks but stays | **right**: 0.62 vs 0.93 (−33 %, half would be 0.47); penalty 0.17 vs 0.23, the shrink's CI spans 0 [−0.16, +0.04], L's penalty stays above 0 [0.07, 0.29]. |
| 5 | MSD/4Dt within 10 % of the ceiling for all | **right**: 1.01–1.06 × the ceiling |

## Reading

1. **Frozen V-JEPA 2 is the weakest encoder at every data size**, on every R², reading sd and
   detection F1. A 1.5 M-parameter CNN trained from scratch beats it by 0.07–0.11 on count.
   The frozen features carry the information (LoRA starts from them, cosine 1.0 at init),
   but not in a form the probes read well.
2. **Adapting V-JEPA 2 is what pays, and it pays most when labels are few.** At 100 % LoRA
   and the CNN tie; at 10 % LoRA leads the CNN by 0.065 on count and 0.06 on radius, and
   **LoRA on 160 clips matches frozen on 1600** (count 0.726 vs 0.726, radius 0.922 vs 0.902).
   Prediction 2 had the right intuition (pretraining helps with few labels) for the wrong
   model: it holds for L, not for F.
3. **Positions were never the bottleneck.** MSD/4Dt is within 6 % of the perfect-detector
   ceiling for every model, as in Gate 1.
4. **Low dose still costs every encoder ~0.2 px** of reading sd. Better encoders lower both
   bins rather than closing the gap: the penalty is a property of the frames.

## Thesis link (pre-registered)

Rule: rerun the longer-context linear forecaster with L's cache iff L's reading sd
(physics, test) ≤ 0.6 px. **L reads 0.621 px [0.571, 0.674]: above the line by 0.02, the CI
straddles it.** By the rule as written the rerun does not trigger, and the reading is that
the encoder is not what limits growth recovery. The margin is far inside the noise; any
decision to rerun anyway is post hoc and must be labelled so.

## Limitations

- One seed per run; the CIs cover test-clip sampling, not training randomness.
- F trains for a fixed number of epochs (Gate 1's protocol); C and L keep the best-val
  epoch. C and L overfit hard after a few epochs (best epochs C 7 / 5 / 3, L 5 / 4 / 6;
  by the last epoch val loss is up to 5× C's minimum and 9× L's at 10 %), so without
  that checkpoint they would score far worse. F's train loss plateaus (~0.38), so it
  likely loses little from the fixed schedule, but the protocols differ.
- Detection and MSD have no CI and are over all strata: the predicted positions were not
  saved, so the matched distance (px) in the plan's scoring is not computed. The checkpoints
  are on the PC if it is needed.
- Synthetic data only. Step 13's real-movie check is not run per model yet.
