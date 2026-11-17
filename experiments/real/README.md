# Step 13: real-data validation on published LPTEM movies

**Date:** 2026-11-16 · **Scripts:** `scripts/real_video_qc.py` (screen), `scripts/real_features.py` (encoding), `scripts/real_validation.py` (analyses); tests `tests/test_real_video_qc.py`, `tests/test_real_validation.py` · **Data:** 132 supplementary movies from 34 LPTEM papers (ACS, via the figshare API, CC BY-NC; `data/real-figshare/`, provenance in `SOURCES.txt` and `pool_sources.tsv`), the SAM-EM Au-nanorod video, `smoke_v3`, 40 fresh simulated clips (seed 13), the pooled cache and the frozen P1 readout · **Compute:** Mac, MPS, ~12 min · **Protocol:** `PLAN.md` Step 13, *Revised 2026-11-16*, written with predictions before any real movie was encoded. `qc.json` and `metrics.json` are here; the log is `logs/real-validation.log` (not in git).

## Question

Everything so far was measured on simulation. Does it carry over to real liquid-cell
video? Four checks, from cheapest to most demanding: do real movies land inside the
simulated feature distribution (A); does the frozen readout measure real particles
about as a pixel estimator does (B); does it see growth where the pixels do (C); does
the label-free forecaster beat the trivial baselines on real snippets (D)?

## Which movies are usable

A published movie is a display copy, not data. It may be upsampled for the screen,
stamped with text, padded with a title card or with repeated frames. Each breaks a
premise of the pipeline: one pixel is one detector pixel, one frame is one time step.
So every movie was screened before it was encoded:

| check | rule | why |
|---|---|---|
| Q1 native grid | frame-difference correlation ≤ 0.10 at lag 2 and ≤ 0.40 at lag 1, both axes | shot noise is white on the detector grid (simulator 0.00 / 0.20, SAM-EM 0.01–0.07 / 0.17–0.26); upsampling or smoothing spreads it |
| Q2 grey | colour channels agree | a false-colour lookup table is not a detector signal |
| Q3 time | ≥ 16 contiguous frames after cutting at scene changes and dropping repeats | title cards and padded frame rates |
| Q4 size | ≥ 256 px both sides | the encoder's crop, never resized |
| by eye | an overlay-free region ≥ 256² | time stamps, scale bars, window edges |

**10 of 132 pass Q1–Q4.** Most fail Q1: their noise is correlated over 2–4 pixels
(lag-2 correlation 0.15–0.95), the signature of a movie rescaled for display. Two more
are screen captures of the microscope software with panels under 256 px. **8 movies
from 7 papers remain**, plus the SAM-EM nanorods:

| movie | paper (DOI) | content | tier |
|---|---|---|---|
| SAM-EM Au nanorods | SAM-EM | the video the simulator's experiment stratum was built on | 3 |
| `aggregative_growth_woehl2014` | 10.1021/nl4043328 | Ag nucleation and aggregative growth, BF-STEM | 3 |
| `am1c20824_si_002` | 10.1021/acsami.1c20824 | Au particle formation and diffusion | 3 |
| `am8b03688_si_004` | 10.1021/acsami.8b03688 | ~10 Au particles, sparse, as in the simulator | 3 |
| `nn5b06632_si_001` | 10.1021/acsnano.5b06632 | dense nucleation and attachment | 3 |
| `cm2c03226_si_004` | 10.1021/acs.chemmater.2c03226 | Pd nucleation, HAADF (bright particles): **positive control** | 1 |
| `ja9b05225_si_001/002` | 10.1021/jacs.9b05225 | dense nucleation, particles ~4 px | 1 |
| `nn6b05270_si_003` | 10.1021/acsnano.6b05270 | nanorod superlattice | 1 |

Tier 1: containment only. Tier 2/3: also readout and forecast, because they are bright
field with apparent radius inside the simulator's 6–15 px.

The screen's own known-answer test caught a gap in it: lag 2 cannot see an exact 2×
nearest-neighbour upsampling (it reads 0.00). The lag-1 bar was added before anything
was encoded; it removed one movie that the look had already dropped.

## Results

### A. Containment: no real movie is inside the simulated distribution

Ratio = median distance from a real sample to its nearest synthetic one, over the median
distance between synthetic clips (`latent_gap.py`'s protocol). The limit is the largest
ratio of a synthetic clip held out against the rest: **1.50** (median 0.93).

| movie | ratio | |
|---|---|---|
| SAM-EM nanorods | **1.51** | at the limit |
| `am8b03688_si_004` | 3.22 | out |
| `nn6b05270_si_003` | 4.85 | out |
| `aggregative_growth_woehl2014` | 7.04 | out |
| `nn5b06632_si_001` | 7.40 | out |
| `ja9b05225_si_001` / `_002` | 7.91 / 8.56 | out |
| `am1c20824_si_002` | 8.35 | out |
| `cm2c03226_si_004` (HAADF control) | 2.58 | out |

The control is detected, so the ratio can see a gap. But six bright-field movies sit
further from the simulation than the polarity-inverted control does.

**This corrects `experiments/latent-gap/`.** Its 0.92 for the nanorods was measured on
2026-08-16 with the processor's default resize (256 → 292 px, then centre crop). That
resize was switched off on 2026-09-28 (`78d73ae`) for everything downstream, and the
containment was never re-measured. On the same samples, the old processor gives 0.83 and
the pipeline's processor gives 1.51 (scratch check, both encoded today). Under the
preprocessing the model actually uses, even the calibration video sits at the edge.

### B. Readout against a pixel estimator: biased low on real movies

Per 16-frame block, the frozen readout's mean radius divided by the validated pixel
estimator's (`compare_to_real.apparent_radius_px`). On 40 fresh simulated clips the ratio
has median 1.11 and 5th–95th percentile **0.79–1.74**. Against truth there, the readout
errs by 0.65 px and the pixel estimator by 1.75 px.

| movie | readout / pixel radius (px) | ratio | counts, readout / detector |
|---|---|---|---|
| SAM-EM nanorods | 5.2 / 7.0 | 0.86, in range | 2.0 / 2.4 |
| `am1c20824_si_002` | 7.8 / 8.0 | 1.00, in range | 15.2 / 6.8 |
| `am8b03688_si_004` | 9.5 / 13.0 | 0.74, **out** | 5.9 / 2.0 |
| `nn5b06632_si_001` | 3.0 / 6.0 | 0.50, **out** | 26.2 / 20.3 |
| `aggregative_growth_woehl2014` | 2.8 / 6.0 | 0.46, **out** | 24.6 / 8.4 |
| *simulated reference* | | *median 1.11* | *4.8 / 5.7* |

Two movies read in range, three out. Every real ratio is below the simulated median: on
real movies the readout reads particles small. The dense movies give readout counts of
15–26, beyond anything in training (about 5 particles per 256² window).

### C. Growth direction: readout and pixels disagree

Spearman ρ of mean radius against time:

| movie | blocks | readout | pixel estimator |
|---|---|---|---|
| `am1c20824_si_002` | 35 | **−0.87** | **+0.79** |
| `am8b03688_si_004` | 4 | −0.90 | +0.77 |
| `nn5b06632_si_001` | 5 | +0.40 | +0.29 |
| SAM-EM nanorods | 63 | +0.01 | −0.13 |
| `aggregative_growth_woehl2014` | 6 | +0.09 | constant (6 px in every block) |

On the one movie with enough blocks to tell, the two disagree in sign. Neither is ground
truth: the detector keeps only blobs of at least 80 px², so it follows the large
particles, and new small nuclei would lower the true mean. What the table does show is
that the readout's reading of change over time on real movies is not anchored to
anything we can check.

### D. Forecast: the label-free forecaster loses to the context mean on real movies

Candidate A (PCA 64 → ridge, context 8), refitted on the cache's train split, applied
unchanged. Mean |forecast − truth| in the readout's z-scored pooled space; gain at +24
over the better baseline (positive = forecaster better), 95 % CI over clusters (tile ×
32-frame block).

| | +1 error: forecaster / ctxmean / persistence | gain at +24 | clusters |
|---|---|---|---|
| simulation, test split | 0.40 / 0.40 / 0.49 | **+0.061 [+0.045, +0.078]** | 200 |
| real, pooled | 0.65 / 0.30 / 0.36 | **−0.418 [−0.511, −0.286]** | 27 |
| SAM-EM nanorods | 0.30 / 0.25 / 0.33 | −0.085 [−0.118, −0.048] | 16 |
| `am1c20824_si_002` | 0.79 / 0.29 / 0.35 | −0.639 [−0.674, −0.599] | 8 |
| `am8b03688_si_004` | 0.73 / 0.46 / 0.48 | −0.269 | 1 |
| `aggregative_growth_woehl2014` | 0.81 / 0.41 / 0.58 | −0.360 | 1 |
| `nn5b06632_si_001` | 0.92 / 0.59 / 0.45 | −0.029 | 1 |

On simulation the forecaster beats holding the context mean. On every real movie it
loses, and it already loses at +1. Real pooled latents change *less* per step than
simulated ones (context-mean error 0.30 vs 0.40), yet the forecaster's one-step error
doubles. So the failure is not growth extrapolated over 24 steps. It happens at the
input: a linear map fitted on the simulated manifold receives states from off it (A).

## Predictions against outcomes

| prediction (PLAN, before encoding) | outcome |
|---|---|
| A: bright-field movies contained (0.8–1.3), HAADF control not (> 1.5) | **Wrong.** Nothing is contained; the nanorods sit at the limit (1.51); the control is out (2.58) but closer than six bright-field movies |
| B: `am8b` and the nanorods in range; the 6–7 px movies read high | **Mostly wrong.** Nanorods in range, `am8b` just out (0.74); the small-particle movies read *low* (0.46–0.50) |
| C: readout agrees in sign with the pixels on the growth movies | **Mixed.** Agrees on `nn5b` (5 blocks); disagrees on `am1c` (35 blocks) and `am8b`; `woehl` undecidable |
| D: beats both baselines on the cache; not on real | **Right**, but for a different reason than predicted: it fails at +1, not through extrapolated growth |

## Reading

1. **The simulator's appearance range does not cover published real LPTEM video.** In
   the frozen encoder's feature space, seven bright-field movies from six papers sit
   3–9× further from the simulation than simulated clips sit from each other. The one
   video the simulator was calibrated on sits at the edge. The domain randomisation of
   Step 4g bracketed one experiment, not real LPTEM in general.
2. **The August "real sits on the manifold" result came from a preprocessing the
   pipeline no longer uses.** It was not re-measured after the processor fix. Under the
   pipeline's own preprocessing it no longer holds, even for the calibration video.
3. **Nothing downstream transfers.** The readout reads real particles small and
   disagrees with the pixels on the direction of change. The forecaster that beats the
   baselines on simulation loses to the context mean on every real movie, from the first
   step.
4. **Most published LPTEM movies are not usable as data.** 122 of 132 fail a basic
   integrity screen, mostly because they were rescaled for display. Anyone training on
   supplementary movies should screen them first.

## Consequence

- The thesis claims are scoped to simulation. Real-data validation is a negative result,
  and it is reported as one: the frozen-encoder pipeline trained on this simulator does
  not transfer to published real video.
- The gap is a coverage hole, not the small offset `latent-gap/` described. Closing it
  needs either a much wider appearance randomisation (contrast, particle density, size
  down to ~2 px, noise texture, pixel scale) or adaptation on real video (the Step 8 LoRA
  run, now with a reason to exist). Both are Future Work unless there is time.
- `experiments/latent-gap/README.md` carries a correction note pointing here.

## Caveats

- Display copies: compressed and contrast-stretched by their authors. Q1 removes the
  rescaled ones, not those effects. A clean raw dataset could look closer to the
  simulation.
- A uses `latent_gap.py`'s protocol unchanged: whole-window, mean-pooled features and 64
  synthetic samples from 25 `smoke_v3` clips (2–3 per clip), so the held-out limit is
  noisy. The other real ratios are 1.7–5.7× the limit, far beyond that noise; the nanorods' 1.51
  against 1.50 is not.
- Most real movies yield one 256² tile and one 32-frame block, so their D CIs are
  degenerate. The pooled CI and the nanorods (16 clusters) and `am1c` (8) carry D.
- B and C have no ground truth on real movies. The pixel estimator is biased (MAE 1.75 px
  on simulation) and ignores small particles. B compares the two instruments against
  their simulated relationship, nothing more.
- Not done: RQ2 on real (the token predictor lives on the PC, and Step 12 had proposed
  skipping it); MSD (frame intervals of display copies are not known for all movies).
