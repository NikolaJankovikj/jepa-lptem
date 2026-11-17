# Latent-space sim-to-real gap (frozen V-JEPA 2)

**Date:** 2026-08-16 · **Script:** `scripts/latent_gap.py` · **Data:** `data/synthetic/smoke_v3` (25 clips, calibrated MTF) vs real Au-nanorod video · **Compute:** Mac, MPS, ~10 min.

## Question

Every pixel statistic in `compare_to_real.py` is a hand-picked proxy: the model never sees
pixels that way — it sees the frozen encoder's features. Do real clips land **inside the
synthetic feature distribution** of V-JEPA 2 ViT-L, the encoder the whole pipeline runs on?

## Protocol

64 real samples (16-frame × 256² crops, tiled from the 1024² video — crop, never resize)
vs 64 synthetic samples balanced across the 25 smoke clips, mean-pooled
`last_hidden_state` → 1024-d per sample. Two numbers, each with a same-pipeline
calibration reference:

| metric | value | reference | reference meaning |
|---|---|---|---|
| containment ratio (real→syn NN dist ÷ cross-clip syn→syn NN dist, cosine) | **0.92** | 1.00 | held-out synthetic clips, ~1 by construction |
| linear separability AUC (group-CV logistic regression) | **1.00** | 0.37 | physics-vs-experiment strata of our own generator (≈ chance) |

Also: real nearest-neighbour distances span 0.007–0.022 (p5–p95) in cosine — tiny; and
real's NNs are physics-stratum clips 97 % of the time (experiment-stratum base rate 20 %).

## Reading

**Real sits on the synthetic feature manifold** (ratio 0.92 — real samples are as close to
synthetic data as held-out synthetic clips are to each other), **but along a direction a
linear probe finds perfectly** (AUC 1.00). Separable-but-adjacent: the domain gap is a
small, *systematic* offset, not a distributional mismatch. Candidate carriers of that
offset, all known and stated: the un-modelled static SiNx texture (the Tier-2 bound), JPEG
compression, and the unknown contrast stretch of the published frames.

Two caveats, honestly:

1. "Real" is one video. The aggregation set's frames are 200×220 — below the 256 px crop,
   and resizing is ruled out (it rescales apparent particle size) — so it contributes
   nothing. A consistent AUC-1.0 signature of *one* video cannot be distinguished from a
   signature of *real data in general*. CV is grouped by spatial tile so the classifier
   cannot score by memorizing a held-out region's window texture, but temporal correlation
   within the single video remains.
2. Mean-pooled features weight background statistics heavily. Token-level probing (Gate 1)
   is the sharper instrument and supersedes this the moment features are extracted (Step 6).

The strata footnote: real's neighbours are overwhelmingly *physics*-stratum clips, not the
experiment stratum designed to mimic real conditions. Mean pooling likely explains it
(experiment clips are near-empty, so their pooled features are dominated by their
particular dose draw), but it is worth re-checking at Gate 1 with token-level features.

## Consequence for the plan

- Frozen-encoder probing on real data (Gate 1 / Step 13) is not doomed from the start:
  real inputs are in-distribution for the encoder as trained on our synthetic conditions.
- The residual gap is the shape LoRA (Step 8) absorbs well — a consistent offset, not a
  coverage hole. No change to the plan's ordering.
- Step 13 remains the arbiter: no feature statistic substitutes for downstream transfer.

## Where this sits in the validation hierarchy

1. **Labels correct, no shortcuts** — the 95-test suite (exponents, MSD, brightness/size
   leaks, persistence-optimality). Independent of realism. ✅
2. **Pixel-level containment** — the Step-4g gate, all three stretch-invariant statistics
   PASS. ✅
3. **Latent-space containment** — this experiment: ratio 0.92, separable-but-adjacent. ✅
4. **Downstream transfer** — Gate 1 on real features; Step 13 full pipeline. *Pending
   (requires GPU phase); the only definitive level.*

**Update 2026-09-30.** Caveat 2 is answered on the synthetic side: token-level probing of the
full frozen cache now exists (Gate 1 PASS — mean radius R² 0.91, count 0.81, MSD at the
perfect-probe ceiling; `experiments/gate1/`), and a single-token linear probe finds
particle-centre cells at held-out AUC 0.98 (`PLAN.md`, Step 6 cache audit). Level 4 is still
open where it matters: none of this has touched real video yet — that is Step 13, and the
strata footnote above (real's neighbours are physics-stratum) is still to re-check with
token-level features.

**Correction 2026-11-16 (`experiments/real/`).** The 0.92 above was measured with the V-JEPA 2
processor's defaults, which resize a 256 crop to 292 px and centre-crop it back. The pipeline
switched that resize off on 2026-09-28 (`78d73ae`, Step 6), and this number was never
re-measured. Re-run on the same video with the pipeline's processor, the ratio is **1.51**
(0.83 with the old processor on the same samples), against a held-out-synthetic limit of 1.50.
Seven other bright-field real movies read 3.2–8.6. The reading "real sits on the synthetic
manifold" does not hold for the preprocessing the model actually uses.
