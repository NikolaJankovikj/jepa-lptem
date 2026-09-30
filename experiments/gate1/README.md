# Gate 1 — probing frozen V-JEPA 2 features on LPTEM

**Date:** 2026-09-30 · **Script:** `scripts/train_probes.py` (commit `3066452`) · **Data:** `dynamics-v1` (2000 clips → 4000 windows of 64 frames, 256² centre crop) · **Encoder:** `facebook/vjepa2-vitl-fpc64-256`, frozen · **Compute:** 1× RTX 3090, ~4 h (disk-bound; GPU ~10 %).

## Question

Can physical observables be read off **frozen** V-JEPA 2 ViT-L features of LPTEM video,
with no adaptation to the domain? PASS makes LoRA (Step 8) a single comparison run; FAIL
makes it the main path.

## Protocol

- Features: `last_hidden_state` per window, `(32 latent frames, 256 tokens, 1024)`;
  ground truth aligned to latent frames in crop coordinates (Step 6).
- Split 80/10/10 **by source clip**: 1600 / 200 / 200 clips (3200 / 400 / 400 windows).
- `AttentiveProbe` → `[count, mean_radius, total_area]` per latent frame (z-scored MSE).
  `CentroidProbe` → per-token occupancy + sub-cell offset (pos-weighted BCE + masked L2).
  Both trained jointly, AdamW 1e-3, 50 epochs; train loss flat at 0.52–0.53 over the last
  10 epochs (converged).
- Centroids decode as 3×3 local maxima; occupancy threshold chosen on **val** by detection
  F1 and applied to test. Decoded centroids → trackpy → drift-removed MSD, pooled across
  windows with n_full weights; the same pipeline on ground-truth centroids gives the
  perfect-probe ceiling (`*_gt`).
- Cache validated beforehand (`scripts/audit_features.py`): 4000 clean windows, 0 visible
  particles outside the crop, overlapping windows agree on shared frames (cos 0.786 vs
  0.590 one step apart), single-token linear probe for particle-centre cells AUC 0.979
  held-out vs 0.559 for labels rolled 5 cells.

## Results

Criteria were fixed in `PLAN.md` before the run. The gate is judged on **test**; val is
shown because it was used to pick the threshold and because it disagrees on count.

| criterion | test | val | |
|---|---|---|---|
| R² mean radius, physics stratum ≥ 0.8 | **0.910** | 0.893 | pass, both splits |
| R² count, physics stratum ≥ 0.8 | **0.806** | 0.699 | pass on test by 0.006; fails on val |
| MSD / 4Dt within [0.8, 1.2], read against the ceiling | **0.995** (ceiling 0.955) | 0.968 (ceiling 0.968) | pass, at the ceiling |
| track coverage (reported, not gated) | 0.051 (ceiling 0.330) | 0.049 (ceiling 0.328) | low |

Supporting numbers (test / val): R² total area 0.926 / 0.892 (physics), all strata count
0.844 / 0.738, mean radius 0.881 / 0.891; detection precision 0.896 / 0.883, recall
0.810 / 0.782 at threshold 0.95; MSD pooled over 57 / 50 windows (ceiling 288 / 296),
0 unlinkable windows.

**Verdict: PASS on the pre-registered criteria, marginal on count.**

## Reading

- **Size is in the frozen features.** Mean radius and total area reach R² ≈ 0.9 on unseen
  clips from an encoder that has never seen electron microscopy.
- **Diffusion decodes correctly where it decodes at all.** The MSD from probe-decoded tracks
  matches the perfect-probe ceiling on both splits.
- **Count is the weak observable.** A 0.1 R² swing between two 200-clip splits says the
  estimate is noisy and 0.806 is not a margin to lean on. Leading hypothesis, untested: the
  label counts a particle only if its *centre* is inside the crop, while a particle just
  outside is partly visible (and one just inside partly cut off). At ~4.5 particles per
  frame each such edge case is a large relative error.
- **Coverage is low for a mechanical reason.** A track counts only if detected in all 32
  latent frames with no gap bridging (`memory=0`); at ~80 % per-frame recall almost every
  track breaks somewhere. The MSD therefore rests on the ~15 % of trackable particles the
  probe holds continuously (57 windows vs 288 for the ceiling).

## Known limitations of this evaluation

1. **The threshold search hit its own edge.** 0.95 is the top of the searched grid
   (0.30–0.95), so the F1 optimum likely lies higher and the reported precision/recall
   trade-off is not the best available.
2. **No gap closing in linking**, which is what caps coverage (above).

Both can be re-evaluated from the saved `probes.pt` without retraining. They go in as
**secondary analyses next to** the numbers above, never in place of them — replacing the
headline after seeing it would move the goalposts.

## Consequence for the plan

- Per the PASS branch, **Step 8 shrinks to one LoRA comparison run.** Its sharpest question
  is now whether adaptation lifts count off the borderline.
- Step 9 (forecasting predictor) proceeds on the frozen cache. Its training script should
  load windows in parallel: this run was disk/CPU-bound at ~10 % GPU.
- `probes.pt` (not in git) is the frozen probe set Gate 2 applies to predicted latents.
