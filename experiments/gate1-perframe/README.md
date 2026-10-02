# Gate 1 on per-frame features — probing look-ahead-free V-JEPA 2 latents

**Date:** 2026-10-02 · **Script:** `scripts/train_probes.py`, unchanged · **Cache:** `extract_features.py --per-frame` of `dynamics-v1` (every 2-frame tubelet encoded alone; `experiments/leak-check/`) · **Split:** identical to Gate 1 (seed 0, same 1600 / 200 / 200 clips) · **Compute:** 1× RTX 3090.

## Question

The Gate-1 cache encoded whole 64-frame windows, so every latent frame had seen the rest
of the window, including its future. Per-frame encoding removes that, and with it the
encoder's averaging of noise across time. Do the frozen features still carry the physics
when each latent sees only its own two frames?

## Results

Same pre-registered criteria as Gate 1, judged on **test**. The whole-window run
(`experiments/gate1/`) is shown alongside it.

| criterion | per-frame test | per-frame val | whole-window test | |
|---|---|---|---|---|
| R² mean radius, physics ≥ 0.8 | **0.902** | 0.888 | 0.910 | pass |
| R² count, physics ≥ 0.8 | **0.726** | 0.575 | 0.806 | **FAIL** |
| MSD / 4Dt in [0.8, 1.2], read against the ceiling | **0.986** (ceiling 0.955) | 0.901 (0.968) | 0.995 (0.955) | pass |
| track coverage (reported) | 0.048 (ceiling 0.330) | 0.049 (0.328) | 0.051 (0.330) | |

Supporting numbers (test): R² total area 0.928 on the physics stratum (whole-window
0.926), detection precision 0.870 and recall 0.797 (0.896 / 0.810), val-picked threshold
0.95 (same as before). MSD pooled over 76 windows (whole-window: 57).

**Verdict: FAIL on count as pre-registered. Mean radius, total area and MSD pass.**

## Reading

- **Taking the look-ahead out cost almost nothing on size and diffusion.** Mean radius
  loses 0.008 R², total area is unchanged, and the MSD still matches the perfect-probe
  ceiling. Two frames are enough to see a particle's size and position.
- **Count is the one observable that relied on time.** It drops by 0.08 on test and 0.12 on
  val. It was already the weak, noisy one (0.806 test / 0.699 val with the whole window).
  Gate 1's secondary analysis showed where it is hard: particles straddling the crop edge,
  and touching or merging particles. Both are ambiguous in a single frame and easier
  across several.
- **What it means downstream.** Gate 2's exponent test reads mean radius, and its position
  test reads the centroid probe; both hold up. The size-distribution distance does depend
  on count: its sd is derived from area / (π · count). Gate-2 size numbers on this cache
  inherit the weaker count reading, so that has to be stated next to them.

## Open

- Cache audit (`logs/pf-audit.log`) not yet synced back. It is the provenance check that
  this run read per-frame features: overlapping windows should agree at cosine 1.000 on
  shared frames (the whole-window cache: 0.786).
- `scripts/gate1_secondary.py --gate1 experiments/gate1-perframe` (CPU, ~15–30 min) to see
  whether the count loss sits in the edge and coalescence buckets.
