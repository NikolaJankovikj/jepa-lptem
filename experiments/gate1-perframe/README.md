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

## Secondary analyses (2026-10-03)

`scripts/gate1_secondary.py --gate1 experiments/gate1-perframe` (commit `6089333`: decodes
at this run's own val threshold). Raw numbers are in `secondary.json`. They sit next to the
verdict and do not change it. The whole-window numbers come from `experiments/gate1/`.

**1. Count vs the crop edge (test, physics stratum).**

| particles near the edge | frames | MAE per-frame | MAE whole-window | bias per-frame |
|---|---|---|---|---|
| 0 | 809 | 0.75 | 0.63 | +0.09 |
| 1 | 1998 | 0.76 | 0.65 | −0.06 |
| 2 | 2575 | 0.84 | 0.74 | −0.06 |
| 3+ | 4730 | 1.01 | 0.91 | −0.27 |

R² 0.741 with no edge particle (whole-window 0.829), 0.719 with one or more (0.800). The
loss is **the same ~0.1 MAE in every bucket**, so the crop edge does not explain it.

**2. Per process (test, all strata).**

| process | R² count per-frame | whole-window | Δ | R² mean radius per-frame | whole-window |
|---|---|---|---|---|---|
| coalescence | 0.675 | 0.751 | −0.08 | 0.844 | 0.865 |
| diffusion | 0.779 | 0.868 | −0.09 | 0.876 | 0.895 |
| reaction | 0.700 | 0.832 | **−0.13** | 0.816 | 0.835 |
| ripening | 0.715 | 0.827 | **−0.11** | 0.739 | 0.796 |
| static | 0.774 | 0.830 | −0.06 | 0.916 | 0.927 |

The merging process (coalescence) does not lose most. Reaction and ripening do: the
processes with many small particles, growing from nucleation or dissolving toward the
visibility floor.

**3. Detection and MSD with gap closing (test).** The fine-grid val optimum is 0.96, just
above the 0.30–0.95 grid used for the headline (whole-window: 0.95).

| linking | MSD / 4Dt | ceiling | windows (probe / ceiling) | coverage |
|---|---|---|---|---|
| no gaps (headline) | 0.986 | 0.955 | 76 / 288 | 0.048 |
| gaps ≤ 1 | 0.980 | 0.950 | 173 / 301 | 0.129 |
| gaps ≤ 2 (Gate 2's linking) | 0.966 | 0.947 | 199 / 307 | 0.158 |

The whole-window probe matched the ceiling to three decimals (0.950 / 0.948). The
per-frame probe sits 2–3 % above it, with fewer tracks.

**Reading.**
- The whole-window guess for count (crop edge, merging) does **not** explain the
  per-frame loss.
- The pattern fits **noise**: a per-frame latent sees one frame pair's shot noise, and the
  whole-window encoder averaged it over many frames.
  - That hurts most where particles are small and faint (reaction, ripening).
  - It adds independent position error to every detection. In tracking that raises the
    MSD by a constant (MSD = 4Dτ + 4σ²), which is the excess over the ceiling seen here.
  - Whole-window errors were correlated in time and cancelled out.
- Not tested directly: count error split by particle size.
- Diffusion still passes comfortably. The MSD bias is known-sign and small.

## Cache audit (provenance)

`scripts/audit_features.py` on the per-frame cache (`logs/pf-audit.log`, not in git):

| check | per-frame | whole-window |
|---|---|---|
| windows, shape | 4000, (32, 256, 1024) | same |
| non-finite / constant windows | 0 | 0 |
| visible particles outside the crop | 0 | 0 |
| cos, same frames in two windows | **1.000** | 0.786 |
| cos, one latent step apart | 0.614 | 0.590 |
| particle-centre token AUC (control, rolled 5) | 0.978 (0.534) | 0.979 (0.559) |

- **The run read per-frame features.** A frame's latent is identical whichever window it
  sits in, so no other frame reaches it.
- Spatial decodability is unchanged.
