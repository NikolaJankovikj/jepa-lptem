# P1: physics read off the pooled (position-free) latent state

**Date:** 2026-10-11 · **Script:** `scripts/pooled_gate1.py` (test: `tests/test_pooled_gate1.py`) · **Data:** `data/features/dynamics-v1-perframe-pooled.npz` (P0, `scripts/pool_features.py` on the PC, 350 s, CPU), the Gate-1 per-frame split (3200 / 400 / 400 windows) · **Compute:** Mac, 43 s for all three states · **Protocol:** `PLAN.md` Step 10, *Position-free forecasting*, fixed 2026-10-11 before any pooled number. `metrics.json` and `readout.npz` (the frozen readout for P2–P3) are here; the run log is `logs/pooled-gate1.log` (not in git).

## Question

Averaging the 256 tokens of a latent frame removes where the particles are. Is the
physics still readable from what is left? This is the precondition for forecasting the
pooled state (P2–P3).

## Results (test, physics stratum; ridge readout from one frame, λ picked on val)

| | mean | mean+std | grid2 | *token probe, per-frame Gate 1* |
|---|---|---|---|---|
| R² mean radius | **0.877** | 0.884 | 0.873 | *0.902* |
| R² total area | 0.942 | 0.950 | 0.938 | *0.928* |
| R² count (reported) | 0.723 | 0.732 | 0.718 | *0.726* |
| λ | 10³ | 10³ | 10⁴ (grid edge) | |

**Growth exponent read from the true future** (frames 8–31, median [95 % clip-bootstrap CI]), `mean` state:

| | readout | true | token floor (follow-up) |
|---|---|---|---|
| diffusion | 0.311 [0.278, 0.349] | 0.333 | 0.348 |
| reaction | 0.473 [0.434, 0.505] | 0.500 | 0.498 |
| static | 0.009 [−0.010, 0.020] | 0.000 | 0.002 |
| ripening | 0.053 | – | 0.037 |
| coalescence | 0.047 | – | 0.055 |

**Verdict: usable, as pre-registered.** `mean` passes both criteria (R² mean radius
0.877 ≥ 0.8; exponents within ±0.1 on diffusion and reaction) and, as the first usable
state, becomes *the pooled state*. The other two also pass; they are reported, not used.

## Reading

1. **Averaging away position costs almost nothing.** A single linear map from the
   1024-d frame average reads mean radius at R² 0.877, against 0.902 for the attentive
   probe on all 256 tokens. Total area reads *better* (0.942 vs 0.928) and count the same
   (0.723 vs 0.726). Size is a global property of the frame, and the average keeps it.
2. **Growth is in the pooled state.** Read from the true future, the exponents land
   within 0.03 of the textbook values (0.311 / 0.473 vs 1/3 and 1/2), and static stays at
   0. A pooled forecast therefore has a target that carries the growth law.
3. **The dilution worry is answered.** Neighbouring frames' averages are 98.8 % alike
   in raw cosine (P0 audit). After per-channel z-scoring, the particle signal is still
   linearly readable.
4. **More spatial detail adds nothing.** mean+std is +0.007 R²; grid2 is slightly
   *worse*, and its λ sits at the grid edge. That is consistent with size being a global
   quantity.
5. **Count stays the weak observable** (0.72), unchanged from the token probe. That
   points to per-frame shot noise on small particles (`experiments/gate1-perframe/`),
   not to the pooling.

## Caveats

- The readout is linear and fitted with labels: an evaluation instrument, as in Gate 1.
  The forecasters in P2–P3 never see labels.
- Val R² is flat in λ from 10⁻² to 10³ (0.852–0.854 for `mean`), so the choice of λ does
  not drive the result.
- The exponents are read from true future latents. Whether a *forecast* carries them
  is P2–P3 (Gate 2′).
