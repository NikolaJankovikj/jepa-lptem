# Where the project stands (2026-10-02)

A review of everything run on the PC so far: what was done, what went wrong, which of it
is our doing, and what the options are now. Numbers are from the experiment READMEs and
JSON files, which were re-checked against each other. Each section links its source.

## 1. What was done

| Step | Result | Source |
|---|---|---|
| Dataset `dynamics-v1`: 2000 simulated LPTEM clips, 5 processes, 2 strata | done, audited | `experiments/dataset-v1/` |
| Feature cache: frozen V-JEPA 2 ViT-L, **each 64-frame window encoded in one pass** | done, audited | `PLAN.md` Step 6 |
| Gate 1 on that cache | **PASS**, marginal on count (R² radius 0.910, count 0.806, MSD at ceiling) | `experiments/gate1/` |
| predictor-v1 (block-causal, 26 M params, L1, rollout 2) | 9 % below the context-mean baseline | `experiments/predictor-v1/` |
| Gate 2 | **FAIL** as pre-registered | `experiments/core/` |
| Gate 2 follow-up (+24, bootstrap CIs) | does not rescue Gate 2; forecasts fade | `experiments/core/` |
| Look-ahead check | **the cache leaks the future into the context** | `experiments/leak-check/` |
| Per-frame re-extraction + Gate 1 again | **FAIL on count** (0.726); radius, area, MSD pass | `experiments/gate1-perframe/` |

## 2. What went wrong, and whose fault it is

### 2a. The look-ahead leak: our design error, now fixed

- **What it was.**
  - The cache was built for Gate 1, where the job is to read the *current* state, so
    encoding whole 64-frame windows looked harmless.
  - The forecasting predictor was then trained on the same cache. Each of its 8 context
    latents had been computed by an encoder that attended over all 64 frames, the 48
    future frames included.
  - Nobody checked whether that cache was causal before reusing it for forecasting.
- **How big it was.**
  - Context latents re-encoded without the future agree with the cached ones at only
    cosine 0.74–0.84. The closer a frame is to the forecast, the lower the agreement.
  - Swapping only the future frames changes the "past" features and the probe's reading of
    them (`leak-check/lookahead.png`).
- **What it invalidated.**
  - v1's one clear win was detection F1 at +1: 0.80 vs persistence 0.56. With honest
    contexts it falls to 0.54 vs 0.56.
  - So the "drift carried forward" result was the encoder reading the future, not learned
    dynamics.
  - The Gate-2 FAIL still stands as recorded, but it is not interpretable.
  - predictor-v2 was planned on the same cache. No v2 results exist in the repo, and it
    should be dropped.
- **The fix.**
  - `extract_features.py --per-frame` encodes each 2-frame tubelet alone. This is how Meta
    uses the same encoder for its world model: V-JEPA 2-AC "encode[s] each frame
    independently" (V-JEPA 2 paper, §AC).
  - On the leak figure, swapping the future changes per-frame context features by exactly 0.
  - **Still to confirm on the full cache:** `logs/pf-audit.log` on the PC should show
    cosine 1.000 between overlapping windows on shared frames. The whole-window cache gave
    0.786.

### 2b. Gate 2's size criterion had no power: our pre-registration error

- The rule required the forecast to beat persistence by 30 % in size distance at +5.
- Even the *true* future beats persistence by only 3–6 %, because radii barely change over
  10 frames compared with the probe's ~2 px reading error.
- That threshold was set without first measuring the floor. The follow-up fixed this
  properly: a test only counts where the floor itself has power. It is recorded as a
  protocol error, not explained away.

### 2c. Forecasts fade on long rollouts: a known limitation, not specific to us

- By +24, v1's forecasts lose 43–74 % of the particles, while the true future stays sharp
  (F1 ≥ 0.85).
- This is what a deterministic L1 forecaster does when the future is uncertain: it
  averages, so the forecast blurs.
- V-JEPA 2-AC uses the same recipe (L1, teacher forcing plus a rollout of T = 2), and its
  authors report the same error accumulation over longer rollouts.
- It has not been measured on honest (per-frame) features yet. The fade may look different
  there.

### 2d. Count fails Gate 1 on per-frame features: not a bug, and partly our choices

| Gate 1 (test, physics stratum) | per-frame | whole-window |
|---|---|---|
| R² mean radius ≥ 0.8 | 0.902 ✔ | 0.910 |
| R² count ≥ 0.8 | **0.726 ✘** (val 0.575) | 0.806 (val 0.699) |
| MSD / 4Dt vs ceiling 0.955 | 0.986 ✔ | 0.995 |
| R² total area | 0.928 | 0.926 |
| detection precision / recall | 0.870 / 0.797 | 0.896 / 0.810 |

**Why it is not a pipeline bug:**
- The script, split and probe are the same, and the val-picked threshold is the same.
- A broken cache or misaligned labels would hurt everything. Here radius and area are
  unchanged, and the MSD still sits at the perfect-detector ceiling.
- Only one observable dropped, on both splits, by a similar amount (0.08 on test, 0.12 on
  val).

**Why count, specifically:**
- Gate 1's secondary analysis had already located count's errors: particles cut by the crop
  edge, and touching or merging particles (count R² was lowest on coalescence, 0.75).
- Both cases are ambiguous in one snapshot and become clear across several frames.
- The whole-window encoder had up to 64 frames to resolve them. A per-frame latent has 2.
- Some of the lost information was *past* frames, which is legitimate. Some was future
  frames, which is not.

**What is our doing:**
- **The probe reads one latent frame at a time.** That was fine when every latent had
  seen 64 frames. On per-frame features it means count is judged from 2 frames. The
  predictor itself sees 8 frames of context, so the gate measures a harder task than the
  model faces.
- **The label convention.** A particle counts only if its centre is inside the crop, while
  particles straddling the edge are half visible.
- **The 0.8 bar on count was set without knowing count's noise.** Val and test disagree by
  0.10–0.15 on both caches. Count's true value is uncertain by about that much.
- None of this changes the verdict. The rule was fixed in advance, and the result is a
  FAIL.

**What is the encoder's doing:**
- V-JEPA 2 was trained on 16–64-frame clips, with images shown as 16 identical frames. A
  2-frame clip is an unusual input for it.
- Meta's follow-up, **V-JEPA 2.1**, changed two things for exactly this use. It added a
  native image tokenizer (a 2D 16×16 patch embedding for single frames) and a loss on every
  token. Frozen dense-probe scores roughly doubled (ADE20K mIoU 24.4 → 47.9, largest model).

## 3. What depends on count downstream

- **Growth exponent** reads mean radius only. Not affected.
- **Positions and F1** read the centroid probe. Slightly weaker (precision 0.870 vs 0.896).
- **Size-distribution distance** takes its spread from area / (π · count). Affected: it
  inherits the noisier count, and that has to be stated next to every size number.

## 4. Is LoRA the only path? No.

As pre-registered, a Gate-1 FAIL makes LoRA (Step 8) the main path. That rule was written
to answer one question: are frozen features able to represent LPTEM physics at all? On
per-frame features the answer is "yes, except count". Count's shortfall points at
**single-frame ambiguity**, which LoRA's training objective does not target. Any deviation
from the rule must be written into `PLAN.md`, dated, before the next result is seen.

| Path | Cost | What it answers | Fixes count? |
|---|---|---|---|
| **A. Stage 2 on the per-frame cache** (predictor from scratch, Gate 2, follow-up) | ~7 h, 1 GPU, ready to launch | The core thesis question on honest features: does a latent predictor learn growth physics? | No. Size numbers carry the weak count |
| **B. Diagnostics** (`gate1_secondary.py`, audit log) | ~30 min CPU | Where count is lost (edge vs coalescence vs density); confirms the cache is per-frame | Locates it |
| **C. V-JEPA 2.1 ViT-L, frozen, per-frame** | Loader via `torch.hub` (not in HF transformers), ~1 day all in | Whether a better frozen dense encoder lifts count. Same size as now (300 M) | Plausibly. Built for single-frame dense features |
| **D. Past-only sliding window** (encode frames t−15…t+1, keep the last tubelet) | ~6–8× the extraction compute, ~70 GB disk, ~1–2 days | Whether *past* temporal context recovers count without any leak | Plausibly, for merges and edges |
| **E. LoRA** (Step 8) | 2–3 days per run, plus first getting Meta's training loop running (not started) | Whether domain adaptation helps; mainly the sim-to-real question (Step 13) | Unclear. Must also be trained per-frame to match how it is used |

**Why A comes first whatever is chosen:** every encoder path ends with "train the predictor,
run Gate 2", and A is that step on the encoder already in hand. If the predictor fails on
radius and positions, which pass Gate 1, then fixing count does not save the thesis. If it
succeeds, C or D is a clean "better encoder" comparison on top.

**Recommended order:** A and B now, together on the PC. Then C if count needs fixing, D as
the fallback. LoRA moves back to what it was best suited for: the sim-to-real step.

## 5. What the thesis can already say

1. Frozen V-JEPA 2 features carry LPTEM physics. Size and diffusion are readable at
   R² ≈ 0.9, and the MSD sits at the perfect-detector ceiling, *even with no temporal
   context*.
2. Encoding whole clips leaks the future into a latent forecaster's context and fakes
   predictive skill. This has been measured, shown and fixed. It is a methodological
   warning anyone building latent world models on video encoders can use.
3. A Gate-2 FAIL whose cause is documented (a protocol with no power, forecasts that fade).
   It has to be re-run on honest features before it means anything about dynamics.

## Open items

- [ ] Sync `logs/pf-audit.log`, `logs/pf-gate1.log` (provenance and convergence).
- [ ] `gate1_secondary.py --gate1 experiments/gate1-perframe` on the PC.
- [x] Decided 2026-10-02: A + B now; V-JEPA 2.1 next as encoder candidate; LoRA planned, timing open (`PLAN.md` Step 8).
- [ ] Superseded notes on the docs that still quote v1 as evidence of learned dynamics
  (`README.md` status, `experiments/core/README.md`, `experiments/predictor-v1/README.md`).

Sources: V-JEPA 2 paper, arXiv 2506.09985 (§ V-JEPA 2-AC: frame-wise encoding, T = 2
rollout, error accumulation); V-JEPA 2.1, arXiv 2603.14482 (image tokenizer, dense loss,
dense-probe results); checkpoints at github.com/facebookresearch/vjepa2.
