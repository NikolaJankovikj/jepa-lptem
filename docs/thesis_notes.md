# Notes for writing the thesis

Points to carry into the thesis text, with where each number comes from. Each entry
is dated; check its source before quoting.

## Related work: why token-level latent forecasts fail on LPTEM (2026-10-11)

**Where it goes:** Discussion of the Gate 2 FAIL, and Related Work (JEPA world models).

- **The claim.** The per-frame Gate 2 FAIL (`experiments/core-perframe/`) is the effect
  recent stochastic-JEPA papers describe: in a random environment, a deterministic latent
  predictor outputs an average of the possible next states, which "corresponds to no
  state at all" (MoP-JEPA). Our contribution is that we show it on a scientific
  instrument and give the mechanism in numbers.
- **The numbers** (`experiments/token-persistence/`: simulator ground truth, 2000
  physics-stratum windows, 256² crop, 16 px tokens, measured from the last context
  frame):

  | | +1 | +2 | +5 | +10 | +24 |
  |---|---|---|---|---|---|
  | particle centre still in the same 16 px token | 0.49 | 0.37 | 0.23 | 0.15 | 0.06 |
  | old centre still inside the particle | 0.63 | 0.49 | 0.30 | 0.19 | 0.08 |
  | median displacement (px) | 6.9 | 9.7 | 15.3 | 21.3 | 33.1 |

  The "same token" fraction is 0.47–0.51 at +1 for every process. Half the particles leave their token after
  one latent step, so the per-token median (the L1-optimal deterministic forecast) is
  background: particles are erased, not just blurred. This matches the measured fade
  (peak occupancy 0.96 → 0.69 → 0.44 at +1/+5/+10, `core-perframe/followup.json`).
  The motion is not a simulator artefact: it is calibrated to real video (~1.5
  radii/frame, `sim/dataset.py`).
- **Why V-JEPA 2-AC and DINO-WM do not hit this.** Their environments are nearly
  deterministic (actions drive the change), and they never decode forecasts. They only
  rank actions by latent distance to a goal, where a blurred forecast is harmless. We
  copied their recipe (per-frame encoding, L1, rollout 2) into a stochastic setting and
  then decoded it.
- **What stays predictable** (`experiments/growth-identifiability/`): from 8 context
  readings of mean radius at probe-like noise (sd 0.8 px), a 20-term ridge with no
  labels cuts the +24 error from 3.26 / 3.90 px (persistence) to 1.31 / 1.86 px on
  diffusion / reaction. Growth is predictable; positions are not. This is the case
  for forecasting a position-free state.
- **Source:** `scripts/token_persistence.py` and `scripts/growth_identifiability.py`
  (ground truth only, seconds on a laptop). Cite the commit that adds them.
- **References:**
  - MoP-JEPA (2026 preprint, arXiv 2607.05238): https://www.roboticscenter.ai/research/papers/mop-jepa-hard-assigned-predictor-mixtures-for-stochastic-jepa-world-models-2607
  - VJEPA, Variational JEPA as probabilistic world models (arXiv 2601.14354): https://arxiv.org/html/2601.14354v1
  - DINO-WM (ICML 2025, arXiv 2411.04983): https://proceedings.mlr.press/v267/zhou25t.html
  - V-JEPA 2 / 2-AC (arXiv 2506.09985)
  - LeCun 2022, *A Path Towards Autonomous Machine Intelligence*: the two ways JEPA
    handles uncertainty are (a) a representation invariant to the unpredictable and
    (b) a latent variable in the predictor.
  - Only abstracts and summaries of the 2026 preprints were read. Read them fully
    before citing.

## Related work: the Gate 1 protocol has precedent (2026-10-11)

**Where it goes:** Methods (evaluation) and Related Work.

- **The claim.** Reading physical quantities with attentive probes on a frozen JEPA
  encoder is an established way to judge whether a self-supervised model learned
  physics. Our Gate 1 (`experiments/gate1-perframe/`) follows it. Unlike that work, we
  also apply the probes to *forecast* latents (Gate 2).
- **Precedent.** *Representation Learning for Spatiotemporal Physical Systems* (Flatiron
  Institute, arXiv 2603.13227, ICLR 2026 AI & PDE workshop):
  - JEPAs on active matter, shear flow and Rayleigh–Bénard convection, with frozen
    encoders and attentive probes regressing the governing parameters.
  - JEPA beats VideoMAE on all three, e.g. active matter MSE 0.079 vs 0.160.
  - They do not forecast; latent prediction is only their pretraining objective.
  - https://arxiv.org/html/2603.13227v1
- **Also relevant:** Garrido et al. 2025, *Intuitive physics understanding emerges from
  self-supervised pretraining on natural videos* (arXiv 2502.11831). V-JEPA-style
  representation-space prediction shows intuitive physics, while pixel prediction stays
  near chance. Supports the brief's argument for latent rather than pixel prediction.
- **Novelty check (not exhaustive).** No JEPA work on microscopy, LPTEM or particle
  forecasting turned up in a search on 2026-10-11.
