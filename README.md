# jepa-lptem

A latent world model for liquid-phase transmission electron microscopy (LPTEM).

We adapt **V-JEPA 2** to noisy LPTEM video and train a lightweight block-causal predictor that
rolls the latent state forward in time. The core contribution is an **evaluation methodology**:
probes that read physical observables (particle size distributions, growth-law exponents, MSD)
off the *predicted future* latent states, tested against exact simulator ground truth, a
persistence baseline, and a pixel-space video-prediction baseline.

See [`PLAN.md`](PLAN.md) for the step-by-step implementation plan and current status and [`thesis_brief.md`](thesis_brief.md)
for the research framing.

## Setup

```bash
uv sync
```

Local machine = coding + CPU unit tests + analysis + writing. GPU work (feature extraction,
LoRA adaptation, predictor/baseline training) runs on the remote 2× RTX 3090 box — see the
"Two machines, one repo" section of `PLAN.md` and the runbook in [`docs/remote.md`](docs/remote.md).

## Status (2026-10-01)

- **Dataset** `dynamics-v1`: 2000 audited clips — [`experiments/dataset-v1/`](experiments/dataset-v1/README.md)
- **Feature cache**: 4000 frozen V-JEPA 2 windows, audited (Step 6 in `PLAN.md`)
- **Gate 1: PASS**, marginal on count — [`experiments/gate1/`](experiments/gate1/README.md)
- **Predictor** `predictor-v1`: 9 % below the context-mean baseline — [`experiments/predictor-v1/`](experiments/predictor-v1/README.md)
- **Gate 2: FAIL** as pre-registered; the size criterion turned out to have no power (even the true future fails it) — [`experiments/core/`](experiments/core/README.md)
- **Next**: Gate-2 follow-up (longer rollout, blur vs growth, CIs)

## Tests

```bash
uv run pytest            # CPU tests (GPU tests skipped via the 'not gpu' default)
uv run pytest -m gpu     # GPU tests (on the remote only)
```

## Layout

```
src/jepa_lptem/
  data/     dataset loaders
  eval/     observables, tracking, probes  (the evaluation spine)
  sim/      simulator dynamics + render bridge
  models/   the latent forecasting predictor
scripts/    runnable jobs (generation, feature extraction, training, evaluation)
configs/    training configs
experiments/  per-experiment results + write-ups (→ thesis Results)
```
