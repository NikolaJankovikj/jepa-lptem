# jepa-lptem

A latent world model for liquid-phase transmission electron microscopy (LPTEM).

We adapt **V-JEPA 2** to noisy LPTEM video and train a lightweight block-causal predictor that
rolls the latent state forward in time. The core contribution is an **evaluation methodology**:
probes that read physical observables (particle size distributions, growth-law exponents, MSD)
off the *predicted future* latent states, tested against exact simulator ground truth, a
persistence baseline, and a pixel-space video-prediction baseline.

See [`PLAN.md`](PLAN.md) for the step-by-step implementation plan and [`thesis_brief.md`](thesis_brief.md)
for the research framing.

## Setup

```bash
uv sync
```

Local machine = coding + CPU unit tests + analysis + writing. GPU work (feature extraction,
LoRA adaptation, predictor/baseline training) runs on the remote 2× RTX 3090 box — see the
"Two machines, one repo" section of `PLAN.md`.

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
