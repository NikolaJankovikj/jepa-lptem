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

## Status (2026-11-16)

Full findings by research question: [`docs/findings.md`](docs/findings.md).

- **Gate 1** (per-frame features): FAIL on count only; size, area and MSD pass — [`experiments/gate1-perframe/`](experiments/gate1-perframe/README.md)
- **Gate 2** (token-level forecasts): FAIL; half the particles leave their token each step, so deterministic forecasts erase them — [`experiments/core-perframe/`](experiments/core-perframe/README.md), [`experiments/token-persistence/`](experiments/token-persistence/README.md)
- **Gate 2′** (frame-averaged latent): size criterion passes, exponents damped by the context's noise — [`experiments/pooled-forecast/`](experiments/pooled-forecast/README.md), [`experiments/longer-context/`](experiments/longer-context/README.md)
- **RQ3** (dose): 0.3–0.6 px worse at low dose at every horizon, through the readout — [`experiments/snr/`](experiments/snr/README.md)
- **RQ2** (merges): the forecaster's error detects merges (AUC 0.71) but does not anticipate them, and simple baselines do as well — [`experiments/events/`](experiments/events/README.md)
- **Real data** (Step 13): negative. 8 screened published movies + SAM-EM nanorods: none inside the simulated feature distribution, and the forecaster loses to the context mean on every one — [`experiments/real/`](experiments/real/README.md)
- **Next**: thesis Results draft

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
