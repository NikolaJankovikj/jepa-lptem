# JEPA-LPTEM Implementation Plan

**Goal:** Show that a V-JEPA-2-based latent world model, adapted to LPTEM video, predicts futures that decode to correct physics (size distributions, growth exponents, MSD) and anticipates events (coalescence).

**Architecture:** Frozen-or-LoRA V-JEPA 2 ViT-L encoder → cached latent features → small block-causal forecasting predictor (teacher forcing + rollout) → probes that read physical observables off predicted latents → compared against a persistence baseline and a pixel ConvLSTM baseline. All probes are validated against simulator ground truth *before* any model exists.

**Tech stack:** Python 3.11, PyTorch 2.x (bf16, DDP), HuggingFace `transformers` (V-JEPA 2) + `peft` (LoRA), official `facebookresearch/vjepa2` repo for SSL adaptation, LPTEMsimulator physics (re-implemented in `render_bridge.py`), pytest, uv, tmux + ssh for remote runs.

---

## How we work

- One step at a time: Nikola gives the green light → implement the step → review together → iterate → next step.
- Every trainable/testable module gets its tests written **first**, verified to fail, then implemented (the code snippets below are starting points, already reviewed once).
- After each step: commit. After each experiment: same-day write-up in `experiments/<name>/README.md` — these become thesis Results subsections.

## Sequencing: local first, PC last

**Everything that can be built and unit-tested on the Mac (Steps 2–4) comes first.** The GPU PC is only provisioned (Step 5) once the whole local pipeline — evaluation spine, simulator, render bridge — is coded, smoke-tested, and green. Nikola gets PC access *when he decides the local side is ready*, not before. Rationale: PC access is the one risk outside our control (IT/VPN/physical), and nothing before Step 5 needs a GPU, so we don't want a remote environment sitting idle or blocking local work.

## Two machines, one repo

| | **Local (this Mac)** | **Remote (work PC, 2× RTX 3090)** |
|---|---|---|
| Role | All coding, unit tests (CPU), analysis, plotting, thesis writing | Heavy batch compute only: dataset generation, feature extraction, LoRA/predictor/baseline training, large evaluations |
| Code | Authored here, pushed to a private GitHub repo | `git pull` before every run — never edited on the remote |
| Data | Small smoke subsets only | Full datasets + feature cache + checkpoints live here (never in git) |
| Results | Metrics/plots/small artifacts rsynced back after each run | Checkpoints stay here; only selected ones synced back |

**Run protocol (every remote job):** local `git push` → `ssh` → `git pull && uv sync` → launch inside `tmux` → detach → monitor (W&B or tail logs) → `rsync` metrics + plots back → analyze locally.

---

## Step 0 — Local repo + environment ✅

**Where:** local. **Project root:** the already-cloned GitHub repo at `~/Documents/masters/JEPA-LPTEM/jepa-lptem` (`github.com/NikolaJankovikj/jepa-lptem`). **Files:** `pyproject.toml`, `src/jepa_lptem/__init__.py`, `tests/__init__.py`, `.gitignore`

- [x] **Fix the remote URL — it embedded a GitHub personal access token in plaintext.** Done via a dual-account SSH setup that leaves the work account (keychain + `Host github.com`) untouched: dedicated key `~/.ssh/id_ed25519_personal`, alias `Host github-personal` in `~/.ssh/config`, remote set to `git@github-personal:NikolaJankovikj/jepa-lptem.git`, repo-local identity `Nikola Jankovikj <NikolaJankovikj@users.noreply.github.com>`. Replicate the same alias trick on the work PC in Step 5.
- [x] **Manual (Nikola):** add `~/.ssh/id_ed25519_personal.pub` to the personal GitHub account (Settings → SSH and GPG keys), verify with `ssh -T git@github-personal`, then **revoke the exposed token** (Settings → Developer settings → Personal access tokens). Done 2026-09-28; no token left in any local git config.
- [x] Set up the uv environment and skeleton inside the existing clone (deps installed, package layout under `src/jepa_lptem/`, `.gitignore`).

  Notes: `code/` (vendored upstream repos) is gitignored — they're pinned upstream, not ours to commit. `uv` resolves torch per-platform automatically (MPS wheel here, CUDA wheel on the remote from the same lockfile). GPU-only tests are marked `@pytest.mark.gpu` and skipped locally. **Gotcha (done):** `uv init` creates an *application* with no build backend, so `import jepa_lptem` fails until a `[build-system]` (hatchling) is added — done. Also `.gitignore` must anchor `/data/` with a leading slash, else it also ignores the `src/jepa_lptem/data/` package — fixed.
- [x] Add a `gpu` marker to `pyproject.toml` (`markers = ["gpu: ..."]`, `addopts = "-m 'not gpu'"` locally; remote runs use `-m gpu`). Verified: `uv run pytest` reads the config and skips gpu by default.
- [x] Move `PLAN.md` and `thesis_brief.md` into the repo so they're versioned and visible on the remote too.
- [x] Commit + push: `chore: project skeleton` (pushed to `origin/master`, commit `78ada5c`).

## Step 1 — Verify sources, start downloads, send the data email ✅

**Where:** local (small subsets only — full data is pulled on the remote in Step 5).

- [x] **Verified the exact repo IDs** (recorded in `docs/data_sources.md` with sizes). Findings: authoritative SAM-EM data = **`papertemp/SAM-EM-Paper-PublicData`** (has checkpoint + trainset + holdout + both experimental sets + ablations); LEONARDO **dataset** = `JamaliLab/LEONDARDO` (the "typo" is the real upstream repo name); LEONARDO **model** = `JamaliLab/LEONARDO`; encoder = `facebook/vjepa2-vitl-fpc64-256`.
- [x] Cloned upstream code (shallow, gitignored) into `code/`: `sam-em` (has the renderer under `application/`+`data/`), `leonardo` (`LEONARDO_*.py` motion sampling), `vjepa2` (`src/` SSL loop). Note: macOS case-insensitive FS shows `vitG*/vitg*` config collisions in the vjepa2 clone — cosmetic, clean on the Linux remote.
- [x] Downloaded small dev subsets (~406 MB, in `data/`, gitignored): SAM-EM `ablationValidationSet.zip` (96 MB synthetic+masks), `experimentalAuNanorods.zip`, `experimentalAggregation.zip`; LEONARDO `datasets.zip` + xlsx + `LEONARDOModel`. Everything ≥ 2 GB (trainset 32 GB, holdout 2.3 GB, checkpoint 2.7 GB, 64 GB ablation) deferred to the remote. **Gotcha:** `hf download REPO --include A B C` mis-parses (first name eaten by `--include`, rest positional); download multi-file with one `hf download REPO FILE` call per file.
- [x] Smoke-loaded V-JEPA 2 (CPU): `facebook/vjepa2-vitl-fpc64-256` → 326M params, `VJEPA2Model` + `VJEPA2VideoProcessor`, weights cached. (Consider `HF_TOKEN` for the remote's 32 GB pulls — unauthenticated HF is throttled.)
- [x] **Send the Jamali-lab data-request email** (`docs/jamali_data_request_email.md`) — sent; **no reply** as of 2026-10-11. Treat the raw videos as unavailable.
- [x] Commit docs (`data_sources.md`, email draft). Code/data are gitignored.

---

# LOCAL PHASE (no GPU needed — Steps 2–4)

## Step 2 — Evaluation spine, tested against ground truth ✅

**Where:** local, pure CPU. *The thesis's contribution is the evaluation — build and validate it before any model exists.* **Done: 15 tests green.** Library-backed where battle-tested (SciPy Wasserstein, `scipy.ndimage` blob centroids, `trackpy` linking); custom only for the probes.

### 2a: Physical observables (`src/jepa_lptem/eval/observables.py`) — done

- [x] Failing tests first (`tests/test_observables.py`):

```python
import numpy as np
from jepa_lptem.eval.observables import compute_msd, fit_growth_exponent, size_dist_distance

def test_msd_recovers_diffusion_coefficient():
    rng = np.random.default_rng(0)
    D, dt, T, N = 0.5, 1.0, 200, 500          # 2D Brownian: MSD(t) = 4 D t
    steps = rng.normal(0, np.sqrt(2 * D * dt), (N, T, 2))
    traj = np.cumsum(steps, axis=1)            # (N particles, T frames, xy)
    msd = compute_msd(traj)                    # (T-1,)
    slope = np.polyfit(np.arange(1, 51), msd[:50], 1)[0]
    assert abs(slope - 4 * D) / (4 * D) < 0.15

def test_growth_exponent_lsw():
    t = np.arange(1, 200, dtype=float)
    r = (3.0 * t) ** (1 / 3)                   # diffusion-limited: r ~ t^(1/3)
    assert abs(fit_growth_exponent(t, r) - 1 / 3) < 0.02

def test_growth_exponent_reaction_limited():
    t = np.arange(1, 200, dtype=float)
    r = (3.0 * t) ** 0.5
    assert abs(fit_growth_exponent(t, r) - 0.5) < 0.02

def test_size_dist_distance_zero_on_identity():
    s = np.random.default_rng(1).gamma(4.0, 1.0, 300)
    assert size_dist_distance(s, s) < 1e-9
```

- [x] Implemented: `compute_msd` (ensemble MSD over lag), `fit_growth_exponent` (log-log slope, drops non-positive samples), `size_dist_distance` (1-Wasserstein via `scipy.stats.wasserstein_distance`).
- [x] 7 tests pass → committed.

### 2b: Centroid extraction + tracking (`src/jepa_lptem/eval/tracking.py`) — done

- [x] Failing tests: heatmap peaks recovered via connected components; two drifting particles linked over 5 frames; **dropout case** proving coverage is reported.
- [x] Implemented `heatmap_to_centroids` (scipy) and `link_tracks` (**trackpy** Crocker–Grier, chosen over hand-rolled Hungarian for gap-closing/robustness). Returns a `LinkResult(tracks, coverage, n_particles, n_full)` — **coverage is always reported**, never silently dropped.
- [x] 4 tests pass → committed.

### 2c: Probes (`src/jepa_lptem/eval/probes.py`) — done

Two probe types, matching the V-JEPA attentive-readout protocol (frozen features, small trainable head):

- **`AttentiveProbe`** — one learned query cross-attends over a latent frame's tokens → regression head for `[count, mean_radius, total_area]`.
- **`CentroidProbe`** — per-token head (with a learned positional embedding) → **occupancy logit + (dy, dx) sub-cell offset**. The token grid is 16×16 over a 256² frame (16 px/cell) — raw grid centroids would quantize Brownian steps to noise; sub-cell offsets fix this: predicted centroid = cell center + offset, trained with BCE (occupancy) + masked L2 (offsets).

- [x] Tests: `AttentiveProbe` overfits a tiny batch (`loss < 1e-2`); `CentroidProbe` shapes `(B,16,16)` + `(B,16,16,2)`; **pure-math** `decode_centroids` recovers an off-center point to <1 px; and a learn-a-target-field test where the trained probe's decoded centroid lands <1 px from intent.
- [x] Implemented both probes + `decode_centroids(occupancy, offsets, threshold, cell_size)` feeding 2b's tracker.
- [x] 4 tests pass → committed. **Note:** `pyproject.toml` now sets `testpaths=["tests"]` so pytest never collects the vendored `code/vjepa2/tests`.

## Step 3 — Simulator extension: verifiable radius dynamics ✅

**Where:** local. **Files:** `src/jepa_lptem/sim/dynamics.py`, `tests/test_dynamics.py`. **Done: 9 tests green.** Each mode reproduces its kinetic law, recovered by 2a's own `fit_growth_exponent` (closing the loop between simulator and evaluation).

- [x] `evolve_radii` — independent single-particle ODEs: `diffusion` (dr/dt=K/r² → t^⅓), `reaction` (dr/dt=K/r → t^½), `dissolution` (dr/dt=−K, clamp 0). Exponents recovered within ±0.05.
- [x] `merge_on_contact` — volume-conserving coalescence (r_new=(r_i³+r_j³)^⅓, volume-weighted position), cascades until stable. Tests: pair merge, no-merge-when-far, 3-particle cascade.
- [x] **`ripen_population`** — the recommended upgrade: *coupled, mass-conserving* Ostwald ripening (LSW). Gibbs–Thomson exchange around r*=⟨r⟩, sub-critical particles dissolve, volume renormalized to initial each step. Verified real coarsening: count 40→6, mean radius 2.50→5.18, total volume constant (851.02). This is the physically-faithful `ripening_coalescence` growth mode; the three `evolve_radii` modes stay simple independent ODEs.
- [x] 9 tests pass; full suite 24 green → committed.

## Step 4 — Render bridge + dynamics dataset generator (code + local smoke) ✅

**Where:** local — write the code, smoke-test on a handful of videos, validate the physics. **The full generation is deferred to the remote (first job after Step 5 PC setup)** — it's a multi-hour batch job whose ~50 GB output is consumed on the PC.

- [x] **Found the real renderer: `github.com/JamaliLab/LPTEMsimulator`** (a Python port of Yao et al.'s MATLAB `chenlabUIUC/simulatedLPTEM`) — *not* SAM-EM, which is only a segmentation GUI. It has no LICENSE file, so we **re-implement** the image-formation physics cleanly (cited) rather than copy it. Cloned to `code/LPTEMsimulator/` for reference; the one reused asset is its measured `MTFmatrix512.mat`.
- [x] Implement `src/jepa_lptem/sim/render_bridge.py` — our own clean version of the LPTEMsimulator physics: `draw_spheres` (our circle front-end + spherical-chord gold-thickness map, replacing their nanorods), `render_frame` (Beer–Lambert attenuation through gold/water/SiNx + Poisson + Gaussian + optional MTF), `render_video`, `normalize_uint8`. Fully TDD'd — `tests/test_render_bridge.py` (12 tests: disk area, chord=diameter, darker-particle invariant, water→dimmer monotonic, seeded-noise determinism, MTF DC-preservation).
- [x] Added `simulate_brownian` to `sim/dynamics.py` (the position ground truth; MSD = 4Dt verified via `compute_msd`, reflecting-box option). +3 tests.
- [x] Implement `src/jepa_lptem/sim/dataset.py` (`build_clip`, the composition layer) + `scripts/generate_dynamics_dataset.py` as a thin CLI on top: processes `{ripening, diffusion, reaction, static}`, one `.npz`/clip (`frames` uint8 `(T,H,W)`, `positions` `(N,T,2)`, `radii` `(N,T)`) + `manifest.jsonl`.
- [x] **Smoke locally (`--n-videos 5`, 512², real MTF):** appearance matches a real SAM-EM frame unprompted (real mean 123 vs ours 126, same Poisson texture); GIFs sent to Nikola. `scripts/verify_clips.py` writes overlay GIFs + physics panels and runs an **independent** classical detector (shares no code with the renderer) for a pixels→physics round trip.
- [x] **Three defects found and fixed before freezing the generator** (see below). Final smoke numbers, all four processes: MSD/4D ratio 0.97–1.13, α = 0.332 / 0.499 / −0.000 / −0.003 (diffusion / reaction / static / ripening), round-trip position error **0.53–0.70 px** on ~20 px radii, detector precision 0.83–1.00.
- [x] `tests/test_dataset_physics.py` — permanent regression guard on the generated ground truth (exponents to ±0.005, MSD ≈ 4Dt, sparsity ≤ 10 % coverage, min radius ≥ 2 px). Full suite **51 passed**.
- [x] Commit scripts. (Full run + `experiments/dataset-v1/README.md` happen in Step 5/6 on the remote.)

**Three defects fixed at the end of Step 4** (all would have silently poisoned the 50 GB remote run):

1. **Negative water thickness.** `render_frame` computed `water = thickness_water − gold_nm`, which goes negative once a particle exceeds the 150 nm cell — subtracting optical depth and rendering large/overlapping particles too bright. Now clamped at 0 (a particle thicker than the cell displaces all the water and no more), with a test asserting intensity stops depending on `thickness_water` past that point.
2. **Uncontrolled density.** Particle count was drawn independently of final size, so growth clips reached ~48 px radii at ~35 % gold coverage — heavy overlap, and *the ground truth stops being recoverable from the frame*, which would make any Gate-1 failure uninterpretable. `rescale_to_density` now scales all radii by one constant so peak coverage hits a target 4–9 % (log-log slope and MSD are both invariant under a uniform radius rescale). Round-trip error on the diffusion clip: 27.65 px → **0.53 px**.
3. **Two exponent biases, both real physics bugs, not tuning.** (a) *Wrong time origin* — a particle of radius `r0` is not newly nucleated, it already has age `r0³/3K`; labelling frame 0 as `t=0` flattened reaction to 0.43. `growth_time_axis` now returns `(t0, dt)` and the clip carries `t0_growth` in its metadata. (b) *Integrator error* — `evolve_radii` was single-step forward Euler, and `dr/dt = K/r²` moved the radius 65 % in one frame at small `r`, so the measured exponent was the integrator rather than the physics. Added `substeps` (default 10, matching `ripen_population`). Combined: diffusion 0.304 → **0.3318**, reaction 0.433 → **0.4993**.

**Step 4b — imaging realism + the normalisation leak.** Prompted by asking how our frames actually compare to real LPTEM, measured with `scripts/compare_to_real.py` (Immerkær noise estimator, edge-insensitive):

| source | σ/background | note |
|---|---|---|
| REAL experimental Au nanorods | 0.238 | JPEG-compressed, unknown preprocessing |
| THEIR synthetic (LPTEMsimulator) | 0.063 | matches our flux=75 prediction to 3 decimals |
| OURS (was, flux=30) | 0.074 | |

Two fixes, one of them a correctness bug:

1. **Per-clip min–max normalisation was leaking the label.** `normalize_uint8` stretched every clip to 0–255, which (a) erased absolute contrast — a 32 nm particle (25 % attenuation) and a 160 nm one (77 %) saved identically — and (b) made overall frame brightness correlate with mean radius at **+0.72/+0.77 on the growth modes, with the sign inverted from the real physics**. A Gate-1 probe could have read mean radius off global brightness without ever localising a particle, on exactly the two modes where the growth exponent is the headline result. Replaced by `to_uint8`, which divides by the analytically known background (`background_intensity`) — one fixed scale for the whole dataset, dose-independent, background at 64/255 for 4× noise headroom. Correlations are now **−0.74/−0.97** (physical attenuation). Ripening reads +0.41, which *is* physical: mass-conserving coarsening shrinks total projected area.
2. **Imaging conditions randomised, not fitted.** `beam_flux` log-uniform over 2–75 (σ/bg 0.27→0.06, bracketing real 0.238) and `pixel_size_nm` 1–4 (magnification, which sets contrast). Fitting the point estimate would overfit the simulator to 12 JPEG frames of one experiment; bracketing puts real data inside the training distribution. Both recorded per clip in the manifest → gives the **physics-recovery-vs-dose** ablation figure for free.

Not closable at any parameter setting: real LPTEM's structural texture (SiNx granularity, contamination, radiolysis bubbles, drift). State this as the Tier-2 bound.

`scripts/verify_clips.py` now reports the brightness-shortcut correlation and the saturated-pixel fraction on every clip. Suite **57 passed**.

**Step 4c — appearance vs real video, and the stratified generator.** `scripts/compare_to_real.py` compares against real experimental video on the three things that survive the fact that the real frames are pre-stretched JPEGs (absolute intensity is *not* comparable): apparent particle size in px, radial power spectrum of I/mean, and frame-to-frame change. Also writes `real_video.gif` and a side-by-side GIF **cropped, never resized** — resizing to a common display size would rescale the very quantity under comparison.

It found that our particles were **~3× larger than real in apparent pixels** (16.7 px radius vs 6.1). That matters specifically for a ViT: at 16×16 patches, real particles occupy under one patch and ours spanned ~2×2. Real data sat *outside* our distribution, which no amount of realism elsewhere would rescue.

Fix: **two strata** (`sim/dataset.py:STRATA`), because "match the real experiment" and "measure physics precisely" genuinely conflict.
- `physics` (4 in 5): 8–22 px particles, 6–16 of them, flux 2–75. Where Gate 1 and Gate 2 are decided — a probe failure here must mean the model can't do it, not that our estimator was too noisy to tell.
- `experiment` (1 in 5): 5–8 px particles, ≤2 % coverage, flux 8–12. Weak statistics by construction; its only job is to put real imaging conditions inside the training distribution. Uses a smaller `growth_ratio` (2.0 vs 5.0) — at 5× a 6 px particle starts at 1.2 px, under the visibility floor, and the opening frames come out empty.

Also added `MIN_VISIBLE_RADIUS_PX`: a particle too small to see is dropped from the ground truth rather than labelled as a target that isn't in the image.

Result — apparent radius **5.6 px** (real 6.1) and frame-to-frame change **0.347** (real 0.397) on the experiment stratum. Suite **69 passed**.

**Known residual — closed later, in two stages (4h operator order, 4j MTF calibration).** At matched temporal noise our frames were spatially *smoother* than real: the ratio of temporal to spatial noise is 3.16 for us at every flux versus 1.67 real, and our high-frequency power sits ~10× low. Pure shot noise fixes that ratio, so the difference is spatial correlation — most likely the measured 512 px MTF over-blurring relative to the real 1024 px detector. Ruled out: JPEG compression (our ratio holds at ~3 for every quality from 100 down to 60). A static-texture term was implemented and **reverted** — it cannot move an Immerkær estimate, which is by construction insensitive to structure. Report as a Tier-2 bound; static structure is the least harmful gap for a video predictor, since it does not change between frames.

**Step 4d — two presentation bugs in the comparison, found by looking at the GIF.** Nikola spotted that our panel "changes scenes" and that our dark spots look much darker than real. Both were artefacts of `compare_to_real.py`, not of the data:
- The GIF concatenated 8 frames from each of 10 clips, so it cut to a different clip every 8 frames. Now animates one continuous clip (experiment-grade by preference); the pooled statistics still use all clips.
- Every panel was matplotlib-autoscaled to its own min/max, so apparent contrast was set by each source's dynamic range rather than compared. `to_common_grey` now puts every source on one scale (background → 128).

Measuring it properly reversed the conclusion: real particles are *darker* than ours, not lighter (depth 0.47 vs 0.25 on the experiment stratum). Two measurement traps on the way: centre-pixel depth is selection-biased (the detector thresholds on darkness), so it must be area-averaged; and an 85th-percentile background over-reads real's very broad histogram (p5=38, p95=179), so the median is the right reference.

**Particle depth is explicitly not a calibration target.** Raising magnification to chase it moved the number by 0.01, which was the signal to stop — the real frames were contrast-stretched to 0-255 before publication, and a stretch does not preserve intensity ratios. Apparent size and frame-to-frame change survive it; depth does not. `pixel_size_nm` for the experiment stratum now spans 2–8 nm/px (a 6 px particle at 25–95 nm of gold, all standard Au sizes) rather than being fitted. Second time in this step that a metric was nearly tuned to an artefact — the first was the static-texture term.

**Step 4e — particle motion was ~5x too slow.** Nikola noticed our clips barely move. Measured by tracking (trackpy) detected blobs through the real video: real particles step **10.6 px/frame at 6.1 px radius**. Control experiment — render literally frozen particles at the same imaging conditions and run the same detect+link pipeline — gives a **4.8 px centroid-jitter floor**, so real's true motion is ~9.5 px/frame = **1.55 radii/frame**. Coherent stage drift accounts for only 7 %, so it is genuine diffusion. Ours was 0.26–0.47 radii/frame.

This is not cosmetic:
- At 0.3 radii/frame, **persistence** (predict the last position) is already an excellent predictor, so the Gate-2 comparison against persistence would have been uninformative on position — and the martingale argument would have looked stronger than it is.
- MSD would have sat barely above the centroid-noise floor.
- At ~10 px/frame a particle crosses 16 px patch boundaries every frame, which is a materially different learning problem from one drifting 2 px/frame.

Fix: `D` is now derived from a **step-to-radius ratio** per stratum (physics 0.2–1.5, experiment 0.8–2.0) rather than an absolute `uniform(1,4)`, inverting the Rayleigh mean step `1.772*sqrt(D)`. `D` stays constant within a clip even as radii grow, so ground-truth `MSD = 4Dt` remains exact. Result: experiment stratum **1.75 radii/frame** (real 1.55).

**Evaluation requirement discovered here:** per-clip MSD is nearly worthless at experiment-grade particle counts. The estimator is unbiased (mean ratio 0.96–1.02 at every N) but its spread is set purely by particle count — **sd 0.25 at N=4 vs 0.03 at N=512**. Gate 2 must **pool MSD across clips**, never score it per clip. `tests/test_dataset_physics.py` now tests it pooled (the way the evaluation must use it) instead of widening a tolerance until one noisy clip fits, and `verify_clips.py` prints the particle count with a spread warning below N=20.

**Step 4f — the "3× too large" finding was wrong, and it had driven the design.** Nikola reported he could no longer tell a particle from noise in our clips. That was correct, and tracing it invalidated an earlier measurement.

`apparent_radius_px` originally sized particles from thresholded connected components. That is biased low — a threshold captures only the dark core of a noisy blob, and the bias *grows as SNR falls*, so it under-measures exactly where it matters most. Replaced with a **radial-profile half-depth fit**, validated against our own ground truth (recovers true 9.5 px → 8.0, true 4.9 px → 5.0). A scale-selection variant was also tried and rejected: it validates at high SNR but reads a true 4.9 px particle as 22 px at low SNR, where noise correlation dominates.

Corrected numbers:

| | old (component) | **new (radial profile)** |
|---|---|---|
| REAL | 6.1 px | **8.0 px** |
| OURS physics | 12.7 px | **7.0 px** |
| OURS experiment | 5.6 px | **4.0 px** |

So our particles were never 3× too large — the physics stratum was already the right size, and the "fix" had shrunk the experiment stratum to *half* of real, which is why it went invisible. Experiment `radius_px` raised to 10–16 → measured **7.5 px** against real 8.0.

**The strata split on density and dose, not particle size.** Size is now matched in both; what genuinely differs is particle count and electron dose. `scripts/compare_to_real.py` writes `detectability.png` (raw / matched-filtered / ground-truth overlay, ours and real side by side) as standing evidence that the signal is present even when the eye and a threshold detector cannot find it — which is precisely the argument for a learned model over classical segmentation.

Lesson worth carrying into the write-up: three separate times in Step 4 a number was nearly tuned to an artefact of its own estimator (static texture vs a structure-insensitive noise metric, particle depth vs an unknown contrast stretch, particle size vs a threshold bias). **Validate the estimator against known ground truth before trusting it on real data** — every one of these was caught that way, and the size bug was caught only because the same estimator could be pointed at clips whose answer we already knew.

**Step 4g — the exit criterion, which was the thing actually missing.** Four commits in a row (`2776936`…`0f8bd1f`) moved appearance parameters toward "looks like real" without ever defining *done*. That loop has no fixed point: each appearance statistic shifts when its estimator is refined, and the reference is 75 frames of one experiment with an unknown contrast stretch, so every refinement reopens the question.

**The dataset is not supposed to look like the real video. It is supposed to contain it.** This is domain randomisation: generate a *distribution* of imaging conditions wide enough that real data falls inside its support, so a model trained here has already seen real conditions. Fitting a point estimate of one experiment fits an artefact, and would also make the dataset *narrower* than the phenomenon — the opposite of what Tier-2 transfer needs.

So the criterion is **containment, not equality**: for each statistic that survives an unknown contrast stretch, real must lie between our minimum and our maximum. Falsifiable, and it terminates. `containment_report` in `scripts/compare_to_real.py`:

| statistic | ours min | ours max | real | |
|---|---|---|---|---|
| apparent radius (px) | 5.0 | 15.0 | 6.0–8.0 | PASS |
| \|dI\|/mean | 0.136 | 0.752 | 0.407 | PASS |

Only two statistics qualify — absolute contrast does not survive the stretch, so it is reported and never gated on. **Step 4 appearance work is closed.** It reopens only if a statistic leaves containment, not because a panel looks off.

Reviewing all nine commits against this: exactly one parameter ever circled — experiment `radius_px`, uncontrolled → (5,8) → (10,16). Every other change was monotone progress (water clamp, fixed intensity scale, exact growth time origin, ODE substeps, radius-scaled diffusion), and (10,16) is the correct value. Nothing to revert; HEAD is the best state the tree has been in.

**Decided here, needed by Step 6:** preprocessing must **crop** to 256², not resize. Real 1024² frames resized to 256² shrink those 6 px particles to 1.5 px — invisible. Our synthetic scale is chosen so both land at the same apparent size after preprocessing.

**Step 4h — logical review of the simulation, before committing 50 GB to it.** Four
things that would each have written a wrong label into every clip. None was caught by
the existing suite, which is the common thread: every one lived in a quantity nothing
graded.

1. **Diffusion scaled the wrong way with size.** `D` was the step-per-radius ratio
   applied at each clip's own radius, so `D ~ r^2` — bigger particles diffused
   *faster*. Stokes–Einstein says `D = kT/6*pi*eta*r`, i.e. slower. A 20x error across
   the 8–22 px range with the sign inverted, and a size→motion shortcut of exactly the
   kind the brightness→size leak was: a probe reading "big blob moves fast" off the
   latents would have scored as learned dynamics. The real measurement (1.55 radii/frame
   at 8 px) is a single `(r, D)` point — it pins the *scale*, not the law. `R_REF_PX`
   anchors it there and `D` falls off as `1/r`; nothing changes at 8 px, so the
   containment result is unaffected. Guarded by `test_bigger_particles_diffuse_slower`.

2. **`ripening` was indistinguishable from `static`.** The coarsening timescale is
   `~r^3/K`; at `K=6` and `r~15` px that is ~560 frames, so a 100-frame clip showed
   mean radius x1.02 and lost one particle in ten. A quarter of the dataset was
   labelled Ostwald ripening and rendered as static particles. Nothing caught it
   because `EXPONENT["ripening"]` is None, so no exponent test touches this process.
   `K=60` puts the timescale inside the clip: count halves, mean radius **x1.30**
   (static x1.00, growth modes x4.96). `K=150` coarsens harder but ends at ~3 particles,
   too few to measure a size distribution on. Guarded by `test_ripening_actually_coarsens`.

3. **The MTF was applied after the read noise — this was the "unclosed residual".**
   Charge spreading happens *at* the sensor: it blurs the electron events that arrived,
   and the readout adds its noise downstream. Blurring the read noise too stripped
   high-frequency power a real detector keeps, and did so *independently of dose* —
   which is why the temporal/spatial noise ratio sat at 2.9 at every flux against real's
   1.67, and why no parameter setting could move it. It was never the 512 px MTF. In the
   correct order the ratio spans **1.65 (flux 4) to 2.69 (flux 75)**, so real's 1.67 is
   now *inside* our span rather than outside it — the same containment criterion Step 4g
   settled on, applied to the statistic that had resisted it. Also `np.real` not
   `np.abs`: ringing beside a dark core is genuinely negative, and folding it positive
   puts a bright halo where the optics put a dark one.

   Two consequences the old blur had been hiding, both fixed here: `READ_NOISE` drops
   1.0 → **0.3 e** (a direct electron detector *counts* electrons; sigma = 1.0 e cannot
   tell zero from one, and at our lowest dose it clipped a third of every frame to
   black), and the physics-stratum `beam_flux` floor rises 2.0 → **4.0**. Zero-clipping
   across 20 clips is now max 1.22 %, mean 0.15 %. Containment still passes on both
   gated statistics, and `|dI|/mean` moved from 0.253 to **0.306** against real 0.397.

4. **`MSD = 4Dt` shipped without the lag it is valid over.** Reflection off the box
   biases MSD down at long lag; the test knew this (`short = 16`, hardcoded) but `D`
   went into the manifest with no ceiling attached. `msd_max_lag` now ships per clip,
   calibrated on measured pooled bias — it bounds *box bias* only, not the estimator
   variance that particle count sets, and both ceilings are real.

Also deleted `merge_on_contact`: 47 lines, tested, advertised in the `dynamics.py`
docstring as a feature of the dataset, and called from no generation path — coalescence
appeared in zero clips.

Suite **72 passed**. What generalises: three of these four hid in a quantity no test
graded, and the fourth was misdiagnosed for four commits because the symptom (spatial
noise too low) was measured but the mechanism was never traced. The Step-4 lesson was
"validate the estimator against known ground truth"; the companion is **check that
every label the dataset ships is graded by something**.

**Step 4i — second logical review: three more defects the suite did not grade, plus the events process** (commit `852d93f`).

1. **Size-distribution collapse.** The growth ODEs are contractions on r, so a population sharing one rate constant converges to a single radius (relative sd 0.095 → 0.0007 over a clip) — and the size *distribution* is the headline observable, so Gate 2's Wasserstein was degenerating to |Δmean| on half the dataset. Fix: per-particle rate constant pinned to r₀ at a common nucleation time (`R0_SPREAD`), which keeps rᵢ/rⱼ exactly constant and tightens the exponent label to ±0.001.
2. **Flat background was a free lunch.** Our background was flat to 0.006 against real's 0.212 illumination swing. V-JEPA predicts masked spatiotemporal blocks, and a byte-identical background makes masked-background prediction free — a shortcut real data does not offer. `thickness_field` models the bulging cell; amplitude derived from sd(ln I) = sd(t)/IMFP_water, not fitted. Consequence: `beam_flux` floor 4 → 6.
3. **Persistence was optimal on position by construction.** Brownian position is a martingale, so the generator handed the Gate-2 baseline an unbeatable strategy on the whole position axis. Coherent stage drift (measured ~7 % on real video; sampled U(0, 0.10) of the frame) is the one motion component a forecaster can get right; applied after the reflecting fold so the eval can remove it exactly.
4. **`coalesce` restored properly**: volume-conserving merge-on-contact with exact merge times in the manifest — the only interacting process, and the only source of events; Step 12 previously had no data at all.

**Step 4j — gate the noise ratio; it failed; the MTF frequency axis was the cause** (commits `debd700`, this one). The temporal/spatial noise ratio that justified the 4h MTF reorder existed only in a commit message. It is stretch-invariant (both terms are differences, so an affine stretch cancels), so it belongs in the containment gate — and once gated with one estimator on both sides it **failed**: ours 1.624–2.371, real 1.284. The reorder had fixed a real bug (a ratio flat across a 19× dose range cannot be a resolution mismatch) but the *other* hypothesis — the one dismissed in 4h — was also true: the measured 512 px matrix, applied verbatim, blurs every spatial frequency at **twice** the rate of the real 1024 px detector. Measured, not guessed: sampling the matrix at half its bin offsets gives 1.27–1.31 (contains real); no MTF gives 1.00; fixed-pattern gain noise ruled out (needs 8 % sd, real DEDs run 0.5–2 %). Fix: `calibrate_mtf` in `render_bridge` resamples the matrix onto the render's frequency grid at the empirically anchored `MTF_FREQ_SCALE = 0.5` — the one hardware calibration number in the renderer. Two lessons compound the 4h one: a statistic that justifies a physics change must live in graded code, not a commit message; and two bugs can share one symptom — fixing the first is not evidence the second was innocent.

**Containment after the calibration (25-clip smoke, all three gated statistics):**

| statistic | ours min | ours max | real | |
|---|---|---|---|---|
| apparent radius (px) | 4.0 | 11.0 | 6.0 | PASS |
| \|dI\|/mean | 0.186 | 0.672 | 0.407 | PASS |
| temporal/spatial noise | 1.267 | 1.326 | 1.284 | PASS |

Physics round-trip on the recalibrated frames: GT exponents exact (0.333/0.500), matched position error 1.1–3.8 px, brightness-shortcut correlations negative on growth modes, saturation ≤ 0.01 %. Suite **90 passed**. **Step 4 is closed by its own 4g criterion: real experimental video is inside the generated distribution on every statistic that survives the contrast stretch.**

**Step 4k — the comparison that matters: latent-space containment** (`scripts/latent_gap.py`, `experiments/latent-gap/README.md`). Pixel statistics are hand-picked proxies; the model only ever sees the frozen encoder's features, so the direct question is whether real clips land inside the synthetic *feature* distribution of V-JEPA 2 ViT-L. Measured on the Mac (MPS, ~10 min): **containment ratio 0.92** (real→synthetic NN distance over cross-clip synthetic→synthetic; held-out-synthetic reference 1.00 — real sits *on* the synthetic feature manifold) with **linear separability AUC 1.00** (group-CV; our own strata read 0.37 ≈ chance). Separable-but-adjacent: the sim-to-real gap is a small systematic offset (candidates: un-modelled SiNx texture, JPEG, the contrast stretch), not a coverage hole — the shape LoRA absorbs, and no reason to reorder the plan. Caveats stated in the write-up: one real video (aggregation frames are 200×220, below the 256 crop; resizing is ruled out), and mean-pooled features weight background heavily — Gate 1's token-level probing supersedes this at Step 6. The validation hierarchy for "is the simulation good enough to train on" now reads: labels ✓ (suite) → pixel containment ✓ (4g gate) → latent containment ✓ (this) → downstream transfer (Gate 1 / Step 13, the only definitive level, pending GPU). Also here: `compare_to_real.py`'s two histogram panels deleted (their gated summaries live in the containment table; visually they showed uint8-quantization combs and a dose *mixture* against real's single dose) and the summary row + PSD panel now pool per-clip / across experiment-stratum clips instead of silently slicing the first frames of one arbitrary clip. The properly measured OURS median radius is 7.0 px against real's 6.0 — the pooled-slice artefact had read 12.0. Suite **95 passed**.

**End of local phase.** At this point the entire evaluation + simulation core is coded and green on CPU. Decide the local side is ready → get PC access → Step 5.

---

# REMOTE PHASE (GPU — Steps 5+, once local pipeline is ready)

## Step 5 — Remote training PC setup ✅ (2026-09-29; runbook: `docs/remote.md`)

**Where:** remote (work PC, 2× RTX 3090). Goal: from the Mac, one command away from launching a training run. Do this only after Steps 2–4 are done and Nikola has confirmed PC access.

- [x] **Access:** confirm SSH reachability from the Mac (`ssh <user>@<host>`); set up key-based auth (`ssh-copy-id`) and a `~/.ssh/config` entry (`Host trainbox`). ~~Reuse the dual-account SSH trick from Step 0~~ **Changed:** the PC account (`mlmonster`) is shared, so it got a **read-only deploy key** for this repo only (alias `github-niki-jepa`) instead of the personal key — a copied key reads one repo and is revoked from GitHub alone. If you need access from outside the work network, ask IT about VPN or Tailscale — otherwise plan launches for when you're at work (jobs run unattended in tmux anyway).
- [x] **Health check** (2× 3090, driver 595.84 / CUDA 13.2, 24 cores, 60 GiB RAM; the machine is shared — other users' jobs on GPU 0 and an idle vLLM server on both cards, so all our GPU work ran on GPU 1): `nvidia-smi` (2 GPUs visible, driver supports CUDA ≥ 12.1), `df -h` (**need ≥ 200 GB free** on one volume: ~50 GB raw dynamics dataset + ~70 GB feature cache + checkpoints + upstream data), `free -h`, `nproc`.
- [x] **Environment** (repo at `/mnt/drive2/niki-masters/jepa-lptem`): install `uv` and `tmux` if missing; `git clone` the repo via the `github-personal` alias; **also `git clone https://github.com/JamaliLab/LPTEMsimulator code/LPTEMsimulator`** — `code/` is gitignored and the generator needs its `MTFmatrix512.mat` (it now refuses to run without it rather than silently rendering un-blurred frames that break the 4j noise calibration); `uv sync` (pulls CUDA torch from the same lockfile); `uv run python -c "import torch; print(torch.cuda.device_count())"` → `2`; `uv run pytest` (CPU suite green on the remote too).
- [x] **GPU smoke test:** superseded in practice — the full extraction and Gate-1 run exercised the encoder on CUDA end to end. (`tests/test_feature_shapes.py -m gpu` still worth one run for the record.) Set `HF_TOKEN` in the remote env (untracked) for fast Hub pulls.
- [x] **Transfer path:** verify `rsync -avz trainbox:~/jepa-lptem/experiments/ ./experiments/` round-trips; decide results-logging. **Decided:** no W&B yet — `python -u … | tee logs/<run>.log` in tmux, `metrics.json` rsynced back. Run rsync from inside the local repo (a relative destination lands wherever you are).
- [ ] **Download the full datasets directly on the remote** (SAM-EM trainset/holdout/experimental, LEONARDO) using the IDs verified in Step 1. *Deferred:* Gates 1–2 run on dynamics-v1 only; first needed by Step 8 (LoRA on SAM-EM trainset) and Step 13.
- [x] **Run the full dataset generation** → 2000 clips, audited clean (`scripts/audit_dataset.py`): 320 physics + 80 experiment per process, 1035 merge events, 0 unreadable, black pixels mean 0.27 % / worst 2.89 % (zero-electron pixels at the lowest dose — physical). Write-up: `experiments/dataset-v1/README.md`. Still open: `verify_clips.py` round trip on a sample of the full set. (Step 4's generator, `--n-videos 2000 --n-frames 100 --out data/dynamics-v1` — `--n-frames` is required: the default 48 is shorter than the 64-frame extraction window, so every clip would be skipped; tmux/overnight); re-run the physics validation on the full set; write `experiments/dataset-v1/README.md` (this dataset is a standalone thesis contribution).
- [x] Document all of it in `docs/remote.md` (host alias, paths, launch commands, tmux conventions). Commit: `docs: remote training environment`.

## Step 6 — Feature extraction + cache

**Where:** code + smoke local (built *before* Step 5, so PC time is pure batch execution); full extraction remote (GPU). **Files:** `scripts/extract_features.py`, `tests/test_feature_shapes.py`

- [x] GPU smoke test (`@pytest.mark.gpu`): load `facebook/vjepa2-vitl-fpc64-256`, feed 64 random frames at 256², assert `last_hidden_state.shape == (1, 32*16*16, 1024)` (64 frames / tubelet 2 = 32 latent frames × 16×16 patches × dim 1024). Written; runs on the remote via `-m gpu`. The pure alignment logic (`window_starts`, `crop_and_align_gt`) is tested locally.
- [x] Implement `scripts/extract_features.py`: center-**crop** to 256² (crop, never resize — Step 4g), window each 100-frame video into 64-frame clips (**stride 36 → exactly 2 windows/video, frames 0–63 and 36–99**, covering late-time growth for exponent fits), save `last_hidden_state` reshaped `(32, 256, 1024)` as fp16 in a per-window `.npz` **together with the ground truth aligned to latent frames** (positions shifted to crop coordinates; a particle whose centre leaves the crop gets radius 0, same convention as the visibility floor; GT sampled at the even frame of each tubelet pair, `frames_per_latent` shipped in the meta). **Budget: ~4,000 clips ≈ 68 GB** for dynamics-v1, plus SAM-EM holdout + experimental. Devices auto (cuda/mps/cpu); both GPUs via `CUDA_VISIBLE_DEVICES` split.
- [x] **Local smoke:** extraction runs end-to-end on `smoke_v3` (25 clips, `--window 48`) on MPS.
- [x] Run extraction on dynamics-v1 (done 2026-09-29, one GPU, unsharded: 4000 windows, 128,000 latent frames; SAM-EM holdout + experimental clips still to do, with Step 13) (remote; a few hours, tmux, can overlap with the dataset-generation tail). One shard per GPU into the same `--out`: `CUDA_VISIBLE_DEVICES=0 ... --shard 0/2` and `CUDA_VISIBLE_DEVICES=1 ... --shard 1/2`. Resumable — re-running skips windows already on disk (writes are atomic, so a crash never leaves a truncated window). Corollary: **delete a cache before re-extracting it with changed settings**, or the old windows are kept.
- [x] Commit script.
- [x] **Cache audit** (`scripts/audit_features.py`, before any probe): 4000 windows `(32, 256, 1024)`, 0 non-finite, 0 visible particles outside the crop, 4.5 particles in view per frame. Temporal alignment: overlapping windows agree on shared frames (cos 0.786) far more than one latent step apart (0.590). Spatial alignment: a single-token linear probe separates particle-centre cells from random empty cells at held-out AUC **0.979**, against **0.559** with labels rolled 5 cells (control slightly above chance; likely a weak positional prior, too small to matter). This is evidence that particle presence is linearly decodable at the right place — not that localisation is solved: the test is balanced and its negatives are mostly easy background.

## Step 7 — GATE 1: probe frozen features

**Where:** code + smoke local (built *before* Step 5); real probe training remote (~30 min/probe on one 3090), analysis local. **Files:** `scripts/train_probes.py`, `tests/test_train_probes.py`, `experiments/gate1/`

- [x] Implement `scripts/train_probes.py`: per-latent-frame tokens `(256, 1024)` → `AttentiveProbe` for `[count, mean_radius, total_area]` (z-scored on train stats) and `CentroidProbe` for occupancy+offsets (targets built as the exact inverse of `decode_centroids` — round-trip tested; cell collisions counted, not silently dropped; BCE pos-weighted for the ~5 % occupancy). Split 80/10/10 **by source clip**. One training sample = one window (all latent frames as a batch), loaded lazily — the ~70 GB remote cache never sits in RAM. Evaluation: R² per observable (overall + physics stratum) and the full decode → trackpy → MSD pipeline, pooled across windows with n_full weights (per-window MSD is noise — Step 4e), drift removed exactly, lags capped by `msd_max_lag`. Writes `probes.pt` + `metrics.json`.
- [x] **Local smoke:** trains end-to-end on the smoke cache; machinery verified (R² numbers from 25 clips are not the gate).
- [x] **Pre-GPU readiness review (Sep 28)** — four fixes before any PC time:
  - **Processor was resizing.** `VJEPA2VideoProcessor` defaults resize 256→292 and centre-crop back to 256, zooming 1.14× and shifting GT by up to 18 px (> one token cell). `load_processor()` switches both off; smoke cache re-extracted.
  - **Centroid decode emitted one detection per covered cell.** A 10–20 px particle spans several 16 px cells and all of them fire; on its own training windows the probe produced ~2 false positives per true one, so linking found no full tracks and the MSD criterion could not pass whatever the features held. `decode_centroids` now keeps only 3×3 local maxima, and the occupancy threshold is picked on val by detection F1 (the BCE pos-weight makes 0.5 uncalibrated). Smoke precision 0.41 → 0.76–0.83.
  - **Perfect-probe ceiling.** Only particles that stay inside the 256 crop for a whole window form full tracks, which biases MSD/4Dt low (0.93–0.98, simulated on GT) and caps coverage at ~0.3–0.5. The same MSD pipeline now also runs on GT centroids (`*_gt` keys in `metrics.json`); the probe is read against that, not against 1. Scoring lives in `eval/centroid_eval.py` so Gate 2 grades identically.
  - The crop itself stays: it keeps the simulator's reflecting walls out of view. Rendering at 256² would not raise the in-view count (~4, set by the coverage cap), only move the walls into the image.
- [x] Run on frozen features of the full set (remote, 1× 3090, ~4 h disk-bound); rsync metrics back; analyze locally.
- [x] **Decide the gate:** → **PASS, marginal on count** (test: R² mean radius 0.910, count 0.806, MSD 0.995 vs ceiling 0.955; val count 0.699). See `experiments/gate1/README.md`.
  - **PASS** (held-out R² ≥ 0.8 for mean radius and count at mid SNR; MSD from decoded tracks within 20% of GT MSD, with track-coverage reported): frozen encoder suffices → **LoRA (Step 8) shrinks to a single comparison run.**
  - **FAIL:** Step 8 becomes the main path.
- [x] Write `experiments/gate1/README.md` same day (a thesis figure either way). Commit.
- **Runtime lesson.** The plan said ~30 min/probe; the run took ~4 h at ~10 % GPU. Every epoch decompresses all 3200 train windows (~54 GB) — the job is disk/CPU-bound, not GPU-bound. The first launch also looked hung: stdout piped into `tee` is block-buffered, so the log stayed empty; long jobs now always run `python -u`. The target-statistics passes had been decompressing every window just to read positions/radii (`load_gt` fixed it, commit `3066452`). Step 9 must load windows in parallel (24 cores idle).
- [x] Secondary analyses from the saved `probes.pt` (no retraining; `scripts/gate1_secondary.py`): the threshold optimum is 0.95 even on a grid to 0.99; with gap closing the probe's MSD equals the perfect-probe ceiling (0.950 vs 0.950 over 193 windows; 0.948 vs 0.947 over 222) and coverage rises from 15 % to 52 % of the ceiling's; count error grows with particles straddling the crop edge (MAE 0.63 → 0.91, slight undercount) but is 0.63 even with none — weakest on coalescence (R² 0.75). Details in `experiments/gate1/README.md`.
- [x] **Gate 1 on the per-frame cache (2026-10-02)** — the whole-window cache leaks the future (Step 10, look-ahead check), so the gate was re-run unchanged (same script, split, criteria) on `extract_features.py --per-frame` features. → **FAIL on count as pre-registered**: test R² count 0.726 (val 0.575; whole-window 0.806 / 0.699); mean radius 0.902, total area 0.928, MSD 0.986 vs ceiling 0.955 all pass; threshold 0.95 again. Only count dropped, on both splits — the observable that needs several frames to resolve crop-edge and merging particles. `experiments/gate1-perframe/README.md`, review in `docs/findings.md`.
  - [x] Provenance (`logs/pf-audit.log`, 2026-10-03): cos **1.000** on shared frames of overlapping windows (whole-window 0.786), 0.614 one step apart; 4000 clean windows; particle-centre token AUC 0.978 vs control 0.534. The cache is per-frame.
  - [x] `gate1_secondary.py --gate1 experiments/gate1-perframe` (2026-10-03): **no** — count MAE rises ~0.1 in every edge bucket (no-edge frames too), and the largest losses are reaction (−0.13 R²) and ripening (−0.11), not coalescence (−0.08). Fits per-frame shot noise on small particles, no longer averaged over the window; the same noise puts MSD 2–3 % above the ceiling with gaps closed (0.966 vs 0.947, memory 2).

## Step 8 — Encoder: per-frame frozen now, V-JEPA 2.1 candidate, LoRA later

**Decided 2026-10-02 (after the per-frame Gate 1 FAIL, before any per-frame forecast exists) — a documented deviation from the Gate-1 rule above.** The rule says FAIL → LoRA becomes the main path. It is not followed immediately, for three reasons: (1) the rule was written to ask whether frozen features represent LPTEM physics at all, and on per-frame features size, area, positions and diffusion all pass — only count fails; (2) count's loss was first read as single-frame ambiguity (edge, merging); the secondary analysis (2026-10-03, Step 7) corrected that to per-frame noise on small particles, which strengthens the case for LoRA (it learns LPTEM's noise) and for past-only averaging, but does not make count a reason to delay the predictor; (3) every encoder path ends in "retrain the predictor, run Gate 2", and that step has never run on honest features. So:

1. **Now:** the predictor and Gate 2 on the frozen per-frame V-JEPA 2 cache (Step 10, per-frame run), plus the Gate-1 diagnostics (Step 7). The Gate-2 size numbers inherit the weak count (sd = area / (π·count)) and are reported with that caveat.
2. **Encoder upgrade candidate: V-JEPA 2.1 ViT-L, frozen, per-frame** (`experiments/encoder-selection/`). Same size (300 M, distilled from ViT-G), same 16 px patches, but built for what we use V-JEPA 2 for: a native 2D image tokenizer with a learned modality token (V-JEPA 2 only saw images as 16 duplicated frames, and our 2-frame clips are an input it never trained on), a predictive loss on every token rather than only masked ones, and deep supervision across layers — frozen dense probes roughly doubled (ADE20K mIoU 24.4 → 47.9, ViT-G). Open points to check before trusting it: weights via `torch.hub` (`vjepa2_1_vit_large_384`), not HF transformers, so `make_encoder` needs a second loader; trained at 256² then cooled down at 384², our 256² crop is in range but has to be confirmed by the smoke audit; single-frame input through the image path (one frame, not a tubelet) changes the latent rate from 2 frames to 1 unless pairs are kept — decide before extracting. Judged exactly like Gate 1, on the same split, against both per-frame V-JEPA 2 numbers. Switch the pipeline only on a clear win; then re-run predictor + Gate 2 on it.
3. **LoRA: planned, timing open.** Most likely run — its strongest case is sim-to-real (Step 13), not count. It adapts whichever frozen encoder wins step 2, and must be trained **the way it is used**: per-frame (or image-path) inputs, otherwise the adapted encoder is again evaluated off its training distribution. Natural slot: after the per-frame Gate 2 and the encoder choice, before Step 13; earlier only if both V-JEPA 2.1 and the predictor stall on count.
4. A past-only sliding window (encode frames t−15…t+1, keep the last tubelet) stays the fallback if count still matters and neither 2.1 nor LoRA lifts it: no leak, temporal context, ~6–8× the per-frame extraction compute.

**Superseded — decided 2026-09-30 (after the whole-window Gate 1 PASS):** the **frozen encoder is the permanent encoder** for the main pipeline (Steps 9–13). Step 8 is one LoRA *comparison* run, judged at the Gate-1 level (probe R², reported per process and per crop-edge bucket, where count is weak) and on real video in Step 13. The cache is re-extracted and the downstream retrained only if adaptation wins by a clear margin — otherwise a 3-day side experiment would force redoing the predictor and Gate 2. Gate 1 left little headroom: MSD is already at the perfect-probe ceiling, mean radius at R² 0.91; the open question is count and, above all, sim-to-real transfer.

**Where:** remote (the one multi-day GPU job: ~2–3 days/run). **Files:** `configs/lora_vitl.yaml`, `scripts/run_lora.py`

- [ ] First get the **unmodified** official V-JEPA 2 pretraining loop running on 10 SAM-EM videos (tiny smoke config, 50 iters, loss decreases) — validates the vendored repo before touching it.
- [ ] Add LoRA injection via `peft.LoraConfig(r=16, lora_alpha=32, target_modules=["qkv","proj"])`; objective unchanged (masked latent prediction); EMA target = frozen base + EMA of LoRA params; assert trainable params < 10 M.
- [ ] Launch (SAM-EM trainset + dynamics-v1, bf16, DDP 2 GPU, 256² crops): `torchrun --nproc_per_node=2 scripts/run_lora.py --config configs/lora_vitl.yaml` in tmux. **While it runs, build Step 9 locally.**
- [ ] Re-run Gate-1 probing on adapted features and compare with frozen (see the decision above: switch only on a clear win).
- [ ] Commit configs + `experiments/encoder-selection/README.md`.

## Step 9 — Block-causal forecasting predictor ✅

**Where:** code + CPU tests local; training remote (cached features → **hours** per run). **Files:** `src/jepa_lptem/models/predictor.py`, `scripts/train_predictor.py`, `tests/test_predictor.py`

- [x] Failing tests (`tests/test_predictor.py`, 7): output shapes; rollout extends horizon and feeds its own forecasts back; **block-causality** (perturb only the last context frame → all earlier per-step predictions unchanged).
- [x] Implemented in `src/jepa_lptem/models/predictor.py`, with three deliberate departures from the draft below: (1) **residual from persistence** — output = current frame + a zero-initialised update, so the untrained model *is* the persistence baseline (tested) and training starts at the floor a martingale imposes; (2) **narrower working width** (`dim` 512, projected in/out of 1024; ~26 M params instead of ~100 M), since Gate 1 showed data, not compute, is the limit; (3) a **learned spatial embedding** beside the temporal one. Dropout 0 (V-JEPA uses none; also keeps the causality test exact). Original draft, for reference (note: temporal embedding broadcasts as `(1, T, 1, D)` — an earlier draft had a `transpose(1,2)` here that only worked when S == T; the shape tests exist to catch exactly this):

```python
class LatentForecaster(nn.Module):
    """Block-causal transformer over latent frames (V-JEPA 2-AC style, no actions).
    Sequence = T latent frames x n_spatial tokens; tokens attend within their own
    frame-block and to earlier blocks. pred[t] = forecast of frame t+1."""
    def __init__(self, dim=1024, depth=8, heads=16, n_spatial=256, max_t=64):
        super().__init__()
        self.n_spatial = n_spatial
        self.t_emb = nn.Parameter(torch.zeros(max_t, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.out = nn.Linear(dim, dim)

    def _mask(self, T, device):
        f = torch.arange(T * self.n_spatial, device=device) // self.n_spatial
        return f[:, None] < f[None, :]              # True = blocked (strictly future frame)

    def forward_all(self, ctx):                      # (B, T, S, D) -> per-step preds
        B, T, S, D = ctx.shape
        x = (ctx + self.t_emb[:T].unsqueeze(0)).reshape(B, T * S, D)   # (1,T,1,D) broadcast
        h = self.blocks(x, mask=self._mask(T, ctx.device))
        return self.out(h).reshape(B, T, S, D)

    def forward(self, ctx):                          # next-frame prediction
        return self.forward_all(ctx)[:, -1]

    @torch.no_grad()
    def rollout(self, ctx, horizon: int):
        outs = []
        for _ in range(horizon):
            nxt = self.forward(ctx)
            outs.append(nxt)
            ctx = torch.cat([ctx[:, 1:], nxt.unsqueeze(1)], dim=1)
        return torch.stack(outs, dim=1)
```

- [x] Implement `scripts/train_predictor.py`: cached clips `(32, 256, 1024)`; context 8 latent frames → next-frame target (teacher forcing, L1 on normalized features, per V-JEPA 2-AC) + 2-step rollout loss; AdamW 1e-4, cosine, bf16, ~50 epochs. Normalization is **per-channel z-score with train statistics** (not per-token LayerNorm) so forecasts map back to raw feature space for the frozen Gate-1 probes; stats ship in the checkpoint. Split = the Gate-1 `split_by_clip` seed, so no Gate-1 val/test clip is ever trained on. Windows load in parallel DataLoader workers, several random snippets per load. A test trains a small model on constant-velocity token dynamics and requires it to beat persistence 5×.
- [x] Train on remote (2026-10-01, 6.4 h, converged by epoch ~45; `experiments/predictor-v1/`): val +1 L1 **0.672** vs persistence 0.957 and **context mean 0.741 — 9.3 % below the context mean** (+2: 7.9 %), but the same ~9 % on every process, static included: feature L1 is dominated by noise/motion shared by all processes, so it cannot show growth physics — Gate 2 does. **Log persistence L1 (`|feat[t+1] − feat[t]|`) alongside model L1 every epoch** — if the model never beats persistence on `growth_*` modes, debug before proceeding.
  **Baseline added 2026-09-30: context mean.** Each latent frame carries fresh shot noise, so simply averaging the 8 context frames beats persistence by a wide margin even on static clips (smoke val: L1 0.74 vs 0.96). Beating persistence is therefore not evidence of learned dynamics; beating the **context mean** is. Trial on the PC (200 windows, 1 epoch): model 0.876 vs persistence 0.955, a uniform +8 % across all five processes including static — the signature of noise averaging, not physics. A 30-epoch smoke model (0.840) was still *worse* than the context mean (0.738). If the full run does not beat the context mean, that is the debug item before Step 10 (candidates: residual from the context mean instead of the last frame; longer training).
- [x] Commit config + curves (`history.json`, README; `predictor.pt` stays out of git).

## Step 10 — GATE 2: the core experiment — probe predicted latents

**Where:** eval remote, analysis + figures local. **Files:** `scripts/eval_forecasts.py`, `experiments/core/`

- [x] **Protocol fixed before the run (2026-10-01), as implemented in `scripts/eval_forecasts.py`:**
  - Test clips of the Gate-1 split (never seen by probes or predictor); per window 3 fixed snippets of 8 context + 10 future latent frames (starts 0, 7, 14); verdict on the **physics stratum**.
  - Methods: model (10-step rollout on its own forecasts), persistence, **context mean**, **floor** (true encoded future); all read by the frozen Gate-1 probes (`experiments/gate1/probes.pt`, threshold 0.95).
  - **(a) Size distribution:** the probes give count, mean radius and total area — not individual radii — so the distribution is taken by its first two moments (sd from E[r²] = area / (π·count)) and compared with the **2-Wasserstein distance between Gaussians**, √(Δmean² + Δsd²), against the GT radii visible at that frame. This replaces the plan's per-particle Wasserstein, which the probes cannot support.
  - **(b) Growth exponent:** log-log slope of probed mean radius vs the clip's physical time (`t0_growth + dt_growth · frame`) over the 10 forecast frames; median per process. True exponent from the clip; the GT-in-crop slope over the same frames is reported as its realisable value.
  - **(c) Positions:** decoded-centroid F1 and matched distance at +h. MSD/4Dt with `memory=2` is reported for every method but **interpretable only for floor and ceiling**: a deterministic forecast predicts the expected position, which does not diffuse, and held frames minus the known drift move ballistically (smoke: persistence scored ~1.0 for that reason alone).
  - **Verdict** unchanged: at +5 on diffusion, reaction and ripening, size-distribution distance ≥ 30 % below persistence AND probed exponent within ±0.1 of the true one where defined. Context-mean comparison reported alongside.
- [x] Implement `scripts/eval_forecasts.py`: for each held-out dynamics-v1 video — encode context, roll out to horizons +1/+5/+10, apply the *frozen Step-7 probes* to predicted latents, compute vs GT-at-that-future-time: (a) size-distribution Wasserstein, (b) growth exponent from probed mean radius vs latent time, (c) MSD from decoded tracks. **Always compute identical metrics for persistence** (predicted latent := last context latent) **and for the context mean** (predicted latent := mean of the context latents) — the second is the bar for "learned dynamics", since averaging alone removes per-frame noise.
- [x] **Linking protocol, fixed now — before any Gate-2 number exists:** decoded tracks are linked with gap closing `memory=2` (skipped frames NaN, MSD over observed pairs), identically for model, persistence, floor and ceiling; the `memory=0` numbers are reported alongside. Chosen from the Gate-1 secondary analysis, where it matched the ceiling and gave the most coverage.
- [x] Also record probe error on **encoded true-future latents** as the floor — this separates "wrong physics" from "predicted latents are off-manifold for the probes" (a known confound; report both numbers).
- [ ] Produce the headline table (horizon × mode; model vs persistence vs floor per observable) — **table done** (`experiments/core/README.md`); money plots still open: predicted vs GT size distribution at +10; probed growth curve overlaid on r ∝ t^⅓.
- [x] **Judge Gate 2.** PASS = on `growth_*` and `ripening` at +5: size-dist Wasserstein ≥ 30% below persistence AND probed growth exponent within ±0.1 of GT. (On `diffusion_only`, parity with persistence on *position* is expected and correct — Brownian motion is a martingale; report MSD spread instead.) FAIL → systematic debugging: verify teacher-forced +1 first, then how error grows with horizon; a genuine collapse-to-persistence is the honest negative result and still a thesis.
  **Result (2026-10-01): FAIL as pre-registered.** Size distance at +5 vs persistence: diffusion +1.7 %, reaction −0.3 %, ripening −28.4 %; exponent diffusion 0.296 (pass), reaction 0.371 vs 0.500 (fail). **The size criterion had no power**: the true encoded future (floor) beats persistence by only 6 % / 3 % and loses on ripening — radii change less over 5 latent steps than the probes' reading error. That is a protocol error, reported as such. The exponent test does have power (floor 0.319 / 0.501): the model's exponents are ordered by process but shrunk toward a common value (static 0.10). Positions: model F1 ≈ 1.5–2× persistence at every horizon (drift carried forward). Rollout smears sizes by +10.
- [x] **Follow-up — POST HOC, written after the Gate-2 FAIL and fixed here (2026-10-01) before any follow-up number exists.** `scripts/gate2_followup.py` → `experiments/core/followup.json`. Same test windows, probes, threshold and methods; one snippet per window: context = latent frames 0–7, forecast = 8–31 (+24 steps, 48 video frames). Physics stratum. All CIs: 95 % percentile bootstrap over **test clips** (2000 resamples, seed 0).
  - **(A) Size at +24.** Per snippet, paired gains over persistence: Δ_floor = W2_persist − W2_floor, Δ_model = W2_persist − W2_model. A process is **powered** iff the CI of mean Δ_floor lies above 0 (the true future measurably beats holding the frame). On powered processes among diffusion / reaction / ripening: **PASS iff the CI of mean Δ_model lies above 0**; unpowered → "untestable", not a fail. Gap closed (mean Δ_model / mean Δ_floor) reported, not judged.
  - **(B) Exponent over the 24 forecast frames.** Same rule as Gate 2: median model exponent within ±0.1 of the true one (diffusion, reaction); bootstrap CI of the median reported.
  - **(C) Blur vs growth — diagnostic, not judged.** Per horizon, model vs floor: mean probed count, mean probed radius, mean peak occupancy. Blur predicts falling count and peak occupancy with rising radius on *static* forecasts. Also reported: each process's model exponent minus the static one (offset-corrected).
  - Caveat to state with the results: the predictor was trained on +1/+2 only; +24 is far outside its training rollout.
  **Result (2026-10-01):** does not rescue Gate 2. Size at +24: diffusion powered and **FAIL** (model significantly *worse* than persistence), reaction and ripening untestable (floor CI not above 0). Exponent over 24 frames: reaction 0.414 pass, diffusion 0.218 FAIL — the reverse of the 10-frame run. Blur confirmed: model forecasts **fade** (count −43–74 % vs floor by +24, peak occupancy falling) while the floor stays sharp (F1 ≥ 0.85). Process ordering of exponents is right (reaction > diffusion > static ≈ 0). **Bottleneck: the predictor's long rollout, not encoder or probes.** Next: predictor-v2 with `--rollout 8`, rerun Gate 2 + follow-up unchanged, report as a new attempt next to v1.
- [x] **Encoder look-ahead check (found 2026-10-01, before Step 11):** the cache encodes whole 64-frame windows and the encoder attends across all of them, so the predictor's 8 context latents were computed with the 48 future frames in view. `scripts/leak_check.py` re-encodes each physics test window's 16 context frames alone and reports (1) token cosine causal vs cached per context position, (2) probe readout on the context frames, cached vs causal, (3) the +24 follow-up with the causal context, vs `experiments/core/followup.json`. If (3) holds, look-ahead did not drive Gate 2; if it drops, the predictor must be retrained on causal contexts and Step 11's comparison must use them. → `experiments/leak-check/`.
  **Result: the look-ahead matters.** Causal vs cached context cosine 0.84 → 0.74 (worst next to the forecast). With causal contexts v1's position advantage vanishes (F1 +1: 0.54 vs persistence 0.56; was 0.80) and its +1 size error is worse than persistence (off-distribution input); long rollouts settle into one generic state with spurious growth in every process (static 0.105). v1's (and v2's) Gate-2 numbers therefore do not measure learned dynamics; the recorded FAIL stands but is uninterpretable. → re-extract with `--per-frame` (each tubelet encoded alone, as V-JEPA 2-AC), then Gate 1 → predictor → Gate 2 + follow-up unchanged on that cache.
- [x] **Per-frame run (fixed 2026-10-02, before training):** `train_predictor.py` with v1's defaults (from scratch — v1's weights and z-score statistics belong to the leaking cache) on `data/features/dynamics-v1-perframe` → `experiments/predictor-perframe/`; then `eval_forecasts.py` and `gate2_followup.py` unchanged into `experiments/core-perframe/`, probes and threshold from `experiments/gate1-perframe/` (`load_gate1`). Same verdicts as Gate 2 and its follow-up. Reported as the first Gate-2 result on honest features, next to v1's uninterpretable FAIL; the size numbers carry the per-frame count caveat (Step 8).
  **Result (2026-10-03): FAIL, and worse than persistence.** Size distance at +5 is 76–82 % *worse* than persistence on the verdict processes; exponents are negative (diffusion −0.83, reaction −1.25; floor 0.348 / 0.498). Follow-up: diffusion and reaction both powered and both FAIL (model gain −4.9 / −4.5 px, CIs well below 0). The forecasts fade: peak occupancy 0.98 → 0.29 by +24, and every process converges to one generic state (radius ~5 px), while the floor stays sharp (F1 ≥ 0.83). At +1 the model is barely above persistence on F1 (+0.02–0.06). Reading: a deterministic L1 forecaster predicts the median of futures that per-frame latents make genuinely uncertain (Brownian motion, shot noise), so particles dissolve; v1's skill was the look-ahead. The encoder and probes are not the bottleneck. `experiments/core-perframe/README.md`. Training converged (`experiments/predictor-perframe/`): 4.2 % below the context mean in feature-space L1 at +1 (v1: 9.3 %), uniform across processes — better in L1, worse in probe space, so the objective is the cause, not the training.
- [x] **Forecast-trained readout — POST HOC diagnostic, fixed 2026-10-03 before it runs.** Question: under the blur, do the per-frame forecasts still carry size growth beyond what the context alone gives? `scripts/forecast_probe.py` → `experiments/forecast-probe/`. Snippets as in the follow-up: context latents 0–7, forecast 8–31 (+24), `predictor-perframe` frozen. Four sources of the latent at +h — model rollout, persistence (frame 7 held), context mean, floor (true future) — and per source a fresh `AttentiveProbe` with a learned horizon embedding, trained to read the GT `[count, mean radius, total area]` at +h from that source's latent at +h. Gate-1 train split, all strata, z-scored MSE, AdamW 1e-3, 15 epochs, each source's epoch picked on its val loss. The persistence and context-mean probes know h, so they are *learned extrapolators from the context*: the bar for "the forecast adds dynamics". Test, physics stratum, per process: mean-radius absolute error at +1/+5/+10/+24, and the growth exponent over the 24 forecast frames (median, clip-bootstrap CI).
  - **Reading (fixed now):** on diffusion and reaction, the reference is whichever of persistence / context mean has the lower mean-radius error at +24 on that process (this favours the baselines). A process is powered iff the floor's paired gain over the reference has a clip-bootstrap CI above 0; on a powered process **"growth is in the forecasts" iff the model's paired gain has a CI above 0**. Exponent within ±0.1 of the true one is reported, not judged.
  - Yes → the forecasts carry growth and the output form (blur) is what fails: next a predictor with sharp/sampled output. No → the predictor adds no size dynamics beyond the context: redesign the objective to target the predictable quantities. Diagnostic only; Gate 2's FAIL stands either way.
  - **Result 2026-10-04: No** (`experiments/forecast-probe/`). Both processes powered (floor gain over persistence +0.91 / +1.45 px, CIs > 0); model gain −0.23 [−0.50, +0.03] on diffusion, +0.04 [−0.21, +0.30] on reaction. The forecast holds the context's sizes (within 0.25 px of persistence, against 4.5–4.9 px worse with the frozen probes) but no growth beyond them. → Redesign the objective; to be fixed here before it runs.
- [ ] **Position-free forecasting (pooled latent state) — fixed 2026-10-11, before any pooled number exists.** The redesign called for above.
  - **Why** (ground truth only, commit `1ab6f65`). Half the particles leave their 16 px token within one latent step, so any deterministic token-level forecast erases them from +1 on, whatever the loss or rollout (`experiments/token-persistence/`). That is why every token-level attempt failed the same way. Growth is predictable from the context: a 20-term ridge on 8 probe-quality readings cuts the +24 mean-radius error from 3.26 / 3.90 px (persistence) to 1.31 / 1.86 px on diffusion / reaction (`experiments/growth-identifiability/`). So the forecast target becomes a position-free summary of the latent: tokens averaged, no labels (JEPA's own answer to an unpredictable future: a representation invariant to it). Positions and MSD are out of scope for this gate (a martingale, as the brief says). A stochastic token forecaster (MoP-JEPA / VJEPA style) is the later extension that could add them. The parked `predictor-obs` draft (branch `worktree-objective-redesign`) is the fallback if P1 fails. It trains through a label-trained probe, so it would be reported as probe-distilled.
  - **P0 (PC, CPU, ~5–15 min, disk-bound):** `scripts/pool_features.py --features data/features/dynamics-v1-perframe --out data/features/dynamics-v1-perframe-pooled.npz`. Per latent frame: `mean` and `std` over the 256 tokens and `grid2` (mean per 8×8-token quadrant), fp16, plus GT `[count, mean radius, total area]` and the window meta (~1.6 GB). rsync to the Mac; P1–P3 run locally.
  - **Common to P1–P3:** the Gate-1 per-frame split (`experiments/gate1-perframe/metrics.json`, `split`): train to fit, val to select, test to judge, physics stratum. Pooled features are z-scored per channel with train statistics. Snippets as in the follow-up: context = latent frames 0–7, forecast = 8–31 (+1..+24), one per window. CIs are 95 % clip-bootstrap (`bootstrap_ci`, 2000 resamples, seed 0).
  - **P1 — pooled readout (Gate 1 for the pooled state).** For each candidate state in this order — `mean`; `mean`+`std` concatenated; `grid2` flattened — a ridge readout from one frame's pooled vector to the z-scored `[count, mean radius, total area]`. λ is picked on val (grid 10^-2…10^4) by mean-radius R². **Usable iff** test R² of mean radius ≥ 0.8 **and** the readout applied to the true future latents (frames 8–31) gives a median growth exponent within ±0.1 of the true one on diffusion and reaction. The first usable state in that order becomes *the pooled state*; its readout is frozen for P2–P3. Count R² is reported, not judged. **None usable → stop:** no Gate 2′. The thesis reports the token-level FAIL with its mechanism, and `predictor-obs` becomes the next decision.
    **Result 2026-10-11: usable — `mean` is the pooled state** (`experiments/pooled-gate1/`). P0: 4000 windows pooled in 350 s on the PC CPU; overlapping windows' shared frames agree at cosine 1.000. P1 test physics: R² mean radius 0.877 (token probe 0.902), area 0.942 (0.928), count 0.723 (0.726). Exponents read from the true future 0.311 / 0.473 (true 0.333 / 0.500), static 0.009. mean+std and grid2 also pass and are not used. Readout frozen in `experiments/pooled-gate1/readout.npz` for P2–P3.
  - **P2 — baselines and the linear candidate.** Each is read by the frozen P1 readout:
    - *persistence* (frame 7 held) and *context mean* (frames 0–7 averaged, held);
    - *pooled trend*: frame 7's readout × the median train ratio of the true mean radius at +h over frame 7, all processes pooled (trend-blind but h-aware, the forecast-probe persistence analogue);
    - *readout ridge*: `growth_identifiability.py`'s ridge on the 8 context readouts of mean radius. It reads a label-trained readout, so it is a reference for how much the context allows, never a candidate;
    - *floor*: the true future pooled latent;
    - **candidate A, pooled ridge** (no labels): the pooled state is projected onto its top 64 PCA components (fit on train frames). The 8 context frames' coefficients, concatenated, are mapped by one ridge per horizon to the coefficients at +h and projected back. λ picked on val by pooled-latent L1 at +24.
    **Result 2026-10-11 (candidate A): FAIL as pre-registered, but the size criterion passes** (`experiments/pooled-forecast/`). Reference on both processes: pooled trend (MAE +24 2.01 / 2.87 px); both powered (floor gain +1.34 / +2.13). Linear gain +0.42 [+0.15, +0.68] (diffusion) and +0.66 [+0.35, +0.98] (reaction), closing 57 % / 68 % of the readout ridge's gain (+0.73 / +0.96). Static exponent 0.008 (no invented growth; pooled trend 0.157). Exponents 0.208 / 0.269 against 1/3 and 1/2: outside ±0.1, so FAIL. Post hoc, not judged: the readout ridge itself reaches only 0.272 / 0.337, so the exponent criterion may be out of reach from 8 context frames. λ = 10⁴, at the grid edge. Projecting onto 64 PCA components costs the readout 0.15–0.2 px on the true future. First label-free forecast that carries clip-specific growth.
  - [ ] **Exponent ceiling from the context — ground-truth diagnostic, fixed 2026-10-11 before it runs** (after P2, before P3). Question: can *any* forecaster that sees only the context meet Gate 2′'s exponent bar (±0.1 over +1..+24)? Candidate A reached 0.208 / 0.269, and even the label-reading readout ridge reached only 0.272 / 0.337. `scripts/exponent_ceiling.py` → `experiments/exponent-ceiling/`.
    - Ground truth only, like `growth-identifiability`: the same simulated physics windows (`token_persistence.gt_windows`, 200 clips per process, seed 1), mean radius of the crop-visible particles, train/test split by clip. Readings are truth + N(0, sd), with sd 0 and 0.8 px.
    - Context-only extrapolators, fitted on train with all processes pooled and no labels: **ridge** (as in `growth_identifiability`, λ = 1) and **kNN** (k = 25, on the context relative to its last reading), both predicting log(r(+h) / last reading) for h = 1..24. References, not forecasters: **process given** (the train-median ratio per process) and **process + window start given**; and the **truth in the crop** (what the exponent fit gives on the true future).
    - Each forecast's exponent is fitted over the forecast frames against the clip's physical time, exactly as Gate 2′ does. Reported: median per process.
    - Main reading at context 8: **the bar is reachable** iff at sd 0.8 at least one context-only extrapolator has median exponents within ±0.1 of the truth on both diffusion and reaction. If it is unreachable, the exponent criterion is recorded as unpowered at context 8, as the original Gate 2 size criterion was. Gate 2′'s recorded FAIL for candidate A stands, and P3 is still judged as pre-registered. The finding only shapes a future, separately pre-registered gate.
    - Also reported, not judged: the same at context 16 with forecast frames 16–31 (+1..+16), the only longer context the 32-frame windows allow. It informs whether a longer-context run is worth pre-registering.
    **Result 2026-10-11: not reachable at context 8** (`experiments/exponent-ceiling/`). At sd 0.8: ridge 0.263 / 0.313, kNN 0.194 / 0.222 (true 1/3, 1/2); noise-free kNN 0.339 / 0.494, so the noise, not the information, limits it. Context 16 at sd 0.8: 0.300 / 0.357, so diffusion is within ±0.1 and reaction is still not. Gate 2′'s exponent criterion is recorded as unpowered at context 8. Candidate A's FAIL stands; P3 is judged as pre-registered. Candidate A's 0.208 / 0.269 lies in the context-only range.
  - **P3 — candidate B, small causal transformer** (no labels). Each latent frame of the pooled state is one input token; dim 256, depth 4, heads 4, causal; a learned horizon embedding; predicts the pooled state at +h directly for h = 1..24. L1 on the z-scored pooled state, AdamW 1e-3, cosine, 100 epochs, batch 64, seed 0. The checkpoint is selected on val L1 at +24. Trained on the Mac.
    **Result 2026-10-11 (candidate B): FAIL as pre-registered; the size criterion passes** (`experiments/pooled-forecast/`, P3 section). Best val epoch 38; 127 s on the Mac CPU. Gain over the pooled trend at +24: +0.54 [+0.30, +0.77] / +0.81 [+0.59, +1.03], closing 73 % / 84 % of the readout ridge's gain. Exponents 0.180 / 0.245 (criterion unpowered at context 8); static 0.110, which invents growth and fails the static criterion. Transformer minus linear at +24: +0.12 [−0.18, +0.41] / +0.16 [−0.12, +0.43], no significant difference. Both label-free candidates carry clip-specific growth; both are limited by what 8 noisy context frames allow.
  - [x] **Mixture of predictors on growth (MoP-JEPA style) — fixed 2026-10-11 before any number exists.** Why: the exponent ceiling shows the damping is shrinkage, and that process + window start gives the law exactly. So the future given the context is a small set of growth curves, the case MoP-JEPA (arXiv 2607.05238) is built for: K heads, the winner-take-all head trained on each window, a router that weights them, and the most likely head as the forecast instead of the average. Growth only; positions stay out of scope (one continuous Gaussian per particle, no small set of modes; `data/thesis_notes.md`, untracked). Branch `worktree-mop-growth`.
    - **M0 — ground-truth premise check** (`scripts/mop_ceiling.py` → `experiments/mop-ceiling/`, Mac, seconds). The data, split, readings (sd 0 and 0.8 px), context 8 and exponent fit are those of `exponent_ceiling.py`. Two mode forecasters, each predicting log(r(+h) / last reading) for h = 1..24:
      - **labelled modes (reference):** heads = the train-median ratio per process + window start (10 branches); router = majority branch among the k = 25 nearest train contexts (kNN features as in `exponent_ceiling.py`).
      - **label-free modes (the MoP analogue, judged):** heads = K-means centroids (K = 6, as MoP-JEPA's AV2 setting; scikit-learn, n_init 10, seed 0) of the train targets; router = the share of the 25 nearest train contexts in each cluster. Forecasts: **top head** (the router's most likely head) and **mixture mean** (router-weighted average of the heads, which should shrink again).
      - Reported: median exponent per process on test, mean-radius MAE at +24, router accuracy (labelled), head usage and head × process counts (label-free).
      - **Decision:** build M1 iff the label-free top head at sd 0.8 has median exponents within ±0.1 of 1/3 and 1/2 on diffusion and reaction and within ±0.1 of 0 on static. Otherwise stop: a mode forecast cannot recover the law from this context either, and that goes in the thesis discussion.
      - Caveat to state with the results: a per-process median of a mode forecast is right whenever most windows pick a head with the right law, so it rewards picking the right mode, not calibrated weights. The +24 MAE shows what committing to one head costs against the mean forecast.
      - **Prediction (written before running):** labelled top head reaches the bar on diffusion and reaction (the median tolerates a minority of wrong picks); label-free top head passes on diffusion and is borderline on reaction (~0.40), static within 0.05; top-head MAE at +24 10–20 % above the mixture mean; mixture mean exponents near the ridge's 0.26 / 0.31.
    - **M1 — candidate C, MoP on the pooled state** (only if M0 passes; `scripts/pooled_mop.py`, Mac). Candidate B's trunk unchanged (pooled tokens, dim 256, depth 4, heads 4, causal, horizon embedding, change from frame 7). K = 6 output heads (Linear, init N(0, 10⁻³) so they start near persistence but distinct) and a router (Linear dim → K) on the last context token. Loss per window: relaxed winner-take-all L1 over all 24 horizons (winner weight 0.95, the others 0.05 / 5, against dead heads, as in multiple-hypothesis prediction) plus the router's cross-entropy to the detached winner. AdamW 1e-3, cosine, 100 epochs, batch 64, seed 0. Checkpoint selected on val L1 at +24 of the top head. No labels.
      - Judged: the **top head** as candidate C, by the same `pooled_forecast.evaluate` and Gate 2′ as A and B. If M0 passes, the exponent criterion counts as powered for a mode forecast. Reported, not judged: the mixture mean, head usage, head × process counts on test.
      - **Prediction:** size criterion passes with a gain at or below B's; exponents above B's (0.180 / 0.245), around 0.25–0.35, so reaction most likely still fails; static within 0.1.
    **Result 2026-10-11 (M0): do not build M1** (`experiments/mop-ceiling/`). At sd 0.8 the label-free top head reaches 0.191 / 0.120, static 0.043: far outside the bar, and below the ridge (0.263 / 0.313). The prediction was wrong: the router, not the averaging, is the limit — it picks the right process 33 % of the time (chance 20 %), and the heads stop splitting by process. Noise-free, mode forecasts do undo the shrinkage (label-free 0.328 / 0.424, labelled exact), so MoP fixes shrinkage only when the context identifies the mode. Post hoc: heads clustered on the true growth ratio reach 0.194 / 0.240, still below the ridge. Mixture mean 0.191 / 0.208. The labelled reference's 0.333 / 0.371 comes from medians over mostly wrong picks (router 0.33). Candidate C is not built; Gate 2′ stays as recorded for A and B.
  - **Gate 2′ verdict, per candidate (A and B).** On diffusion and reaction:
    - The **reference** is whichever of persistence / context mean / pooled trend has the lowest test mean-radius MAE at +24 on that process (this favours the baselines).
    - A process is **powered** iff the floor's paired gain over the reference has a CI above 0.
    - **PASS** iff (1) on every powered growing process the candidate's paired gain over the reference has a CI above 0, (2) its median exponent over +1..+24 is within ±0.1 of the true one on diffusion and on reaction, and (3) its median exponent on static is within ±0.1 of 0, so it does not invent growth.
    - Both growing processes unpowered → untestable, not a FAIL.
    - Reported, not judged: the gap to readout ridge (candidate gain ÷ readout-ridge gain over the reference), all horizons, all processes, the Gaussian-W2 size distance (count is weak), and the exponents of every method.
  - **Reporting.** `experiments/pooled-gate1/` (P1) and `experiments/pooled-forecast/` (P2–P3, Gate 2′). Reported next to the token-level Gate 2 FAILs, never in place of them. If candidate A passes, it is the result, and B is kept only if it also beats A's paired gain with a CI above 0.
- [x] **Longer context — fixed 2026-10-11 before any candidate number exists** (after P3). `scripts/longer_context.py` → `experiments/longer-context/`. Branch `worktree-longer-context`.
  - **Question.** At context 8 both candidates' exponents (A 0.208 / 0.269, B 0.180 / 0.245) sit inside the context-only ceiling, so the shortfall is either the data (8 noisy frames) or the model. The ceiling rises with context. Do the candidates rise with it? If they do, the latent forecaster uses what the context offers and the exponent shortfall is the data's. If they do not, the pooled latent or the label-free fit is the bottleneck.
  - **Data, no new extraction.** Per-frame features encode each tubelet alone, so a clip's two cached windows (video frames 0–63 and 36–99) stitch into one 50-latent-frame sequence. Checked on the pooled cache: the 14 shared frames agree at cosine 1.000 and their targets are identical. Snippets are cut so that every context of a design forecasts the *same* frames (+1..+24), which makes the contexts a paired comparison:
    - **design A (primary):** contexts 8, 12, 16; forecast begins at latent frame 16, 21 or 26 (3 snippets per clip, 6000 in all);
    - **design B (reported):** contexts 8, 16, 24; forecast begins at latent frame 24 or 26. Only 2 nearby forecast positions fit a 24-frame context, so the forecaster nearly knows where on the growth curve it is. Its absolute exponents are inflated for that reason, at every context alike.
    Split, strata and readout unchanged: the Gate-1 per-frame split by clip, physics stratum for the verdict, the frozen P1 readout.
  - **Methods, unchanged apart from the context length.** Candidate A (linear: PCA 64 → ridge, λ on val) and candidate B (causal transformer, same hyperparameters, checkpoint on val L1 at +24), both label-free, each trained afresh per snippet set. Judged by `pooled_forecast.evaluate`, so every Gate 2′ criterion and reference is computed as before. The pooled scripts read the context length off the snippet (commit `f80a7b6`; rerun on the 32-frame windows reproduces P2 and P3 exactly).
  - **Primary reading (design A).** Per candidate, per test-physics snippet: Δ = exponent over +1..+24 at context 16 minus at context 8 (same snippet, same forecast frames). A candidate **tracks the context** iff the median Δ has a clip-bootstrap CI above 0 on both diffusion and reaction.
    - Tracks → the context-8 shortfall is (at least partly) data-limited. The thesis states the exponent result against the ceiling curve.
    - Does not track (while the ceiling rises, which it does, below) → model-limited. The next question is the pooled readout and latent, not more context.
  - **Reported, not judged:** the candidate's rise as a fraction of the ceiling's rise (best of ridge / kNN at sd 0.8); the gap to the ceiling at each context; static's exponent at each context (inventing growth); design B's 8 → 24 paired reading; every Gate 2′ criterion at every context, reported as "criteria met at context c". **The Gate 2′ FAIL at context 8 stands**; this is a separate question, not a re-run of the gate.
  - **Ceiling, computed before any candidate ran** (simulator ground truth, 200 clips per process, seed 1, same snippet design, `experiments/longer-context/ceiling.json`). Median exponent over +1..+24 at sd 0.8, ridge / kNN (true 0.333 / 0.500 / 0):

    | design A | diffusion | reaction | static |
    |---|---|---|---|
    | context 8 | 0.228 / 0.202 | 0.279 / 0.263 | 0.121 / 0.096 |
    | context 12 | 0.262 / 0.268 | 0.331 / 0.335 | 0.082 / 0.047 |
    | context 16 | 0.291 / 0.315 | 0.359 / 0.385 | 0.064 / 0.025 |

    Design B, context 8 / 16 / 24 (best method): diffusion 0.211 / 0.315 / 0.353, reaction 0.266 / 0.381 / 0.425.
  - **Predictions (written before the run).** On design A both candidates **track**: they rise by about +0.05 to +0.09 on each growing process, 50–100 % of the ceiling's rise (+0.09 / +0.11), with their median Δ CIs above 0. Candidate A ends near 0.27 / 0.33 at context 16 and B a little lower, as at context 8. Static stays below 0.1 for A, and B's spurious static growth (0.110 at context 8) shrinks. Gate 2′ criteria at context 16: size passes; the diffusion exponent comes within ±0.1 but reaction does not, as for the ceiling. Design B at context 24: reaction can reach ±0.1 only if a candidate matches the ceiling. I expect it to fall short by a few hundredths.
  **Result 2026-10-11: the linear candidate tracks the context, the transformer does not** (`experiments/longer-context/`, 22 min on the Mac CPU). Design A, paired rise from context 8 to 16: linear +0.097 [+0.079, +0.107] / +0.104 [+0.083, +0.125], about the ceiling's own rise (+0.087 / +0.106), reaching 0.287 / 0.340 at context 16. The gap to the ceiling does not widen (0.00–0.07 at every context). Transformer +0.014 / +0.027: CIs above 0, so it tracks by the rule, but that is 16–25 % of the ceiling's rise, and on design B (8 → 24) it does not track (+0.003 [−0.002, +0.011]). It still invents static growth (~0.11). The transformer's +24 size gain equals the linear's, but its +1 error is worse, so its flat exponents look like forecast-curve shape (checkpoint selected at +24 only). That is a hypothesis, not tested. Linear +24 size gain over the pooled trend grows with context (A: +0.22 → +0.55 px diffusion, +0.29 → +0.88 px reaction). Gate 2′ criteria are not met at any context: linear misses only the reaction exponent (best 0.363 at B 24). Predictions: right for the linear candidate and the context-16 criteria, wrong for the transformer. **Reading: for the linear candidate the context-8 exponent shortfall is the data's; the transformer is model-limited.** Gate 2′'s FAIL at context 8 stands.
- [x] **RQ3 — forecast quality vs dose — fixed 2026-10-24 before any binned number exists.** `scripts/snr_split.py` → `experiments/snr/`, Mac, no new training beyond the linear map.
  - **Question.** RQ3 asks how forecast quality degrades with horizon and with SNR. The horizon half is already answered: every pooled table reports +1, +5, +10 and +24. This answers the SNR half. The exponent ceiling found the context-8 shortfall is set by reading noise, not by missing information (noise-free kNN 0.339 / 0.494, at sd 0.8 0.194 / 0.222). If that holds for the model, high-dose clips should forecast better.
  - **Variable.** Beam flux (`beam_flux` in the clip meta, electrons per pixel per frame), the dose an experimentalist controls. The physics stratum draws it log-uniform on 6–75. Two bins split at the geometric middle of that range, √(6·75) = 21.2, fixed by the design rather than the data. Mean water thickness is the same in every clip (150 nm), so dose is the only SNR knob in dynamics-v1. Thickness *variation* (`thickness_sd_nm`, uniform 15–55) adds background structure, not shot noise: reported the same way, split at 35 nm, not judged.
  - **Setting.** Longer-context design A (the stitched 50-frame clips; forecast frames start at latent frame 16, 21, 26), contexts 8 and 16, the frozen P1 readout, the Gate-1 per-frame split, test physics snippets. **Candidate A (linear) only**: the longer-context result made it the pooled result, and the transformer is model-limited, so splitting it by dose would measure its curve shape, not dose. Trained on all doses as before (deterministic, so it reproduces `experiments/longer-context/`), and split only for scoring.
  - **Measures, per bin and per process (diffusion, reaction, static):**
    1. **Reading noise:** the readout on the true frames of the test physics clips (all 50 frames, frames with no particle dropped): MAE, bias, and sd-equivalent MAE / 0.80, the convention that gave sd 0.8 px (`growth_identifiability.py`).
    2. **Ceiling per bin:** the longer-context ground-truth ceiling (ridge and kNN, simulator clips, seed 1, 200 per process, design A) with the reading noise set to each bin's measured sd-equivalent (all processes pooled). Dose enters the ceiling only through that noise. That is an assumption (iid Gaussian reading error), stated with the result.
    3. **Forecasts:** mean-radius MAE at +1, +5, +10, +24 of the linear candidate, the pooled trend and the floor; the linear candidate's paired +24 gain over the pooled trend (clip CI); median exponent over +1..+24 of the linear candidate and the floor (clip CI); clips and snippets per bin; the bins' mean true radius as a balance check.
  - **Primary reading (context 16, where the linear candidate's growth signal is largest).** Per growing process: Δ = the linear candidate's +24 mean-radius absolute error, mean over low-dose snippets minus mean over high-dose snippets, with a 95 % two-sample clip-bootstrap CI (clips resampled within each bin, 2000 draws, seed 0). **Forecasts degrade at low dose** on a process iff the CI lies above 0. The same Δ for the floor says how much is the readout's: if the floor's Δ is as large as the linear candidate's, the dose effect is in reading the latent, not in forecasting it. Context 8 is reported the same way, not judged.
  - **Predictions (written before the run).**
    - Reading noise: higher at low dose, sd-equivalent ~0.9 vs ~0.65 px. The bins differ in shot-noise SNR by about √(40/12) ≈ 1.8×, but the readout averages 256 tokens, so the gap is smaller than that.
    - Ceiling: the high-dose bin's exponents a few hundredths closer to the truth; the two bins within 0.05 of each other at both contexts.
    - Primary: Δ > 0 on both processes, about +0.2 to +0.5 px, CI above 0 on at most one. With ~15 test clips per bin and process, a two-sample split is close to unpowered. The floor's Δ is about half the linear candidate's.
    - Exponents: high dose closer to the truth by 0.02–0.05 with overlapping CIs. Static stays within ±0.05 in both bins.
    - Thickness variation: no effect on either measure.
  - **Reported, not judged:** everything above at context 8, the thickness split, all horizons.
  **Result 2026-10-24: not shown at +24; dose costs 0.3–0.6 px at every horizon, through the readout** (`experiments/snr/`, 40 s on the Mac). Reading noise sd-equivalent 1.21 (low dose) vs 0.88 px (high), bias −0.23 vs +0.01. Ceiling per bin within 0.034 at both contexts, against 0.07–0.10 from context 8 → 16. Primary Δ at context 16: +0.32 [−0.20, +1.03] (diffusion), +0.28 [−0.61, +1.27] (reaction), so **forecasts degrade at low dose: not shown** on either. Post hoc at +1: +0.41 / +0.64 / +0.49 px with CIs above 0 on all three processes; the penalty does not grow with horizon, so at +24 the extrapolation error swamps it. Exponents at high dose closer by 0.021 / 0.054, CIs overlapping; on reaction the floor itself reads 0.423 at low dose against 0.502. Thickness variation: no consistent effect. Mean water thickness is fixed in dynamics-v1, so RQ3's thickness half cannot be answered with this dataset. Predictions: right on direction, the ceiling, the primary and the exponents; reading noise larger than predicted at both doses; the floor's share wrong on reaction.
- [ ] ~~**predictor-v2**~~ **Dropped 2026-10-02:** it continues v1 on the leaking cache, so it inherits the look-ahead. Kept below as written. **(fixed before training, 2026-10-01):** continue v1 (`--init experiments/predictor-v1/predictor.pt`, which also fixes its standardization and architecture) with `--rollout 8 --epochs 15 --lr 5e-5`; checkpoint selected on val L1 at **+8** (v1 was selected at +1). Then `eval_forecasts.py` and `gate2_followup.py` unchanged into `experiments/core-v2/`; reported as a second attempt next to v1's FAIL, never in place of it.
- [x] Write `experiments/core/README.md`. Commit.
- [ ] Draft the thesis Results section from it.

## Step 11 — Pixel-space baseline (ConvLSTM)

**Where:** remote (~1 day on one GPU; can run concurrently with Step 10 coding).

- [ ] Train an off-the-shelf ConvLSTM/PredRNN next-frame predictor on the same clips at 256².
- [ ] Extract observables from its predicted frames via SAM-EM-checkpoint segmentations; add the column to the Gate-2 headline table.
- [ ] Commit + extend `experiments/core/README.md`.

## Step 12 — Event anticipation

**Where:** remote compute, local analysis. **Files:** `scripts/event_anticipation.py`, `experiments/events/`

- [ ] On held-out `ripening_coalescence` videos: per-step teacher-forced latent prediction error as a time series, aligned to GT merge times from `gt.npz`.
- [ ] Metrics: lead-time distribution (smoothed error exceeds its per-video 95th percentile *k* frames before the merge) + ROC vs matched no-event windows.
- [ ] Qualitative repeat on real aggregation clips (hand-annotate ~10 visible onsets): one figure, error trace with annotated event.
- [ ] Note for the write-up: an error spike at an event means the model did **not** fully learn event timing — frame this correctly (timing is plausibly stochastic even when deterministic dynamics are learned; anticipation = usable surprisal signal, not a failure of RQ1).
- [ ] Write `experiments/events/README.md`. Commit.

## Step 13 — Real-data validation

**Where:** remote compute, local analysis. **Files:** `scripts/eval_real.py`, `experiments/real/`

- [ ] Full pipeline (encode → rollout → probes) on the Au-nanorod experimental clips. Evaluation GT: SAM-EM-checkpoint segmentations of the *actual* future frames — **state the circularity caveat explicitly**.
- [ ] Report probed-MSD vs segmentation-MSD; if the Jamali raw videos have arrived, swap in their published trajectories as independent GT (same script, new GT loader).
- [ ] Optional, only if data + time permit: one LoRA continuation run on real video. Not on the critical path.
- [ ] Write-up. Commit.

## Step 14 — Thesis assembly + workshop paper (continuous)

- [ ] Ongoing from Step 7: every `experiments/*/README.md` written same-day; these become Results subsections nearly verbatim.
- [ ] Assemble: Intro/Related Work (SAM-EM = segmentation, LEONARDO = trajectories, PNNL = pixels, this = latent world model), Methods (the eval spine, simulator, encoder, and predictor — Steps 2–9), Results (gate READMEs), the martingale argument as its own subsection, honest negatives included. Also: one explicit paragraph on sim-vs-real scope (the model learns the extended simulator's physics; real-data validation bounds the transfer claim).
- [ ] Advisor review cycle; revise.
- [ ] Compress to 4–8 pages for NeurIPS AI4Mat / ICLR ML4Materials (check current deadlines when Gate 2 lands).

---

## Decision gates recap

- **Gate 1 (after Step 7):** frozen V-JEPA 2 features probe-able on LPTEM? → decides whether LoRA is the main path or a single comparison run. Whole-window cache: PASS (2026-09-30), invalidated for forecasting by the look-ahead. Per-frame cache: FAIL on count only (2026-10-02) → Step 8 order: per-frame Gate 2 first, V-JEPA 2.1 as encoder candidate, LoRA later (documented deviation).
- **Gate 2 (after Step 10):** predicted latents decode to correct physics on growth modes? → PASS = thesis floor secured; FAIL = the honest negative result, still a thesis — pivot the writing accordingly.

## Standing risks

- ~~**Remote access friction**~~ **Resolved** (Step 5): SSH from the Mac works; jobs run unattended in tmux.
- **Shared GPU machine.** Other users' jobs and a vLLM server share both cards and the `mlmonster` account; our runs take whichever card is idle, which may not always be one. Etiquette and checks in `docs/remote.md`.
- ~~**SAM-EM renderer integration** (Step 4)~~ **Resolved:** the renderer was found (`JamaliLab/LPTEMsimulator`) and re-implemented in `render_bridge.py`; local render matches real SAM-EM appearance. No longer a risk.
- **Gate 2 failure** is the scientific risk, and it is survivable by design.
- **Disk on the remote:** dataset (~40 GB) + feature cache (~67 GB) fit on `/mnt/drive2`. Every encoder candidate (per-frame V-JEPA 2, V-JEPA 2.1, LoRA) is another ~67 GB cache: delete the whole-window cache once nothing reads it, and check `df -h` before each extraction.
