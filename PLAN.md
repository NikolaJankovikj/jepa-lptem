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
- [ ] **Manual (Nikola):** add `~/.ssh/id_ed25519_personal.pub` to the personal GitHub account (Settings → SSH and GPG keys), verify with `ssh -T git@github-personal`, then **revoke the exposed token** (Settings → Developer settings → Personal access tokens).
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
- [ ] **Send the Jamali-lab data-request email** (draft ready in `docs/jamali_data_request_email.md`; off critical path, send now for lead time) — **manual (Nikola)**.
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

**Known residual, not closed.** At matched temporal noise our frames are spatially *smoother* than real: the ratio of temporal to spatial noise is 3.16 for us at every flux versus 1.67 real, and our high-frequency power sits ~10× low. Pure shot noise fixes that ratio, so the difference is spatial correlation — most likely the measured 512 px MTF over-blurring relative to the real 1024 px detector. Ruled out: JPEG compression (our ratio holds at ~3 for every quality from 100 down to 60). A static-texture term was implemented and **reverted** — it cannot move an Immerkær estimate, which is by construction insensitive to structure. Report as a Tier-2 bound; static structure is the least harmful gap for a video predictor, since it does not change between frames.

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

**Decided here, needed by Step 6:** preprocessing must **crop** to 256², not resize. Real 1024² frames resized to 256² shrink those 6 px particles to 1.5 px — invisible. Our synthetic scale is chosen so both land at the same apparent size after preprocessing.

**End of local phase.** At this point the entire evaluation + simulation core is coded and green on CPU. Decide the local side is ready → get PC access → Step 5.

---

# REMOTE PHASE (GPU — Steps 5+, once local pipeline is ready)

## Step 5 — Remote training PC setup (when local pipeline is green)

**Where:** remote (work PC, 2× RTX 3090). Goal: from the Mac, one command away from launching a training run. Do this only after Steps 2–4 are done and Nikola has confirmed PC access.

- [ ] **Access:** confirm SSH reachability from the Mac (`ssh <user>@<host>`); set up key-based auth (`ssh-copy-id`) and a `~/.ssh/config` entry (`Host trainbox`). Reuse the dual-account SSH trick from Step 0 (dedicated personal key + `github-personal` alias) so the clone authenticates as the personal account. If you need access from outside the work network, ask IT about VPN or Tailscale — otherwise plan launches for when you're at work (jobs run unattended in tmux anyway).
- [ ] **Health check:** `nvidia-smi` (2 GPUs visible, driver supports CUDA ≥ 12.1), `df -h` (**need ≥ 200 GB free** on one volume: ~50 GB raw dynamics dataset + ~70 GB feature cache + checkpoints + upstream data), `free -h`, `nproc`.
- [ ] **Environment:** install `uv` and `tmux` if missing; `git clone` the repo via the `github-personal` alias; `uv sync` (pulls CUDA torch from the same lockfile); `uv run python -c "import torch; print(torch.cuda.device_count())"` → `2`; `uv run pytest` (CPU suite green on the remote too).
- [ ] **GPU smoke test:** a 30-second matmul-on-both-GPUs script now; the real `tests/test_feature_shapes.py -m gpu` once Step 6 exists. Set `HF_TOKEN` in the remote env (untracked) for fast Hub pulls.
- [ ] **Transfer path:** verify `rsync -avz trainbox:~/jepa-lptem/experiments/ ./experiments/` round-trips; decide results-logging (recommended: free W&B academic account so runs are monitorable from anywhere; fallback: CSV logs + rsync).
- [ ] **Download the full datasets directly on the remote** (SAM-EM trainset/holdout/experimental, LEONARDO) using the IDs verified in Step 1.
- [ ] **Run the full dataset generation** (Step 4's generator, `--n 2000 --out data/dynamics-v1`, tmux/overnight); re-run the physics validation on the full set; write `experiments/dataset-v1/README.md` (this dataset is a standalone thesis contribution).
- [ ] Document all of it in `docs/remote.md` (host alias, paths, launch commands, tmux conventions). Commit: `docs: remote training environment`.

## Step 6 — Feature extraction + cache

**Where:** remote (GPU). **Files:** `scripts/extract_features.py`, `tests/test_feature_shapes.py`

- [ ] GPU smoke test (`@pytest.mark.gpu`): load `facebook/vjepa2-vitl-fpc64-256`, feed 64 random frames at 256², assert `last_hidden_state.shape == (1, 32*16*16, 1024)` (64 frames / tubelet 2 = 32 latent frames × 16×16 patches × dim 1024).
- [ ] Implement `scripts/extract_features.py`: center-crop/resize to 256², window each 100-frame video into 64-frame clips (**stride 36 → exactly 2 windows/video, frames 0–63 and 36–99**, covering late-time growth for exponent fits), save `last_hidden_state` reshaped `(32, 256, 1024)` as fp16 `.npy` (~17 MB each). **Budget: ~4,000 clips ≈ 68 GB** for dynamics-v1, plus SAM-EM holdout + experimental. bf16 inference, both GPUs via `CUDA_VISIBLE_DEVICES` split.
- [ ] Run extraction on dynamics-v1 + SAM-EM holdout + experimental clips (a few hours, tmux, can overlap with the dataset-generation tail).
- [ ] Commit script.

## Step 7 — GATE 1: probe frozen features

**Where:** probe training remote (~30 min/probe on one 3090), analysis local. **Files:** `scripts/train_probes.py`, `experiments/gate1/`

- [ ] Implement `scripts/train_probes.py`: per-latent-frame tokens `(256, 1024)` → `AttentiveProbe` for `[count, mean_radius, total_area]` (targets from `gt.npz`) and `CentroidProbe` for occupancy+offsets (Gaussian splats + offset targets at GT centroids). Split 80/10/10 **by video**.
- [ ] Run on frozen features; rsync metrics back; analyze locally.
- [ ] **Decide the gate:**
  - **PASS** (held-out R² ≥ 0.8 for mean radius and count at mid SNR; MSD from decoded tracks within 20% of GT MSD, with track-coverage reported): frozen encoder suffices → **LoRA (Step 8) shrinks to a single comparison run.**
  - **FAIL:** Step 8 becomes the main path.
- [ ] Write `experiments/gate1/README.md` same day (a thesis figure either way). Commit.

## Step 8 — Encoder adaptation (LoRA) — scope set by Gate 1

**Where:** remote (the one multi-day GPU job: ~2–3 days/run). **Files:** `configs/lora_vitl.yaml`, `scripts/run_lora.py`

- [ ] First get the **unmodified** official V-JEPA 2 pretraining loop running on 10 SAM-EM videos (tiny smoke config, 50 iters, loss decreases) — validates the vendored repo before touching it.
- [ ] Add LoRA injection via `peft.LoraConfig(r=16, lora_alpha=32, target_modules=["qkv","proj"])`; objective unchanged (masked latent prediction); EMA target = frozen base + EMA of LoRA params; assert trainable params < 10 M.
- [ ] Launch (SAM-EM trainset + dynamics-v1, bf16, DDP 2 GPU, 256² crops): `torchrun --nproc_per_node=2 scripts/run_lora.py --config configs/lora_vitl.yaml` in tmux. **While it runs, build Step 9 locally.**
- [ ] Re-run Gate-1 probing on adapted features; keep whichever encoder maximizes held-out probe R² (max 3 runs total incl. variants); **freeze the choice permanently** and re-extract the feature cache with it.
- [ ] Commit configs + `experiments/encoder-selection/README.md`.

## Step 9 — Block-causal forecasting predictor

**Where:** code + CPU tests local; training remote (cached features → **hours** per run). **Files:** `src/jepa_lptem/models/predictor.py`, `scripts/train_predictor.py`, `tests/test_predictor.py`

- [ ] Failing tests: output shapes; rollout extends horizon; **block-causality** (perturb only the last context frame → all earlier per-step predictions unchanged).
- [ ] Implement (note: temporal embedding broadcasts as `(1, T, 1, D)` — an earlier draft had a `transpose(1,2)` here that only worked when S == T; the shape tests exist to catch exactly this):

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

- [ ] Implement `scripts/train_predictor.py`: cached clips `(32, 256, 1024)`; context 8 latent frames → next-frame target (teacher forcing, L1 on normalized features, per V-JEPA 2-AC) + 2-step rollout loss; AdamW 1e-4, cosine, bf16, ~50 epochs.
- [ ] Train on remote; **log persistence L1 (`|feat[t+1] − feat[t]|`) alongside model L1 every epoch** — if the model never beats persistence on `growth_*` modes, debug before proceeding.
- [ ] Commit config + curves.

## Step 10 — GATE 2: the core experiment — probe predicted latents

**Where:** eval remote, analysis + figures local. **Files:** `scripts/eval_forecasts.py`, `experiments/core/`

- [ ] Implement `scripts/eval_forecasts.py`: for each held-out dynamics-v1 video — encode context, roll out to horizons +1/+5/+10, apply the *frozen Step-7 probes* to predicted latents, compute vs GT-at-that-future-time: (a) size-distribution Wasserstein, (b) growth exponent from probed mean radius vs latent time, (c) MSD from decoded tracks. **Always compute identical metrics for persistence** (predicted latent := last context latent).
- [ ] Also record probe error on **encoded true-future latents** as the floor — this separates "wrong physics" from "predicted latents are off-manifold for the probes" (a known confound; report both numbers).
- [ ] Produce the headline table (horizon × mode; model vs persistence vs floor per observable) and money plots: predicted vs GT size distribution at +10; probed growth curve overlaid on r ∝ t^⅓.
- [ ] **Judge Gate 2.** PASS = on `growth_*` and `ripening` at +5: size-dist Wasserstein ≥ 30% below persistence AND probed growth exponent within ±0.1 of GT. (On `diffusion_only`, parity with persistence on *position* is expected and correct — Brownian motion is a martingale; report MSD spread instead.) FAIL → systematic debugging: verify teacher-forced +1 first, then how error grows with horizon; a genuine collapse-to-persistence is the honest negative result and still a thesis.
- [ ] Write `experiments/core/README.md` + draft the thesis Results section immediately. Commit.

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

- **Gate 1 (after Step 7):** frozen V-JEPA 2 features probe-able on LPTEM? → decides whether LoRA is the main path or a single comparison run.
- **Gate 2 (after Step 10):** predicted latents decode to correct physics on growth modes? → PASS = thesis floor secured; FAIL = the honest negative result, still a thesis — pivot the writing accordingly.

## Standing risks

- **Remote access friction** (VPN/IT policy): the reason PC setup is deferred to Step 5 and kept off the local critical path; mitigated by tmux-everything + W&B monitoring; worst case, launches happen on-site and run unattended.
- ~~**SAM-EM renderer integration** (Step 4)~~ **Resolved:** the renderer was found (`JamaliLab/LPTEMsimulator`) and re-implemented in `render_bridge.py`; local render matches real SAM-EM appearance. No longer a risk.
- **Gate 2 failure** is the scientific risk, and it is survivable by design.
- **Disk on the remote** (~200 GB) — checked in Step 5 before anything is generated.
