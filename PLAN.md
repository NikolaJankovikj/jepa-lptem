# JEPA-LPTEM Implementation Plan

**Goal:** Show that a V-JEPA-2-based latent world model, adapted to LPTEM video, predicts futures that decode to correct physics (size distributions, growth exponents, MSD) and anticipates events (coalescence).

**Architecture:** Frozen-or-LoRA V-JEPA 2 ViT-L encoder → cached latent features → small block-causal forecasting predictor (teacher forcing + rollout) → probes that read physical observables off predicted latents → compared against a persistence baseline and a pixel ConvLSTM baseline. All probes are validated against simulator ground truth *before* any model exists.

**Tech stack:** Python 3.11, PyTorch 2.x (bf16, DDP), HuggingFace `transformers` (V-JEPA 2) + `peft` (LoRA), official `facebookresearch/vjepa2` repo for SSL adaptation, SAM-EM simulator (extended), pytest, uv, tmux + ssh for remote runs.

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

## Step 2 — Evaluation spine, tested against ground truth

**Where:** local, pure CPU. *The thesis's contribution is the evaluation — build and validate it before any model exists.*

### 2a: Physical observables (`src/jepa_lptem/eval/observables.py`)

- [ ] Failing tests first (`tests/test_observables.py`):

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

- [ ] Implement: `compute_msd` (ensemble MSD over lag), `fit_growth_exponent` (log-log slope), `size_dist_distance` (1-Wasserstein via `scipy.stats.wasserstein_distance`).
- [ ] Tests pass → commit.

### 2b: Centroid extraction + tracking (`src/jepa_lptem/eval/tracking.py`)

- [ ] Failing tests: heatmap peaks recovered via connected components; two drifting particles linked over 5 frames by Hungarian assignment (`scipy.optimize.linear_sum_assignment`) with a `max_dist` gate.
- [ ] Implement `heatmap_to_centroids` and `link_tracks`. **Requirement:** `link_tracks` must *report* coverage (fraction of tracks surviving full-length) rather than silently dropping lost tracks — evaluation coverage must be visible in every MSD number we publish.
- [ ] Tests pass → commit.

### 2c: Probes (`src/jepa_lptem/eval/probes.py`)

Two probe types, matching the V-JEPA attentive-readout protocol (frozen features, small trainable head):

- **`AttentiveProbe`** — one learned query cross-attends over a latent frame's tokens → regression head for `[count, mean_radius, total_area]`.
- **`CentroidProbe`** — per-token head with **3 output channels: occupancy logit + (dx, dy) sub-cell offset**. The token grid is 16×16 over a 256² frame (16 px/cell) — raw grid centroids would quantize Brownian steps to noise, which would corrupt MSD exactly where we compare against persistence. Sub-cell offsets fix this: predicted centroid = cell center + offset, trained with BCE (occupancy) + masked L2 (offsets at occupied cells).

- [ ] Failing tests: (a) `AttentiveProbe` overfits a tiny batch (8 samples, 300 Adam steps, `loss < 1e-2` — if flaky across seeds, fix the seed and/or loosen to 3e-2, it's a sanity check not a benchmark); (b) `CentroidProbe` output shapes `(B,16,16)` occupancy + `(B,16,16,2)` offsets; (c) a synthetic case where the true centroid sits off-center in a cell and the decoded centroid recovers it to <1 px after overfitting.
- [ ] Implement both probes + a `decode_centroids(occupancy, offsets, threshold)` helper feeding into 2b's tracker.
- [ ] Tests pass → commit.

## Step 3 — Simulator extension: verifiable radius dynamics

**Where:** local. **Files:** `src/jepa_lptem/sim/dynamics.py`, `tests/test_dynamics.py`

- [ ] Failing tests — each mode must reproduce its kinetic law, recovered by 2a's own `fit_growth_exponent` (closing the loop between simulator and evaluation):

```python
def test_diffusion_limited_growth_exponent():   # dr/dt = K/r^2  ->  r ~ t^(1/3)  (LSW)
    t, r = evolve_radii(r0=np.array([2.0]), n_steps=500, dt=1.0, mode="diffusion")
    assert abs(fit_growth_exponent(t[50:], r[0, 50:]) - 1 / 3) < 0.05

def test_reaction_limited_growth_exponent():    # dr/dt = K/r    ->  r ~ t^(1/2)
    ...

def test_dissolution_shrinks_to_zero():          # dr/dt = -K, clamped at 0
    ...

def test_coalescence_conserves_volume():         # r_new^3 = r_i^3 + r_j^3, volume-weighted position
    ...
```

- [ ] Implement `evolve_radii` (per-particle radius ODEs, Euler steps, clamp ≥ 0) and `merge_on_contact` (merge overlapping pairs until stable).
- [ ] Tests pass → commit.

## Step 4 — Render bridge + dynamics dataset generator (code + local smoke)

**Where:** local — write the code, smoke-test on a handful of videos, validate the physics. **The full 2,000-video generation is deferred to the remote (first job after Step 5 PC setup)** — it's a multi-hour batch job whose ~50 GB output is consumed on the PC.

- [ ] Read SAM-EM's synthetic-video generation code (`code/sam-em`) and locate the renderer entry point; record its signature in the `render_bridge.py` docstring. *(This is the main unknown-shaped task — do it before wiring anything downstream.)*
- [ ] Implement `src/jepa_lptem/sim/render_bridge.py` — one function, `render_video(positions, radii, liquid_thickness_nm, size=512) -> (T,H,W)`, delegating to SAM-EM's renderer; keep it <100 lines.
- [ ] Implement `scripts/generate_dynamics_dataset.py`: 4 modes × 500 videos (`diffusion_only`, `growth_diffusion`, `growth_reaction`, `ripening_coalescence`), 100 frames, 512², liquid thickness sampled 20–120 nm (the SNR axis for RQ3); motion sampled from LEONARDO statistics as upstream does. Per video: `frames.npy` + `gt.npz` (positions, radii, event times).
- [ ] **Smoke locally:** `--n 5`, eyeball the videos (particles visibly grow/merge), then `tests/test_dataset_physics.py`: fit growth exponents on generated `gt.npz` → within ±0.05 of each mode's law. **This is the Step-4 completion bar** — the generator is proven correct on tiny output before we ever run it at scale.
- [ ] Commit scripts. (Full run + `experiments/dataset-v1/README.md` happen in Step 5/6 on the remote.)

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
- **SAM-EM renderer integration** (Step 4) is the main unknown-shaped task — its entry-point reading comes *before* any dependent scheduling.
- **Gate 2 failure** is the scientific risk, and it is survivable by design.
- **Disk on the remote** (~200 GB) — checked in Step 5 before anything is generated.
