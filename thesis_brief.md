# Thesis Proposal Brief

**Working title:** *Does a self-supervised video model learn the physics of nanoparticles? A latent world model for liquid-phase transmission electron microscopy*

**Student:** Nikola Jankovikj
**Program:** MSc Data Science, FINKI
**Proposed duration:** ~5–7 months full-time (detailed week-by-week plan available)

---

## Abstract

Liquid-phase transmission electron microscopy (LPTEM) records video of nanoparticles moving, growing, and merging in liquid at nanometer resolution, but the footage is extremely noisy and current machine-learning tools for it are limited to *descriptive* tasks: segmenting particles or modeling extracted trajectories. This thesis asks a predictive question: **can a self-supervised video model, trained only by watching unlabeled LPTEM footage, implicitly learn the underlying physics of nanoparticle dynamics?** We adapt V-JEPA 2 — a joint-embedding predictive architecture that learns by predicting future video content in an abstract latent space rather than in pixels — to the LPTEM domain, and train a lightweight forecasting module that rolls the latent state forward in time. The core contribution is an evaluation methodology rather than a new architecture: we design probes that read physical observables (particle size distributions, growth-law exponents, mean-squared displacement) directly off the *predicted future* latent states and test them against exact simulator ground truth, against a persistence baseline, and against pixel-space video prediction. We further test whether the model's prediction error rises *before* visually identifiable events such as particle coalescence, enabling event anticipation. The hypothesis is falsifiable in both directions, and either outcome — latent forecasts that decode to correct physics, or a demonstration that they collapse to trivial persistence — is a reportable scientific result.

---

## 1. The science: what is LPTEM and what is the open problem?

**The instrument.** A transmission electron microscope images matter at sub-nanometer resolution by passing an electron beam through a sample. Classically the sample must be solid and static. *Liquid-phase* TEM seals a droplet of liquid between two membranes, so for the first time we can record **video of nanoscale processes as they happen in liquid**: nanoparticles diffusing, nucleating, growing, dissolving, and merging. This matters because these processes determine how catalysts form, how drugs are delivered, and how new materials are synthesized — and they cannot be observed any other way.

**Why the data is hard.** LPTEM video has very low signal-to-noise ratio: the liquid layer scatters electrons (the thicker the liquid, the worse the contrast) and the electron dose must be kept low because the beam itself perturbs the chemistry. The result is video where particles are barely distinguishable from shot noise — a setting where most computer-vision methods degrade badly.

**The physics we test against.** Nanoparticle dynamics follow well-characterized quantitative laws, which is what makes this thesis empirically grounded rather than qualitative:

- **Diffusion:** a particle's mean-squared displacement (MSD) grows linearly with time for Brownian motion, with deviations (anomalous diffusion) that encode the viscoelastic liquid environment.
- **Growth kinetics:** particle radius follows r ∝ t^1/3 when growth is diffusion-limited (Lifshitz–Slyozov–Wagner coarsening) versus r ∝ t^1/2 when reaction-limited — a measurable exponent that distinguishes mechanisms.
- **Coalescence:** merging particles conserve volume, and merge events change particle counts and size distributions in characteristic ways.

**The state of the art is descriptive, not predictive.** Three recent systems define the landscape: SAM-EM (NeurIPS 2025) fine-tunes the Segment Anything Model to *segment and track* particles in LPTEM video; LEONARDO (Nature Communications 2025) is a physics-informed generative model of *extracted trajectories* (coordinate sequences, not video); and a pixel-space LSTM (PNNL) predicts future *frames* but spends its capacity reproducing noise. **Nobody has tested whether a modern self-supervised video model can learn a forward model of the system — i.e., predict how the physical state will evolve — directly from raw, unlabeled footage.** That is the gap this thesis fills.

---

## 2. The machine learning: what we build and why this architecture

**Self-supervised learning and JEPA.** The dominant way to learn from unlabeled video is to predict withheld parts of the data. *Generative* approaches predict missing pixels; for LPTEM this is wasteful, because most pixel-level variation is irreducible shot noise that carries no information about the dynamics. A **Joint-Embedding Predictive Architecture (JEPA)** instead learns an encoder mapping video into an abstract latent space, and a predictor that predicts the *latent representation* of withheld content. The model is free to discard unpredictable noise and keep only what has predictive structure — which, in LPTEM, is precisely the physics (where particles are, how big they are, how they move). This is the one domain where JEPA's defining design choice is not a convenience but the correct inductive bias.

**The world-model framing.** Meta's V-JEPA 2 (2025) demonstrated that the predictor of a video-JEPA, retrained in a temporally causal fashion, functions as a *world model*: given the latent state of the recent past, it forecasts latent states of the future, and those forecasts are accurate enough to plan robot actions. We transfer this idea to a scientific instrument:

1. **Adapt the encoder** to the LPTEM domain. V-JEPA 2's ViT-L encoder (307M parameters, pretrained on ~1M hours of natural video) is adapted with LoRA — low-rank adapters that train <10M parameters — using the unchanged self-supervised objective on unlabeled LPTEM video. (A week-3 experiment first checks whether the frozen encoder already suffices.)
2. **Train a forecasting predictor**: a small block-causal transformer (~100M parameters) that takes the latent states of frames 1…N and predicts the latent state at N+1, applied autoregressively to reach horizons N+5, N+10. It trains on *cached* encoder features, so experiments run in hours on the available hardware (2× RTX 3090).
3. **Never use a label during training.** Segmentation masks and trajectories are used only at evaluation time, to test what the model learned.

**The actual contribution: an evaluation methodology.** Training the model is engineering; the science is proving the predictions *mean* something. We design probing functions — small attentive readout heads trained on held-out data with frozen features — that map a latent state to physical observables: particle count, mean radius, size distribution, and an occupancy map from which trajectories and MSD are computed. Applying these probes to *predicted future* latents lets us ask quantitatively: does the forecasted future have the right size distribution (Wasserstein distance)? The right growth exponent (t^1/3 vs t^1/2)? The right MSD? Ground truth comes from an extended physics simulator (below), so every comparison is exact.

**Controls and baselines — where the thesis is honest.**

- **Persistence baseline** (predict "nothing changes"): the critical control. For purely diffusive motion, theory says persistence is near-optimal for *positions* (Brownian motion is a martingale — the future increment is unpredictable in expectation), so the model is *expected* to match, not beat, persistence there; what it must beat persistence on is everything with deterministic structure: growth, size distributions, event statistics. Building this distinction into the evaluation prevents both false claims of success and false claims of failure.
- **Pixel-space prediction baseline** (ConvLSTM): tests the central claim that latent prediction beats pixel prediction on physics metrics in low-SNR video.
- **Event anticipation:** prediction-error spikes preceding coalescence events, evaluated by lead-time and ROC analysis against ground-truth event times, with a qualitative demonstration on real aggregation footage.

**A data contribution.** Public LPTEM datasets contain only *diffusing* particles. We extend the open SAM-EM simulator with verifiable radius dynamics — diffusion-limited growth, reaction-limited growth, dissolution, and volume-conserving coalescence (each unit-tested to reproduce its known kinetic law) — producing ~2,000 ground-truthed videos. This dataset is releasable and useful to the community independently of our model results.

---

## 3. Research questions and hypotheses

- **RQ1:** Do latent states predicted by a JEPA world model decode to physically correct future observables (size distribution, growth exponent, MSD) on low-SNR LPTEM video?
  *H1:* At short horizons they do, with significantly lower distributional error than persistence and pixel-space baselines on dynamics with deterministic structure.
- **RQ2:** Does the model's prediction error anticipate physical events (coalescence) before they are visually identifiable?
  *H2:* Smoothed prediction error exceeds its baseline significantly earlier than the visible event onset.
- **RQ3:** How does forecast quality degrade with prediction horizon and with SNR (liquid thickness)?

All hypotheses are falsifiable; a negative result on RQ1 ("latent forecasting on LPTEM collapses to persistence") is itself a publishable finding about the limits of video world models in scientific imaging.

---

## 4. Data, feasibility, and timeline

| Resource | Role | Status |
|---|---|---|
| SAM-EM synthetic LPTEM (~50k frames + masks, CC BY 4.0) | Pretraining + probe training | Public, downloaded |
| SAM-EM simulator (extended by us) | Ground-truthed growth/coalescence data | Public code |
| SAM-EM real experimental clips (Au nanorods, aggregation) | Real-data validation | Public |
| LEONARDO trajectories + model (Apache 2.0) | Real motion statistics; trajectory-level comparison | Public |
| V-JEPA 2 ViT-L pretrained weights | Encoder backbone | Public |

No private data is required. All compute fits on the available 2× RTX 3090 workstation (LoRA adaptation ~2–3 days per run; all other training operates on cached features in hours). The work plan is ~18–20 weeks with two decision gates (week 3: frozen vs adapted encoder; week ~10: core physics-probing result), and a written risk plan for each gate.

**Expected outcomes:** an MSc thesis; an open-source codebase and ground-truthed dynamics dataset; and a workshop-paper submission (target: NeurIPS AI4Mat or ICLR ML4Materials) at the result floor, with a journal submission (e.g., *npj Computational Materials*) plausible at the ceiling.

---

## 5. Key references

1. Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, arXiv:2506.09985 (2025).
2. *SAM-EM: Real-Time Segmentation for Automated Liquid Phase Transmission Electron Microscopy*, NeurIPS 2025, arXiv:2501.03153.
3. *Learning the diffusion of nanoparticles in liquid phase TEM via physics-informed generative AI* (LEONARDO), Nature Communications 16 (2025), s41467-025-61632-1.
4. *Deep-learning-based prediction of nanoparticle phase transitions during in situ TEM*, arXiv:2205.11407 (PNNL/UW; the pixel-space prediction baseline).
5. LeCun, *A Path Towards Autonomous Machine Intelligence* (2022) — the JEPA/world-model research program.
