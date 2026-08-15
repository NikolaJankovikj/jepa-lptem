"""Clip construction: turn a named physical process into ground-truth arrays.

This is the composition layer between ``dynamics.py`` (the physics) and
``render_bridge.py`` (the pixels). It owns the choices that make a *dataset*
rather than a simulation -- how many particles, how big, how fast they grow --
and nothing else. ``scripts/generate_dynamics_dataset.py`` is a thin CLI on top.

The one non-obvious constraint here is **density**. The dataset exists to give
the evaluation an exact answer key, and an answer key is worthless if the answer
isn't visible: once particles grow until they overlap, no centroid is recoverable
from the frame and a probe failure can no longer be attributed to the model.
So particle size is set by a target area fraction, not by the growth law -- see
:func:`rescale_to_apparent_size`, and :func:`build_clip` for the coalescence
process, which has to solve the same cap up front because its radii and positions
are coupled.
"""

from __future__ import annotations

import numpy as np

from .dynamics import GROWTH_K, coalesce, evolve_radii, ripen_population, simulate_brownian

# Growth processes and their known exponent (None = coupled/coarsening, no single alpha).
PROCESSES = ("ripening", "diffusion", "reaction", "static", "coalescence")
EXPONENT = {"diffusion": 1 / 3, "reaction": 1 / 2, "static": 0.0,
            "ripening": None, "coalescence": None}

# Final/initial radius over a clip for the growth modes. Sets how much *visible*
# growth a clip contains; it no longer has to be large for the exponent's sake,
# because each particle is placed exactly on its own pure power law from a common
# nucleation time (see R0_SPREAD) rather than approximately, so there is no
# finite-r0 offset for a long clip to wash out. Calibration knob: raise it for more
# visible growth, lower it for better early-frame visibility.
GROWTH_RATIO = 3.0
R0_UNITS = 2.0  # arbitrary pre-rescale radius unit; only the ratio matters

# Per-particle spread in initial radius, log-uniform over [1/S, S].
#
# The growth ODEs are contractions on r -- dr_final/dr0 = (r0/r_final)^(p-1) -- so a
# population sharing one rate constant converges to a single radius: measured
# relative sd 0.095 -> 0.0007 over a 5x diffusion clip, and 0.079 -> 0.0037 for
# reaction. That is a delta function, and the size *distribution* is the headline
# observable of the whole evaluation, so on the two growth processes Gate 2 was
# comparing two spikes and the Wasserstein distance degenerated to |delta mean|.
#
# The fix is a per-particle rate constant k_i, pinned to r0_i by k_i = r0_i^p/(p t0)
# so every particle is on r = (p k t)^(1/p) from the *same* t=0. Then r_i/r_j is
# constant over the clip -- the spread neither collapses nor blows up -- and the
# ensemble mean is still exactly t^(1/p). Set so the resulting spread sits near the
# 0.155 that the `static` process already carries, rather than at a value that
# would make the growth clips the widest in the dataset.
R0_SPREAD = 1.35

# A particle smaller than this is not visible under shot noise, so labelling it as
# present would hand the evaluation a target that is not in the image.
MIN_VISIBLE_RADIUS_PX = 2.0

# Diffusion is set by how far a particle moves *relative to its own radius* per
# frame, not by an absolute D. Real experimental video measures ~1.5 radii/frame
# (apparent 10.6 px at 6.1 px radius, minus a 4.8 px centroid-jitter floor measured
# on a frozen-particle control; coherent stage drift accounts for only 7 %).
#
# This is not cosmetic. At the 0.3 radii/frame we had, persistence -- predict the
# last position -- is already an excellent predictor, so the Gate-2 comparison
# against persistence would have been uninformative on position, and MSD would sit
# barely above the centroid-noise floor. It also changes the learning problem: at
# ~10 px/frame a particle crosses 16 px patch boundaries constantly, which is a
# different task from one drifting 2 px/frame.
#
# For a Rayleigh-distributed 2D step the mean magnitude is 1.772*sqrt(D), which
# inverts to give D. D stays *constant within a clip* even while radii grow, so the
# ground truth MSD = 4Dt remains exact -- that is what the evaluation grades.
STEP_MEAN_PER_SQRT_D = 1.772

# The real measurement is a single (radius, step) point: 1.55 radii/frame at an
# 8 px particle. That pins the *scale* of D; it says nothing about how D varies with
# size, and Stokes-Einstein does: D = kT / (6 pi eta r), so bigger particles diffuse
# SLOWER. Applying the ratio at each clip's own radius instead made D ~ r^2 -- the
# scaling inverted, a 20x error across the 8-22 px range, and a size->motion shortcut
# ("big blob moves fast") of exactly the kind the brightness->size leak was. Anchor
# the ratio here and let D fall off as 1/r from it; at 8 px nothing changes, so the
# containment result on the experiment stratum is unaffected.
R_REF_PX = 8.0

# Coherent stage drift, as a fraction of the frame travelled over the whole clip.
#
# Brownian position is a martingale, so on the position axis the generator was
# handing the persistence baseline an optimal strategy *by construction* -- the
# Gate-2 comparison would then have been decided by our choice of simulator rather
# than by evidence. Real LPTEM has coherent stage drift (measured at ~7 % of the
# apparent frame-to-frame displacement on the Au-nanorod video) and it is the one
# motion component a forecaster can actually get right.
#
# Parameterised as total excursion over the clip rather than as a per-frame speed,
# because that is the quantity that has to stay inside the frame: the box is inset
# by the excursion so no particle drifts out and the ground truth never labels a
# particle that is not in the image. Sampling U(0, 0.10) of the frame puts the
# per-frame drift at 0-12 % of the Brownian step across the two strata -- it
# brackets the measured 7 % rather than fitting a point estimate of it, for the
# same reason STRATA does.
DRIFT_FRACTION = 0.10

# Coalescence sizes its particles so that a survivor of this many merges lands at
# the stratum's target apparent radius. Sizing for the worst case instead -- every
# particle in the clip merging into one -- is what the volume arithmetic allows but
# not what happens: clips average 1-2 merges, so that bound shrank the starting
# particles to ~3.5 px, the smallest in the dataset, on the one process whose whole
# point is that you can see the event. A rare heavy cascade now overshoots the
# apparent-size target instead, which is cosmetic; the coverage cap is enforced
# separately and cannot be overshot, because merging only ever reduces total area.
MERGE_HEADROOM = 4.0

# Reflection off the box walls biases MSD downward at long lag, so the MSD = 4Dt
# label is only true up to some lag. Ship that lag with the label rather than
# leaving each consumer to rediscover it: valid while the free MSD 4*D*lag stays
# under this fraction of the box area. Calibrated, not derived -- measured pooled
# MSD/4D over 8 clips per stratum: 0.02 reads 0.97 physics / 0.85 experiment, 0.005
# reads 1.00 / 0.91. The fast experiment-stratum clips are what set it; they land at
# lag 16, which is the ceiling test_msd_of_generated_positions_matches_4Dt had
# hardcoded. Lower it further only if a stratum diffuses faster still -- below ~6
# lags the fit runs out of points and variance replaces the bias it removed.
MSD_BIAS_FRACTION = 0.005

# Two strata, because "match the real experiment" and "measure physics precisely"
# pull in opposite directions and the dataset has to serve both.
#
#   physics     Enough particles, large enough, to estimate an MSD or a size
#               distribution with tight error bars. This is where Gate 1 and
#               Gate 2 are actually decided: if a probe fails here we need to
#               know it failed because the model cannot do it, not because our
#               estimator was too noisy to tell.
#   experiment  Pinned to what we measured on real experimental video: ~8 px
#               particles, under 2 % coverage, and a beam flux set from the
#               *temporal* noise statistic (|dI|/mean ~0.40 -> flux ~8-12), which
#               is what a video model actually consumes. Weak statistics per clip
#               by construction -- its job is to put real imaging conditions
#               *inside* the training distribution, not to measure anything.
#
# Known residual: at matched temporal noise our frames are spatially smoother
# than real (ratio of temporal to spatial noise 3.16 for us, 1.67 real; our
# high-frequency power spectrum sits ~10x low). Pure shot noise fixes that ratio,
# so the difference is spatial correlation -- most likely the measured 512 px MTF
# over-blurring relative to the real 1024 px detector. Not closed here; it is a
# Tier-2 bound, and static structure is the least harmful kind of gap for a video
# predictor since it does not change frame to frame. Ruled out as the cause:
# JPEG compression (our ratio holds at ~3 for every quality from 100 down to 60).
#
# Randomising over a range that brackets real data beats fitting a point estimate
# of it: our picture of "real" is 75 frames of one Au-nanorod experiment, exported
# by one person, and Step 13 also tests on a different (aggregation) condition.
# The two strata differ in *density and dose*, not particle size. An earlier
# version split them on size, on the strength of a measurement that turned out to
# be wrong: a threshold-component size estimator read real video as 6.1 px when a
# radial-profile fit (validated against our own ground truth) says 8.0 px, and read
# our clips as 3x larger than real when they were already the same size. Particle
# size is now matched in both strata; what genuinely differs is how many particles
# there are and at what dose.
STRATA = {
    # beam_flux floor is 6.0. It was 4.0 (up from 2.0, where the background was 0.45
    # electrons per pixel and the frame clipped to black over 13 % of its area --
    # a clip that has lost an eighth of its dynamic range to a floor is not a
    # low-dose sample, it is a censored one, and no probe can recover physics from
    # it). The thickness field then darkened the thick regions by up to
    # exp(2 sigma / IMFP_WATER) ~ 1.5x, which put the *effective* floor back at ~2.7
    # and returned zero-clipping to 2.7 % on the worst clip. Measured: clipping
    # tracks flux, not field amplitude -- the clip with the largest sd (52.7 nm) at
    # flux 8.3 clipped 0.19 %, while flux 4.2 at sd 40 clipped 2.74 %. So the fix is
    # the floor, raised by the same factor the field darkens by. Costs the 4-6 band.
    # thickness_sd_nm brackets the 39 nm that reproduces real's measured background
    # structure (see render_bridge.THICKNESS_SD_NM). The physics stratum runs the
    # same range as experiment rather than a gentler one: a flat background is the
    # free lunch we are removing, so exempting half the dataset from it would leave
    # the model most of its free lunch. Verified not to cost the answer key --
    # round-trip position error and detector precision are unchanged.
    "physics": dict(radius_px=(8.0, 22.0), n=(6, 16), beam_flux=(6.0, 75.0),
                    pixel_size_nm=(1.5, 4.0), max_coverage=0.09, growth_ratio=3.0,
                    step_per_radius=(0.2, 1.5), thickness_sd_nm=(15.0, 55.0)),
    # growth_ratio is smaller here because the clip *ends* around 6 px: at 5x the
    # particle would start at 1.2 px, under the visibility floor, and the opening
    # frames would contain no labelled particles at all. The exact time origin
    # (growth_time_axis) keeps the exponent accurate over the shorter range.
    # pixel_size_nm spans a wide range here rather than a fitted value. Contrast is
    # gold path length = 2*r*pixel_size_nm, so this range puts a 6 px particle
    # anywhere from ~25 to ~95 nm of gold -- all standard Au nanoparticle sizes.
    # Deliberately NOT tuned to the measured particle depth of real video: those
    # frames were contrast-stretched to 0-255 before publication, which inflates
    # depth by an unknown factor. Apparent size and frame-to-frame change survive
    # that stretch; absolute depth does not, so it is not a calibration target.
    "experiment": dict(radius_px=(10.0, 16.0), n=(3, 8), beam_flux=(8.0, 12.0),
                       pixel_size_nm=(2.0, 8.0), max_coverage=0.02, growth_ratio=2.0,
                       step_per_radius=(0.8, 2.0), thickness_sd_nm=(15.0, 55.0)),
}
EXPERIMENT_EVERY = 5  # one clip in five is experiment-grade


def growth_time_axis(mode: str, r0: float, n_steps: int, ratio: float = GROWTH_RATIO):
    """Return ``(t0, dt)``: the clip's true time origin and timestep.

    Inverts each mode's exact solution -- ``r^3 = r0^3 + 3Kt`` (diffusion-limited)
    and ``r^2 = r0^2 + 2Kt`` (reaction-limited) -- for a ``dt`` that grows ``r0`` by
    ``ratio`` over the clip, so every clip spans a fixed amount of *growth* rather
    than a fixed amount of time.

    ``t0`` is the age the particle already has at frame 0. A particle of radius
    ``r0`` is not newly nucleated: it has been growing for ``r0^3 / 3K`` already.
    Labelling frame 0 as ``t=0`` therefore mislabels the whole clip and flattens
    the measured exponent (reaction reads ~0.43 instead of 0.5). With the origin at
    nucleation the solution is a pure power law, ``r = (3Kt)^(1/3)``, and the fitted
    exponent recovers the textbook value. We know ``t0`` exactly because we chose
    ``r0``; an experimentalist would have to fit it.
    """
    K = GROWTH_K[mode]
    if mode == "diffusion":
        t0 = r0 ** 3 / (3 * K)
    elif mode == "reaction":
        t0 = r0 ** 2 / (2 * K)
    else:
        raise ValueError(f"{mode!r} is not a single-particle growth mode")
    return t0, t0 * (ratio ** (1 / EXPONENT[mode]) - 1) / max(n_steps - 1, 1)


def rescale_to_apparent_size(radii: np.ndarray, image_size, target_radius_px: float,
                             max_coverage: float) -> np.ndarray:
    """Scale radii to a target apparent size, backing off if that would overcrowd.

    Apparent size *in pixels* is the knob that matters, because V-JEPA tokenizes
    into 16x16 patches: a particle spanning half a patch and one spanning four are
    different problems however correct the nanometres are. So the target is the
    largest radius in the clip, with peak coverage as a safety rail rather than the
    primary control (the other way round, apparent size fell out of whatever count
    was drawn, and drifted 3x above real data).

    One constant scales the whole ``(N, T)`` array, so nothing about the physics
    moves: log-log growth slope and MSD are invariant under a uniform rescale of
    radius. Coverage is checked at the *densest* frame, which for mass-conserving
    ripening is t=0.
    """
    peak_radius = float(radii.max())
    if peak_radius <= 0:
        return radii
    scale = target_radius_px / peak_radius
    peak_area = float(np.pi * ((radii * scale) ** 2).sum(axis=0).max())
    cap = max_coverage * image_size[0] * image_size[1]
    if peak_area > cap:
        scale *= np.sqrt(cap / peak_area)
    return radii * scale


def build_clip(process: str, n_frames: int, size: tuple[int, int],
               rng: np.random.Generator, stratum: str = "physics"):
    """Return ``(positions_t, radii_t, meta, pos, radii)`` for one clip.

    ``positions_t``/``radii_t`` are the per-frame lists the renderer consumes;
    ``pos`` ``(N, T, 2)`` and ``radii`` ``(N, T)`` are the same data as arrays,
    saved verbatim as the ground truth. ``meta`` carries the physical parameters
    the evaluation needs to grade against (``D``, ``exponent``, ``t0_growth``,
    ``dt_growth``) plus the imaging conditions the renderer needs.

    ``stratum`` selects a row of :data:`STRATA`: ``"physics"`` for clips sized to
    measure on, ``"experiment"`` for clips pinned to real imaging conditions.
    """
    cfg = STRATA[stratum]
    h, w = size
    n = int(rng.integers(cfg["n"][0], cfg["n"][1] + 1))
    # D from the step-to-radius ratio (see STEP_MEAN_PER_SQRT_D), using the clip's
    # typical radius so motion is scaled to the particles that are actually drawn.
    step_ratio = float(rng.uniform(*cfg["step_per_radius"]))
    t0_growth, dt_growth = 0.0, 1.0
    if process == "coalescence":
        # Radii are set in pixels up front, not rescaled afterwards: contact is a
        # geometric test between radii and *positions*, and rescale_to_apparent_size
        # scales only the radii, so running it after the merges would silently
        # change which particles had touched. Volume is conserved and nothing
        # grows, so total area only ever falls (see dynamics.coalesce) -- peak
        # coverage is at frame 0 and both caps can be solved for exactly here.
        spread = rng.uniform(0.7, 1.3, n)
        target = float(rng.uniform(*cfg["radius_px"]))
        radii = spread * min(
            target / MERGE_HEADROOM ** (1 / 3),   # a MERGE_HEADROOM-way survivor
            float(np.sqrt(cfg["max_coverage"] * h * w / (np.pi * np.sum(spread ** 2)))),
        )
        radii = np.tile(radii[:, None], (1, n_frames))
    elif process == "ripening":
        # K sets the coarsening timescale, which is ~r^3/K. At K=6 and r~15 px that
        # is ~560 frames, so a 100-frame clip showed mean radius x1.02 and lost one
        # particle in ten -- labelled "Ostwald ripening", indistinguishable from
        # "static", and graded by nothing (EXPONENT["ripening"] is None). K=60 puts
        # the timescale inside the clip: count halves, mean radius x1.30. Higher
        # coarsens harder but leaves too few particles to measure a size
        # distribution on (K=150 ends at ~3).
        radii = ripen_population(rng.uniform(8.0, 22.0, n), n_steps=n_frames, dt=1.0,
                                 K=60.0, min_radius=3.0)[1]
    elif process in ("diffusion", "reaction"):
        # One common nucleation time, a per-particle radius, and the rate constant
        # derived from the two -- so every particle sits exactly on r = (p k t)^(1/p)
        # from t=0 rather than approximately (see R0_SPREAD). The spread is then
        # constant over the clip instead of collapsing to a delta, and the ensemble
        # mean is still an exact power law with the labelled exponent.
        p = 3 if process == "diffusion" else 2
        t0_growth, dt_growth = growth_time_axis(process, R0_UNITS, n_frames,
                                                ratio=cfg["growth_ratio"])
        r0 = R0_UNITS * np.exp(rng.uniform(-np.log(R0_SPREAD), np.log(R0_SPREAD), n))
        k = r0 ** p / (p * t0_growth)
        radii = evolve_radii(r0, n_steps=n_frames, dt=dt_growth, mode=process, k=k)[1]
    else:  # static: fixed radii, pure diffusion of position
        radii = np.tile(rng.uniform(8.0, 16.0, (n, 1)), (1, n_frames))

    if process != "coalescence":
        # Once radii have a real spread, the smallest particle at frame 0 is
        # target / (R0_SPREAD^2 * growth_ratio) -- and if that falls under the
        # visibility floor it gets zeroed, so the ground-truth mean radius is taken
        # over a subset that *grows with time* as particles cross the floor. That
        # bends the mean-radius curve away from the power law it is labelled with:
        # measured |alpha - 1/3| up to 0.031 (and 0.057 for reaction) before this
        # bound, against a ±0.005 label. Costs the 8-11 px band on growth clips
        # only; the experiment stratum already cleared it via its smaller ratio.
        lo, hi = cfg["radius_px"]
        if process in ("diffusion", "reaction"):
            lo = max(lo, MIN_VISIBLE_RADIUS_PX * cfg["growth_ratio"] * R0_SPREAD ** 2)
        radii = rescale_to_apparent_size(radii, size, float(rng.uniform(lo, hi)),
                                         cfg["max_coverage"])
    # A particle below the visibility floor is gone as far as the image is
    # concerned, so drop it from the ground truth rather than label a target that
    # is not there. Only ripening reaches this, on its dissolving tail.
    radii[radii < MIN_VISIBLE_RADIUS_PX] = 0.0

    typical_radius = float(np.median(radii[radii > 0])) if (radii > 0).any() else 1.0
    # step_ratio is the step-per-radius at R_REF_PX; D falls off as 1/r from there.
    D = float((step_ratio * R_REF_PX / STEP_MEAN_PER_SQRT_D) ** 2 * (R_REF_PX / typical_radius))

    # Coherent stage drift (see DRIFT_FRACTION). The walk runs in a box inset by the
    # total excursion and the drift is added after the fold, so particles land back
    # inside [0, size] and never leave the frame -- the ground truth never labels a
    # particle that is not in the image.
    angle = float(rng.uniform(0.0, 2 * np.pi))
    excursion = (float(rng.uniform(0.0, DRIFT_FRACTION)) * min(h, w)
                 * np.array([np.sin(angle), np.cos(angle)]))
    drift = excursion / max(n_frames - 1, 1)
    box = (h - abs(excursion[0]), w - abs(excursion[1]))
    origin = np.where(excursion < 0, np.abs(excursion), 0.0)

    margin = 0.15 * min(box)
    if process == "coalescence":
        init_pos = _place_without_overlap(radii[:, 0], box, margin, rng)
    else:
        init_pos = rng.uniform(margin, np.array(box) - margin, (n, 2))
    pos = simulate_brownian(n, n_frames, dt=1.0, D=D, box=box, drift=drift,
                            init_positions=init_pos, rng=rng)[1] + origin

    merge_events: list[list[int]] = []
    if process == "coalescence":
        pos, radii, merge_events = coalesce(pos, radii[:, 0])

    # max_lag bounds *box bias*, so it has to use the inset box the walk actually
    # ran in, not the frame.
    max_lag = max(1, min(n_frames - 1,
                         int(MSD_BIAS_FRACTION * box[0] * box[1] / (4.0 * D))))

    meta = {"process": process, "stratum": stratum, "exponent": EXPONENT[process],
            "n_particles": n, "D": D, "step_per_radius": step_ratio,
            "msd_max_lag": max_lag,
            # MSD = 4Dt holds on drift-removed positions: subtract drift * t exactly
            # (we know it; an experimentalist would estimate it from the ensemble mean)
            "drift": [float(drift[0]), float(drift[1])],
            "drift_per_step": float(np.hypot(*drift) / (STEP_MEAN_PER_SQRT_D * np.sqrt(D))),
            "merge_events": merge_events, "n_merges": len(merge_events),
            "t0_growth": t0_growth, "dt_growth": dt_growth,
            "beam_flux": float(np.exp(rng.uniform(*np.log(cfg["beam_flux"])))),
            "thickness_sd_nm": float(rng.uniform(*cfg["thickness_sd_nm"])),
            "pixel_size_nm": float(rng.uniform(*cfg["pixel_size_nm"]))}
    return [pos[:, i, :] for i in range(n_frames)], [radii[:, i] for i in range(n_frames)], \
        meta, pos, radii


def _place_without_overlap(radii: np.ndarray, box, margin: float,
                           rng: np.random.Generator, attempts: int = 200) -> np.ndarray:
    """Initial centres at least ``r_i + r_j`` apart, so no clip opens mid-merge.

    Only coalescence needs this: elsewhere a frame-0 overlap is cosmetic, but here
    it would fire a merge event at frame 0 with no lead-up, which is exactly the
    thing event anticipation cannot be graded on. Sequential placement, keeping the
    best-separated candidate if a particle cannot be placed cleanly -- at the
    coverage this stratum caps at, that fallback effectively never fires
    (``test_coalescence_clips_do_not_open_mid_merge`` is what says so).
    """
    pos = np.empty((len(radii), 2))
    hi = np.asarray(box, dtype=float) - margin
    for i, r in enumerate(radii):
        best, best_gap = None, -np.inf
        for _ in range(attempts):
            p = rng.uniform(margin, hi)
            gap = (np.inf if i == 0 else
                   float(np.min(np.hypot(*(pos[:i] - p).T) - (radii[:i] + r))))
            if gap > best_gap:
                best, best_gap = p, gap
            if gap > 0:
                break
        pos[i] = best
    return pos
