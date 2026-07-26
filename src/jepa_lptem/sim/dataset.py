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
:func:`rescale_to_density`.
"""

from __future__ import annotations

import numpy as np

from .dynamics import GROWTH_K, evolve_radii, ripen_population, simulate_brownian

# Growth processes and their known exponent (None = coupled/coarsening, no single alpha).
PROCESSES = ("ripening", "diffusion", "reaction", "static")
EXPONENT = {"diffusion": 1 / 3, "reaction": 1 / 2, "static": 0.0, "ripening": None}

# Final/initial radius over a clip for the growth modes. Large enough that the
# asymptotic exponent dominates the finite-r0 offset, small enough that particles
# are still several pixels wide at t=0 after the density rescale. Calibration knob:
# raise it for purer exponents, lower it for better early-frame visibility.
GROWTH_RATIO = 5.0
R0_UNITS = 2.0  # arbitrary pre-rescale radius unit; only the ratio matters

# A particle smaller than this is not visible under shot noise, so labelling it as
# present would hand the evaluation a target that is not in the image.
MIN_VISIBLE_RADIUS_PX = 2.0

# Two strata, because "match the real experiment" and "measure physics precisely"
# pull in opposite directions and the dataset has to serve both.
#
#   physics     Enough particles, large enough, to estimate an MSD or a size
#               distribution with tight error bars. This is where Gate 1 and
#               Gate 2 are actually decided: if a probe fails here we need to
#               know it failed because the model cannot do it, not because our
#               estimator was too noisy to tell.
#   experiment  Pinned to what we measured on real experimental video: ~6 px
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
# What is NOT optional is coverage -- before this split our particles were ~3x
# larger than real in apparent pixels, so real data sat outside our distribution
# entirely, and no amount of realism elsewhere rescues that.
STRATA = {
    "physics": dict(radius_px=(8.0, 22.0), n=(6, 16), beam_flux=(2.0, 75.0),
                    pixel_size_nm=(1.5, 4.0), max_coverage=0.09, growth_ratio=5.0),
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
    "experiment": dict(radius_px=(5.0, 8.0), n=(3, 8), beam_flux=(8.0, 12.0),
                       pixel_size_nm=(2.0, 8.0), max_coverage=0.02, growth_ratio=2.0),
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
    n = int(rng.integers(*cfg["n"]))
    margin = 0.15 * min(h, w)
    init_pos = rng.uniform(margin, min(h, w) - margin, (n, 2))
    D = float(rng.uniform(1.0, 4.0))  # px^2 per frame
    pos = simulate_brownian(n, n_frames, dt=1.0, D=D, box=size,
                            init_positions=init_pos, rng=rng)[1]

    t0_growth, dt_growth = 0.0, 1.0
    if process == "ripening":
        radii = ripen_population(rng.uniform(8.0, 22.0, n), n_steps=n_frames, dt=1.0,
                                 K=6.0, min_radius=3.0)[1]
    elif process in ("diffusion", "reaction"):
        r0 = rng.uniform(0.8 * R0_UNITS, 1.2 * R0_UNITS, n)
        t0_growth, dt_growth = growth_time_axis(process, float(r0.mean()), n_frames,
                                                ratio=cfg["growth_ratio"])
        radii = evolve_radii(r0, n_steps=n_frames, dt=dt_growth, mode=process)[1]
    else:  # static: fixed radii, pure diffusion of position
        radii = np.tile(rng.uniform(8.0, 16.0, (n, 1)), (1, n_frames))

    radii = rescale_to_apparent_size(radii, size, float(rng.uniform(*cfg["radius_px"])),
                                     cfg["max_coverage"])
    # A particle below the visibility floor is gone as far as the image is
    # concerned, so drop it from the ground truth rather than label a target that
    # is not there. Only ripening reaches this, on its dissolving tail.
    radii[radii < MIN_VISIBLE_RADIUS_PX] = 0.0
    meta = {"process": process, "stratum": stratum, "exponent": EXPONENT[process],
            "n_particles": n, "D": D, "t0_growth": t0_growth, "dt_growth": dt_growth,
            "beam_flux": float(np.exp(rng.uniform(*np.log(cfg["beam_flux"])))),
            "pixel_size_nm": float(rng.uniform(*cfg["pixel_size_nm"]))}
    return [pos[:, i, :] for i in range(n_frames)], [radii[:, i] for i in range(n_frames)], \
        meta, pos, radii
