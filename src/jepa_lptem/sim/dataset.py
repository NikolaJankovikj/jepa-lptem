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

# Fraction of the frame covered by gold at the clip's densest frame. Real LPTEM is
# sparse; this keeps particles resolvable end to end. Drawn per clip for variety.
AREA_FRACTION = (0.04, 0.09)
# Final/initial radius over a clip for the growth modes. Large enough that the
# asymptotic exponent dominates the finite-r0 offset, small enough that particles
# are still several pixels wide at t=0 after the density rescale. Calibration knob:
# raise it for purer exponents, lower it for better early-frame visibility.
GROWTH_RATIO = 5.0
R0_UNITS = 2.0  # arbitrary pre-rescale radius unit; only the ratio matters


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


def rescale_to_density(radii: np.ndarray, image_size, frac: float) -> np.ndarray:
    """Scale all radii so peak gold coverage is ``frac`` of the frame.

    Applied to the whole ``(N, T)`` array by a single constant, so it changes the
    pixel size of the physics and nothing else: the log-log growth slope and the
    MSD are both invariant under a uniform rescale of radius. Normalizing at the
    *densest* frame (not the last) means ripening, which conserves mass while the
    count drops, is bounded at t=0 where it is fullest.
    """
    peak_area = float(np.pi * (radii ** 2).sum(axis=0).max())
    if peak_area <= 0:
        return radii
    return radii * np.sqrt(frac * image_size[0] * image_size[1] / peak_area)


def build_clip(process: str, n_frames: int, size: tuple[int, int], rng: np.random.Generator):
    """Return ``(positions_t, radii_t, meta, pos, radii)`` for one clip.

    ``positions_t``/``radii_t`` are the per-frame lists the renderer consumes;
    ``pos`` ``(N, T, 2)`` and ``radii`` ``(N, T)`` are the same data as arrays,
    saved verbatim as the ground truth. ``meta`` carries the physical parameters
    the evaluation needs to grade against (``D``, ``exponent``, ``dt_growth``).
    """
    h, w = size
    n = int(rng.integers(6, 16))
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
        t0_growth, dt_growth = growth_time_axis(process, float(r0.mean()), n_frames)
        radii = evolve_radii(r0, n_steps=n_frames, dt=dt_growth, mode=process)[1]
    else:  # static: fixed radii, pure diffusion of position
        radii = np.tile(rng.uniform(8.0, 16.0, (n, 1)), (1, n_frames))

    radii = rescale_to_density(radii, size, float(rng.uniform(*AREA_FRACTION)))
    meta = {"process": process, "exponent": EXPONENT[process], "n_particles": n,
            "D": D, "t0_growth": t0_growth, "dt_growth": dt_growth}
    return [pos[:, i, :] for i in range(n_frames)], [radii[:, i] for i in range(n_frames)], \
        meta, pos, radii
