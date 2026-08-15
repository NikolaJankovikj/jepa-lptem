"""Verifiable nanoparticle radius dynamics — the ground-truth physics engine.

This module produces the *numbers* (radii and positions over time) that the render
bridge later turns into video. It contains no rendering and no model. Every process
reproduces a known kinetic law, checked in ``tests/test_dynamics.py`` via the same
``fit_growth_exponent`` used to grade the model — closing the loop between the
simulator and the evaluation.

Processes
---------
- ``evolve_radii``: independent single-particle ODEs for clean, exponent-controlled
  data — diffusion-limited growth (r ~ t^1/3), reaction-limited growth (r ~ t^1/2),
  and dissolution (linear shrink to 0).
- ``ripen_population``: coupled, mass-conserving Ostwald ripening (LSW) for a whole
  population — small particles feed large ones, count drops, the mean coarsens, and
  total volume is conserved. This is the physically faithful growth mode.
- ``coalesce``: volume-conserving merge-on-contact over a pre-generated walk. The
  only process where particles interact, and the only source of *events* — a
  discrete, exactly-timestamped thing that happens, which is what RQ2 (does
  prediction error rise before an event?) needs to be gradeable at all.
"""

from __future__ import annotations

import numpy as np

# Rate constants per single-particle mode (arbitrary units; set for clean exponents).
GROWTH_K = {"diffusion": 1.0, "reaction": 1.0, "dissolution": 0.05}


def simulate_brownian(
    n_particles: int,
    n_steps: int,
    dt: float,
    D: float,
    init_positions: np.ndarray | None = None,
    box: tuple[float, float] | None = None,
    drift: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
):
    """2D Brownian trajectories plus coherent stage drift — the position ground truth.

    Each coordinate takes Gaussian steps of variance ``2 D dt``, so the ensemble
    mean-squared displacement grows as ``MSD(t) = 4 D t`` (checked against
    ``eval.observables.compute_msd`` in the tests) *once the drift is removed*.

    The Brownian part is a martingale: the best predictor of the next frame is the
    current one, which is exactly why the model can only be graded on *statistics*
    (MSD, displacement distribution), never on predicting an individual step. Drift
    is the opposite — a rigid, constant-velocity translation of the whole field,
    and the one component of real LPTEM motion that a forecaster can actually get
    right. Without it the generator hands persistence an optimal strategy on the
    entire position axis, which would make the Gate-2 comparison against
    persistence unwinnable by construction rather than by evidence.

    Drift is applied *after* the reflecting fold, so it stays an exact rigid
    translation: pairwise distances are untouched (coalescence is unaffected) and
    the eval removes it exactly by subtracting ``drift * t``. The caller is
    responsible for insetting ``box`` so the drifted particles stay in frame — see
    :func:`jepa_lptem.sim.dataset.build_clip`.

    Parameters
    ----------
    n_particles, n_steps : int
        Number of particles ``N`` and frames ``T``.
    dt, D : float
        Timestep and diffusion coefficient (px^2 per unit time).
    init_positions : np.ndarray or None
        Shape ``(N, 2)`` starting ``(row, col)``; defaults to the origin. If
        ``box`` is given without ``init_positions``, particles start at the centre.
    box : (float, float) or None
        ``(height, width)``; when set, coordinates reflect off ``[0, size]`` walls
        so particles stay in frame. ``None`` = free diffusion (for clean MSD).
    drift : np.ndarray or None
        Shape ``(2,)`` stage-drift velocity in px per unit time, added coherently
        to every particle. ``None`` = no drift.
    rng : np.random.Generator or None
        Seeded generator for reproducibility.

    Returns
    -------
    (t, pos) : tuple[np.ndarray, np.ndarray]
        ``t`` shape ``(n_steps,)``; ``pos`` shape ``(N, n_steps, 2)``.
    """
    if rng is None:
        rng = np.random.default_rng()
    if init_positions is None:
        if box is not None:
            init_positions = np.tile(
                [box[0] / 2.0, box[1] / 2.0], (n_particles, 1)
            )
        else:
            init_positions = np.zeros((n_particles, 2))
    init_positions = np.asarray(init_positions, dtype=float)

    sigma = np.sqrt(2.0 * D * dt)
    steps = rng.normal(0.0, sigma, size=(n_particles, n_steps, 2))
    steps[:, 0, :] = 0.0
    pos = init_positions[:, None, :] + np.cumsum(steps, axis=1)

    if box is not None:
        # reflect into [0, size] on each axis via the triangle-wave folding map
        for axis, size in enumerate(box):
            period = 2.0 * size
            folded = np.mod(pos[..., axis], period)
            pos[..., axis] = np.where(folded > size, period - folded, folded)

    t = np.arange(n_steps) * dt
    if drift is not None:
        # after the fold, so drift stays a rigid translation rather than something
        # the walls can reverse for one particle and not another
        pos = pos + np.asarray(drift, dtype=float)[None, None, :] * t[None, :, None]

    return t, pos


def evolve_radii(r0: np.ndarray, n_steps: int, dt: float, mode: str,
                 k: np.ndarray | float | None = None, substeps: int = 10):
    """Integrate per-particle radius ODEs (independent particles).

    Parameters
    ----------
    r0 : np.ndarray
        Shape ``(N,)`` initial radii.
    n_steps, dt : int, float
        Number of frames and timestep.
    mode : {"diffusion", "reaction", "dissolution"}
        ``diffusion``: ``dr/dt = K / r^2`` -> ``r ~ t^(1/3)`` (diffusion-limited).
        ``reaction`` : ``dr/dt = K / r``   -> ``r ~ t^(1/2)`` (reaction-limited).
        ``dissolution``: ``dr/dt = -K``    -> linear shrink, clamped at 0.

        These are *single-particle* ODEs chosen to produce a clean, exactly-known
        exponent -- they are not the coupled LSW population law, which lives in
        :func:`ripen_population`.
    k : np.ndarray, float or None
        Rate constant. Scalar, or shape ``(N,)`` for a **per-particle** rate.
        ``None`` = ``GROWTH_K[mode]``. A per-particle ``k`` is what stops the size
        distribution collapsing: these ODEs are contractions on ``r``
        (``dr_final/dr0 = (r0/r_final)^(p-1)``), so a population sharing one ``k``
        converges to a single radius -- measured relative sd fell 0.095 -> 0.0007
        over a 5x growth clip, i.e. a delta function, on the exact observable
        (size distribution) the evaluation grades. With ``r0_i`` fixed to
        ``(p k_i t0)^(1/p)`` every particle sits on the same pure power law from a
        common nucleation time, so the ratio ``r_i / r_j`` is *constant* over the
        clip and the ensemble mean is exactly ``t^(1/p)``.
    substeps : int
        Euler substeps per frame (as in :func:`ripen_population`). These ODEs are
        stiffest at small ``r`` -- ``dr/dt = K/r^2`` can move the radius tens of
        percent in a single frame -- and the integration error shows up directly as
        a biased growth exponent. Substepping is what makes the measured exponent
        the physics rather than the integrator.

    Returns
    -------
    (t, r) : tuple[np.ndarray, np.ndarray]
        ``t`` shape ``(n_steps,)``; ``r`` shape ``(N, n_steps)``.
    """
    if mode not in GROWTH_K:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(GROWTH_K)}")
    K = GROWTH_K[mode] if k is None else np.asarray(k, dtype=float)
    r0 = np.asarray(r0, dtype=float)
    r = np.zeros((len(r0), n_steps))
    r[:, 0] = r0
    cur = r0.copy()
    sub_dt = dt / substeps
    for i in range(1, n_steps):
        for _ in range(substeps):
            safe = np.maximum(cur, 1e-6)
            if mode == "diffusion":
                dr = K / safe ** 2
            elif mode == "reaction":
                dr = K / safe
            else:  # dissolution
                dr = -K * np.ones_like(cur)
            cur = np.maximum(cur + dr * sub_dt, 0.0)
        r[:, i] = cur
    return np.arange(n_steps) * dt, r


def ripen_population(
    r0: np.ndarray,
    n_steps: int,
    dt: float,
    K: float = 0.3,
    min_radius: float = 0.1,
    substeps: int = 10,
):
    """Coupled, mass-conserving Ostwald ripening (LSW) of a whole population.

    Gibbs--Thomson exchange drives growth around the critical radius ``r* = <r>``
    (the mean over surviving particles): ``dr/dt = (K / r) (1/r* - 1/r)``, so
    sub-critical particles shrink and super-critical ones grow. Particles that fall
    below ``min_radius`` dissolve (set to 0). Total volume is renormalized to its
    initial value each step, so mass is conserved exactly while it redistributes
    from vanishing small particles to surviving large ones — the mean coarsens.

    Returns
    -------
    (t, r) : tuple[np.ndarray, np.ndarray]
        ``t`` shape ``(n_steps,)``; ``r`` shape ``(N, n_steps)`` with dissolved
        particles held at 0.
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = float(np.sum(r0 ** 3))
    r = np.zeros((len(r0), n_steps))
    r[:, 0] = r0
    cur = r0.copy()
    sub_dt = dt / substeps
    for i in range(1, n_steps):
        for _ in range(substeps):
            alive = cur > 0
            if alive.sum() <= 1:
                break
            rstar = cur[alive].mean()
            dr = np.zeros_like(cur)
            dr[alive] = (K / cur[alive]) * (1.0 / rstar - 1.0 / cur[alive])
            cur = cur + dr * sub_dt
            cur[cur < min_radius] = 0.0
        # renormalize surviving volume back to v0 (mass conserved, redistributed)
        v_alive = float(np.sum(cur ** 3))
        if v_alive > 0:
            cur = cur * (v0 / v_alive) ** (1 / 3)
        r[:, i] = cur
    return np.arange(n_steps) * dt, r


def coalesce(pos: np.ndarray, r0: np.ndarray):
    """Volume-conserving merge-on-contact over a pre-generated set of walks.

    Two particles touch when their centres are closer than ``r_i + r_j``. The
    larger absorbs the smaller: ``r_new = (r_i^3 + r_j^3)^(1/3)`` (volume conserved
    exactly) and the survivor's centre jumps to the volume-weighted centroid, which
    is a real, visible displacement rather than a bookkeeping detail. The absorbed
    particle is set to radius 0 for the rest of the clip — the same "0 = absent"
    convention :func:`ripen_population` already uses — and its stored positions
    past that frame are meaningless.

    This is a *post-process on independent walks*, not a many-body simulation: the
    survivor keeps its own trajectory (shifted by the centroid jump) instead of
    both parents' momenta being combined. At the sizes and densities we generate,
    the centroid jump is the part that shows up in a frame; a proper two-body
    update would move the merged particle by a sub-pixel amount more. What this
    buys is that the walk still comes from :func:`simulate_brownian`, so the box
    reflection, the drift, and the ``MSD = 4Dt`` label all keep holding.

    Because volume is conserved and nothing grows, total *area* strictly decreases
    at every merge — ``(a + b)^(2/3) < a^(2/3) + b^(2/3)`` — so peak coverage is at
    frame 0 and the caller can size particles against the coverage cap up front.

    Parameters
    ----------
    pos : np.ndarray
        Shape ``(N, T, 2)`` pre-generated trajectories.
    r0 : np.ndarray
        Shape ``(N,)`` initial radii in pixels.

    Returns
    -------
    (pos, radii, events) : tuple[np.ndarray, np.ndarray, list[list[int]]]
        ``pos`` ``(N, T, 2)`` with centroid jumps applied, ``radii`` ``(N, T)``
        (0 = absorbed), and ``events`` as ``[frame, survivor, absorbed]`` rows in
        chronological order — the exact merge times RQ2 grades lead-time against.
    """
    pos = np.array(pos, dtype=float, copy=True)
    n, T, _ = pos.shape
    cur = np.asarray(r0, dtype=float).copy()
    radii = np.zeros((n, T))
    events: list[list[int]] = []

    for frame in range(T):
        while True:  # cascade: a merge can bring the survivor into contact with a third
            alive = np.flatnonzero(cur > 0)
            hit = None
            for a in range(len(alive)):
                for b in range(a + 1, len(alive)):
                    i, j = int(alive[a]), int(alive[b])
                    if np.hypot(*(pos[i, frame] - pos[j, frame])) < cur[i] + cur[j]:
                        hit = (i, j) if cur[i] >= cur[j] else (j, i)
                        break
                if hit:
                    break
            if not hit:
                break
            i, j = hit  # i survives (the larger), j is absorbed
            vi, vj = cur[i] ** 3, cur[j] ** 3
            # volume-weighted centroid: the survivor shifts toward what it ate, and
            # keeps that offset for the rest of the clip
            pos[i, frame:] += (vj / (vi + vj)) * (pos[j, frame] - pos[i, frame])
            cur[i] = (vi + vj) ** (1 / 3)
            cur[j] = 0.0
            events.append([frame, i, j])
        radii[:, frame] = cur

    return pos, radii, events
