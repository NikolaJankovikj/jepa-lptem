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
- ``merge_on_contact``: volume-conserving coalescence of overlapping particles.
- ``ripen_population``: coupled, mass-conserving Ostwald ripening (LSW) for a whole
  population — small particles feed large ones, count drops, the mean coarsens, and
  total volume is conserved. This is the physically faithful growth mode.
"""

from __future__ import annotations

import numpy as np

# Rate constants per single-particle mode (arbitrary units; set for clean exponents).
GROWTH_K = {"diffusion": 1.0, "reaction": 1.0, "dissolution": 0.05}


def evolve_radii(r0: np.ndarray, n_steps: int, dt: float, mode: str):
    """Integrate per-particle radius ODEs (independent particles).

    Parameters
    ----------
    r0 : np.ndarray
        Shape ``(N,)`` initial radii.
    n_steps, dt : int, float
        Number of frames and timestep.
    mode : {"diffusion", "reaction", "dissolution"}
        ``diffusion``: ``dr/dt = K / r^2`` -> ``r ~ t^(1/3)`` (LSW, diffusion-limited).
        ``reaction`` : ``dr/dt = K / r``   -> ``r ~ t^(1/2)`` (reaction-limited).
        ``dissolution``: ``dr/dt = -K``    -> linear shrink, clamped at 0.

    Returns
    -------
    (t, r) : tuple[np.ndarray, np.ndarray]
        ``t`` shape ``(n_steps,)``; ``r`` shape ``(N, n_steps)``.
    """
    if mode not in GROWTH_K:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(GROWTH_K)}")
    K = GROWTH_K[mode]
    r0 = np.asarray(r0, dtype=float)
    r = np.zeros((len(r0), n_steps))
    r[:, 0] = r0
    for i in range(1, n_steps):
        cur = r[:, i - 1]
        safe = np.maximum(cur, 1e-6)
        if mode == "diffusion":
            dr = K / safe ** 2
        elif mode == "reaction":
            dr = K / safe
        else:  # dissolution
            dr = -K * np.ones_like(cur)
        r[:, i] = np.maximum(cur + dr * dt, 0.0)
    return np.arange(n_steps) * dt, r


def merge_on_contact(pos: np.ndarray, rad: np.ndarray):
    """Merge every overlapping pair into one volume-conserving particle.

    Two particles overlap when ``dist < r_i + r_j``. The merged particle has
    ``r_new = (r_i^3 + r_j^3)^(1/3)`` (volume conserved) at the volume-weighted mean
    position. Repeats until no overlaps remain (a merge can create a new overlap).

    Parameters
    ----------
    pos : np.ndarray
        Shape ``(N, 2)`` positions.
    rad : np.ndarray
        Shape ``(N,)`` radii.

    Returns
    -------
    (pos, rad) : tuple[np.ndarray, np.ndarray]
        Reduced arrays after all merges.
    """
    pos = np.asarray(pos, dtype=float).copy()
    rad = np.asarray(rad, dtype=float).copy()
    merged = True
    while merged and len(rad) > 1:
        merged = False
        for i in range(len(rad)):
            for j in range(i + 1, len(rad)):
                if np.linalg.norm(pos[i] - pos[j]) < rad[i] + rad[j]:
                    vi, vj = rad[i] ** 3, rad[j] ** 3
                    v = vi + vj
                    pos[i] = (vi * pos[i] + vj * pos[j]) / v
                    rad[i] = v ** (1 / 3)
                    pos = np.delete(pos, j, axis=0)
                    rad = np.delete(rad, j)
                    merged = True
                    break
            if merged:
                break
    return pos, rad


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
