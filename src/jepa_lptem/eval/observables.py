"""Physical observables read off particle trajectories / radii / sizes.

These are the *measurement instruments* of the thesis: thin, tested wrappers over
NumPy/SciPy that map raw physical quantities to the numbers we compare against
ground truth (MSD, growth-law exponent, size-distribution distance). They contain
no model — they are validated against closed-form synthetic input in
``tests/test_observables.py`` so that, once model outputs flow through them, the
measurement itself is trusted.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def compute_msd(traj: np.ndarray) -> np.ndarray:
    """Ensemble mean-squared displacement over lag time.

    Parameters
    ----------
    traj : np.ndarray
        Shape ``(N, T, 2)`` — ``N`` particles, ``T`` frames, ``xy`` coordinates.

    Returns
    -------
    np.ndarray
        Shape ``(T - 1,)``. ``msd[k]`` is the mean squared displacement over a lag
        of ``k + 1`` frames, averaged over every particle and every valid start
        frame. NaN positions (frames a gap-closing linker skipped) drop out of
        every pair they touch rather than being filled in. For 2D Brownian motion
        this grows as ``4 D t``.
    """
    traj = np.asarray(traj, dtype=float)
    if traj.ndim != 3 or traj.shape[2] != 2:
        raise ValueError(f"traj must have shape (N, T, 2); got {traj.shape}")
    _, T, _ = traj.shape
    return np.array(
        [
            np.nanmean(np.sum((traj[:, lag:] - traj[:, :-lag]) ** 2, axis=-1))
            for lag in range(1, T)
        ]
    )


def fit_growth_exponent(t: np.ndarray, r: np.ndarray) -> float:
    """Power-law exponent of mean radius vs. time, via a log-log linear fit.

    ``r ~ t**alpha``; returns ``alpha``. ``1/3`` = diffusion-limited coarsening
    (LSW), ``1/2`` = reaction-limited growth. Non-positive ``t`` or ``r`` samples
    (e.g. ``t = 0`` or a fully dissolved particle) are dropped before taking logs.
    """
    t = np.asarray(t, dtype=float)
    r = np.asarray(r, dtype=float)
    mask = (t > 0) & (r > 0)
    if mask.sum() < 2:
        raise ValueError("need >= 2 positive (t, r) samples to fit an exponent")
    return float(np.polyfit(np.log(t[mask]), np.log(r[mask]), 1)[0])


def size_dist_distance(pred_sizes: np.ndarray, true_sizes: np.ndarray) -> float:
    """1-Wasserstein (earth-mover) distance between two particle-size distributions.

    Symmetric, zero iff the distributions are identical, and grows monotonically
    as one distribution is shifted away from the other. This is the headline
    distributional metric for forecast quality (Gate 2).
    """
    pred_sizes = np.asarray(pred_sizes, dtype=float)
    true_sizes = np.asarray(true_sizes, dtype=float)
    return float(wasserstein_distance(pred_sizes, true_sizes))
