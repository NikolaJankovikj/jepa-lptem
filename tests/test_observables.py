import numpy as np

from jepa_lptem.eval.observables import (
    compute_msd,
    fit_growth_exponent,
    size_dist_distance,
)


def test_msd_recovers_diffusion_coefficient():
    # 2D Brownian motion: ensemble MSD(t) = 4 D t. Recover D from the slope.
    rng = np.random.default_rng(0)
    D, dt, T, N = 0.5, 1.0, 200, 500
    steps = rng.normal(0, np.sqrt(2 * D * dt), (N, T, 2))
    traj = np.cumsum(steps, axis=1)  # (N particles, T frames, xy)
    msd = compute_msd(traj)  # (T-1,)
    assert msd.shape == (T - 1,)
    slope = np.polyfit(np.arange(1, 51), msd[:50], 1)[0]
    assert abs(slope - 4 * D) / (4 * D) < 0.15


def test_msd_is_zero_for_static_particles():
    traj = np.tile(np.array([[3.0, -1.0]]), (10, 20, 1))  # 10 particles, never move
    msd = compute_msd(traj)
    assert np.allclose(msd, 0.0)


def test_growth_exponent_lsw():
    # diffusion-limited coarsening (LSW): r ~ t^(1/3)
    t = np.arange(1, 200, dtype=float)
    r = (3.0 * t) ** (1 / 3)
    assert abs(fit_growth_exponent(t, r) - 1 / 3) < 0.02


def test_growth_exponent_reaction_limited():
    # reaction-limited growth: r ~ t^(1/2)
    t = np.arange(1, 200, dtype=float)
    r = (3.0 * t) ** 0.5
    assert abs(fit_growth_exponent(t, r) - 0.5) < 0.02


def test_growth_exponent_ignores_nonpositive():
    # zeros/negatives (e.g. a dissolved particle) must not crash the log-log fit
    t = np.arange(0, 200, dtype=float)  # starts at t=0
    r = np.concatenate([[0.0], (3.0 * np.arange(1, 200, dtype=float)) ** (1 / 3)])
    assert abs(fit_growth_exponent(t, r) - 1 / 3) < 0.02


def test_size_dist_distance_zero_on_identity():
    s = np.random.default_rng(1).gamma(4.0, 1.0, 300)
    assert size_dist_distance(s, s) < 1e-9


def test_size_dist_distance_grows_with_shift():
    rng = np.random.default_rng(2)
    a = rng.gamma(4.0, 1.0, 500)
    near = size_dist_distance(a, a + 1.0)
    far = size_dist_distance(a, a + 5.0)
    assert 0 < near < far
