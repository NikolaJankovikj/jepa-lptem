import numpy as np

from jepa_lptem.eval.observables import compute_msd, fit_growth_exponent
from jepa_lptem.sim.dynamics import (
    GROWTH_K,
    evolve_radii,
    merge_on_contact,
    ripen_population,
    simulate_brownian,
)


# --- Brownian motion (the position ground truth for MSD) ----------------------

def test_brownian_msd_matches_4Dt():
    # 2D free diffusion: MSD(t) = 4 D t, so d(MSD)/dt = 4 D
    rng = np.random.default_rng(0)
    D, dt = 0.5, 1.0
    t, pos = simulate_brownian(n_particles=3000, n_steps=60, dt=dt, D=D, rng=rng)
    assert pos.shape == (3000, 60, 2)
    msd = compute_msd(pos)
    lags = np.arange(1, 60) * dt
    slope = np.polyfit(lags, msd, 1)[0]
    assert abs(slope - 4 * D) < 0.15


def test_brownian_starts_at_given_positions():
    init = np.array([[10.0, 20.0], [30.0, 40.0]])
    _, pos = simulate_brownian(
        n_particles=2, n_steps=5, dt=1.0, D=1.0,
        init_positions=init, rng=np.random.default_rng(1),
    )
    assert np.allclose(pos[:, 0], init)


def test_brownian_stays_inside_box_when_bounded():
    _, pos = simulate_brownian(
        n_particles=50, n_steps=200, dt=1.0, D=5.0,
        box=(64, 64), rng=np.random.default_rng(2),
    )
    assert pos.min() >= 0.0
    assert pos[..., 0].max() <= 64.0
    assert pos[..., 1].max() <= 64.0


# --- single-particle kinetic laws (independent ODEs) --------------------------

def test_diffusion_limited_growth_exponent():
    # dr/dt = K / r^2  ->  r ~ t^(1/3)  (LSW)
    t, r = evolve_radii(r0=np.array([2.0]), n_steps=500, dt=1.0, mode="diffusion")
    assert abs(fit_growth_exponent(t[50:], r[0, 50:]) - 1 / 3) < 0.05


def test_reaction_limited_growth_exponent():
    # dr/dt = K / r  ->  r ~ t^(1/2)
    t, r = evolve_radii(r0=np.array([2.0]), n_steps=500, dt=1.0, mode="reaction")
    assert abs(fit_growth_exponent(t[50:], r[0, 50:]) - 0.5) < 0.05


def test_dissolution_shrinks_to_zero():
    # dr/dt = -K  ->  linear shrink, clamped at 0
    t, r = evolve_radii(r0=np.array([5.0]), n_steps=2000, dt=1.0, mode="dissolution")
    assert r[0, -1] == 0.0
    assert np.all(np.diff(r[0]) <= 0)  # monotonically non-increasing


def test_evolve_radii_shapes_and_multiple_particles():
    t, r = evolve_radii(r0=np.array([1.0, 2.0, 3.0]), n_steps=100, dt=0.5, mode="diffusion")
    assert t.shape == (100,)
    assert r.shape == (3, 100)
    assert np.allclose(r[:, 0], [1.0, 2.0, 3.0])


# --- coalescence (volume-conserving fusion) -----------------------------------

def test_coalescence_conserves_volume():
    pos = np.array([[0.0, 0.0], [1.0, 0.0]])
    rad = np.array([3.0, 4.0])
    pos2, rad2 = merge_on_contact(pos, rad)
    assert len(rad2) == 1
    assert abs(rad2[0] ** 3 - (3.0 ** 3 + 4.0 ** 3)) < 1e-9
    # position is the volume-weighted mean
    expected_x = (27 * 0.0 + 64 * 1.0) / 91
    assert abs(pos2[0, 0] - expected_x) < 1e-9


def test_no_merge_when_far_apart():
    pos = np.array([[0.0, 0.0], [100.0, 0.0]])
    rad = np.array([3.0, 4.0])
    pos2, rad2 = merge_on_contact(pos, rad)
    assert len(rad2) == 2


def test_multiple_merges_cascade():
    # three touching particles collapse to one; total volume conserved
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    rad = np.array([2.0, 2.0, 2.0])
    pos2, rad2 = merge_on_contact(pos, rad)
    assert len(rad2) == 1
    assert abs(rad2[0] ** 3 - 3 * 2.0 ** 3) < 1e-9


# --- mass-conserving Ostwald ripening (coupled population) --------------------

def test_ripening_conserves_total_volume():
    rng = np.random.default_rng(0)
    r0 = rng.uniform(1.0, 4.0, 40)
    _, r = ripen_population(r0, n_steps=800, dt=1.0)
    v0 = np.sum(r0 ** 3)
    vT = np.sum(r[:, -1] ** 3)
    assert abs(vT - v0) / v0 < 1e-6  # mass conserved by construction


def test_ripening_coarsens_and_reduces_count():
    rng = np.random.default_rng(1)
    r0 = rng.uniform(1.0, 4.0, 40)
    _, r = ripen_population(r0, n_steps=800, dt=1.0)
    alive0 = int(np.sum(r[:, 0] > 0))
    aliveT = int(np.sum(r[:, -1] > 0))
    assert aliveT < alive0  # small particles dissolved away
    mean0 = r[:, 0][r[:, 0] > 0].mean()
    meanT = r[:, -1][r[:, -1] > 0].mean()
    assert meanT > mean0  # surviving mean radius coarsened upward


def test_evolve_radii_matches_closed_form_solution():
    """Analytic oracle: both growth ODEs have exact solutions, so compare to them.

    ``dr/dt = K/r^2`` integrates to ``r^3 = r0^3 + 3Kt`` and ``dr/dt = K/r`` to
    ``r^2 = r0^2 + 2Kt``. Fitting an exponent only says the curve is *a* power law;
    this pins the actual trajectory. It is also the check that catches integrator
    error -- single-step Euler misses by ~7-11% here, which silently biases the
    growth exponent the whole dataset is labelled with.
    """
    for mode, power in (("diffusion", 3), ("reaction", 2)):
        r0 = np.array([2.0, 5.0, 11.0])
        t, r = evolve_radii(r0, n_steps=60, dt=3.0, mode=mode)
        exact = (r0[:, None] ** power + power * GROWTH_K[mode] * t[None, :]) ** (1 / power)
        assert np.abs(r - exact).max() / exact.max() < 0.01, mode
