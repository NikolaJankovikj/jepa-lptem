import numpy as np

from jepa_lptem.eval.observables import compute_msd, fit_growth_exponent
from jepa_lptem.sim.dynamics import (
    GROWTH_K,
    coalesce,
    evolve_radii,
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


def test_per_particle_rate_constant_keeps_the_size_ratio_fixed():
    """``r_i / r_j`` must be constant when ``r0_i`` is pinned to ``k_i``.

    That is the whole point of the per-particle rate: with a shared ``k`` these ODEs
    are contractions and the population converges to one radius, wiping out the size
    distribution the evaluation grades. Pinning ``r0_i = (p k_i t0)^(1/p)`` puts
    every particle on the same pure power law, so the spread is invariant.
    """
    for mode, p in (("diffusion", 3), ("reaction", 2)):
        t0 = 40.0
        k = np.array([0.5, 1.0, 2.0])
        r0 = (p * k * t0) ** (1 / p)
        _, r = evolve_radii(r0, n_steps=80, dt=5.0, mode=mode, k=k)
        ratio = r / r[0]                       # each particle relative to the first
        assert np.allclose(ratio, ratio[:, :1], rtol=2e-3), mode


def test_evolve_radii_shapes_and_multiple_particles():
    t, r = evolve_radii(r0=np.array([1.0, 2.0, 3.0]), n_steps=100, dt=0.5, mode="diffusion")
    assert t.shape == (100,)
    assert r.shape == (3, 100)
    assert np.allclose(r[:, 0], [1.0, 2.0, 3.0])


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


# --- coalescence (the only interacting process, and the only source of events) ---

def test_coalesce_merges_touching_particles_conserving_volume():
    # two particles closing head-on: they must merge exactly when the centres come
    # within r_i + r_j, and the survivor must carry both volumes
    T = 10
    pos = np.zeros((2, T, 2))
    pos[0, :, 1] = np.linspace(0.0, 12.0, T)    # approaching along the column axis
    pos[1, :, 1] = 20.0
    r0 = np.array([4.0, 6.0])
    out, radii, events = coalesce(pos, r0)
    assert len(events) == 1
    frame, survivor, absorbed = events[0]
    assert survivor == 1 and absorbed == 0      # the larger one survives
    assert radii[absorbed, frame] == 0.0
    assert abs(radii[survivor, frame] ** 3 - (4.0 ** 3 + 6.0 ** 3)) < 1e-9
    # centroid jumps toward what it ate, by the volume-weighted fraction
    gap = pos[0, frame, 1] - pos[1, frame, 1]
    assert np.isclose(out[survivor, frame, 1] - pos[1, frame, 1],
                      gap * 4.0 ** 3 / (4.0 ** 3 + 6.0 ** 3))


def test_coalesce_leaves_distant_particles_alone():
    pos = np.tile(np.array([[0.0, 0.0], [100.0, 100.0]])[:, None, :], (1, 12, 1))
    _, radii, events = coalesce(pos, np.array([5.0, 5.0]))
    assert events == []
    assert np.all(radii == 5.0)


def test_coalesce_cascades_within_one_frame():
    # three mutually touching particles must collapse to one in the same frame,
    # not one per frame -- a merged survivor is bigger and can reach a third
    # 0 and 1 touch (gap 7 < 8); the third is out of reach at 12 until the merged
    # survivor grows to r=5.04 and its centroid shifts to 3.5, closing the gap
    pos = np.tile(np.array([[0.0, 0.0], [0.0, 7.0], [0.0, 12.0]])[:, None, :], (1, 4, 1))
    _, radii, events = coalesce(pos, np.array([4.0, 4.0, 4.0]))
    assert [e[0] for e in events] == [0, 0]
    assert int((radii[:, -1] > 0).sum()) == 1
    assert abs(radii[:, -1].max() ** 3 - 3 * 4.0 ** 3) < 1e-9


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
