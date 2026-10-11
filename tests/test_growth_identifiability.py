"""The growth extrapolators on trajectories with a known answer."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from growth_identifiability import HORIZONS, extrapolate, growth_detection, mean_radius

T = 32


def trajectories(n: int = 200):
    """Half static, half growing at a per-clip rate: noise-free, so the context fully
    determines each future."""
    rng = np.random.default_rng(0)
    rate = np.r_[np.zeros(n // 2), rng.uniform(0.02, 0.06, n - n // 2)]
    r0 = rng.uniform(8.0, 16.0, n)
    truth = r0[:, None] * (1 + rate[:, None] * np.arange(T)) ** 0.5
    process = np.where(rate > 0, "reaction", "static")
    return truth, process, rng.random(n) < 0.5


def test_mean_radius_ignores_absent_particles():
    rad = np.array([[10.0, 0.0], [20.0, 0.0]])
    out = mean_radius(rad)
    assert out[0] == 15.0 and np.isnan(out[1])


def test_ridge_reads_each_clip_rate_and_the_pooled_trend_cannot():
    truth, process, train = trajectories()
    f = extrapolate(truth, truth, process, train)
    err = {
        k: np.abs(v - truth[:, 7 + HORIZONS])[~train, -1].mean() for k, v in f.items()
    }
    assert err["ridge"] < 0.4 * err["persistence"]  # 0.29 measured: a 20-term ridge, not an exact law fit
    assert err["ridge"] < 0.5 * err["pooled_trend"]
    static = ~train & (process == "static")
    assert np.allclose(f["persistence"][static], truth[static, 7:8])


def test_growth_detection_is_perfect_without_noise():
    truth, process, train = trajectories()
    assert growth_detection(truth, process, train)["accuracy"] == 1.0
