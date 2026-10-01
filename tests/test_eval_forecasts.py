"""Step 10 / Gate 2: the pure pieces of the forecast evaluation."""

import sys
from pathlib import Path

import numpy as np
import torch

from jepa_lptem.models.predictor import LatentForecaster

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from eval_forecasts import (
    forecasts,
    gaussian_w2,
    gt_moments,
    moments_from_observables,
    snippet_exponent,
)

S, D = 4, 8


def test_gaussian_w2_is_zero_on_identity_and_adds_both_moments():
    assert gaussian_w2(10.0, 2.0, 10.0, 2.0) == 0.0
    assert np.isclose(gaussian_w2(10.0, 2.0, 13.0, 6.0), 5.0)  # sqrt(3^2 + 4^2)


def test_moments_from_observables_recover_mean_and_spread():
    # count, mean radius and total area pin the first two moments: E[r^2] = A / (pi n)
    radii = np.array([6.0, 10.0, 14.0])
    obs = np.array([len(radii), radii.mean(), np.pi * (radii**2).sum()])
    mean, sd = moments_from_observables(obs)
    assert np.isclose(mean, radii.mean()) and np.isclose(sd, radii.std())
    assert moments_from_observables(np.array([0.0, 0.0, 0.0])) == (
        0.0,
        0.0,
    )  # empty frame


def test_gt_moments_use_visible_particles_only():
    assert gt_moments(np.array([5.0, 0.0, 7.0])) == (6.0, 1.0)
    assert gt_moments(np.zeros(3)) == (0.0, 0.0)


def test_snippet_exponent_recovers_a_power_law():
    t = np.linspace(50.0, 120.0, 10)
    assert abs(snippet_exponent(t, 2.0 * t**0.5) - 0.5) < 1e-6
    assert np.isnan(snippet_exponent(t, np.zeros(10)))  # nothing to fit


def test_forecasts_build_every_baseline_from_the_right_frames():
    torch.manual_seed(0)
    feats = np.random.default_rng(0).normal(size=(20, S, D)).astype(np.float32)
    stats = (np.zeros(D, np.float32), np.ones(D, np.float32))
    model = LatentForecaster(enc_dim=D, dim=16, depth=1, heads=2, n_spatial=S)
    out = forecasts(model, feats, start=3, ctx=4, horizon=5, stats=stats)
    context = feats[3:7]
    assert all(v.shape == (5, S, D) for v in out.values())
    assert np.allclose(out["persistence"], context[-1])
    assert np.allclose(out["ctxmean"], context.mean(axis=0))
    assert np.allclose(out["floor"], feats[7:12])  # the true future frames
    # an untrained model is persistence by construction: the rollout must agree
    assert np.allclose(out["model"], out["persistence"], atol=1e-5)


def test_bootstrap_ci_resamples_clips_not_snippets():
    from gate2_followup import bootstrap_ci

    assert bootstrap_ci([2.0] * 6, list("aabbcc")) == [2.0, 2.0]
    # duplicating every snippet inside its clip adds no information: same CI
    per_clip = bootstrap_ci([0.0, 1.0, 2.0, 3.0], list("abcd"))
    duplicated = bootstrap_ci([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0], list("aabbccdd"))
    assert np.allclose(per_clip, duplicated)


def test_size_test_is_untestable_when_the_floor_has_no_edge():
    from gate2_followup import size_test

    clips = [f"c{i}" for i in range(20)]
    persist = np.full(20, 3.0)
    noise = np.random.default_rng(0).normal(0, 0.1, 20)
    unpowered = size_test(persist, persist - 1.0, persist + noise, clips)
    assert not unpowered["powered"] and unpowered["pass"] is None
    powered = size_test(persist, persist - 0.5 + noise, persist - 1.0 + noise, clips)
    assert powered["powered"] and powered["pass"] is True
    assert abs(powered["gap_closed"] - 0.5) < 0.1
