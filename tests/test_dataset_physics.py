"""Regression guard: generated clips must still carry the physics they claim.

`scripts/verify_clips.py` proves this by hand on a rendered clip, with pictures.
This is the automated half: it asserts that the *ground truth* coming out of
`build_clip` still obeys the kinetic laws the dataset is labelled with, so a
tweak to `dynamics.py` or to the generation parameters fails here rather than
silently poisoning a 50 GB overnight run on the remote.

Pixel-level correctness is covered separately by `tests/test_render_bridge.py`, so
most of these skip rendering and stay fast. The exception is the brightness-leak
test, which has to look at actual pixels to see the leak at all.
"""

import numpy as np
import pytest

from jepa_lptem.eval.observables import compute_msd, fit_growth_exponent
from jepa_lptem.sim.dataset import PROCESSES, build_clip
from jepa_lptem.sim.render_bridge import render_video, to_uint8


@pytest.mark.parametrize("process,expected", [("diffusion", 1 / 3), ("reaction", 1 / 2)])
def test_growth_exponent_recovered_from_generated_clip(process, expected):
    rng = np.random.default_rng(0)
    _, _, meta, _, radii = build_clip(process, 64, (512, 512), rng)
    # time-since-nucleation, not time-since-frame-0 (see growth_time_axis)
    t = meta["t0_growth"] + np.arange(64) * meta["dt_growth"]
    alive = radii > 0
    mean_r = np.array([radii[alive[:, k], k].mean() for k in range(64)])
    alpha = fit_growth_exponent(t, mean_r)
    assert abs(alpha - expected) < 0.005, f"{process}: alpha={alpha:.4f}"


def test_msd_of_generated_positions_matches_4Dt():
    rng = np.random.default_rng(1)
    _, _, meta, pos, _ = build_clip("static", 64, (512, 512), rng)
    msd = compute_msd(pos)
    lags = np.arange(1, 64)
    short = 16  # few particles per clip make long lags noisy; short lags give 4D
    slope = float(np.polyfit(lags[:short], msd[:short], 1)[0])
    assert abs(slope / (4 * meta["D"]) - 1.0) < 0.4, f"slope={slope:.3f} D={meta['D']:.3f}"


@pytest.mark.parametrize("process", PROCESSES)
def test_particles_stay_sparse_enough_to_be_resolvable(process):
    # The whole point of the dataset is a recoverable answer key. If particles grow
    # until they overlap, no centroid is extractable and a probe failure becomes
    # uninterpretable. Cap peak coverage well below the touching limit.
    rng = np.random.default_rng(2)
    _, _, _, _, radii = build_clip(process, 48, (512, 512), rng)
    peak = float(np.pi * (radii ** 2).sum(axis=0).max()) / (512 * 512)
    assert peak <= 0.10, f"{process}: peak gold coverage {peak:.3f}"


@pytest.mark.parametrize("process", PROCESSES)
def test_particles_are_several_pixels_wide_in_every_frame(process):
    # Sub-pixel particles are invisible under Poisson noise -> unlearnable targets.
    rng = np.random.default_rng(3)
    _, _, _, _, radii = build_clip(process, 48, (512, 512), rng)
    alive = radii[radii > 0]
    assert alive.min() >= 2.0, f"{process}: smallest radius {alive.min():.2f}px"


@pytest.mark.parametrize("process", ["diffusion", "reaction"])
def test_global_brightness_does_not_leak_particle_size(process):
    """Overall frame brightness must track particle size the way physics says.

    Growing particles add gold, so the frame gets *darker* -- a negative
    correlation. The per-clip min-max normalisation this replaced inverted that
    to +0.72, meaning a Gate-1 probe could read mean radius off overall brightness
    with the sign the physics does not have. That is a label leak from the save
    path, and it lived on exactly the two modes where the growth exponent is the
    headline result.
    """
    rng = np.random.default_rng(4)
    positions_t, radii_t, meta, _, radii = build_clip(process, 16, (128, 128), rng)
    video = render_video(positions_t, radii_t, (128, 128),
                         pixel_size_nm=meta["pixel_size_nm"],
                         beam_flux=meta["beam_flux"], rng=rng)
    frames = np.stack([to_uint8(f, beam_flux=meta["beam_flux"]) for f in video])
    mean_r = radii.mean(axis=0)
    corr = np.corrcoef(mean_r, frames.reshape(16, -1).mean(axis=1))[0, 1]
    assert corr < 0, f"{process}: corr={corr:+.3f} -- brightness leaks size"
