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
from jepa_lptem.sim.dataset import (
    MIN_VISIBLE_RADIUS_PX,
    PROCESSES,
    STRATA,
    build_clip,
)
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
    """MSD must recover 4Dt -- but only when *pooled across clips*.

    The estimator is unbiased (mean ratio 0.96-1.02 for every particle count), yet
    its spread is entirely set by how many particles contribute: sd 0.25 at N=4
    versus 0.03 at N=512. A single experiment-grade clip holds 3-8 particles, so
    its MSD is worth little on its own. This is an evaluation requirement, not a
    generator defect: Gate 2 must pool MSD across clips rather than score per clip.
    Testing it the way the evaluation has to use it, instead of widening a
    tolerance until one noisy clip fits.
    """
    rng = np.random.default_rng(1)
    lags = np.arange(1, 64)
    short = 16  # long lags are biased downward by the reflecting box
    ratios = []
    for _ in range(6):
        _, _, meta, pos, _ = build_clip("static", 64, (512, 512), rng)
        slope = float(np.polyfit(lags[:short], compute_msd(pos)[:short], 1)[0])
        ratios.append(slope / (4 * meta["D"]))
    pooled = float(np.mean(ratios))
    assert abs(pooled - 1.0) < 0.15, f"pooled MSD/4D = {pooled:.3f} from {ratios}"


@pytest.mark.parametrize("stratum", list(STRATA))
@pytest.mark.parametrize("process", PROCESSES)
def test_particles_stay_sparse_enough_to_be_resolvable(process, stratum):
    # The whole point of the dataset is a recoverable answer key. If particles grow
    # until they overlap, no centroid is extractable and a probe failure becomes
    # uninterpretable. Cap peak coverage well below the touching limit.
    rng = np.random.default_rng(2)
    _, _, _, _, radii = build_clip(process, 48, (512, 512), rng, stratum=stratum)
    peak = float(np.pi * (radii ** 2).sum(axis=0).max()) / (512 * 512)
    assert peak <= STRATA[stratum]["max_coverage"] + 1e-6, \
        f"{process}/{stratum}: peak gold coverage {peak:.3f}"


@pytest.mark.parametrize("stratum", list(STRATA))
@pytest.mark.parametrize("process", PROCESSES)
def test_no_particle_is_labelled_below_the_visibility_floor(process, stratum):
    # A particle too small to see under Poisson noise is not a target, it is a
    # mislabel. Ripening is the one that reaches here, on its dissolving tail.
    rng = np.random.default_rng(3)
    _, _, _, _, radii = build_clip(process, 48, (512, 512), rng, stratum=stratum)
    alive = radii[radii > 0]
    assert alive.min() >= MIN_VISIBLE_RADIUS_PX, \
        f"{process}/{stratum}: smallest labelled radius {alive.min():.2f}px"


@pytest.mark.parametrize("process", PROCESSES)
def test_experiment_stratum_covers_measured_real_conditions(process):
    """The experiment stratum exists to put real imaging conditions inside the
    training distribution. Measured on real experimental video (see
    scripts/compare_to_real.py): apparent particle radius ~6 px, coverage well
    under 2 %, and frame-to-frame change |dI|/mean ~0.40, which corresponds to
    beam flux ~8-12. Before this
    stratum existed our particles were ~3x larger than real, so real data sat
    outside the distribution entirely -- the failure mode this guards against.
    """
    rng = np.random.default_rng(5)
    _, _, meta, _, radii = build_clip(process, 48, (512, 512), rng, stratum="experiment")
    alive = radii[radii > 0]
    assert 3.0 <= alive.max() <= 9.0, f"{process}: max radius {alive.max():.1f}px"
    coverage = float(np.pi * (radii ** 2).sum(axis=0).max()) / (512 * 512)
    assert coverage <= 0.02, f"{process}: coverage {coverage:.4f}"
    assert 8.0 <= meta["beam_flux"] <= 12.0, f"{process}: flux {meta['beam_flux']:.1f}"


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
