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


def undrift(pos, meta):
    """Remove the coherent stage drift, which is what `MSD = 4Dt` is labelled on.

    Exact rather than estimated: the generator knows the drift, so the test grades
    the Brownian part without also grading a drift estimator. Evaluation on real
    video has to estimate it from the ensemble mean instead.
    """
    t = np.arange(pos.shape[1])
    return pos - np.asarray(meta["drift"])[None, None, :] * t[None, :, None]


@pytest.mark.parametrize("process,expected", [("diffusion", 1 / 3), ("reaction", 1 / 2)])
def test_growth_exponent_recovered_from_generated_clip(process, expected):
    rng = np.random.default_rng(0)
    _, _, meta, _, radii = build_clip(process, 64, (512, 512), rng)
    # time-since-nucleation, not time-since-frame-0 (see growth_time_axis)
    t = meta["t0_growth"] + np.arange(64) * meta["dt_growth"]
    alive = radii > 0
    mean_r = np.array([radii[alive[:, k], k].mean() for k in range(64)])
    alpha = fit_growth_exponent(t, mean_r)
    # 0.001, not the 0.005 this used to carry: with r0_i pinned to k_i at a common
    # nucleation time every particle is on an exact power law, so the ensemble mean
    # is too, and the residual is integrator error alone (measured max 3e-4).
    assert abs(alpha - expected) < 0.001, f"{process}: alpha={alpha:.4f}"


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
        msd = compute_msd(undrift(pos, meta))
        slope = float(np.polyfit(lags[:short], msd[:short], 1)[0])
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
    scripts/compare_to_real.py): apparent particle radius ~8 px, coverage well
    under 2 %, and frame-to-frame change |dI|/mean ~0.40, which corresponds to
    beam flux ~8-12. Before this
    stratum existed our particles were ~3x larger than real, so real data sat
    outside the distribution entirely -- the failure mode this guards against.
    """
    rng = np.random.default_rng(5)
    _, _, meta, _, radii = build_clip(process, 48, (512, 512), rng, stratum="experiment")
    alive = radii[radii > 0]
    assert 8.0 <= alive.max() <= 17.0, f"{process}: max radius {alive.max():.1f}px"
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
    # with the background field on, i.e. what actually ships -- the field adds
    # variance to overall brightness, so this is the harder version of the test
    video = render_video(positions_t, radii_t, (128, 128),
                         pixel_size_nm=meta["pixel_size_nm"],
                         beam_flux=meta["beam_flux"],
                         thickness_sd_nm=meta["thickness_sd_nm"], rng=rng)
    frames = np.stack([to_uint8(f, beam_flux=meta["beam_flux"]) for f in video])
    mean_r = radii.mean(axis=0)
    corr = np.corrcoef(mean_r, frames.reshape(16, -1).mean(axis=1))[0, 1]
    assert corr < 0, f"{process}: corr={corr:+.3f} -- brightness leaks size"


@pytest.mark.parametrize("stratum", list(STRATA))
def test_background_structure_brackets_real_video(stratum):
    """Generated frames must carry real LPTEM's large-scale illumination structure.

    Measured on the Au-nanorod video with the same 40-px lowpass used here: the
    smooth background has sd/mean = 0.212 and carries 26 % of the frame's variance.
    Our flat background read 0.006 -- a 34x gap, and the entire low-frequency
    deficit in the radial power spectrum. Graded as *containment* (real inside our
    generated range) rather than a match to the point estimate, for the reason
    STRATA gives: our picture of "real" is 75 frames of one experiment.

    This is not cosmetic. A byte-identical uniform background makes predicting a
    masked background block free, which is a free lunch in the V-JEPA objective
    that real data does not offer.
    """
    from scipy import ndimage
    rng = np.random.default_rng(25)
    vals = []
    for _ in range(6):
        positions_t, radii_t, meta, _, _ = build_clip("static", 2, (256, 256), rng,
                                                      stratum=stratum)
        video = render_video(positions_t, radii_t, (256, 256),
                             pixel_size_nm=meta["pixel_size_nm"],
                             beam_flux=meta["beam_flux"],
                             thickness_sd_nm=meta["thickness_sd_nm"], rng=rng)
        f = np.stack([to_uint8(x, beam_flux=meta["beam_flux"]) for x in video])[0]
        vals.append(ndimage.gaussian_filter(f.astype(float), 10).std() / f.mean())
    assert min(vals) <= 0.212 <= max(vals), \
        f"{stratum}: real 0.212 outside generated range {min(vals):.3f}-{max(vals):.3f}"


def test_bigger_particles_diffuse_slower():
    """Stokes--Einstein, ``D = kT / (6 pi eta r)``: D must fall with radius.

    This ran inverted -- ``D`` was the step-per-radius ratio applied at each clip's
    own radius, giving ``D ~ r^2``, a 20x error across the 8-22 px range with the
    sign backwards. It is also a shortcut: a probe reading "big blob moves fast"
    off the latents would score as learned dynamics while having learned an
    artefact of the generator. Same failure mode as the brightness->size leak.

    Holds ``step_per_radius`` fixed by seeding identically and comparing strata
    would not isolate it, so compare ``D`` against radius across many clips.
    """
    rng = np.random.default_rng(9)
    radii_and_D = []
    for i in range(24):
        _, _, meta, _, radii = build_clip("static", 32, (512, 512), rng)
        # divide out the per-clip step ratio; what is left must be a function of r
        radii_and_D.append((float(np.median(radii[radii > 0])),
                            meta["D"] / meta["step_per_radius"] ** 2))
    r, d = np.array(radii_and_D).T
    corr = np.corrcoef(np.log(r), np.log(d))[0, 1]
    assert corr < -0.9, f"D vs radius correlation {corr:+.2f} -- expected D ~ 1/r"


def test_ripening_actually_coarsens():
    """A clip labelled "Ostwald ripening" must show ripening.

    Nothing else grades this process -- ``EXPONENT["ripening"]`` is None, so no
    exponent test touches it -- and at the original rate constant it did not
    coarsen at all: mean radius x1.02, one particle lost in ten, statistically
    indistinguishable from the ``static`` process over a 100-frame clip. The
    coarsening timescale is ~r^3/K, so K has to put it inside the clip.
    """
    rng = np.random.default_rng(3)
    growth, survival = [], []
    for _ in range(6):
        _, _, _, _, radii = build_clip("ripening", 100, (512, 512), rng)
        alive = radii > 0
        mean_r = np.array([radii[alive[:, k], k].mean() for k in range(100)])
        growth.append(mean_r[-1] / mean_r[0])
        survival.append(alive[:, -1].sum() / alive[:, 0].sum())
    assert np.mean(growth) > 1.2, f"mean radius only x{np.mean(growth):.2f}"
    assert np.mean(survival) < 0.8, f"{np.mean(survival):.0%} of particles survived"


@pytest.mark.parametrize("process", ["diffusion", "reaction"])
def test_growth_does_not_collapse_the_size_distribution(process):
    """The size distribution is the headline observable -- it must survive the clip.

    The growth ODEs are contractions on ``r``, so a population sharing one rate
    constant converges to a single radius: this read 0.095 -> 0.0007 relative sd
    over a 5x diffusion clip, a delta function, which made the Gate-2 Wasserstein
    distance degenerate to ``|delta mean|`` on half the dataset. With a per-particle
    ``k`` pinned to ``r0`` at a common nucleation time (see ``R0_SPREAD``), the
    ratio ``r_i/r_j`` is invariant, so the spread must hold to within a few percent
    end to end -- and must be comparable to the 0.155 that `static` already carries.
    """
    rng = np.random.default_rng(21)
    first, last = [], []
    for _ in range(8):
        *_, radii = build_clip(process, 64, (512, 512), rng)
        for out, k in ((first, 0), (last, 63)):
            alive = radii[radii[:, k] > 0, k]
            out.append(alive.std() / alive.mean())
    f, l = float(np.mean(first)), float(np.mean(last))
    # invariant by construction, so grade it as one: anything that bends it (a
    # particle crossing the visibility floor mid-clip, an integrator drifting) shows
    # up here before it shows up as a biased exponent
    assert abs(l - f) / f < 0.02, f"{process}: rel.sd {f:.4f} -> {l:.4f} (collapsed)"
    assert 0.08 < l < 0.30, f"{process}: rel.sd {l:.4f} outside the useful band"


def test_drift_is_a_coherent_predictable_component_of_motion():
    """Real LPTEM stage drift, and the one thing on the position axis a forecaster
    can beat persistence on.

    Brownian position is a martingale, so without drift the generator hands
    persistence an optimal strategy by construction and Gate 2 is decided by our
    choice of simulator rather than by evidence. Two things must hold: the drift is
    actually there (ensemble-mean displacement grows linearly, not as a random
    walk), and it is *rigid* -- identical for every particle -- so the eval can
    remove it exactly and `coalesce` sees untouched pairwise distances.
    """
    rng = np.random.default_rng(22)
    fracs = []
    for _ in range(8):
        _, _, meta, pos, _ = build_clip("static", 100, (512, 512), rng)
        drift = np.asarray(meta["drift"])
        t = np.arange(100)
        # rigid: every particle carries exactly the same coherent term
        residual = undrift(pos, meta)
        rigid = residual - residual[:, :1, :]
        assert np.allclose((pos - pos[:, :1, :]) - rigid, drift[None, None, :] * t[None, :, None])
        fracs.append(meta["drift_per_step"])
    assert 0.0 < np.mean(fracs) < 0.25, f"drift/step = {np.mean(fracs):.3f}"
    assert max(fracs) > 0.02, "drift never appreciably present"


def test_coalescence_produces_timed_volume_conserving_merges():
    """RQ2 needs events: discrete, exactly-timestamped things that happen.

    Before this, `PROCESSES` had none -- `merge_on_contact` had been deleted and
    Step 12's `ripening_coalescence` clips did not exist, so event anticipation had
    no data at all. Each merge must conserve volume exactly (r^3 additive) and drop
    the particle count by one, and clips must contain enough of them to fit a
    lead-time distribution on.
    """
    rng = np.random.default_rng(23)
    counts = []
    for _ in range(8):
        *_, meta, _, radii = build_clip("coalescence", 100, (512, 512), rng)
        counts.append(meta["n_merges"])
        vol = (radii ** 3).sum(axis=0)
        assert np.allclose(vol, vol[0]), "coalescence must conserve total volume"
        for frame, survivor, absorbed in meta["merge_events"]:
            assert radii[absorbed, frame] == 0.0 and radii[absorbed, frame - 1] > 0.0
            assert radii[survivor, frame] > radii[survivor, frame - 1]
        alive = (radii > 0).sum(axis=0)
        assert alive[0] - alive[-1] == meta["n_merges"]
    assert np.mean(counts) >= 1.0, f"only {np.mean(counts):.1f} merges per clip"


def test_coalescence_clips_do_not_open_mid_merge():
    # A merge at frame 0 has no lead-up, so it cannot be graded for anticipation --
    # it is a mislabelled event, not a hard one.
    rng = np.random.default_rng(24)
    for stratum in STRATA:
        for _ in range(8):
            *_, meta, _, _ = build_clip("coalescence", 48, (512, 512), rng, stratum=stratum)
            opening = [e for e in meta["merge_events"] if e[0] == 0]
            assert not opening, f"{stratum}: merge at frame 0 {opening}"


@pytest.mark.parametrize("stratum", list(STRATA))
def test_msd_label_ships_the_lag_it_is_valid_over(stratum):
    """``MSD = 4Dt`` is only true before particles feel the reflecting box.

    The generator knows where that is; consumers should not have to rediscover it,
    so ``msd_max_lag`` ships in the metadata. This asserts the label is honest.

    Pooled across clips, for the reason ``test_msd_of_generated_positions_matches_4Dt``
    documents -- and note ``msd_max_lag`` bounds *box bias* only. It cannot bound
    estimator variance, which is set by particle count and by the fact that a lag
    approaching the clip length has almost no independent start frames left. Both
    ceilings are real; this one is the one the generator can compute.
    """
    rng = np.random.default_rng(12)
    ratios = []
    for _ in range(8):
        _, _, meta, pos, _ = build_clip("static", 100, (512, 512), rng, stratum=stratum)
        lag = meta["msd_max_lag"]
        assert lag >= 1
        msd = compute_msd(undrift(pos, meta))
        slope = float(np.polyfit(np.arange(1, lag + 1), msd[:lag], 1)[0])
        ratios.append(slope / (4 * meta["D"]))
    pooled = float(np.mean(ratios))
    assert abs(pooled - 1.0) < 0.15, f"{stratum}: pooled MSD/4D = {pooled:.3f}"
