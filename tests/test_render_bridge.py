import numpy as np

from jepa_lptem.sim.render_bridge import (
    calibrate_mtf,
    draw_spheres,
    render_frame,
    render_video,
    to_uint8,
    background_intensity,
    BACKGROUND_LEVEL,
    CONVERSION_FACTOR,
)


# --- geometry front-end: our contribution (circles, not nanorods) -------------

def test_draw_spheres_occupancy_area_matches_disk():
    # a single disk of radius 10 px covers ~pi r^2 pixels
    occ, _ = draw_spheres(
        positions=np.array([[32.0, 32.0]]),
        radii=np.array([10.0]),
        image_size=(64, 64),
    )
    assert occ.shape == (64, 64)
    assert occ.dtype == bool
    count = int(occ.sum())
    assert abs(count - np.pi * 10 ** 2) < 40  # rasterization tolerance


def test_draw_spheres_center_gold_thickness_is_diameter_in_nm():
    # spherical chord at the center = full diameter = 2 r, in nm
    _, gold = draw_spheres(
        positions=np.array([[32.0, 32.0]]),
        radii=np.array([10.0]),
        image_size=(64, 64),
        pixel_size_nm=2.0,
    )
    assert abs(gold[32, 32] - 2 * 10.0 * 2.0) < 1e-6  # 40 nm
    # falls to zero outside the disk
    assert gold[0, 0] == 0.0


def test_draw_spheres_empty_is_all_zero():
    occ, gold = draw_spheres(
        positions=np.zeros((0, 2)),
        radii=np.zeros((0,)),
        image_size=(16, 16),
    )
    assert not occ.any()
    assert np.all(gold == 0.0)


# --- physics renderer: adapted Beer-Lambert + Poisson + MTF -------------------

def test_render_frame_preserves_shape():
    gold = np.zeros((32, 32))
    img = render_frame(gold, add_noise=False)
    assert img.shape == (32, 32)


def test_render_frame_uniform_when_no_gold_and_no_noise():
    gold = np.zeros((32, 32))
    img = render_frame(gold, add_noise=False)
    assert np.allclose(img, img.flat[0])  # every pixel identical


def test_render_frame_particle_is_darker_than_background():
    # gold attenuates more than water (shorter IMFP) -> particle pixels darker
    gold = np.zeros((32, 32))
    gold[12:20, 12:20] = 40.0  # 40 nm gold blob
    img = render_frame(gold, add_noise=False)
    particle = img[12:20, 12:20].mean()
    background = np.concatenate([img[:12].ravel(), img[20:].ravel()])
    assert particle < background.mean()


def test_render_frame_thicker_water_is_dimmer():
    gold = np.zeros((32, 32))
    thin = render_frame(gold, thickness_water=50.0, add_noise=False).mean()
    thick = render_frame(gold, thickness_water=150.0, add_noise=False).mean()
    assert thick < thin  # more water -> more attenuation -> dimmer


def test_render_frame_water_thickness_clamped_at_zero():
    # A particle thicker than the liquid cell displaces all the water and no more.
    # Once gold exceeds the channel, the water thickness must stop mattering --
    # otherwise the term goes negative and silently brightens big/overlapping particles.
    gold = np.full((8, 8), 400.0)  # nm, far thicker than either channel below
    a = render_frame(gold, thickness_water=150.0, add_noise=False)
    b = render_frame(gold, thickness_water=100.0, add_noise=False)
    assert np.allclose(a, b)


def test_render_frame_noise_is_deterministic_under_seed():
    gold = np.zeros((32, 32))
    gold[10:14, 10:14] = 30.0
    a = render_frame(gold, rng=np.random.default_rng(0))
    b = render_frame(gold, rng=np.random.default_rng(0))
    c = render_frame(gold, rng=np.random.default_rng(1))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_render_frame_mtf_preserves_dc_on_uniform_image():
    # DC gain is 1.0, so a flat image is unchanged by the MTF low-pass
    gold = np.zeros((16, 16))
    mtf = np.ones((16, 16))
    img = render_frame(gold, add_noise=False, mtf=mtf)
    assert np.allclose(img, img.flat[0])


# --- video stitching ----------------------------------------------------------

def test_render_video_shape():
    positions_t = [np.array([[8.0, 8.0]]) for _ in range(3)]
    radii_t = [np.array([4.0]) for _ in range(3)]
    vid = render_video(positions_t, radii_t, image_size=(16, 16), add_noise=False)
    assert vid.shape == (3, 16, 16)


def test_render_video_particle_darker_each_frame():
    positions_t = [np.array([[16.0, 16.0]]) for _ in range(2)]
    radii_t = [np.array([6.0]) for _ in range(2)]
    vid = render_video(positions_t, radii_t, image_size=(32, 32), add_noise=False)
    for frame in vid:
        assert frame[16, 16] < frame[0, 0]


# --- uint8 quantization against the analytic background -----------------------

def _disk_gold(radius_px, pixel_size_nm=2.0):
    _, gold = draw_spheres([[64.0, 64.0]], [float(radius_px)], (128, 128),
                           pixel_size_nm=pixel_size_nm)
    return gold


def test_to_uint8_puts_background_at_a_fixed_level_regardless_of_dose():
    # A fixed scale is the whole point: clean water must land on the same value
    # whatever the electron dose, or clips are not comparable to each other.
    for flux in (2.0, 30.0, 75.0):
        img = render_frame(np.zeros((16, 16)), beam_flux=flux, add_noise=False)
        u = to_uint8(img, beam_flux=flux)
        assert np.all(u == BACKGROUND_LEVEL)


def test_to_uint8_preserves_absolute_contrast_between_particle_sizes():
    # The bug this replaced: a per-frame min-max stretch mapped every clip to
    # 0-255, so a thin particle and a thick one came out identical.
    small = to_uint8(render_frame(_disk_gold(8), add_noise=False))
    large = to_uint8(render_frame(_disk_gold(32), add_noise=False))
    assert small[64, 64] > large[64, 64]  # more gold -> genuinely darker on disk
    assert small.max() == large.max() == BACKGROUND_LEVEL  # background is shared


def test_to_uint8_leaves_fourfold_headroom_above_background():
    # Clipping is a nonlinearity that breaks the Beer-Lambert relation we grade
    # against. Putting the background at 64 of 255 buys a 4x upward excursion
    # before a pixel saturates -- what the near-binary lowest-dose frames need.
    # (The realized clipping fraction on generated clips is reported by
    # scripts/verify_clips.py, where the real MTF is in play.)
    bg = background_intensity(beam_flux=2.0)
    assert to_uint8(np.array([[bg]]), beam_flux=2.0)[0, 0] == BACKGROUND_LEVEL
    assert to_uint8(np.array([[3.9 * bg]]), beam_flux=2.0)[0, 0] < 255
    assert to_uint8(np.array([[4.1 * bg]]), beam_flux=2.0)[0, 0] == 255


def test_render_frame_matches_upstream_lptemsimulator_reference():
    """External oracle: reproduce LPTEMsimulator's `simulate_tem_image` exactly.

    `render_bridge` is a re-implementation, so the thing worth testing is not that
    it is self-consistent but that it agrees with the published simulator that
    produced the real SAM-EM training data. Their nanorod is a flat 4.5 px slab at
    2 nm/px = 9 nm of gold; fed the same gold map, the two must agree to float
    precision. (Noise and MTF are omitted so this is deterministic -- they are
    identical operations applied afterwards in both implementations.)
    """
    gold_nm = 9.0
    # transcribed from JamaliLab/LPTEMsimulator, LPTEM_simulated_data.ipynb
    expected = 30 * np.exp(-(gold_nm / (20 + 50)
                             + (150 - gold_nm) / 185
                             + 2 * 50 / 144)) * 37.46
    got = render_frame(np.full((8, 8), gold_nm), mtf=None, add_noise=False)
    assert np.allclose(got, expected, rtol=1e-12)


def test_mtf_blurs_shot_noise_but_not_read_noise():
    """Read noise must be added *after* the MTF, not before.

    Charge spreading happens at the sensor: it blurs the electron events that
    arrived, and the readout electronics add their noise downstream of it. When the
    read noise was blurred too, the ratio of temporal to spatial noise sat at ~2.9
    at *every* flux -- dose-independent, which is the signature of a noise term
    being low-passed that shouldn't be. Real video measures 1.67.

    Detects the ordering directly: with the MTF applied to a noise-free image, the
    only thing left un-blurred is the read noise, so the frame must still carry
    pixel-to-pixel variance of about ``read_noise`` after a heavy low-pass.
    """
    gold = np.zeros((64, 64))
    mtf = np.zeros((64, 64))
    mtf[30:34, 30:34] = 1.0  # brutal low-pass: kills anything it is applied to
    img = render_frame(gold, mtf=mtf, read_noise=3.0, beam_flux=0.0,
                       rng=np.random.default_rng(0)) / CONVERSION_FACTOR
    assert abs(img.std() - 3.0) < 0.3, f"read noise was blurred: std={img.std():.2f}"


def test_calibrate_mtf_reads_matrix_at_half_offsets():
    """The detector calibration: output bin at offset u from DC = matrix at u/2.

    Applied verbatim, the measured 512 px matrix attenuates each frequency at
    twice the rate of the real 1024 px detector (temporal/spatial noise ratio
    1.62-2.40 vs real's 1.284; at half-offsets, 1.27-1.31 -- containing real).
    A radially known synthetic matrix makes the resampling exact to check.
    """
    n = 64
    u = np.arange(n) - n // 2
    dist = np.hypot(u[:, None], u[None, :])
    mtf = 1.0 / (1.0 + dist)  # known radial profile, DC = 1
    out = calibrate_mtf(mtf, n)
    assert abs(out[n // 2, n // 2] - 1.0) < 1e-12          # DC preserved
    got = out[n // 2, n // 2 + 20]                          # offset 20 from DC
    assert abs(got - 1.0 / (1.0 + 10)) < 1e-6              # = matrix at offset 10


def test_calibrate_mtf_resamples_to_other_render_sizes():
    # Same cycles/px must get the same gain whatever the render size: offset 16
    # on a 64 px grid and offset 8 on a 32 px grid are both 0.25 cycles/px.
    n = 64
    u = np.arange(n) - n // 2
    mtf = 1.0 / (1.0 + np.hypot(u[:, None], u[None, :]))
    big, small = calibrate_mtf(mtf, 64), calibrate_mtf(mtf, 32)
    assert abs(big[32, 32 + 16] - small[16, 16 + 8]) < 1e-6


def test_mtf_ringing_stays_negative():
    """``np.real``, not ``np.abs``: ringing beside a dark core is genuinely
    negative, and folding it positive puts a bright halo where the optics put a
    dark one."""
    gold = np.zeros((32, 32))
    gold[16, 16] = 400.0  # sharp dark point -> guaranteed ringing under a low-pass
    mtf = np.zeros((32, 32))
    mtf[12:21, 12:21] = 1.0
    img = render_frame(gold, mtf=mtf, add_noise=False, beam_flux=100.0)
    flat = render_frame(np.zeros((32, 32)), mtf=mtf, add_noise=False, beam_flux=100.0)
    assert (img - flat).min() < 0, "no negative excursion -- abs() folded the ringing"
