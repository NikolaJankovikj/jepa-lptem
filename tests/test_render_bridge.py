import numpy as np

from jepa_lptem.sim.render_bridge import (
    draw_spheres,
    render_frame,
    render_video,
    normalize_uint8,
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


# --- uint8 helper for saving / visualization ----------------------------------

def test_normalize_uint8_spans_full_range():
    x = np.array([[1.0, 2.0], [3.0, 5.0]])
    u = normalize_uint8(x)
    assert u.dtype == np.uint8
    assert u.min() == 0
    assert u.max() == 255
