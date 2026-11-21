"""Step 11: the dark-mass reading -- near zero on background, blur-invariant, volume-like."""

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from dark_mass import background, dark_mass, radius_exponent

N = 256
REFERENCE_SEED = 99
# Pixel noise sd. The frame sum's noise is sd x 256, so the tests keep it small; on
# simulated clips the premise check (P0b) measures what real shot noise costs.
NOISE = 1.0


def frame_with_disks(radii, seed=0) -> np.ndarray:
    """Bright tilted background, dark spheres (darkening ∝ chord length), noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:N, :N].astype(float)
    f = 150 + 0.1 * xx
    for k, r in enumerate(radii):
        cy, cx = 50 + 50 * (k % 3), 60 + 60 * (k // 3)
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        f -= 4 * r * np.sqrt(np.clip(1 - d2 / r**2, 0, None))
    return f + rng.normal(0, NOISE, f.shape)


def reference(radii) -> np.ndarray:
    """The background a snippet would take from its last context frame."""
    return background(frame_with_disks(radii, seed=REFERENCE_SEED))


def test_empty_frame_has_almost_no_mass():
    one = dark_mass(frame_with_disks([10]), reference([10]))
    # A tilted background leaves a small offset (~0.006 per pixel), well below one particle.
    assert abs(dark_mass(frame_with_disks([]), reference([]))) < 0.1 * one


def test_blur_does_not_change_the_mass():
    f, bg = frame_with_disks([8, 12, 10]), reference([8, 12, 10])
    sharp = dark_mass(f, bg)
    for s in (1, 2, 4):
        assert abs(dark_mass(gaussian_filter(f, s), bg) - sharp) < 0.01 * sharp


def test_growth_exponent_is_recovered():
    # Four spheres growing as r ∝ t^(1/2): the mass goes as r^3, the exponent as 1/2.
    times = np.linspace(4, 12, 12)
    radii = [[2.5 * t**0.5] * 4 for t in times]
    bg = reference(radii[0])
    masses = np.array([
        dark_mass(frame_with_disks(r, seed=i), bg) for i, r in enumerate(radii)
    ])
    assert abs(radius_exponent(times, masses) - 0.5) < 0.1  # the plan's ±0.1 bar
