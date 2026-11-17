"""The integrity screen on movies with a known answer."""

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from real_video_qc import (
    NEIGHBOUR_CORR_MAX,
    NOISE_CORR_MAX,
    noise_correlation,
    segments,
)


def test_white_noise_passes_and_upsampled_noise_fails_the_native_grid_check():
    rng = np.random.default_rng(0)
    white = rng.normal(128, 20, (21, 128, 128))
    # A display copy upsampled 2x: each detector pixel spans two screen pixels.
    upsampled = np.repeat(np.repeat(white[:, :64, :64], 2, axis=1), 2, axis=2)
    smoothed = ndimage.gaussian_filter(white, (0, 1.5, 1.5))
    assert max(noise_correlation(white)) < NOISE_CORR_MAX
    assert max(noise_correlation(white, lag=1)) < NEIGHBOUR_CORR_MAX
    # Exact 2x nearest-neighbour: lag 2 cannot see it, lag 1 must.
    assert max(noise_correlation(upsampled)) < NOISE_CORR_MAX
    assert min(noise_correlation(upsampled, lag=1)) > NEIGHBOUR_CORR_MAX
    assert min(noise_correlation(smoothed)) > NOISE_CORR_MAX


def test_segments_split_at_a_cut_and_drop_repeated_frames():
    # Steps between frames: ordinary noise ~10, a repeat ~0 at 3->4, a cut ~100 at 6->7.
    steps = np.full(9, 10.0)
    steps[3] = 0.1
    steps[6] = 100.0
    segs = segments(steps)
    assert [s.tolist() for s in segs] == [[0, 1, 2, 3, 5, 6], [7, 8, 9]]
