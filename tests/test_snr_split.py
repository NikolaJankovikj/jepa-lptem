"""RQ3's dose split: the two-sample clip CI and the reading-noise measure."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from snr_split import diff_ci, reading_noise


def test_diff_ci_separates_shifted_groups_and_not_equal_ones():
    rng = np.random.default_rng(0)
    clips = np.repeat(np.arange(20), 3)  # 3 snippets per clip
    low = rng.normal(1.0, 0.3, 60)
    high = rng.normal(0.0, 0.3, 60)
    assert diff_ci(low, clips, high, clips + 100)[0] > 0
    same = rng.normal(0.0, 0.3, 60)
    ci = diff_ci(high, clips, same, clips + 100)
    assert ci[0] < 0 < ci[1]


def test_reading_noise_recovers_gaussian_sd_and_drops_empty_frames():
    rng = np.random.default_rng(0)
    truth = rng.uniform(5, 15, 20000)
    count = np.ones_like(truth)
    count[:1000] = 0
    pred = truth + rng.normal(0.5, 1.0, truth.shape)
    pred[:1000] = 1e6  # an empty frame's reading must not count
    noise = reading_noise(pred, truth, count)
    assert abs(noise["bias"] - 0.5) < 0.05
    # MAE / 0.80 is the sd of zero-mean noise; a bias inflates it, as it should
    assert 1.0 < noise["sd_equivalent"] < 1.25
    assert noise["n_frames"] == 19000
