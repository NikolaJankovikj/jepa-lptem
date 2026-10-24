"""The PC part of RQ2: pixel change per token, stitching, crop coordinates."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from event_signals import crop_gt, latent_sequence, pixel_signal


def test_pixel_signal_marks_only_the_token_that_changed():
    frames = np.zeros((8, 256, 256), np.uint8)
    frames[4:, 32:48, 80:96] = 100  # token (2, 5) brightens from latent frame 2 on
    signal = pixel_signal(frames)
    assert signal.shape == (4, 256)
    assert np.isnan(signal[0]).all()
    assert signal[2, 2 * 16 + 5] == 100
    assert np.count_nonzero(signal[1:]) == 1


def test_latent_sequence_stitches_windows_and_refuses_gaps(tmp_path):
    whole = np.arange(50, dtype=np.float16)[:, None, None] * np.ones(
        (1, 2, 3), np.float16
    )
    for start in (0, 36):
        np.savez(
            tmp_path / f"c__w{start:03d}.npz",
            features=whole[start // 2 : start // 2 + 32],
        )
    assert np.array_equal(latent_sequence(tmp_path, "c", 50), whole)
    with pytest.raises(SystemExit):
        latent_sequence(tmp_path, "c", 60)


def test_crop_gt_zeroes_centres_outside_the_crop():
    positions = np.array(
        [[[300.0, 300.0], [100.0, 300.0]]]
    )  # inside, then above the crop
    pos, rad = crop_gt(positions, np.array([[5.0, 5.0]]), 512, 256)
    assert np.allclose(pos[0, 0], [172, 172])
    assert rad.tolist() == [[5.0, 0.0]]
