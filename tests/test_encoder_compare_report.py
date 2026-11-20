"""Step 15 M4: the reading-sd dose split and the prediction verdicts."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from encoder_compare_report import TestSet, comparison, run_stats


def test_reading_sd_splits_by_dose():
    # 4 clips x 3 frames, one particle of radius 5; low-dose clips read 1 px off
    truth = np.zeros((4, 3, 3))
    truth[..., 0], truth[..., 1], truth[..., 2] = 1, 5, np.pi * 25
    truth[:, :, 1] += np.arange(4)[:, None]  # radius varies, so R² is defined
    preds = {r: truth[r].copy() for r in range(4)}
    dose_low = np.array([True, True, False, False])
    for r in (0, 1):
        preds[r][:, 1] += 1.0
    test = TestSet(np.arange(4), truth, np.arange(3), dose_low)
    s = run_stats(test, preds, test.rows)
    assert np.isclose(s["reading_sd_low"], 1 / 0.80)
    assert s["reading_sd_high"] == 0
    assert np.isclose(s["low_dose_penalty"], 1 / 0.80)


def test_comparison_verdicts():
    tie = {"value": 0.0, "ci": [-0.01, 0.01]}
    win = {"value": 0.1, "ci": [0.05, 0.15]}
    assert comparison("a", [tie, win], strict=False)["right"]
    assert not comparison("a", [tie, win], strict=True)["right"]
    assert not comparison("a", [tie, win], strict=False)["resolved"]
    assert comparison("a", [win], strict=True)["resolved"]
