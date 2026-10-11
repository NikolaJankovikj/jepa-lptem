"""P2's linear candidate and verdict logic on cases with a known answer."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from pooled_forecast import CTX, HORIZON, gate_result, linear_candidate, paired_gain


def test_linear_candidate_extrapolates_a_per_window_trend():
    # Each window drifts at its own rate along one direction: the future is a linear
    # function of the context, so candidate A should recover it almost exactly.
    rng = np.random.default_rng(0)
    w, d = 300, 64
    rate = rng.uniform(-1, 1, (w, 1, 1))
    direction = rng.normal(size=d) / 8
    z = rate * np.arange(CTX + HORIZON)[None, :, None] * direction + rng.normal(
        size=(w, 1, d)
    )
    split = {
        "train": np.arange(200),
        "val": np.arange(200, 250),
        "test": np.arange(250, 300),
    }
    forecast, _ = linear_candidate(z, split)
    held = np.repeat(z[:, CTX - 1 : CTX], HORIZON, axis=1)
    err = np.abs(forecast[250:] - z[250:, CTX:]).mean()
    assert err < 0.05 * np.abs(held[250:] - z[250:, CTX:]).mean()


def test_paired_gain_is_positive_when_the_method_is_closer():
    truth = np.full((2, HORIZON), 10.0)
    radius = {"ref": np.full((2, HORIZON), 7.0), "m": np.full((2, HORIZON), 9.0)}
    assert paired_gain(radius, truth, np.arange(2), ("ref", "m")).tolist() == [2.0, 2.0]


def test_gate_needs_every_criterion_and_a_powered_process():
    ok = {"diffusion": True, "reaction": True}
    assert gate_result({"diffusion": True, "reaction": None}, ok, True) == "PASS"
    assert gate_result(ok, {"diffusion": True, "reaction": False}, True) == "FAIL"
    assert gate_result(ok, ok, False) == "FAIL"
    assert gate_result({"diffusion": None, "reaction": None}, ok, True) == "untestable"
