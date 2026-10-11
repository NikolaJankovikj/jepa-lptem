"""P1's readout and floor exponent on cases with a known answer."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from pooled_gate1 import CTX, HORIZON, Readout, floor_exponents


def test_readout_recovers_a_linear_map_in_physical_units():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 32, 16)).astype(np.float32)
    true_w = rng.normal(size=(16, 3))
    y = x @ true_w * np.array([2.0, 5.0, 300.0]) + np.array([5.0, 10.0, 2000.0])
    readout = Readout()
    readout.fit(x, y, np.arange(40), np.arange(40, 50))
    assert np.allclose(readout.predict(x[50:]), y[50:], rtol=1e-2, atol=1e-1)


def power_law_windows(laws):
    """One physics window per ``(process, alpha)``: radius exactly ``4 t**alpha``."""
    frames = np.arange(CTX + HORIZON)
    meta, radius = [], []
    for clip, (process, alpha) in enumerate(laws):
        m = {"stratum": "physics", "process": process, "clip": f"c{clip}",
             "start": 0, "frames_per_latent": 2, "t0_growth": 10.0,
             "dt_growth": 0.5, "exponent": alpha}
        meta.append(m)
        radius.append(4.0 * (m["t0_growth"] + m["dt_growth"] * 2 * frames) ** alpha)
    return np.array(radius), meta


def test_floor_exponent_recovers_a_pure_power_law():
    radius, meta = power_law_windows([("reaction", 0.5)] * 3 + [("static", 0.0)] * 3)
    out = floor_exponents(radius, meta, np.arange(len(meta)))
    assert np.isclose(out["reaction"]["median"], 0.5)
    assert np.isclose(out["static"]["median"], 0.0, atol=1e-9)
