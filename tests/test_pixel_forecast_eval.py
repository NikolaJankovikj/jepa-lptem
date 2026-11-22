"""Step 11 P3: the judged pixel reading recovers a known growth exponent, and reads a
held frame as no growth."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from pixel_forecast_eval import pixel_columns

N, FRAMES, FIRST = 256, 48, 16
CENTRES = np.array([[64.0, 64.0], [64.0, 192.0], [192.0, 128.0]])
META = {"t0_growth": 10.0, "dt_growth": 1.0}


def disks(radius: float) -> np.ndarray:
    yy, xx = np.mgrid[:N, :N]
    f = np.full((N, N), 200.0)
    for cy, cx in CENTRES:
        f[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2] = 80.0
    return f


def growing_clip() -> tuple[np.ndarray, tuple]:
    """Disks with r = t^(1/2), t the growth time of each video frame."""
    t = META["t0_growth"] + FIRST + np.arange(FRAMES)
    frames = np.stack([disks(r) for r in 1.5 * t**0.5])
    lat = FRAMES // 2
    pos = np.repeat(CENTRES[:, None], lat, axis=1)
    return frames, (pos, np.full((len(CENTRES), lat), 5.0))


def test_growth_exponent_is_recovered_and_a_held_frame_reads_none():
    frames, gt = growing_clip()
    held = np.repeat(frames[:1], FRAMES, axis=0)
    cols = pixel_columns({"true": frames, "conv": held}, gt, META, FIRST)
    assert abs(cols["pixel/true"] - 0.5) < 0.1
    assert abs(cols["pixel/convlstm"]) < 1e-9


def test_an_erased_forecast_has_no_particle_depth():
    frames, gt = growing_clip()
    flat = np.full_like(frames, 200.0)
    cols = pixel_columns({"true": frames, "conv": flat}, gt, META, FIRST)
    assert cols["depth/true"].min() > 50  # disks are 120 grey levels dark
    assert np.abs(cols["depth/convlstm"]).max() < 1e-9
