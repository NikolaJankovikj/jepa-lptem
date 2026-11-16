"""Region statistics on a case with a known answer."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from region_predictability import region_stats


def test_particles_land_in_their_region_and_absent_ones_are_dropped():
    pos = np.array([[[10.0, 10.0]], [[10.0, 200.0]], [[200.0, 200.0]]])  # (N=3, T=1)
    rad = np.array([[2.0], [1.0], [0.0]])  # the third left the crop
    stats = region_stats(pos, rad, cell=128)[0]  # (4 regions, count/area)
    assert stats[:, 0].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert np.allclose(stats[:, 1], [4 * np.pi, np.pi, 0.0, 0.0])
