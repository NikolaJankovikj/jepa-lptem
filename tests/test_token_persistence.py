"""The token-persistence measure on cases with a known answer."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from token_persistence import persistence


def one_particle(start, end, radius=10.0):
    pos = np.array([[start, end]], dtype=float)  # (N=1, T=2, 2)
    return pos, np.full((1, 2), radius)


def test_still_particle_stays_in_its_cell():
    pos, rad = one_particle([40.0, 40.0], [40.0, 40.0])
    out = persistence(pos, rad, ref=0, h=1)
    assert out["same_cell"].tolist() == [True]
    assert out["inside"].tolist() == [True]
    assert out["displacement"].tolist() == [0.0]


def test_one_cell_move_leaves_the_cell_but_stays_inside_a_big_particle():
    pos, rad = one_particle([40.0, 40.0], [40.0, 56.0], radius=20.0)
    out = persistence(pos, rad, ref=0, h=1)
    assert out["same_cell"].tolist() == [False]
    assert out["inside"].tolist() == [True]  # 16 px < radius 20
    assert out["displacement"].tolist() == [16.0]


def test_particle_absent_at_either_end_is_not_counted():
    pos, rad = one_particle([40.0, 40.0], [40.0, 40.0])
    rad[0, 1] = 0.0  # left the crop
    assert persistence(pos, rad, ref=0, h=1)["same_cell"].size == 0
