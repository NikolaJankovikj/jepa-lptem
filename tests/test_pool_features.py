"""Pooling must remove token position and keep the quadrant layout row-major."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from pool_features import pool

G = 16


def tokens_from_grid(grid: np.ndarray) -> np.ndarray:
    """``(T, G, G, D)`` token image -> ``(T, G*G, D)`` as the cache stores it."""
    return grid.reshape(grid.shape[0], G * G, grid.shape[-1])


def test_quadrants_are_row_major():
    grid = np.zeros((1, G, G, 1))
    grid[0, :8, 8:] = 1.0  # top-right
    grid[0, 8:, :8] = 2.0  # bottom-left
    grid[0, 8:, 8:] = 3.0  # bottom-right
    assert pool(tokens_from_grid(grid))["grid2"][0, :, 0].tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
    ]


def test_mean_and_std_ignore_where_a_particle_is_but_grid2_does_not():
    a, b = np.zeros((1, G, G, 2)), np.zeros((1, G, G, 2))
    a[0, 2, 3], b[0, 12, 9] = 5.0, 5.0  # one "particle", top-left vs bottom-right
    pa, pb = pool(tokens_from_grid(a)), pool(tokens_from_grid(b))
    assert np.allclose(pa["mean"], pb["mean"]) and np.allclose(pa["std"], pb["std"])
    assert not np.allclose(pa["grid2"], pb["grid2"])
