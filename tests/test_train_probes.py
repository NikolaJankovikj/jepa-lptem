"""Step 7 target/metric logic — pure numpy, no encoder, no training."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from train_probes import (GRID, attentive_targets, centroid_targets, r_squared,
                          split_by_clip)

from jepa_lptem.eval.probes import decode_centroids


def test_attentive_targets_counts_only_present_particles():
    radii = np.array([[5.0, 0.0], [3.0, 3.0]])  # particle 0 vanishes at t=1
    t = attentive_targets(radii)
    assert t.shape == (2, 3)
    assert np.allclose(t[:, 0], [2, 1])                    # count
    assert np.allclose(t[:, 1], [4.0, 3.0])                # mean radius of present
    assert np.allclose(t[:, 2], np.pi * np.array([34.0, 9.0]))  # total area


def test_centroid_targets_round_trip_through_decode():
    # Building targets then decoding them must recover the positions exactly:
    # centroid_targets is the inverse of decode_centroids by construction.
    crop = 256
    pos = np.array([[[37.3, 200.9]], [[141.0, 15.5]]])  # (N=2, T=1, 2)
    rad = np.ones((2, 1))
    occ, off, collisions = centroid_targets(pos, rad, crop)
    assert collisions == 0
    got = decode_centroids(occ[0], off[0], cell_size=crop / GRID)
    assert got.shape == (2, 2)
    want = pos[:, 0, :][np.lexsort((pos[:, 0, 1], pos[:, 0, 0]))]
    got = got[np.lexsort((got[:, 1], got[:, 0]))]
    assert np.allclose(got, want, atol=1e-5)


def test_centroid_targets_reports_cell_collisions():
    pos = np.array([[[20.0, 20.0]], [[21.0, 21.0]]])  # same 16 px cell
    rad = np.ones((2, 1))
    occ, _, collisions = centroid_targets(pos, rad, 256)
    assert collisions == 1
    assert occ[0].sum() == 1.0


def test_split_by_clip_is_deterministic_and_disjoint():
    clips = [f"clip_{i:04d}" for i in range(25)] * 2  # two windows per clip
    a = split_by_clip(clips, seed=0)
    b = split_by_clip(clips, seed=0)
    assert a == b
    assert not (a["train"] & a["val"]) and not (a["train"] & a["test"])
    assert not (a["val"] & a["test"])
    assert a["train"] | a["val"] | a["test"] == set(clips)


def test_r_squared_perfect_and_mean_predictor():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(y, y) == 1.0
    assert abs(r_squared(np.full(4, y.mean()), y)) < 1e-12
