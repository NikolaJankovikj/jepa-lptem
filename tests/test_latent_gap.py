"""Metric logic of scripts/latent_gap.py, on synthetic Gaussians (no encoder)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from latent_gap import containment_ratio, separability_auc, subsample, tile_windows


def _blobs(rng, n, d=32, shift=0.0):
    return rng.normal(shift, 1.0, (n, d)) + 5.0  # offset so cosine dist is informative


def test_containment_ratio_near_one_for_same_distribution():
    rng = np.random.default_rng(0)
    real = _blobs(rng, 40)
    syn = _blobs(rng, 80)
    groups = [f"c{i // 4}" for i in range(80)]  # 20 clips of 4 samples
    ratio, _, _ = containment_ratio(real, syn, groups)
    assert 0.5 < ratio < 2.0


def test_containment_ratio_large_for_off_manifold_real():
    # Off-manifold must change *direction*, not just magnitude -- the metric is
    # cosine, so a uniform shift along the shared mean vector is invisible to it.
    rng = np.random.default_rng(1)
    real = _blobs(rng, 40)
    real[:, :8] += 20.0  # push real along a subspace synthetic never occupies
    syn = _blobs(rng, 80)
    groups = [f"c{i // 4}" for i in range(80)]
    ratio, _, _ = containment_ratio(real, syn, groups)
    assert ratio > 3.0


def test_separability_auc_chance_for_identical_and_high_for_separated():
    rng = np.random.default_rng(2)
    ga = [f"a{i // 4}" for i in range(40)]
    gb = [f"b{i // 4}" for i in range(40)]
    same = separability_auc(_blobs(rng, 40), _blobs(rng, 40), ga, gb)
    apart = separability_auc(_blobs(rng, 40), _blobs(rng, 40, shift=4.0), ga, gb)
    assert same < 0.75
    assert apart > 0.95


def test_tile_windows_shapes_count_and_positions():
    frames = np.zeros((35, 512, 512), dtype=np.uint8)
    tiles, where = tile_windows(frames, n_frames=16, tile=256)
    assert len(tiles) == 2 * 4  # 2 temporal windows x 2x2 spatial tiles
    assert tiles[0].shape == (16, 256, 256)
    assert set(where) == {"r0c0", "r0c256", "r256c0", "r256c256"}


def test_subsample_caps_and_balances_groups():
    rng = np.random.default_rng(3)
    samples = list(range(100))
    groups = ["a"] * 90 + ["b"] * 10
    kept, kept_groups = subsample(samples, groups, 20, rng)
    assert len(kept) == 20
    assert kept_groups.count("b") == 10  # minority group fully kept, not swamped
