"""Step 7 target/metric logic — pure numpy, no encoder, no training."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from train_probes import (GRID, attentive_targets, centroid_targets, r_squared,
                          split_by_clip)

from jepa_lptem.eval.centroid_eval import match_count, pick_threshold, pool_msd, window_msd

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


def test_match_count_is_one_to_one_within_tolerance():
    gt = np.array([[10.0, 10.0], [50.0, 50.0]])
    pred = np.array([[11.0, 10.0], [12.0, 10.0], [50.0, 70.0]])
    # two detections near the first particle claim it once; the third is too far
    assert match_count(pred, gt, tol=8.0) == 1
    assert match_count(np.zeros((0, 2)), gt, tol=8.0) == 0


def test_pick_threshold_cuts_between_true_and_spurious_peaks():
    occ = np.zeros((1, GRID, GRID))
    occ[0, 3, 3] = 0.92          # the real particle
    occ[0, 10, 10] = 0.62        # a spurious peak far from anything
    gt = [np.array([[3.5 * 16, 3.5 * 16]])]
    windows = [(occ, np.zeros((1, GRID, GRID, 2)), gt, {"crop": 256})]
    th = pick_threshold(windows)
    assert 0.62 <= th < 0.92


def test_window_msd_recovers_4dt_on_ground_truth_tracks():
    from jepa_lptem.sim.dynamics import simulate_brownian
    D = 0.5
    rng = np.random.default_rng(0)
    pos = simulate_brownian(300, 33, dt=2.0, D=D, box=(4000, 4000),
                            init_positions=rng.uniform(200, 3800, (300, 2)),
                            rng=rng)[1]
    meta = {"frames_per_latent": 2, "D": D, "drift": [0.0, 0.0], "msd_max_lag": 20}
    ratio, link = window_msd([pos[:, t] for t in range(pos.shape[1])], meta)
    assert link.n_full == 300
    assert abs(ratio - 1.0) < 0.1


def test_pool_msd_weights_by_full_tracks_and_skips_empty_windows():
    from jepa_lptem.eval.tracking import LinkResult
    empty = np.zeros((0, 2, 2))
    results = [(1.0, LinkResult(empty, 1.0, 3, 3)),
               (2.0, LinkResult(empty, 0.5, 2, 1)),
               (None, LinkResult(empty, 0.0, 1, 0))]  # < 2 full tracks: no ratio
    out = pool_msd(results, "_gt")
    assert out["msd_ratio_pooled_gt"] == 1.25          # (3*1 + 1*2) / 4
    assert out["msd_windows_used_gt"] == 2
    assert out["track_coverage_mean_gt"] == 0.5        # coverage counts every window


def test_edge_counts_count_particles_straddling_the_crop_border():
    from gate1_secondary import edge_counts
    pos = np.array([
        [[128.0, 128.0], [128.0, 128.0], [5.0, 128.0]],     # centre, centre, 5 px inside top edge
        [[128.0, 128.0], [5.0, 100.0], [128.0, 128.0]],     # inside, 5 px from the top edge
        [[128.0, 128.0], [128.0, 128.0], [100.0, 262.0]],   # outside, 6 px past the right edge
    ])                                                       # (N=3, T=3, 2)
    assert edge_counts(pos, crop=256).tolist() == [0, 1, 2]
    far = np.array([[[128.0, 300.0]]])                       # well outside: not ambiguous
    assert edge_counts(far, crop=256).tolist() == [0]


def test_clip_predictions_stitch_windows_by_start():
    from train_probes import clip_predictions

    class Cache:  # two overlapping windows of one clip: latents 0-3 and 2-5
        def load(self, i):
            t0 = 2 * i
            feats = np.arange(t0, t0 + 4, dtype=np.float32)[:, None, None]
            meta = {"clip": "c", "start": 2 * t0, "frames_per_latent": 2}
            return feats, None, None, meta

    def att(x):  # reads the latent index back out
        return x[:, 0, :].repeat(1, 3)

    att.eval = lambda: None
    out = clip_predictions(att, Cache(), [0, 1], (np.zeros(3), np.ones(3)), "cpu")
    assert out["c"].shape == (6, 3)
    assert np.allclose(out["c"][:, 0], np.arange(6))
