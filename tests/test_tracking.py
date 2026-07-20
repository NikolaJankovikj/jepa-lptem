import numpy as np

from jepa_lptem.eval.tracking import heatmap_to_centroids, link_tracks


def test_heatmap_peaks_recovered():
    hm = np.zeros((16, 16))
    hm[3, 4] = 1.0
    hm[12, 9] = 0.8
    cents = heatmap_to_centroids(hm, threshold=0.5)
    got = sorted(map(tuple, np.round(cents).astype(int).tolist()))
    assert got == [(3, 4), (12, 9)]


def test_heatmap_empty_below_threshold():
    hm = np.full((8, 8), 0.1)
    cents = heatmap_to_centroids(hm, threshold=0.5)
    assert cents.shape == (0, 2)


def test_linking_two_drifting_particles():
    # two particles drift by (+1, +1) per frame; well separated, small steps
    frames = [np.array([[0.0, 0.0], [10.0, 10.0]]) + t for t in range(5)]
    res = link_tracks(frames, max_dist=3.0)
    assert res.tracks.shape == (2, 5, 2)
    assert res.coverage == 1.0
    # tracks are sorted by first-frame position, so track 0 starts near the origin
    assert np.allclose(res.tracks[0, :, 0], np.arange(5))
    assert np.allclose(res.tracks[1, :, 0], 10 + np.arange(5))


def test_linking_reports_dropout_coverage():
    # 3 particles in frames 0-1; the third vanishes from frame 2 on
    frames = [
        np.array([[0.0, 0.0], [10.0, 10.0], [5.0, 5.0]]),
        np.array([[1.0, 1.0], [11.0, 11.0], [6.0, 6.0]]),
        np.array([[2.0, 2.0], [12.0, 12.0]]),
        np.array([[3.0, 3.0], [13.0, 13.0]]),
        np.array([[4.0, 4.0], [14.0, 14.0]]),
    ]
    res = link_tracks(frames, max_dist=3.0)
    # only the two full-length tracks are returned...
    assert res.tracks.shape == (2, 5, 2)
    # ...but the dropout is *reported*, not silently hidden
    assert res.n_particles == 3
    assert res.n_full == 2
    assert abs(res.coverage - 2 / 3) < 1e-9
