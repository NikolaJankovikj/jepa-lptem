"""RQ2's events, negatives and scores on cases with a known answer."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from event_anticipation import (
    CTX,
    ClipGT,
    auc,
    auc_table,
    events,
    lead_times,
    one_step_error,
    sample_points,
    score,
    token_negatives,
)


def still_clip(centres, merges, n_video=100) -> ClipGT:
    """Particles that never move, all present, radius 10."""
    centres = np.asarray(centres, dtype=float)
    pos = np.repeat(centres[:, None], n_video, axis=1)
    return ClipGT(
        pos,
        np.full(pos.shape[:2], 10.0),
        np.asarray(merges).reshape(-1, 3),
        n_video // 2,
    )


def test_events_need_both_particles_in_the_crop_and_enough_context():
    gt = still_clip(
        [[100, 100], [110, 110], [200, 200]], [[41, 0, 1], [61, 2, 1], [15, 0, 2]]
    )
    gt.rad[1, 60] = 0  # absorbed particle outside the crop just before the second merge
    found = events(gt)
    assert [e["t"] for e in found] == [20]  # 41 // 2; the third merge is before t = 10
    assert found[0]["token"] == (6, 6)


def test_negatives_keep_far_particles_only():
    gt = still_clip([[100, 100], [110, 110], [130, 130], [240, 20]], [[41, 0, 1]])
    near_t = [n["token"] for n in token_negatives(gt) if abs(n["t"] - 20) <= 3]
    assert near_t and all(
        tok == (15, 1) for tok in near_t
    )  # 0, 1 merge; 2 is 28 px away
    far_t = {n["token"] for n in token_negatives(gt) if n["t"] == 40}
    assert far_t == {(6, 6), (8, 8), (15, 1)}


def test_auc_known_values():
    assert auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert auc(np.array([1.0]), np.array([2.0, 3.0])) == 0.0
    assert auc(np.array([1.0, 1.0]), np.array([1.0])) == 0.5


def test_a_signal_that_spikes_at_merges_beats_one_that_does_not():
    rng = np.random.default_rng(0)
    gts, spike, flat = [], [], []
    for c in range(20):
        gt = still_clip(
            [[60, 60], [70, 70], [200, 200]], [[2 * (15 + c % 10) + 1, 0, 1]]
        )
        noise = rng.normal(0, 1, (gt.n_lat, 256))
        loud = noise.copy()
        loud[15 + c % 10] += 3.0
        gts.append(gt)
        spike.append(loud)
        flat.append(rng.normal(0, 1, (gt.n_lat, 256)))
    points = sample_points(gts, np.arange(20), per_token=True)
    table = auc_table(
        score(points, {"spike": spike, "flat": flat}), (("spike", "flat"),)
    )
    assert table["auc"]["spike"][0]["ci"][0] > 0.5
    assert table["auc"]["spike"][-1]["ci"][0] < 0.5 < table["auc"]["spike"][-1]["ci"][1]
    assert table["diff"]["spike-flat"][0]["ci"][0] > 0


def test_frame_level_points_carry_no_token():
    gt = still_clip([[60, 60], [70, 70]], [[41, 0, 1]])
    points = sample_points([gt], [0], per_token=False)
    assert points.events == [(0, 20, None)]
    assert all(k is None for _, _, k in points.negatives)
    frames = score(points, {"s": [np.arange(gt.n_lat, dtype=float)]})
    assert frames.pos["s"][0].tolist() == [20.0]


def test_lead_time_is_the_first_lag_over_the_clip_bar():
    values = {lag: np.array([0.0, 0.0, np.nan]) for lag in (-4, -3, -2, -1, 0)}
    values[-1][0] = values[0][0] = 5.0  # event 0 crosses at -1; event 1 never
    leads = lead_times(values, np.zeros(10), np.zeros(10, int), np.zeros(3, int))
    assert leads == [-1, None]  # event 2 lacks a lead lag and is left out


def test_one_step_error_learns_a_linear_drift():
    rng = np.random.default_rng(0)
    rate = rng.uniform(-1, 1, (120, 1, 1))
    z = rate * np.arange(30)[None, :, None] * rng.normal(size=(1, 1, 80)) / 4
    z = z + rng.normal(0, 0.05, z.shape)
    split = {
        "train": np.arange(80),
        "val": np.arange(80, 100),
        "test": np.arange(100, 120),
    }
    err = one_step_error(z, split)
    assert np.isnan(err[:, :CTX]).all()
    change = np.abs(z[:, 1:] - z[:, :-1]).mean(-1)
    assert np.nanmean(err[100:, CTX:]) < 0.5 * change[100:, CTX - 1 :].mean()
