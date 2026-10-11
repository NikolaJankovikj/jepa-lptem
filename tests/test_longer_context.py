"""Longer-context snippets line up across contexts, and the paired reading needs a rise."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from longer_context import HORIZON, Stitched, cut, stitch, tracks


def test_stitch_rebuilds_the_whole_clip_from_its_two_windows():
    whole = np.arange(2 * 50, dtype=float).reshape(2, 50, 1)
    windows, meta = [], []
    for clip in (1, 0):  # cache order need not be clip order
        for start, first in ((36, 18), (0, 0)):
            windows.append(whole[clip, first : first + 32])
            meta.append({"clip": f"c{clip}", "start": start})
    windows = np.stack(windows)
    seq, targets, clip_meta = stitch(windows, windows, meta)
    assert np.array_equal(seq, whole)
    assert np.array_equal(targets, whole)
    assert [m["clip"] for m in clip_meta] == ["c0", "c1"]


def test_every_context_forecasts_the_same_frames():
    seq = np.tile(np.arange(50, dtype=float)[None, :, None], (3, 1, 1))
    meta = [{"clip": f"c{i}", "frames_per_latent": 2} for i in range(3)]
    clips = Stitched(seq, seq, meta, Path())
    short, _, short_meta = cut(clips, 8, (16, 26))
    long, _, long_meta = cut(clips, 16, (16, 26))
    assert short.shape == (6, 8 + HORIZON, 1)
    assert long.shape == (6, 16 + HORIZON, 1)
    assert np.array_equal(short[:, -HORIZON:], long[:, -HORIZON:])
    assert [m["start"] for m in short_meta[:2]] == [16, 36]  # video frames of the first context frame
    assert [m["start"] for m in long_meta[:2]] == [0, 20]


def test_tracks_needs_a_rise_on_every_judged_process():
    rng = np.random.default_rng(0)
    process = np.repeat(["diffusion", "reaction", "static"], 60)
    clips = np.arange(len(process)) // 2
    short = {"process": process, "clips": clips}
    short |= {c: rng.normal(0.2, 0.05, len(process)) for c in ("linear", "transformer")}
    rise = np.where(process == "reaction", 0.0, 0.1)  # reaction does not rise
    long = short | {"linear": short["linear"] + 0.1, "transformer": short["transformer"] + rise}
    result = tracks(short, long)
    assert result["linear"]["tracks"]
    assert not result["transformer"]["tracks"]
