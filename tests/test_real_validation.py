"""Step 13's bookkeeping on cases with a known answer."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from pooled_gate1 import CTX, HORIZON
from real_features import process_and_stratum, tiles
from real_validation import SNIPPET_STRIDE, gain_entry, snippets


def test_tiles_are_non_overlapping_crops_with_an_even_frame_count():
    out = tiles(np.zeros((11, 600, 300), np.uint8))
    assert sorted(out) == ["r0c0", "r256c0"]
    assert all(t.shape == (10, 256, 256) for t in out.values())


def test_fresh_clips_put_one_in_five_on_the_experiment_stratum_for_every_process():
    rows = [process_and_stratum(i) for i in range(25)]
    experiment = [p for p, s in rows if s == "experiment"]
    assert len(experiment) == 5
    assert len(set(experiment)) == 5


def test_snippets_cover_the_sequence_at_the_stride_and_skip_short_ones():
    z = np.arange(50)[:, None] * np.ones((1, 3))
    snip, starts = snippets(z)
    assert starts.tolist() == list(range(0, 50 - CTX - HORIZON + 1, SNIPPET_STRIDE))
    assert snip[1, 0, 0] == SNIPPET_STRIDE
    assert len(snippets(z[:20])[0]) == 0


def test_gain_is_measured_against_the_better_baseline():
    n = 6
    err = {
        "linear": np.full((n, HORIZON), 1.0),
        "persistence": np.full((n, HORIZON), 3.0),
        "ctxmean": np.full((n, HORIZON), 2.0),
    }
    entry = gain_entry(err, np.arange(n))
    assert entry["baseline"] == "ctxmean"
    assert entry["gain_24"] == 1.0
