"""Step 15 M2: the pieces that make the CNN / LoRA comparison fair."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from extract_features import load_processor
from train_encoder_compare import TubeletCNN, Windowing, vjepa_input
from train_probes import train_clips


def test_vjepa_input_matches_the_cache_processor():
    # LoRA must see exactly what the frozen cache was encoded from
    frames = np.random.default_rng(0).integers(0, 256, (2, 32, 32), dtype=np.uint8)
    rgb = np.repeat(frames[..., None], 3, axis=-1)
    want = load_processor()(list(rgb), return_tensors="pt")["pixel_values_videos"]
    got = vjepa_input(torch.from_numpy(frames)[None])
    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=1e-5)


def test_cnn_gives_one_token_per_16px_cell():
    tokens = TubeletCNN()(torch.zeros(3, 2, 256, 256, dtype=torch.uint8))
    assert tokens.shape == (3, 256, 256)


def test_train_fractions_are_nested():
    split = {"train": {f"clip_{i:04d}" for i in range(50)}}
    small, mid, full = (train_clips(split, f, seed=0) for f in (0.1, 0.3, 1.0))
    assert len(small) == 5 and len(mid) == 15 and full == split["train"]
    assert small < mid < full


def test_windows_are_the_cache_windows():
    # dynamics-v1: 100 frames, windows of 64 frames at stride 36 -> w000, w036
    win = Windowing(64, 36)
    assert win.slices(50) == [slice(0, 32), slice(18, 50)]
    assert len(win.frame_index(50)) == 64
