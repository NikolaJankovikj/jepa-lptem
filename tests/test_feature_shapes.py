"""Step 6 extraction: pure alignment logic locally, encoder shape check on GPU."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from extract_features import TUBELET, crop_and_align_gt, window_starts


def test_window_starts_plan_default_covers_100_frames():
    # 100-frame videos, window 64, stride 36 -> exactly frames 0-63 and 36-99.
    assert window_starts(100, 64, 36) == [0, 36]
    assert window_starts(48, 48, 48) == [0]
    assert window_starts(40, 64, 36) == []  # too short -> skipped, not padded


def test_crop_and_align_gt_shifts_samples_and_zeroes_outside():
    # One particle drifting right across a 512 frame; centre crop 256 -> offset 128.
    t_total = 8
    pos = np.zeros((1, t_total, 2))
    pos[0, :, 0] = 256.0                       # row: centre of the frame
    pos[0, :, 1] = np.linspace(120, 400, t_total)  # col: enters and exits the crop
    rad = np.full((1, t_total), 5.0)
    p, r = crop_and_align_gt(pos, rad, start=0, window=t_total,
                             full_size=512, crop=256)
    assert p.shape == (1, t_total // TUBELET, 2)
    # sampled at even frames, shifted by 128
    assert np.allclose(p[0, :, 1], pos[0, ::TUBELET, 1] - 128.0)
    inside = (p[0, :, 1] >= 0) & (p[0, :, 1] < 256)
    assert np.all(r[0, inside] == 5.0)
    assert np.all(r[0, ~inside] == 0.0)
    assert (~inside).any() and inside.any()  # the case actually exercises both


def test_crop_and_align_gt_respects_window_start():
    pos = np.zeros((1, 10, 2))
    pos[0, :, 0] = np.arange(10) + 200.0  # distinct value per frame
    pos[0, :, 1] = 256.0
    rad = np.ones((1, 10))
    p, _ = crop_and_align_gt(pos, rad, start=4, window=4, full_size=512, crop=256)
    assert np.allclose(p[0, :, 0], np.array([204.0, 206.0]) - 128.0)  # frames 4, 6


@pytest.mark.gpu
def test_encoder_shape_64_frames():
    """The plan's Step-6 smoke: 64 frames at 256^2 -> (1, 32*16*16, 1024)."""
    import torch
    from transformers import VJEPA2Model, VJEPA2VideoProcessor

    from extract_features import MODEL_NAME
    model = VJEPA2Model.from_pretrained(MODEL_NAME).eval().cuda()
    proc = VJEPA2VideoProcessor.from_pretrained(MODEL_NAME)
    vid = np.random.randint(0, 255, (64, 256, 256, 3), dtype=np.uint8)
    with torch.no_grad():
        out = model(**proc(list(vid), return_tensors="pt").to("cuda"))
    assert out.last_hidden_state.shape == (1, 32 * 16 * 16, 1024)
