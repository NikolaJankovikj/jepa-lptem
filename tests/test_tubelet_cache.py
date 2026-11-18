"""Step 15 M1: raw tubelets line up with the per-frame feature cache."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from extract_features import crop_and_align_gt
from tubelet_cache import TubeletSet, build

T, FULL, CROP = 8, 64, 32


def write_clip(path: Path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    frames = np.arange(T, dtype=np.uint8)[:, None, None] * np.ones(
        (1, FULL, FULL), np.uint8
    )
    frames[:, FULL // 2, FULL // 2] = 200 + seed  # marks the crop centre
    np.savez(
        path,
        frames=frames,
        positions=rng.uniform(0, FULL, (n, T, 2)),
        radii=rng.uniform(2, 5, (n, T)),
        process=np.array("diffusion"),
        drift=np.zeros(2),
    )


def test_tubelets_match_the_cache_window_convention(tmp_path):
    write_clip(tmp_path / "clip_0000_diffusion_physics.npz", n=3, seed=0)
    write_clip(tmp_path / "clip_0001_diffusion_physics.npz", n=5, seed=1)
    build(tmp_path, tmp_path / "out", CROP)

    ds = TubeletSet(tmp_path / "out", clips={"clip_0001_diffusion_physics"})
    assert len(ds) == T // 2
    frames, attentive, occ, _ = ds[2]
    assert frames.shape == (2, CROP, CROP)
    assert list(frames[:, 0, 0]) == [4, 5]  # latent 2 = input frames 4, 5
    assert frames[0, CROP // 2, CROP // 2] == 201  # centre crop, this clip

    # cache window starting at frame 4: its latent 0 is latent 2 here
    with np.load(tmp_path / "clip_0001_diffusion_physics.npz") as z:
        pos, rad = crop_and_align_gt(z["positions"], z["radii"], 4, 4, FULL, CROP)
    gt = np.load(tmp_path / "out" / "gt.npz")
    assert np.allclose(gt["radii"][1, :, 2:4], rad)
    assert np.allclose(gt["positions"][1, :, 2:4], pos)
    assert gt["radii"][0, 3:].sum() == 0  # padding of the 3-particle clip
    assert attentive[0] == (rad[:, 0] > 0).sum()
    assert occ.sum() <= attentive[0]
