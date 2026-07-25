"""How does our synthetic video compare to real LPTEM video?

Three comparisons, chosen because they survive the one thing we cannot undo:
the real frames are 8-bit JPEGs that someone already contrast-stretched, so
*absolute* intensity is not comparable. These are not.

1. Apparent particle size, in pixels. The single most important appearance
   parameter for a ViT: V-JEPA tokenizes into 16x16 patches, so a particle
   spanning 2 patches and one spanning 6 are different problems regardless of
   how physically correct the nanometres are.
2. Radial power spectrum of I / mean(I). Scale-free and stretch-invariant.
   The high-frequency plateau is the noise floor; the low-frequency bump is
   particle structure. One curve carries both.
3. Frame-to-frame difference. Real LPTEM video changes enormously between
   frames -- almost all of it shot noise, not motion. This is what a video
   model actually sees, and single-frame statistics say nothing about it.

Usage:
    uv run python scripts/compare_to_real.py [--out data/synthetic/smoke]
"""
from __future__ import annotations

import argparse
import glob
import re
import zipfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from verify_clips import detect_dark_blobs


def load_sequence(zip_path, pred, limit=40):
    """Decode a frame sequence from a zip, ordered by trailing frame number."""
    zf = zipfile.ZipFile(zip_path)
    names = sorted((x for x in zf.namelist() if pred(x)),
                   key=lambda s: int(re.findall(r"(\d+)", s)[-1]))
    out = []
    for n in names[:limit]:
        a = cv2.imdecode(np.frombuffer(zf.read(n), np.uint8), cv2.IMREAD_GRAYSCALE)
        if a is not None:
            out.append(a.astype(float))
    return np.stack(out) if out else np.zeros((0, 1, 1))


def radial_psd(frames, k=8):
    """Azimuthally averaged power spectrum of I/mean(I), vs cycles per pixel.

    Averaged over frames: a single frame from one clip would report whichever
    beam flux that clip happened to draw, not the distribution we generate.
    """
    profs = [_one_psd(f) for f in frames[:k]]
    freq = np.arange(len(profs[0])) / min(frames.shape[1:])
    return freq, np.mean(profs, axis=0)


def _one_psd(frame):
    f = frame / frame.mean()
    p = np.abs(np.fft.fftshift(np.fft.fft2(f - f.mean()))) ** 2
    h, w = p.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - h // 2, xx - w // 2).astype(int)
    nbins = min(h, w) // 2
    return np.bincount(r.ravel(), p.ravel())[:nbins] / np.bincount(r.ravel())[:nbins]


def apparent_radius_px(frames, k=12):
    """Median equivalent-disk radius of detected dark blobs, in pixels."""
    radii = []
    for fr in frames[:k]:
        _, lab = detect_dark_blobs(fr)
        areas = np.bincount(lab.ravel())[1:]
        radii += [np.sqrt(a / np.pi) for a in areas if a >= 80]
    return float(np.median(radii)) if radii else float("nan")


def temporal_delta(frames):
    """Mean |I(t+1) - I(t)| relative to the mean level -- what a video model sees."""
    if len(frames) < 2:
        return float("nan")
    return float(np.abs(np.diff(frames, axis=0)).mean() / frames.mean())


def centre_crop(frames, size=512):
    """Crop to a common pixel window -- NOT a resize.

    Resizing to a common display size would silently rescale apparent particle
    size, which is the very quantity being compared. Cropping keeps one pixel
    meaning one pixel in every panel.
    """
    if frames.shape[1] <= size:
        return frames
    top = (frames.shape[1] - size) // 2
    left = (frames.shape[2] - size) // 2
    return frames[:, top:top + size, left:left + size]


def write_gif(panels, path, n_frames=40, fps=7, suptitle=None):
    """Animate one or more equally-sized frame stacks side by side."""
    n = min([n_frames] + [len(f) for _, f in panels])
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6))
    axes = np.atleast_1d(axes)
    ims = []
    for ax, (name, fr) in zip(axes, panels):
        ims.append(ax.imshow(fr[0], cmap="gray"))
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)

    def upd(k):
        for im, (_, fr) in zip(ims, panels):
            im.set_data(fr[k % len(fr)])
        return ims

    anim = animation.FuncAnimation(fig, upd, frames=n, interval=1000 // fps, blit=True)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[wrote] {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/synthetic/smoke"))
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    real = load_sequence("data/sam-em/experimentalAuNanorods.zip",
                         lambda x: x.endswith(".jpeg"), limit=75)
    theirs = load_sequence("data/sam-em/ablationValidationSet.zip",
                           lambda x: "/jpeg_images/" in x and x.endswith(".jpg"))
    clips = sorted(glob.glob(str(args.out / "*.npz")))
    ours = np.concatenate([np.load(f)["frames"][:8].astype(float) for f in clips])
    fluxes = [float(np.load(f)["beam_flux"]) for f in clips]

    sources = [("REAL experimental", real), ("THEIRS (LPTEMsimulator)", theirs),
               ("OURS", ours)]
    print(f"{'source':26s} {'n':>4s} {'size':>12s} {'radius px':>10s} {'|dI|/mean':>11s}")
    for name, fr in sources:
        print(f"{name:26s} {len(fr):4d} {str(fr.shape[1:]):>12s} "
              f"{apparent_radius_px(fr):10.1f} {temporal_delta(fr):11.3f}")
    print(f"\n  our clips span beam_flux {min(fluxes):.1f}-{max(fluxes):.1f}")
    print("  NOTE: absolute intensity is NOT comparable -- the real frames are "
          "JPEGs that\n        were already contrast-stretched. Only shape, scale "
          "and dynamics are.")

    fig = plt.figure(figsize=(14, 8))
    for i, (name, fr) in enumerate(sources):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.imshow(fr[0], cmap="gray")
        ax.set_title(f"{name}\n{fr.shape[1]}x{fr.shape[2]}", fontsize=9)
        ax.axis("off")

    ax = fig.add_subplot(2, 3, 4)
    for name, fr in sources:
        f, p = radial_psd(fr)
        ax.loglog(f[1:], p[1:], label=name, lw=1.4)
    ax.set_xlabel("cycles / pixel"); ax.set_ylabel("power (I/mean)")
    ax.set_title("Radial power spectrum", fontsize=10); ax.legend(fontsize=7)

    ax = fig.add_subplot(2, 3, 5)
    for name, fr in sources:
        d = np.abs(np.diff(fr[:12], axis=0)).ravel() / fr.mean()
        ax.hist(d, bins=60, range=(0, 1.2), histtype="step", density=True, label=name)
    ax.set_xlabel("|I(t+1) - I(t)| / mean"); ax.set_ylabel("density")
    ax.set_title("Frame-to-frame change", fontsize=10); ax.legend(fontsize=7)

    ax = fig.add_subplot(2, 3, 6)
    for name, fr in sources:
        v = (fr[0] / fr[0].mean()).ravel()
        ax.hist(v, bins=80, range=(0, 2.2), histtype="step", density=True, label=name)
    ax.set_xlabel("I / mean"); ax.set_ylabel("density")
    ax.set_title("Intensity distribution (stretch-normalised)", fontsize=10)
    ax.legend(fontsize=7)

    plt.tight_layout()
    path = args.out / "compare_to_real.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    print(f"\n[wrote] {path}")

    if args.no_gif:
        return
    # Real video on its own, full field of view.
    write_gif([("REAL experimental Au nanorods  (1024x1024)", real)],
              args.out / "real_video.gif", n_frames=75)
    # Side by side at MATCHED PIXEL SCALE: 512 px cropped from each, never resized,
    # so a particle that looks bigger really is bigger in the pixels the model sees.
    write_gif([("REAL (512px crop)", centre_crop(real)),
               ("THEIRS (512px crop)", centre_crop(theirs)),
               ("OURS (512px native)", ours)],
              args.out / "compare_side_by_side.gif",
              suptitle="matched pixel scale -- cropped, not resized")


if __name__ == "__main__":
    main()
