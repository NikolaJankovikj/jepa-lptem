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
from matplotlib import animation, patches
from scipy import ndimage
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
    """Per-pixel power, so frames of different sizes are actually comparable.

    ``np.fft.fft2`` is unnormalized, so ``E|F(k)|^2 = N sigma^2`` -- the level
    scales with the pixel count. Without the ``/ f.size`` this panel compared our
    512^2 renders against 1024^2 real and LPTEMsimulator frames and read a flat
    **4x** deficit at every frequency that was purely image size: cropping the same
    real frames 1024 -> 512 reproduced it (measured ratio 3.7-4.1 across the band).
    That is the wrong way for a plot to be wrong -- it made the one statistic we
    had *not* closed look worse than it is.
    """
    f = frame / frame.mean()
    p = np.abs(np.fft.fftshift(np.fft.fft2(f - f.mean()))) ** 2 / f.size
    h, w = p.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - h // 2, xx - w // 2).astype(int)
    nbins = min(h, w) // 2
    return np.bincount(r.ravel(), p.ravel())[:nbins] / np.bincount(r.ravel())[:nbins]


def radial_profile(frames, centres_fn, k=8, rmax=40):
    """Mean intensity vs distance from a particle centre, pooled over particles."""
    prof = np.zeros(rmax)
    cnt = np.zeros(rmax)
    yy, xx = np.mgrid[-rmax:rmax + 1, -rmax:rmax + 1]
    dist = np.hypot(yy, xx).astype(int)
    for i, f in enumerate(frames[:k]):
        g = f.astype(float)
        g = g - ndimage.gaussian_filter(g, 40)  # flatten illumination gradient
        for (r, c) in centres_fn(f, i):
            r, c = int(round(r)), int(round(c))
            if (r - rmax < 0 or c - rmax < 0
                    or r + rmax + 1 > g.shape[0] or c + rmax + 1 > g.shape[1]):
                continue
            patch = g[r - rmax:r + rmax + 1, c - rmax:c + rmax + 1]
            for d in range(rmax):
                m = dist == d
                prof[d] += patch[m].sum()
                cnt[d] += m.sum()
    return prof / np.maximum(cnt, 1)


def apparent_radius_px(frames, centres_fn=None, k=8):
    """Particle radius from the half-depth point of the radial intensity profile.

    NOT from thresholded connected components. That was the first approach and it
    is biased low -- a threshold only captures the dark core of a noisy blob, and
    the bias grows as SNR falls, so it under-measures exactly where it matters.
    Validated against known ground truth on generated clips: it recovers
    true 9.5 px -> 8.0 and true 4.9 px -> 5.0, whereas the component method read
    real video as 6.1 px when the profile says 8.0.

    A scale-selection variant (which disk radius maximises the matched-filter
    response) was also tried and rejected: it validates on high-SNR clips but reads
    a true 4.9 px particle as 22 px at low SNR, where noise correlation dominates.
    """
    if centres_fn is None:
        centres_fn = lambda f, i: detect_dark_blobs(f)[0]
    p = radial_profile(frames, centres_fn, k=k)
    base = p[-8:].mean()
    depth = base - p[0]
    if depth <= 0:
        return float("nan")
    for d in range(len(p)):
        if p[d] >= base - 0.5 * depth:
            return float(d)
    return float("nan")


def particle_depth(frames, k=8):
    """Median ``1 - I_particle / I_background``, averaged over each blob's area.

    Area-averaged rather than centre-pixel because the detector thresholds on
    darkness, so single pixels are selection-biased towards dark noise. Background
    is the median for the reason given in :func:`to_common_grey`.

    **Reported for information, not as a calibration target.** The real frames were
    contrast-stretched to 0-255 before publication, and such a stretch does not
    preserve intensity ratios -- it inflates depth by an unknown factor. Apparent
    size and frame-to-frame change survive the stretch; this does not.
    """
    deps = []
    for fr in frames[:k]:
        _, lab = detect_dark_blobs(fr)
        bg = float(np.median(fr))
        for i in np.unique(lab):
            if i == 0:
                continue
            m = lab == i
            if m.sum() >= 80:
                deps.append(1.0 - fr[m].mean() / bg)
    return float(np.median(deps)) if deps else float("nan")


def temporal_delta(frames):
    """Mean |I(t+1) - I(t)| relative to the mean level -- what a video model sees."""
    if len(frames) < 2:
        return float("nan")
    return float(np.abs(np.diff(frames, axis=0)).mean() / frames.mean())


def matched_filter(frame, radius):
    """Correlate with a disk of the expected radius, in units of the noise sigma.

    The point of this is evidentiary. Experiment-grade clips look like pure noise to
    the eye and defeat the crude threshold detector, which invites the conclusion
    that the data is unusable. It is not: a particle's signal is spread over its
    whole area, so integrating over that area recovers it. Any competent detector --
    and a ViT patch -- does this implicitly. Real video needs exactly the same
    treatment, which is the real point of comparison.
    """
    f = frame.astype(float)
    f = f - ndimage.gaussian_filter(f, 40)  # flatten illumination gradient, which
    # otherwise dominates the response on real frames and hides the particles
    yy, xx = np.mgrid[-int(2 * radius):int(2 * radius) + 1,
                      -int(2 * radius):int(2 * radius) + 1]
    disk = (yy ** 2 + xx ** 2 <= radius ** 2).astype(float)
    disk /= disk.sum()
    smooth = ndimage.correlate(f, disk, mode="reflect")
    return (smooth - np.median(smooth)) / (smooth.std() + 1e-9)


def write_detectability(our_clip, our_pos, our_rad, phys_clip, real, path):
    """Six panels: is the signal there, and is real any easier?"""
    fig, ax = plt.subplots(2, 3, figsize=(13, 8.6))
    rr = float(np.median(our_rad[our_rad > 0])) or 6.0
    panels = [
        (ax[0, 0], to_common_grey(our_clip[None])[0], "OURS experiment-grade\nraw", None),
        (ax[0, 1], -matched_filter(our_clip, rr), f"OURS matched-filtered (r={rr:.0f}px)\n"
         "the same frame", None),
        (ax[0, 2], to_common_grey(our_clip[None])[0], "OURS raw + ground truth\n"
         "the particles were always there", (our_pos, our_rad)),
        (ax[1, 0], to_common_grey(real[None])[0], "REAL\nraw (512px crop)", None),
        (ax[1, 1], -matched_filter(real, 6.1), "REAL matched-filtered (r=6px)\n"
         "just as invisible raw, just as recoverable", None),
        (ax[1, 2], to_common_grey(phys_clip[None])[0], "OURS physics-grade\n"
         "4 of 5 clips look like this", None),
    ]
    for a, img, title, gt in panels:
        a.imshow(img, cmap="gray")
        a.set_title(title, fontsize=9)
        a.axis("off")
        if gt is not None:
            pos, rad = gt
            for (r, c), q in zip(pos, rad):
                if q > 0:
                    a.add_patch(patches.Circle((c, r), max(q, 4) * 2.2, fill=False,
                                               ec="lime", lw=1.1))
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {path}")


def containment_report(clip_files, real):
    """The generator's exit criterion: is real *inside* our distribution?

    Not "does our mean match real's mean". This dataset is domain randomisation:
    its job is to span a range of imaging conditions wide enough that real data
    falls inside it, so a model trained here has already seen real conditions.
    Chasing a point estimate of one 75-frame experiment -- exported by one person,
    contrast-stretched by an unknown factor -- fits an artefact, and worse, it has
    no fixed point: every appearance statistic moves when its estimator is
    refined, so "close enough to real" never terminates. Containment terminates,
    and it is falsifiable: for each statistic that survives an unknown stretch,
    real must lie between our min and our max.

    Only two statistics qualify. Absolute contrast does not survive the stretch
    (see :func:`particle_depth`), so it is reported but never gated on.

    Our side measures from ground-truth centres, since we have them; real has no
    choice but the detector. The size estimator is the same on both sides and was
    validated against known radii before being trusted here.
    """
    rows = []
    for f in clip_files:
        z = np.load(f)
        fr = z["frames"].astype(float)
        pos, rad = z["positions"], z["radii"]
        centres = lambda _f, i, p=pos, r=rad: p[:, i, :][r[:, i] > 0]
        rows.append((apparent_radius_px(fr[:8], centres), temporal_delta(fr)))
    ours = np.array(rows)  # (n_clips, 2)
    real_c = centre_crop(real)
    targets = [("apparent radius (px)", apparent_radius_px(real_c)),
               ("|dI|/mean", temporal_delta(real_c))]

    print(f"\n{'statistic':22s} {'ours min':>9s} {'ours max':>9s} {'real':>8s}")
    ok = True
    for j, (name, target) in enumerate(targets):
        lo, hi = np.nanmin(ours[:, j]), np.nanmax(ours[:, j])
        inside = lo <= target <= hi
        ok &= inside
        print(f"{name:22s} {lo:9.3f} {hi:9.3f} {target:8.3f}   "
              f"{'PASS' if inside else 'FAIL'}")
    print(f"\n  {'CONTAINED' if ok else 'NOT CONTAINED'}: real experimental video "
          f"{'is' if ok else 'is NOT'} inside the generated distribution"
          f" ({len(clip_files)} clips).")
    return ok


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


def to_common_grey(frames):
    """Put every source on one grey scale: background -> 128, whatever its units.

    Without this each panel is autoscaled to its own min/max, so apparent contrast
    is set by the source's dynamic range rather than compared between sources --
    and the real frames (already contrast-stretched to 0-255) get judged against
    ours (physical scale, background at 64). Median is the background reference
    because the real histogram is very broad (p5=38, p95=179): a high percentile
    over-reads its background and exaggerates particle depth.
    """
    bg = np.median(frames)
    return np.clip(frames / max(bg, 1e-6) * 128.0, 0, 255)


def write_gif(panels, path, n_frames=40, fps=7, suptitle=None):
    """Animate one or more equally-sized frame stacks side by side."""
    panels = [(name, to_common_grey(fr)) for name, fr in panels]
    n = min([n_frames] + [len(f) for _, f in panels])
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6))
    axes = np.atleast_1d(axes)
    ims = []
    for ax, (name, fr) in zip(axes, panels):
        ims.append(ax.imshow(fr[0], cmap="gray", vmin=0, vmax=255))
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
    # Pooled across clips for the statistics...
    ours = np.concatenate([np.load(f)["frames"][:8].astype(float) for f in clips])
    fluxes = [float(np.load(f)["beam_flux"]) for f in clips]
    # ...but ONE continuous clip for the animation. Concatenating clips makes the
    # GIF cut between scenes every 8 frames, which reads as a rendering artefact.
    exp_clips = [f for f in clips if f.endswith("_experiment.npz")] or clips
    one_clip = np.load(exp_clips[0])["frames"].astype(float)

    sources = [("REAL experimental", real), ("THEIRS (LPTEMsimulator)", theirs),
               ("OURS", ours)]
    print(f"{'source':26s} {'n':>4s} {'size':>12s} {'radius px':>10s} "
          f"{'depth*':>7s} {'|dI|/mean':>11s}")
    for name, fr in sources:
        print(f"{name:26s} {len(fr):4d} {str(fr.shape[1:]):>12s} "
              f"{apparent_radius_px(fr):10.1f} {particle_depth(fr):7.3f} "
              f"{temporal_delta(fr):11.3f}")
    containment_report(clips, real)

    print(f"\n  our clips span beam_flux {min(fluxes):.1f}-{max(fluxes):.1f}")
    print("  * depth is informational only: the real frames were contrast-stretched "
          "to 0-255\n    before publication, which inflates it by an unknown factor. "
          "Apparent size and\n    frame-to-frame change survive that stretch and are "
          "the comparable numbers.")

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

    exp_z = np.load(exp_clips[0])
    phys = [f for f in clips if f.endswith("_physics.npz")] or clips
    write_detectability(exp_z["frames"][20].astype(float), exp_z["positions"][:, 20, :],
                        exp_z["radii"][:, 20],
                        np.load(phys[0])["frames"][20].astype(float),
                        centre_crop(real)[0], args.out / "detectability.png")

    if args.no_gif:
        return
    # Real video on its own, full field of view.
    write_gif([("REAL experimental Au nanorods  (1024x1024)", real)],
              args.out / "real_video.gif", n_frames=75)
    # Side by side at MATCHED PIXEL SCALE: 512 px cropped from each, never resized,
    # so a particle that looks bigger really is bigger in the pixels the model sees.
    write_gif([("REAL (512px crop)", centre_crop(real)),
               ("THEIRS (512px crop)", centre_crop(theirs)),
               ("OURS experiment-grade", one_clip),
               ("OURS physics-grade", np.load(phys[0])["frames"].astype(float))],
              args.out / "compare_side_by_side.gif",
              suptitle="matched pixel scale (cropped, not resized) and matched grey scale "
                       "(background -> 128)")


if __name__ == "__main__":
    main()
