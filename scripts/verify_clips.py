"""Verify a generated clip is 'good': can the physics be recovered from the pixels?

A render is only trustworthy if (1) each rendered particle sits exactly under its
saved ground-truth position/radius, and (2) an independent detector run on the
pixels recovers the same trajectories/statistics as the answer key. This script
checks both and writes visual proof.

Usage:
    uv run python scripts/verify_clips.py data/synthetic/smoke/clip_0003_static.npz
Outputs (next to a chosen --out dir): an overlay GIF and a physics panel PNG,
plus printed pass/fail numbers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, patches
from scipy import ndimage

from jepa_lptem.eval.observables import compute_msd, fit_growth_exponent


def detect_dark_blobs(frame: np.ndarray, k: float = 1.8, min_area: int = 80):
    """Fair classical baseline detector: find dark particles in a rendered frame.

    Gaussian-blur to suppress per-pixel Poisson noise (particles are many px wide,
    noise is 1 px), threshold below ``mean - k*std``, take connected components,
    and drop specks below ``min_area``. Shares no code with the renderer. A simple
    detector like this is *expected* to struggle at high noise / heavy overlap —
    that gap is the motivation for a learned model, not a defect of the video.

    Returns ``(centroids (M,2) as (row,col), labels image)``.
    """
    f = ndimage.gaussian_filter(frame.astype(float), sigma=2.5)
    mask = f < (f.mean() - k * f.std())
    mask = ndimage.binary_opening(mask, iterations=1)
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros((0, 2)), lab
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_area]
    if not keep:
        return np.zeros((0, 2)), lab
    coms = ndimage.center_of_mass(mask, lab, keep)
    return np.array(coms), lab


def match_error(gt_pos: np.ndarray, gt_rad: np.ndarray, det: np.ndarray) -> list[float]:
    """Nearest-neighbour distance (px) from each *present* GT particle to a detection."""
    errs = []
    for (r, c), rad in zip(gt_pos, gt_rad):
        if rad <= 0 or det.shape[0] == 0:
            continue
        d = np.hypot(det[:, 0] - r, det[:, 1] - c)
        errs.append(float(d.min()))
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/synthetic/smoke"))
    args = ap.parse_args()

    d = np.load(args.npz)
    frames, pos, rad = d["frames"], d["positions"], d["radii"]  # (T,H,W),(N,T,2),(N,T)
    process = str(d["process"]) if "process" in d else "?"
    D = float(d["D"]) if "D" in d else None
    T = frames.shape[0]
    stem = args.npz.stem

    # --- (1) GROUND-TRUTH PHYSICS -------------------------------------------------
    # MSD from saved positions vs the theoretical 4 D t (dt = 1 frame).
    msd = compute_msd(pos)  # (N,T,2)
    lags = np.arange(1, T)
    # fit slope on short lags only: few particles/clip make long lags noisy and
    # the reflecting box biases them downward. Short-lag slope -> 4D cleanly.
    short = max(4, T // 4)
    msd_slope = float(np.polyfit(lags[:short], msd[:short], 1)[0])
    # growth exponent from saved radii
    alive = rad > 0
    meanr = np.array([rad[alive[:, k], k].mean() if alive[:, k].any() else np.nan
                      for k in range(T)])
    m = ~np.isnan(meanr)
    # time-since-nucleation: frame 0 already has an age t0 (see sim.dataset)
    t0 = float(d["t0_growth"]) if "t0_growth" in d else 0.0
    dt_growth = float(d["dt_growth"]) if "dt_growth" in d else 1.0
    t_growth = t0 + np.arange(T) * dt_growth
    try:
        alpha = fit_growth_exponent(t_growth[m], meanr[m])
    except Exception:
        alpha = float("nan")

    # --- (2) ROUND-TRIP: recover positions from PIXELS ONLY -----------------------
    all_err, n_det, n_true, n_precise = [], 0, 0, 0
    for k in range(T):
        det, _ = detect_dark_blobs(frames[k])
        all_err += match_error(pos[:, k, :], rad[:, k], det)
        n_det += det.shape[0]
        n_true += int(alive[:, k].sum())
        # a detection is "precise" if it lands within one radius of a present GT particle
        gp, gr = pos[:, k, :][alive[:, k]], rad[:, k][alive[:, k]]
        for dr, dc in det:
            if gp.shape[0] and (np.hypot(gp[:, 0] - dr, gp[:, 1] - dc) <= gr).any():
                n_precise += 1
    med_err = float(np.median(all_err)) if all_err else float("nan")
    precision = n_precise / n_det if n_det else float("nan")   # fraction of detections that are real
    recall_ratio = n_det / n_true if n_true else float("nan")  # detected count / true count

    print(f"\n=== {stem}  (process={process}, D={D}) ===")
    n_part = int(alive.any(axis=1).sum())
    print(f"[GT physics]   MSD short-lag slope = {msd_slope:.3f}"
          + (f"   vs 4D = {4*D:.3f}   ratio {msd_slope/(4*D):.2f}" if D else "")
          + f"   [N={n_part}]")
    if n_part < 20:
        print(f"               ^ a per-clip MSD from {n_part} particles has sd ~"
              f"{0.5 / max(n_part, 1) ** 0.5:.2f} on this ratio.\n"
              "                 The estimator is unbiased; pool across clips before "
              "drawing conclusions.")
    print(f"[GT physics]   growth exponent alpha = {alpha:.3f}")
    print(f"[round-trip]   position error (matched): median {med_err:.2f}px  "
          f"(radii ~{np.nanmean(rad[alive]):.0f}px)")
    stratum = str(d["stratum"]) if "stratum" in d else "?"
    print(f"[round-trip]   detector precision = {precision:.2f}  "
          f"(detected/true count ratio = {recall_ratio:.2f})")
    if stratum == "experiment":
        print("               ^ low precision is EXPECTED here: experiment-grade "
              "clips have ~6px\n                 particles at real dose, below "
              "what a threshold detector can resolve.\n                 That gap "
              "is why SAM-EM needed a trained model in the first place.")

    # Shortcut check: how much of mean radius is readable off overall brightness
    # alone, with no localization? That is the trivial baseline any Gate-1 probe
    # must beat. The sign is diagnostic, but the physical sign is mode-dependent:
    # growing particles add gold, so the frame darkens (negative). Ripening
    # conserves mass while coarsening, which *shrinks* total projected area, so
    # the frame brightens (positive). A positive value on a growth mode is the
    # signature of the min-max normalisation leak this pipeline no longer has.
    gmean = frames.reshape(T, -1).mean(axis=1)[m]
    leak = float(np.corrcoef(meanr[m], gmean)[0, 1]) if m.sum() > 2 else float("nan")
    clipped = float((frames >= 255).mean())
    print(f"[shortcut]     corr(mean_radius, frame_mean) = {leak:+.3f}"
          f"   (expect < 0 for growth modes)")
    print(f"[render]       saturated pixels = {100 * clipped:.2f}%  "
          f"flux={float(d['beam_flux']) if 'beam_flux' in d else float('nan'):.1f}")

    # --- overlay GIF: rendered frame + GT circle + GT centroid + detections -------
    fig, ax = plt.subplots(figsize=(5, 5)); ax.axis("off")
    im = ax.imshow(frames[0], cmap="gray", vmin=0, vmax=255)
    arts = []
    def upd(kk):
        for a in arts: a.remove()
        arts.clear()
        im.set_data(frames[kk])
        for (r, c), rr in zip(pos[:, kk, :], rad[:, kk]):
            if rr <= 0: continue
            circ = patches.Circle((c, r), rr, fill=False, ec="lime", lw=1.2)
            ax.add_patch(circ); arts.append(circ)
            dot, = ax.plot(c, r, "+", color="lime", ms=6); arts.append(dot)
        det, _ = detect_dark_blobs(frames[kk])
        if det.shape[0]:
            sc, = ax.plot(det[:, 1], det[:, 0], "x", color="red", ms=7, mew=1.5)
            arts.append(sc)
        ax.set_title(f"{process}  frame {kk}/{T-1}   green=GT  red=detected", fontsize=9)
        return [im, *arts]
    anim = animation.FuncAnimation(fig, upd, frames=T, interval=140, blit=False)
    gif_path = args.out / f"verify_{stem}.gif"
    anim.save(gif_path, writer=animation.PillowWriter(fps=7)); plt.close(fig)

    # --- physics panel PNG --------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].plot(lags, msd, "o-", ms=3, label="MSD from GT positions")
    if D: ax[0].plot(lags, 4 * D * lags, "--", label=f"4Dt (D={D:.2f})")
    ax[0].set_xlabel("lag (frames)"); ax[0].set_ylabel("MSD (px^2)"); ax[0].legend(); ax[0].set_title("Diffusion")
    ax[1].plot(t_growth[m], meanr[m], "o-", ms=3)
    ax[1].set_xlabel("time"); ax[1].set_ylabel("mean radius (px)")
    ax[1].set_title(f"Growth: fitted alpha = {alpha:.3f}")
    r0v, rTv = rad[:, 0][rad[:, 0] > 0], rad[:, -1][rad[:, -1] > 0]
    present = np.concatenate([r0v, rTv]) if (r0v.size or rTv.size) else np.array([1.0])
    bins = np.linspace(0, present.max() + 2, 16)
    ax[2].hist(r0v, bins=bins, alpha=0.6, label="t=0"); ax[2].hist(rTv, bins=bins, alpha=0.6, label="t=end")
    ax[2].set_xlabel("radius (px)"); ax[2].set_ylabel("count"); ax[2].legend(); ax[2].set_title("Size distribution")
    plt.tight_layout()
    png_path = args.out / f"verify_{stem}.png"
    plt.savefig(png_path, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"[wrote] {gif_path}\n[wrote] {png_path}")


if __name__ == "__main__":
    main()
