"""Screen published LPTEM movies before any of them touches the model.

Supplementary movies are display copies: re-encoded, often rescaled, padded with
title cards and repeated frames, and stamped with text. Each of those breaks a
premise of the pipeline (one pixel is one detector pixel, one frame is one time
step), so a movie is screened on data integrity first and on fit to the simulator
second. The criteria were fixed before any movie was encoded (`PLAN.md` Step 13).

Integrity, per movie (all must hold):

- **Q1 native pixel grid.** Shot noise is white on the detector grid. A movie that
  was upsampled or smoothed for display spreads it over several pixels, which
  rescales apparent particle size and changes the noise a frozen encoder sees.
  Measured as the lag-2 correlation of the frame-to-frame difference (horizontal
  and vertical; structure cancels, noise stays). The simulator reads <= 0.02 and the
  SAM-EM real frames <= 0.07, so the bar is <= 0.10 on both axes. Lag 2 is blind
  to an exact 2x nearest-neighbour upsampling (its lag-2 pairs come from different
  source pixels), so lag 1 must also stay <= 0.40: that upsampling reads 0.5, the
  detector's own MTF 0.17-0.26.
- **Q2 grey scale.** Colour channels agree (a false-colour LUT is not a detector
  signal).
- **Q3 contiguous time.** Title cards and scene cuts (a frame difference > 4x the
  median) split a movie into segments; repeated frames (< 0.2x the median) are
  dropped, since a display copy repeats frames to pad its frame rate. The longest
  segment must keep >= 16 frames (one containment sample).
- **Q4 size.** >= 256 px on both sides, so a 256 crop needs no resize.

The overlay-free region (text, scale bars, panel borders) is set by hand per movie
in `REAL_MOVIES` after looking at it; this script only reports what it measures.

Usage (Mac, CPU, ~2 min for ~130 movies):
    uv run python scripts/real_video_qc.py data/real-figshare data/real-figshare/pool \
        --out experiments/real/qc.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

NOISE_CORR_MAX = 0.10  # lag 2
NEIGHBOUR_CORR_MAX = 0.40  # lag 1
CUT_FACTOR, DUP_FACTOR = 4.0, 0.2
MIN_FRAMES, MIN_SIDE = 16, 256
GREY_MAX_CHANNEL_DIFF = 2.0  # grey levels; codec chroma noise on a grey movie stays below
LAG = 2  # pixels; the simulator's MTF correlates neighbours (lag 1) but not lag 2
MIN_STEPS = 2
COLOUR_DIMS = 3
CHECKS = ("q1_native_grid", "q2_grey", "q3_frames", "q4_size")
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".wmv", ".mpg", ".gif"}
REAL_DIR = Path("data/real-figshare")
NANORODS = "samem_au_nanorods"  # the SAM-EM video, a zip of JPEG frames

# Movies that passed Q1-Q4 and the look: overlay-free region (y0, y1, x0, x1) and
# tier (Step 13: 1 containment, 2 + readout, 3 + forecast).
REAL_MOVIES = {
    NANORODS: {"roi": (0, 1024, 0, 1024), "tier": 3},
    "aggregative_growth_woehl2014.avi": {"roi": (64, 472, 0, 496), "tier": 3},
    "am1c20824_si_002.mp4": {"roi": (56, 430, 4, 512), "tier": 3},
    "am8b03688_si_004.avi": {"roi": (70, 440, 0, 480), "tier": 3},
    "nn5b06632_si_001.avi": {"roi": (0, 480, 0, 500), "tier": 3},
    "cm2c03226_si_004.avi": {"roi": (0, 680, 0, 764), "tier": 1},  # HAADF: control
    "ja9b05225_si_001.avi": {"roi": (0, 400, 0, 400), "tier": 1},
    "ja9b05225_si_002.avi": {"roi": (0, 400, 0, 400), "tier": 1},
    "nn6b05270_si_003.avi": {"roi": (0, 900, 4, 1024), "tier": 1},
}


def grey_frames(path: Path, keep=None):
    """Yield ``(index, uint8 grey frame, |channel difference|)``, streaming: some
    movies decode to more frames than fit in memory. ``keep`` limits which frames
    are yielded (all are still decoded, in order)."""
    cap = cv2.VideoCapture(str(path))
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            return
        if keep is None or i in keep:
            colour = 0.0
            if f.ndim == COLOUR_DIMS:
                colour = float(np.abs(f[..., 0].astype(int) - f[..., 2]).mean())
                f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            yield i, f, colour
        i += 1


def frame_steps(path: Path) -> tuple[np.ndarray, tuple, float]:
    """Mean |frame difference| per step, frame shape, mean channel difference."""
    steps, colour, prev = [], [], None
    for _, f, c in grey_frames(path):
        g = f.astype(np.float32)
        if prev is not None:
            steps.append(float(np.abs(g - prev).mean()))
        prev = g
        colour.append(c)
    shape = prev.shape if prev is not None else (0, 0)
    return np.array(steps), shape, float(np.mean(colour)) if colour else 0.0


def segments(steps: np.ndarray) -> list[np.ndarray]:
    """Frame indices of each contiguous segment, repeated frames dropped."""
    med = np.median(steps)
    out, current = [], [0]
    for i, step in enumerate(steps, start=1):
        if step > CUT_FACTOR * med:
            out.append(np.array(current))
            current = [i]
        elif step >= DUP_FACTOR * med:
            current.append(i)
    out.append(np.array(current))
    return out


def noise_correlation(frames: np.ndarray, lag: int = LAG, n: int = 20) -> tuple:
    """Lag-``lag`` correlation of the frame difference, along x and along y."""
    if len(frames) <= LAG:
        return np.nan, np.nan
    d = np.diff(frames[: n + 1].astype(np.float32), axis=0)

    def lagged(a: np.ndarray) -> float:
        return float(np.corrcoef(a[..., :-lag].ravel(), a[..., lag:].ravel())[0, 1])

    return lagged(d), lagged(d.transpose(0, 2, 1))


def measure(path: Path, steps: np.ndarray, shape: tuple, colour: float) -> dict:
    """The four integrity checks and what they were computed from."""
    segs = segments(steps)
    longest = max(segs, key=len)
    first = set(longest[:21].tolist())
    frames = np.stack([f for _, f, _ in grey_frames(path, first)])
    corr, neighbour = noise_correlation(frames), noise_correlation(frames, lag=1)
    checks = {
        "q1_native_grid": bool(
            max(corr) <= NOISE_CORR_MAX and max(neighbour) <= NEIGHBOUR_CORR_MAX
        ),
        "q2_grey": colour < GREY_MAX_CHANNEL_DIFF,
        "q3_frames": len(longest) >= MIN_FRAMES,
        "q4_size": min(shape) >= MIN_SIDE,
    }
    return {
        "segments": [[int(s[0]), int(s[-1]), len(s)] for s in segs if len(s) > 1],
        "longest": [int(longest[0]), int(longest[-1]), len(longest)],
        "dropped_repeats": int(len(steps) + 1 - sum(len(s) for s in segs)),
        "noise_corr_lag2": [round(c, 3) for c in corr],
        "noise_corr_lag1": [round(c, 3) for c in neighbour],
        **checks,
        "pass": all(checks.values()),
    }


def screen(path: Path) -> dict:
    steps, shape, colour = frame_steps(path)
    if len(steps) < MIN_STEPS:
        return {"movie": path.name, "readable": False, "pass": False}
    return {
        "movie": path.name,
        "readable": True,
        "frames": len(steps) + 1,
        "height": shape[0],
        "width": shape[1],
        "colour": round(colour, 1),
    } | measure(path, steps, shape, colour)


def load_movie(name: str) -> np.ndarray:
    """A screened movie as ``(T, H, W)`` uint8: its longest clean segment, repeated
    frames dropped, cropped to its overlay-free region."""
    y0, y1, x0, x1 = REAL_MOVIES[name]["roi"]
    if name == NANORODS:
        from compare_to_real import load_sequence

        frames = load_sequence(
            "data/sam-em/experimentalAuNanorods.zip",
            lambda x: x.endswith(".jpeg"),
            limit=10_000,
        ).astype(np.uint8)
        return frames[:, y0:y1, x0:x1]
    path = next(REAL_DIR.rglob(name))
    longest = max(segments(frame_steps(path)[0]), key=len)
    keep = set(longest.tolist())
    return np.stack([f[y0:y1, x0:x1] for _, f, _ in grey_frames(path, keep)])


def report_line(r: dict) -> str:
    if not r["readable"]:
        return f"{r['movie']:40s} unreadable"
    fails = [k for k in CHECKS if not r[k]]
    return (
        f"{r['movie']:40s} {r['width']:5d}x{r['height']:<5d} T={r['frames']:5d} "
        f"longest={r['longest'][2]:5d} corr={r['noise_corr_lag2']} "
        + ("PASS" if r["pass"] else "fail: " + ",".join(fails))
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out: Path = args.out
    paths = sorted(
        p for d in args.dirs for p in d.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES
    )
    rows = [screen(p) for p in paths]
    print("\n".join(report_line(r) for r in rows))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"\n{sum(r['pass'] for r in rows)} of {len(rows)} movies pass the integrity screen")


if __name__ == "__main__":
    main()
