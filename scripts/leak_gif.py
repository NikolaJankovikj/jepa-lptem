"""The look-ahead as an animation, for explaining it: two movies with the same
past and different futures, played side by side, while two "computers" -- the
whole-window encoder (Gate-2 cache) and the per-frame one -- report how
different they find each moment. In the past, a fair computer must say "the same".

Usage (local, MPS ~1 min):
    uv run python scripts/leak_gif.py --data data/synthetic/smoke_v3 \
        --clip-a clip_0001_diffusion_physics --clip-b clip_0006_diffusion_physics \
        --out experiments/leak-check/lookahead.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from extract_features import TUBELET, make_encoder, pick_device
from leak_check import token_cosine
from leak_figure import CTX_FRAMES, swap_future, token_cosine_map
from matplotlib.animation import FuncAnimation, PillowWriter

COMPUTERS = (("Computer A: watches the whole movie at once\n(how our old features were made)", False),
             ("Computer B: looks at one picture at a time\n(the fix, --per-frame)", True))
HOLD = 6  # extra frames on the last step, so the ending can be read
GIF_DPI = 60  # 600 x 660 px: noisy frames compress badly, so size grows fast with dpi


def differences(original, swapped, device: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per computer: per-token (T, 16, 16) and per-frame (T,) 1 - cosine, movie 1 vs 2."""
    out = []
    for _, per_frame in COMPUTERS:
        encode = make_encoder(device, per_frame)
        a, b = encode(original), encode(swapped)
        out.append((np.stack([1 - token_cosine_map(x, y) for x, y in zip(a, b, strict=True)]),
                    1 - token_cosine(a, b)))
    return out


def animate(original, swapped, diffs, out: Path) -> None:
    n_lat = len(original) // TUBELET
    ctx = CTX_FRAMES // TUBELET
    vmax = float(np.percentile(np.concatenate([d[0][ctx:].ravel() for d in diffs]), 95))
    fig = plt.figure(figsize=(10, 11))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.75])
    movie = [fig.add_subplot(grid[0, k]) for k in range(2)]
    seen = [fig.add_subplot(grid[1, k]) for k in range(2)]
    chart = fig.add_subplot(grid[2, :])

    for ax in (*movie, *seen):
        ax.axis("off")
    imgs = [ax.imshow(original[0], cmap="gray", vmin=0, vmax=255) for ax in movie]
    maps = [ax.imshow(d[0][0], cmap="inferno", vmin=0, vmax=vmax) for ax, d in zip(seen, diffs, strict=True)]
    for ax, (title, _) in zip(seen, COMPUTERS, strict=True):
        ax.set_title(title + "\nblack = 'the same', bright = 'different'", fontsize=10)

    chart.axvspan(-0.5, ctx - 0.5, color="green", alpha=0.12)
    chart.text(ctx / 2 - 0.5, 0.97, "PAST: identical in both movies", ha="center", va="top",
               transform=chart.get_xaxis_transform(), color="green")
    chart.text((ctx + n_lat) / 2 - 0.5, 0.97, "FUTURE: different movies", ha="center", va="top",
               transform=chart.get_xaxis_transform(), color="firebrick")
    lines = [chart.plot([], [], "o-", label=title.split(":")[0])[0] for title, _ in COMPUTERS]
    chart.set(xlim=(-0.5, n_lat - 0.5), ylim=(0, max(d[1].max() for d in diffs) * 1.15),
              xlabel="moment in the movie", ylabel="how different?")
    chart.legend(loc="center right")

    def draw(step: int):
        t = min(step, n_lat - 1)
        past = t < ctx
        for ax, img, frames, name in zip(movie, imgs, (original, swapped), ("Movie 1", "Movie 2"), strict=True):
            img.set_data(frames[TUBELET * t])
            ax.set_title(f"{name} — " + ("PAST (the same)" if past else "FUTURE (different!)"),
                         color="green" if past else "firebrick", fontsize=13)
        for m, line, (tok, frame) in zip(maps, lines, diffs, strict=True):
            m.set_data(tok[t])
            line.set_data(range(t + 1), frame[: t + 1])
        return (*imgs, *maps, *lines)

    draw(0)
    fig.tight_layout()  # after the first draw, so the movie titles get room
    anim = FuncAnimation(fig, draw, frames=n_lat + HOLD, blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=2), dpi=GIF_DPI)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--clip-a", required=True)
    ap.add_argument("--clip-b", required=True, help="supplies movie 2's future")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    za, zb = (np.load(args.data / f"{c}.npz") for c in (args.clip_a, args.clip_b))
    original, swapped = swap_future(za, zb)
    animate(original, swapped, differences(original, swapped, pick_device()), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
