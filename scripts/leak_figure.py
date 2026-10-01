"""Show the encoder look-ahead: change only the FUTURE, watch the PAST's features move.

Two versions of one window share their 16 context frames pixel for pixel; from
frame 16 on, one carries a different clip's frames. Whole-window encoding (the
Gate-2 cache) lets the context features change anyway -- and with them what the
frozen Gate-1 probe reads in the past. Per-frame encoding (``--per-frame``) leaves
the context features bit-identical.

Usage (local, MPS ~1 min):
    uv run python scripts/leak_figure.py --data data/synthetic/smoke_v3 \
        --clip-a clip_0001_diffusion_physics --clip-b clip_0006_diffusion_physics \
        --probes experiments/gate1/probes.pt --out experiments/leak-check/lookahead.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from extract_features import TUBELET, make_encoder, pick_device
from gate1_secondary import load_probes
from leak_check import token_cosine

CTX_FRAMES = 16  # 8 latent frames, as the predictor's context
CROP = 256


def crop(frames: np.ndarray) -> np.ndarray:
    off = (frames.shape[1] - CROP) // 2
    return frames[:, off : off + CROP, off : off + CROP]


def visible(z, frame: int) -> np.ndarray:
    """Crop-frame (row, col) of the particles in view at ``frame``."""
    pos = z["positions"][:, frame] - (z["frames"].shape[1] - CROP) / 2
    keep = (z["radii"][:, frame] > 0) & ((pos >= 0) & (pos < CROP)).all(axis=1)
    return pos[keep]


def occupancy(cen, feats: np.ndarray, latent: int) -> np.ndarray:
    with torch.no_grad():
        occ, _ = cen(torch.from_numpy(feats[latent : latent + 1].astype(np.float32)))
    return torch.sigmoid(occ)[0].numpy()


def mark(ax, pts, style: str, label: str) -> None:
    cell = CROP / 16  # token maps are 16 x 16: place points in token units
    ax.plot(
        pts[:, 1] / cell - 0.5, pts[:, 0] / cell - 0.5, style, ms=9, mew=2, label=label
    )


def swap_future(za, zb) -> tuple[np.ndarray, np.ndarray]:
    """Clip A, and clip A with clip B's frames from ``CTX_FRAMES`` on."""
    original = crop(za["frames"])
    swapped = original.copy()
    swapped[CTX_FRAMES:] = crop(zb["frames"])[CTX_FRAMES : len(original)]
    assert np.array_equal(original[:CTX_FRAMES], swapped[:CTX_FRAMES])
    return original, swapped


def context_features(original, swapped, device: str) -> dict:
    """Context latents of both versions, under each encoding."""
    ctx = CTX_FRAMES // TUBELET
    out = {}
    for name, per_frame in (("whole window (Gate-2 cache)", False), ("per frame (--per-frame)", True)):
        encode = make_encoder(device, per_frame)
        out[name] = (encode(original)[:ctx], encode(swapped)[:ctx])
    return out


def plot_frames(axes, original, swapped) -> None:
    shown = ((original[CTX_FRAMES - 2], f"context frame {CTX_FRAMES - 2}: identical in both"),
             (original[CTX_FRAMES], f"future frame {CTX_FRAMES}: original"),
             (swapped[CTX_FRAMES], f"future frame {CTX_FRAMES}: swapped in"))
    for ax, (img, title) in zip(axes, shown, strict=True):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")


def plot_change(axes, fig, change: dict, cen, particles) -> None:
    """Feature change per context position; per token and as probe occupancy at the last."""
    future_b, past_a = particles
    for name, (fa, fb) in change.items():
        axes[0].plot(range(len(fa)), 1 - token_cosine(fa, fb), "o-", label=name)
    axes[0].set(xlabel="context latent frame (7 = next to the future)", ylabel="feature change, 1 − cosine",
                title="Only the future changed. Did the past's features?")
    axes[0].legend()

    whole_a, whole_b = change["whole window (Gate-2 cache)"]
    last = len(whole_a) - 1
    im = axes[1].imshow(1 - token_cosine_map(whole_a[last], whole_b[last]), cmap="magma")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    mark(axes[1], future_b, "c+", "swapped-in future particles")
    axes[1].set_title(f"feature change per token, context latent {last}\n(whole-window encoding)")
    axes[1].legend(loc="lower right", fontsize=8)

    occ_diff = occupancy(cen, whole_b, last) - occupancy(cen, whole_a, last)
    lim = np.abs(occ_diff).max()
    im = axes[2].imshow(occ_diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    mark(axes[2], future_b, "k+", "swapped-in future particles")
    mark(axes[2], past_a, "gx", "particles actually in the context")
    axes[2].set_title(f"Gate-1 probe occupancy at context latent {last}:\n"
                      "swapped − original (red = sees a particle more)")
    axes[2].legend(loc="lower right", fontsize=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--clip-a", required=True)
    ap.add_argument("--clip-b", required=True, help="supplies the swapped-in future")
    ap.add_argument("--probes", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    za, zb = (np.load(args.data / f"{c}.npz") for c in (args.clip_a, args.clip_b))
    original, swapped = swap_future(za, zb)
    change = context_features(original, swapped, pick_device())
    _, cen, _ = load_probes(args.probes)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    plot_frames(axes[0], original, swapped)
    plot_change(axes[1], fig, change, cen, (visible(zb, CTX_FRAMES), visible(za, CTX_FRAMES - 2)))
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    for name, (fa, fb) in change.items():
        print(f"{name:30s} max feature change over the context: {float((1 - token_cosine(fa, fb)).max()):.4f}")
    print(f"wrote {args.out}")


def token_cosine_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-token cosine of one frame, as the 16 x 16 token grid."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return cos.reshape(16, 16)


if __name__ == "__main__":
    main()
