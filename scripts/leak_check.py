"""Does the predictor's context carry the future? -- encoder look-ahead check.

The cache encodes 64-frame windows in one pass, and the encoder attends across
all of them: the features of the 8 context latent frames were computed with the
48 future frames in view. Every Gate-2 forecast started from such a context.

Here each physics test window's 16 context frames are re-encoded **alone** (no
future in view) and compared with the cached context:

1. feature agreement -- token cosine per context position, causal vs cached;
2. probe readout on the context frames themselves, causal vs cached -- separates
   "causal features are off-distribution for the probes" from lost future info;
3. the Gate-2 follow-up (+24, unchanged) with the causal context swapped in, to
   compare with ``experiments/core/followup.json`` (cached context).

If (3) matches the cached-context follow-up, look-ahead did not drive Gate 2.

Usage (remote, one GPU):
    uv run python -u scripts/leak_check.py --data data/dynamics-v1 \
        --features data/features/dynamics-v1 --gate1 experiments/gate1 \
        --predictor experiments/predictor-v1/predictor.pt --out experiments/leak-check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from eval_forecasts import (
    horizon_summary,
    load_gate1,
    load_predictor,
    score_method,
    snippet_truth,
)
from extract_features import TUBELET, make_encoder
from gate1_secondary import split_indices
from gate2_followup import analyse, collect, print_report
from train_predictor import pick_device
from train_probes import WindowCache


def token_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean cosine over tokens, per frame: ``(T, S, D)`` x2 -> ``(T,)``."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return cos.mean(axis=-1)


def causal_context(data_dir: Path, meta: dict, ctx: int, encode) -> np.ndarray:
    """The window's first ``ctx`` latent frames, encoded from their own frames only."""
    frames = np.load(data_dir / f"{meta['clip']}.npz")["frames"]
    off, crop, start = (
        (frames.shape[1] - meta["crop"]) // 2,
        meta["crop"],
        meta["start"],
    )
    return encode(
        frames[start : start + ctx * TUBELET, off : off + crop, off : off + crop]
    )


def context_readout(cache, contexts: dict, probes, ctx: int) -> dict:
    """Probes on the context frames, cached vs causal, against the GT there."""
    device = str(next(probes[0].parameters()).device)
    rows = []
    for i, causal in contexts.items():
        feats, pos, rad, meta = cache.load(i)
        truth = snippet_truth(pos, rad, meta, np.arange(ctx))
        methods = {
            name: score_method(probes, lat, truth, device)
            for name, lat in (("cached", feats[:ctx]), ("causal", causal))
        }
        rows.append({"meta": meta, "truth": truth, "methods": methods})
    return {
        name: {f"pos{h - 1}": horizon_summary(rows, name, h) for h in range(1, ctx + 1)}
        for name in ("cached", "causal")
    }


def print_agreement(cosine: np.ndarray, readout: dict) -> None:
    print("\n(1) token cosine, causal vs cached context, by context position 0..7:")
    print("    " + "  ".join(f"{c:.3f}" for c in cosine))
    print(
        "\n(2) probes on the context frames, cached / causal -- F1 | size distance (px):"
    )
    for pos in readout["cached"]:
        c, k = readout["cached"][pos], readout["causal"][pos]
        print(
            f"    {pos}: {c['det_f1']:.2f} / {k['det_f1']:.2f} | {c['w2']:.2f} / {k['w2']:.2f}"
        )
    print(
        "\n(3) Gate-2 follow-up with the CAUSAL context (compare experiments/core/followup.json):"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data", required=True, type=Path, help="the clips the cache was built from"
    )
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--gate1", required=True, type=Path)
    ap.add_argument("--predictor", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    device = pick_device()
    cache = WindowCache(args.features)
    physics = [
        i
        for i in split_indices(cache, args.gate1 / "metrics.json")["test"]
        if cache.meta[i]["stratum"] == "physics"
    ]
    probes = load_gate1(args.gate1, device)
    predictor = load_predictor(args.predictor, device)
    ctx = predictor[2]
    encode = make_encoder(device)
    print(
        f"physics test windows {len(physics)}  device {device}  ctx {ctx} latent frames"
    )

    contexts = {
        i: causal_context(args.data, cache.meta[i], ctx, encode) for i in physics
    }
    cosine = np.mean(
        [token_cosine(cache.load(i)[0][:ctx], c) for i, c in contexts.items()], axis=0
    )
    readout = context_readout(cache, contexts, probes, ctx)
    print_agreement(cosine, readout)
    followup = analyse(collect(cache, physics, predictor, probes, contexts))
    print_report(followup)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "leak.json").write_text(
        json.dumps(
            {
                "cosine_by_position": cosine.tolist(),
                "context_readout": readout,
                "followup_causal": followup,
            },
            indent=2,
            default=float,
        )
    )
    print(f"\nwrote {args.out}/leak.json")


if __name__ == "__main__":
    main()
