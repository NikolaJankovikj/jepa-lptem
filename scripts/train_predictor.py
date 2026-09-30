"""Step 9: train the block-causal latent forecaster on the frozen feature cache.

Each training sample is a snippet of ``ctx + rollout`` consecutive latent frames
from one window (default 8 + 2 of 32). Two losses, both L1 on per-channel
standardized features:

- **teacher forcing** -- every context position forecasts its next frame from
  true frames only (V-JEPA 2-AC);
- **rollout** -- the model's own forecast is fed back to predict one more frame,
  so it trains on the inputs it will see at inference.

Standardization is per channel with train-split statistics -- invertible, so a
forecast maps back to raw feature space and the frozen Gate-1 probes can read it
(Gate 2). A per-token LayerNorm would not map back.

Every epoch logs the **persistence baseline** (next frame := this frame) next to
the model, overall and per process, at +1 and at the rollout horizon. The model
starts *as* persistence (see ``models/predictor.py``); on the growth processes it
has to pull away from it, and if it never does, debug before Step 10.

The split is ``split_by_clip`` with the Gate-1 seed, so the predictor never trains
on a Gate-1 val/test clip -- Gate 2 evaluates on exactly those.

Windows load in parallel worker processes: Gate-1 probe training ran at ~10 % GPU,
bound by decompressing one window at a time.

Usage (local smoke):
    uv run python -u scripts/train_predictor.py --features data/features/smoke_v3 \
        --out experiments/predictor-smoke --epochs 2 --dim 64 --depth 2 --workers 0
Remote:
    uv run python -u scripts/train_predictor.py --features data/features/dynamics-v1 \
        --out experiments/predictor-v1
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from train_probes import WindowCache, split_by_clip

from jepa_lptem.models.predictor import LatentForecaster

STATS_WINDOWS = 64  # train windows sampled for the per-channel mean/std
EVAL_SNIPPETS = 3  # fixed snippets per val window: first, middle, last


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def unstandardize(z, mean, std):
    return z * std + mean


def eval_starts(n_frames: int, length: int, n: int) -> np.ndarray:
    """``n`` evenly spaced snippet starts, first to last possible, deduplicated."""
    return np.unique(np.linspace(0, n_frames - length, n).round().astype(int))


def channel_stats(
    cache: WindowCache, idx: list[int], rng
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over every token of a sample of train windows."""
    sample = rng.choice(idx, min(STATS_WINDOWS, len(idx)), replace=False)
    tokens = []
    for i in sample:
        feats = cache.load(int(i))[0]
        tokens.append(feats.reshape(-1, feats.shape[-1]))
    tokens = np.concatenate(tokens)
    return tokens.mean(axis=0), tokens.std(axis=0) + 1e-6


class Snippets(Dataset):
    """One item = one window -> ``(k, length, S, D)`` standardized snippets.

    Several snippets per load amortize decompressing the whole window. Train
    draws ``per_window`` random starts; ``fixed`` draws evenly spaced ones.
    """

    def __init__(self, cache, idx, stats, length, *, per_window: int, fixed: bool):
        self.cache, self.idx, self.length = cache, idx, length
        self.mean, self.std = stats
        self.per_window, self.fixed = per_window, fixed
        self.processes = sorted({m["process"] for m in cache.meta})

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k: int):
        feats, _, _, meta = self.cache.load(self.idx[k])
        n = len(feats)
        if self.fixed:
            starts = eval_starts(n, self.length, self.per_window)
        else:
            starts = np.random.default_rng().integers(
                0, n - self.length + 1, self.per_window
            )
        snips = np.stack([feats[s : s + self.length] for s in starts])
        label = self.processes.index(meta["process"])
        return (
            torch.from_numpy(standardize(snips, self.mean, self.std)),
            torch.full((len(starts),), label),
        )


def collate(items):
    return torch.cat([x for x, _ in items]), torch.cat([y for _, y in items])


def forecast_losses(model, x: torch.Tensor, ctx: int, rollout: int):
    """``x`` ``(B, ctx + rollout, S, D)`` -> (training loss, per-sample L1 parts).

    Parts are ``(B,)`` tensors for model and persistence at +1 and at +``rollout``;
    persistence at +k is the last context frame held k frames forward.
    """
    preds = model.forward_all(x[:, :ctx])  # [:, t] forecasts x[:, t + 1]
    teacher = (preds - x[:, 1 : ctx + 1]).abs().mean()
    seq, nxt, roll = torch.cat([x[:, 1:ctx], preds[:, -1:]], dim=1), preds[:, -1], []
    for k in range(1, rollout):
        nxt = model(seq)
        roll.append((nxt - x[:, ctx + k]).abs().mean())
        seq = torch.cat([seq[:, 1:], nxt[:, None]], dim=1)
    loss = teacher + (torch.stack(roll).mean() if roll else 0.0)

    def per_sample(a, b):
        return (a - b).abs().mean(dim=(1, 2)).detach()

    last = x[:, ctx - 1]
    parts = {
        "model_l1_1": per_sample(preds[:, -1], x[:, ctx]),
        "persistence_l1_1": per_sample(last, x[:, ctx]),
        f"model_l1_{rollout}": per_sample(nxt, x[:, ctx + rollout - 1]),
        f"persistence_l1_{rollout}": per_sample(last, x[:, ctx + rollout - 1]),
    }
    return loss, parts


def autocast(device: str):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")


def evaluate(model, loader, args, device) -> dict:
    """Mean L1 parts over the loader, overall and per process."""
    processes = loader.dataset.processes
    model.eval()
    chunks: dict[str, list[torch.Tensor]] = {}
    labels = []
    with torch.no_grad(), autocast(device):
        for x, y in loader:
            _, parts = forecast_losses(model, x.to(device), args.ctx, args.rollout)
            for key, v in parts.items():
                chunks.setdefault(key, []).append(v.float().cpu())
            labels.append(y)
    labels = torch.cat(labels)
    out = {}
    for key, pieces in chunks.items():
        v = torch.cat(pieces)
        out[key] = float(v.mean())
        for i, p in enumerate(processes):
            out[f"{key}_{p}"] = (
                float(v[labels == i].mean()) if (labels == i).any() else None
            )
    return out


def split_windows(cache: WindowCache, max_train: int) -> dict[str, list[int]]:
    """Window indices per split, by clip, with the Gate-1 seed."""
    clips = [m["clip"] for m in cache.meta]
    split = split_by_clip(clips, seed=0)
    idx = {part: [i for i, c in enumerate(clips) if c in split[part]] for part in split}
    if max_train:
        idx["train"] = idx["train"][:max_train]
    return idx


def make_loaders(cache, idx, stats, args):
    length = args.ctx + args.rollout
    common = {
        "collate_fn": collate,
        "num_workers": args.workers,
        "persistent_workers": args.workers > 0,
        "batch_size": args.windows_per_batch,
    }
    train_loader = DataLoader(
        Snippets(
            cache, idx["train"], stats, length, per_window=args.per_window, fixed=False
        ),
        shuffle=True,
        **common,
    )
    val_loader = DataLoader(
        Snippets(
            cache, idx["val"], stats, length, per_window=EVAL_SNIPPETS, fixed=True
        ),
        **common,
    )
    return train_loader, val_loader


def train_epoch(model, loader, opt, args, device) -> float:
    """One pass; ``opt`` is ``(optimizer, per-step scheduler)``."""
    optimizer, sched = opt
    model.train()
    losses = []
    for x, _ in loader:
        with autocast(device):
            loss, _ = forecast_losses(
                model, x.to(device, non_blocking=True), args.ctx, args.rollout
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sched.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def cosine_with_warmup(optimizer, total_steps: int, warmup: int) -> LambdaLR:
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, factor)


def report(row: dict, rollout: int, processes) -> None:
    """One epoch's line, then +1 model vs persistence per process."""
    print(
        f"epoch {row['epoch']:3d}  train {row['train_loss']:.4f}  "
        f"val +1 model {row['model_l1_1']:.4f} / persist {row['persistence_l1_1']:.4f}  "
        f"+{rollout} model {row[f'model_l1_{rollout}']:.4f} / persist "
        f"{row[f'persistence_l1_{rollout}']:.4f}  "
        f"(train {row['train_seconds']:.0f}s, val {row['val_seconds']:.0f}s)"
    )
    for p in processes:
        m, q = row[f"model_l1_1_{p}"], row[f"persistence_l1_1_{p}"]
        if m is not None:
            print(f"    {p:12s} +1 model {m:.4f} / persist {q:.4f}  ({1 - m / q:+.1%})")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def save_checkpoint(path: Path, model, stats, args, epoch: int, processes) -> None:
    """Weights plus everything Gate 2 needs to use them: the standardization
    statistics (to map forecasts back for the probes) and the config."""
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(n_spatial=model.n_spatial, enc_dim=model.out.out_features)
    torch.save(
        {
            "model": model.state_dict(),
            "mean": stats[0],
            "std": stats[1],
            "config": config,
            "epoch": epoch,
            "processes": processes,
        },
        path,
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--ctx", type=int, default=8)
    ap.add_argument("--rollout", type=int, default=2)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--windows-per-batch", type=int, default=2)
    ap.add_argument(
        "--per-window", type=int, default=4, help="snippets drawn per window load"
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--max-train-windows", type=int, default=0, help="0 = all (smoke runs cap it)"
    )
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device()
    cache = WindowCache(args.features)
    idx = split_windows(cache, args.max_train_windows)
    stats = channel_stats(cache, idx["train"], np.random.default_rng(args.seed))
    train_loader, val_loader = make_loaders(cache, idx, stats, args)
    n_spatial, enc_dim = cache.load(0)[0].shape[1:]
    model = LatentForecaster(
        enc_dim=enc_dim,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        n_spatial=n_spatial,
    ).to(device)
    print(
        f"windows { {k: len(v) for k, v in idx.items()} }  device {device}  "
        f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  "
        f"snippet {args.ctx}+{args.rollout} frames"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = cosine_with_warmup(
        optimizer, args.epochs * len(train_loader), warmup=min(500, len(train_loader))
    )
    args.out.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, train_loader, (optimizer, sched), args, device)
        t1 = time.time()
        row = {"epoch": epoch, "train_loss": loss, **evaluate(model, val_loader, args, device)}
        # timed apart: val is a fixed cost, so only the train time scales with --max-train-windows
        row.update(train_seconds=t1 - t0, val_seconds=time.time() - t1)
        history.append(row)
        report(row, args.rollout, val_loader.dataset.processes)
        if row["model_l1_1"] < best:
            best = row["model_l1_1"]
            save_checkpoint(
                args.out / "predictor.pt",
                model,
                stats,
                args,
                epoch,
                val_loader.dataset.processes,
            )
        (args.out / "history.json").write_text(json.dumps(history, indent=1))
    print(
        f"\nbest val +1 model L1 {best:.4f}; wrote {args.out}/predictor.pt, history.json"
    )


if __name__ == "__main__":
    main()
