"""Step 11 P2: train the ConvLSTM pixel baseline, forecast the test snippets, encode them.

PLAN Step 11, fixed 2026-11-21. Same 256² crops as the feature cache (the tubelet
memory map of Step 15 M1), ``split_by_clip`` seed 0, train clips only, no labels:

- train: random 64-frame windows, 16 context frames, 48-frame rollout, pixel MSE;
  teacher forcing ramps linearly from 1 to 0 over the first half of the epochs
- select: the epoch with the lowest val MSE over the 48-frame rollout (no teacher)
- forecast (test clips): ``ctx8`` = frames s..s+15 -> s+16..s+63 for s in (0, 36),
  the snippets of pooled-forecast; ``ctx16`` = frames 20..51 -> 52..99, the context-16
  snippet of longer-context design A
- encode (``--encode``): every forecast per frame with the frozen V-JEPA 2; saves the
  pooled means and the Gate-1 attentive readings, so the Mac never needs the tokens

Usage (PC; local smoke with --tubelets data/tubelets/smoke_v3 --epochs 1):
    uv run python -u scripts/train_convlstm.py --tubelets data/tubelets/dynamics-v1 \
        --out experiments/pixel-baseline --encode
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

import numpy as np
import torch
from extract_features import make_encoder, pick_device
from torch.optim.lr_scheduler import CosineAnnealingLR
from train_probes import load_probes, split_by_clip

from jepa_lptem.models.convlstm import PixelForecaster, rollout_loss

LOG = logging.getLogger(__name__)
WINDOW = 64
CTX_FRAMES = 16
STARTS = (0, 36)
LONG_CTX = slice(20, 52)  # context 16 latent frames, forecast 52..99 (longer-context A)
FORECAST = WINDOW - CTX_FRAMES
PIXEL_MAX = 255.0


def to_unit(frames: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.asarray(frames, dtype=np.float32) / PIXEL_MAX).to(device)


def to_uint8(frames: torch.Tensor) -> np.ndarray:
    return (frames.float().cpu().numpy() * PIXEL_MAX).round().astype(np.uint8)


class Clips:
    """The tubelet memory map's frames, by split (rows of the map)."""

    def __init__(self, root: Path):
        self.frames = np.load(root / "frames.npy", mmap_mode="r")
        meta = [json.loads(line) for line in (root / "meta.jsonl").open()]
        split = split_by_clip([m["clip"] for m in meta], seed=0)
        self.meta = meta
        self.rows = {
            k: [i for i, m in enumerate(meta) if m["clip"] in v]
            for k, v in split.items()
        }

    def batch(self, rows: list[int], starts: list[int]) -> np.ndarray:
        return np.stack([
            self.frames[r, s : s + WINDOW] for r, s in zip(rows, starts, strict=True)
        ])


def train_epoch(model, clips: Clips, opt, cfg: dict) -> float:
    """One pass over the train clips in random order, a random window each."""
    rng, rows, losses = cfg["rng"], list(clips.rows["train"]), []
    rng.shuffle(rows)
    model.train()
    for i in range(0, len(rows), cfg["batch"]):
        chunk = rows[i : i + cfg["batch"]]
        starts = rng.integers(
            0, clips.frames.shape[1] - WINDOW + 1, len(chunk)
        ).tolist()
        x = to_unit(clips.batch(chunk, starts), cfg["device"])
        with torch.autocast(cfg["device"].split(":")[0], dtype=torch.bfloat16):
            loss = rollout_loss(model, x, ctx=CTX_FRAMES, teacher=cfg["teacher"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def val_mse(model, clips: Clips, cfg: dict) -> float:
    """Pixel MSE over the 48-frame rollout on the val clips' cache windows."""
    model.eval()
    errs = []
    for r in clips.rows["val"]:
        x = to_unit(clips.batch([r] * len(STARTS), list(STARTS)), cfg["device"])
        pred = model.rollout(x[:, :CTX_FRAMES], FORECAST)
        errs.append((pred - x[:, CTX_FRAMES:]).pow(2).mean().item())
    return float(np.mean(errs))


def teacher_ratio(epoch: int, epochs: int) -> float:
    """Teacher forcing from 1 down to 0 over the first half of training, then 0."""
    return max(0.0, 1.0 - epoch / max(1, epochs // 2))


def run_epoch(model, clips: Clips, opt, cfg: dict) -> dict:
    return {
        "epoch": cfg["epoch"] + 1,
        "teacher": cfg["teacher"],
        "train_mse": train_epoch(model, clips, opt, cfg),
        "val_mse": val_mse(model, clips, cfg),
    }


def train(model, clips: Clips, args, device: str) -> tuple:
    """Train with the teacher-forcing ramp; keep the epoch with the lowest val MSE."""
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = CosineAnnealingLR(opt, args.epochs)
    cfg = {"rng": np.random.default_rng(args.seed), "batch": args.batch, "device": device}
    best, history = (np.inf, None), []
    for epoch in range(args.epochs):
        cfg |= {"epoch": epoch, "teacher": teacher_ratio(epoch, args.epochs)}
        entry = run_epoch(model, clips, opt, cfg)
        sched.step()
        history.append(entry)
        LOG.info(json.dumps(entry))
        if entry["val_mse"] < best[0]:
            best = (entry["val_mse"], copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[1])
    return model, history


@torch.no_grad()
def forecast_test(model, clips: Clips, device: str) -> dict:
    """uint8 forecasts of every test clip: ``ctx8`` (2, 48, H, W) and ``ctx16`` (48, H, W)."""
    model.eval()
    out = {}
    for r in clips.rows["test"]:
        x = to_unit(clips.batch([r] * len(STARTS), list(STARTS)), device)
        full = to_unit(clips.frames[r][None], device)
        out[clips.meta[r]["clip"]] = {
            "ctx8": to_uint8(model.rollout(x[:, :CTX_FRAMES], FORECAST)),
            "ctx16": to_uint8(model.rollout(full[:, LONG_CTX], FORECAST)[0]),
        }
    return out


def encode_forecasts(forecasts: dict, probes: Path, device: str) -> dict:
    """Per clip and context: pooled means ``(.., 24, 1024)`` and attentive readings ``(.., 24, 3)``, leading axis = snippets (2 for ctx8, 1 for ctx16)."""
    encode = make_encoder(device, per_frame=True)
    att, _, (mu, sd) = load_probes(probes, device)

    def read(frames: np.ndarray) -> tuple:
        tokens = encode(frames)
        with torch.no_grad():
            reading = (
                att(torch.from_numpy(tokens.astype(np.float32)).to(device))
                .cpu()
                .numpy()
            )
        return tokens.astype(np.float32).mean(1), reading * sd + mu

    out = {}
    for i, (clip, by_ctx) in enumerate(forecasts.items()):
        LOG.info(f"[encode {i + 1}/{len(forecasts)}] {clip}")
        for ctx, frames in by_ctx.items():
            pairs = [read(f) for f in frames.reshape(-1, FORECAST, *frames.shape[-2:])]
            out[f"{clip}/{ctx}/pooled"] = np.stack([p for p, _ in pairs])
            out[f"{clip}/{ctx}/attentive"] = np.stack([a for _, a in pairs])
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tubelets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encode", action="store_true")
    ap.add_argument(
        "--encode-only", action="store_true", help="skip training; encode --out/forecasts.npz"
    )
    ap.add_argument(
        "--shard", default="0/1", help="i/n: encode every n-th clip from the i-th (one per GPU)"
    )
    ap.add_argument(
        "--probes", type=Path, default=Path("experiments/gate1-perframe/probes.pt")
    )
    return ap.parse_args()


def save_forecasts(forecasts: dict, out: Path) -> None:
    flat = {f"{c}/{k}": v for c, d in forecasts.items() for k, v in d.items()}
    np.savez(out / "forecasts.npz", **flat)


def load_forecasts(out: Path) -> dict:
    """``forecasts.npz`` back to ``{clip: {ctx: frames}}``."""
    z = np.load(out / "forecasts.npz")
    forecasts: dict = {}
    for key in z.files:
        clip, ctx = key.rsplit("/", 1)
        forecasts.setdefault(clip, {})[ctx] = z[key]
    return forecasts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    device = pick_device()
    torch.manual_seed(args.seed)
    if args.encode_only:
        forecasts = load_forecasts(args.out)
    else:
        clips = Clips(args.tubelets)
        model, history = train(PixelForecaster().to(device), clips, args, device)
        args.out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.out / "convlstm.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=1))
        forecasts = forecast_test(model, clips, device)
        save_forecasts(forecasts, args.out)
    if args.encode or args.encode_only:
        i, n = map(int, args.shard.split("/"))
        mine = {c: f for k, (c, f) in enumerate(forecasts.items()) if k % n == i}
        name = "encoded.npz" if n == 1 else f"encoded-{i}of{n}.npz"
        np.savez(args.out / name, **encode_forecasts(mine, args.probes, device))


if __name__ == "__main__":
    main()
