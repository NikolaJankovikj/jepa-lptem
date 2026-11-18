"""Step 15 M2/M3: train a CNN or LoRA V-JEPA 2 end to end with the Gate-1 heads.

Same input as the per-frame cache (one 2-frame tubelet, 256² centre crop), same
heads (``AttentiveProbe``, ``CentroidProbe``), same loss (``probe_loss``), same
split (``split_by_clip`` seed 0) and same scoring as ``train_probes.py``: only the
encoder changes. Scoring runs on the cache's windows (``--window`` / ``--stride``
in input frames), so frozen V-JEPA's Gate-1 numbers are directly comparable.

- ``cnn``  -- a small conv net, stride 16 -> 16x16 tokens, trained from scratch
- ``lora`` -- V-JEPA 2 ViT-L with LoRA on the encoder's attention, base frozen

The checkpoint is the epoch with the lowest val loss. ``preds.npz`` keeps the
val/test predictions per clip for the M4 analysis (dose split, bootstrap CIs).

Usage (local smoke):
    uv run python scripts/train_encoder_compare.py --model cnn \
        --tubelets data/tubelets/smoke_v3 --out experiments/encoder-compare/smoke-cnn \
        --epochs 3 --window 48 --stride 48
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from extract_features import MODEL_NAME, TUBELET, pick_device, window_starts
from torch import nn
from torch.utils.data import DataLoader
from train_probes import (
    GRID,
    OBS_NAMES,
    probe_loss,
    r_squared,
    split_by_clip,
    train_clips,
)
from tubelet_cache import TubeletSet

from jepa_lptem.eval.centroid_eval import eval_centroid_msd, pick_threshold
from jepa_lptem.eval.probes import AttentiveProbe, CentroidProbe
from itertools import starmap

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CNN_DIM = 256
LORA_RANK, LORA_ALPHA = 16, 32
LORA_TARGETS = r"encoder\.layer\.\d+\.attention\.(query|key|value|proj)"
ENCODER_LR = {"cnn": 1e-3, "lora": 1e-4}
HEAD_LR = 1e-3
PREDICT_BATCH = 16


# --------------------------------- encoders -----------------------------------


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Halve the resolution, then one more conv at the new size."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=2, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class TubeletCNN(nn.Module):
    """(B, 2, H, W) uint8 -> (B, (H/16)², CNN_DIM) tokens: four stride-2 blocks,
    so each token covers the same 16 px cell as a V-JEPA token."""

    def __init__(self, dim: int = CNN_DIM):
        super().__init__()
        widths = (TUBELET, 32, 64, 128, dim)
        self.net = nn.Sequential(
            *starmap(conv_block, zip(widths, widths[1:], strict=False))
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x = self.net(frames.float() / 127.5 - 1.0)
        return x.flatten(2).transpose(1, 2)


def vjepa_input(frames: torch.Tensor) -> torch.Tensor:
    """(B, 2, H, W) uint8 -> (B, 2, 3, H, W): grey to RGB, rescale and ImageNet
    normalization, what ``load_processor`` does with its resize switched off."""
    x = frames.float().div(255.0).unsqueeze(2).expand(-1, -1, 3, -1, -1)
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    return (x - mean) / std


class LoraVJEPA(nn.Module):
    """V-JEPA 2 ViT-L encoder with LoRA adapters; at init it equals the frozen
    encoder (LoRA's B starts at zero)."""

    def __init__(self):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import VJEPA2Model

        base = VJEPA2Model.from_pretrained(MODEL_NAME)
        cfg = LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS
        )
        self.model = get_peft_model(base, cfg)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values_videos=vjepa_input(frames), skip_predictor=True)
        return out.last_hidden_state


def make_encoder(name: str) -> tuple[nn.Module, int]:
    if name == "cnn":
        return TubeletCNN(), CNN_DIM
    return LoraVJEPA(), 1024


class Reader(nn.Module):
    """Encoder plus the Gate-1 heads."""

    def __init__(self, name: str):
        super().__init__()
        self.encoder, dim = make_encoder(name)
        self.att = AttentiveProbe(dim=dim, out_dim=3)
        self.cen = CentroidProbe(dim=dim, grid=GRID)

    def forward(self, frames: torch.Tensor):
        tokens = self.encoder(frames)
        return self.att(tokens), *self.cen(tokens)

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def optimizer(self, name: str) -> torch.optim.Optimizer:
        enc = [p for p in self.encoder.parameters() if p.requires_grad]
        heads = [*self.att.parameters(), *self.cen.parameters()]
        return torch.optim.AdamW(
            [
                {"params": enc, "lr": ENCODER_LR[name]},
                {"params": heads, "lr": HEAD_LR},
            ]
        )


# --------------------------------- training -----------------------------------


@dataclass
class Objective:
    """The Gate-1 loss with its train-split statistics, on one device."""

    mu: np.ndarray
    sd: np.ndarray
    bce: nn.Module
    device: str

    @classmethod
    def from_train(cls, ds: TubeletSet, rows: list[int], device: str) -> Objective:
        att = ds.attentive[rows].reshape(-1, 3)
        occ_frac = float(ds.occupancy[rows].mean())
        pos_weight = torch.tensor((1 - occ_frac) / max(occ_frac, 1e-4), device=device)
        return cls(
            att.mean(axis=0),
            att.std(axis=0) + 1e-9,
            nn.BCEWithLogitsLoss(pos_weight=pos_weight),
            device,
        )

    def loss(self, model: Reader, batch) -> torch.Tensor:
        frames, att, occ, off = (b.to(self.device) for b in batch)
        mu, sd = (
            torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for v in (self.mu, self.sd)
        )
        with torch.autocast("cuda", torch.bfloat16, enabled=self.device == "cuda"):
            pred, occ_p, off_p = model(frames)
        y = (att.float() - mu) / sd
        predicted = (pred.float(), occ_p.float(), off_p.float())
        return probe_loss(predicted, (y, occ.float(), off.float()), self.bce)


def run_epoch(model: Reader, loader, objective: Objective, opt=None) -> float:
    """One pass; trains when ``opt`` is given, else only measures."""
    model.train(opt is not None)
    losses = []
    for batch in loader:
        with torch.set_grad_enabled(opt is not None):
            loss = objective.loss(model, batch)
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def train(model: Reader, loaders: dict, objective: Objective, args) -> list[dict]:
    """Train for ``args.epochs``; leaves the best-val-loss weights in ``model``."""
    opt = model.optimizer(args.model)
    history, best = [], (math.inf, None)
    for ep in range(args.epochs):
        tr = run_epoch(model, loaders["train"], objective, opt)
        va = run_epoch(model, loaders["val"], objective)
        history.append({"epoch": ep + 1, "train_loss": tr, "val_loss": va})
        print(
            f"  epoch {ep + 1:3d}/{args.epochs}  train {tr:.4f}  val {va:.4f}",
            flush=True,
        )
        if va < best[0]:
            best = (va, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[1])
    return history


# --------------------------------- scoring ------------------------------------


@torch.no_grad()
def predict_clip(model: Reader, frames: np.ndarray, objective: Objective):
    """All latent frames of one clip -> observables (T, 3), occupancy probability
    (T, G, G) and offsets (T, G, G, 2)."""
    model.eval()
    tubelets = torch.from_numpy(np.array(frames)).reshape(
        -1, TUBELET, *frames.shape[1:]
    )
    outs = [
        model(tubelets[i : i + PREDICT_BATCH].to(objective.device))
        for i in range(0, len(tubelets), PREDICT_BATCH)
    ]
    att, occ, off = (
        torch.cat(o).float().cpu().numpy() for o in zip(*outs, strict=True)
    )
    return att * objective.sd + objective.mu, 1 / (1 + np.exp(-occ)), off


@dataclass
class Windowing:
    """The feature cache's windows, in input frames."""

    window: int
    stride: int

    def slices(self, n_lat_total: int) -> list[slice]:
        n_lat = self.window // TUBELET
        starts = window_starts(n_lat_total * TUBELET, self.window, self.stride)
        return [slice(s // TUBELET, s // TUBELET + n_lat) for s in starts]

    def frame_index(self, n_lat_total: int) -> np.ndarray:
        """Latent frames of all windows, concatenated: overlaps repeat, as in
        Gate 1's per-window R²."""
        return np.concatenate(
            [np.arange(s.start, s.stop) for s in self.slices(n_lat_total)]
        )


def windows_of(pred, ds: TubeletSet, row: int, win: Windowing):
    """One clip's predictions cut into cache windows, in the
    ``(occ, off, gt centroids, meta)`` form ``centroid_eval`` scores."""
    _, occ, off = pred
    pos, rad = ds.positions[row], ds.radii[row]
    for t in win.slices(rad.shape[1]):
        cents = [pos[rad[:, k] > 0, k] for k in range(t.start, t.stop)]
        yield occ[t], off[t], cents, ds.meta[row]


def r2_scores(preds: dict, ds: TubeletSet, rows: list[int], frames_idx) -> dict:
    """R² per observable, overall and on the physics stratum."""
    p = np.concatenate([preds[r][0][frames_idx] for r in rows])
    y = np.concatenate([ds.attentive[r][frames_idx] for r in rows])
    phys = np.repeat(
        [ds.meta[r]["stratum"] == "physics" for r in rows], len(frames_idx)
    )
    out = {}
    for j, name in enumerate(OBS_NAMES):
        out[f"r2_{name}"] = r_squared(p[:, j], y[:, j])
        if phys.any():
            out[f"r2_{name}_physics"] = r_squared(p[phys, j], y[phys, j])
    return out


def score(preds: dict, ds: TubeletSet, rows: dict, win: Windowing) -> dict:
    """Gate-1 metrics on val and test, threshold picked on val."""
    frames_idx = win.frame_index(ds.attentive.shape[1])

    def wins(part):
        for r in rows[part]:
            yield from windows_of(preds[r], ds, r, win)

    threshold = pick_threshold(wins("val"))
    return {
        part: {
            **r2_scores(preds, ds, rows[part], frames_idx),
            **eval_centroid_msd(wins(part), threshold),
        }
        for part in ("val", "test")
    }


# ----------------------------------- main -------------------------------------


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("cnn", "lora"), required=True)
    ap.add_argument("--tubelets", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=36)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def split_rows(ds: TubeletSet, args) -> tuple[dict, dict]:
    """Clip names and dataset rows per split; train shrunk to ``--train-frac``."""
    split = split_by_clip([m["clip"] for m in ds.meta], seed=0)
    clips = {**split, "train": train_clips(split, args.train_frac, args.seed)}
    rows = {
        k: [i for i, m in enumerate(ds.meta) if m["clip"] in v]
        for k, v in clips.items()
    }
    return clips, rows


def loaders_for(clips: dict, args) -> dict:
    def loader(part, shuffle):
        ds = TubeletSet(args.tubelets, clips[part])
        return DataLoader(ds, args.batch, shuffle=shuffle, num_workers=args.workers)

    return {"train": loader("train", True), "val": loader("val", False)}


def save(model: Reader, preds: dict, ds: TubeletSet, out: Path) -> None:
    """Trainable weights (LoRA + heads, or the whole CNN) and the per-clip
    val/test predictions."""
    out.mkdir(parents=True, exist_ok=True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    state = model.state_dict()
    keep = {k: v for k, v in state.items() if k in trainable or "encoder.net" in k}
    torch.save(keep, out / "reader.pt")
    np.savez(
        out / "preds.npz",
        clips=np.array([ds.meta[r]["clip"] for r in preds]),
        attentive=np.stack([p[0] for p in preds.values()]),
    )


def describe(model: Reader, args, clips: dict) -> dict:
    """What was trained, for metrics.json."""
    return {
        "model": args.model,
        "train_frac": args.train_frac,
        "n_train_clips": len(clips["train"]),
        "n_trainable": model.n_trainable(),
        "split": {k: sorted(v) for k, v in clips.items()},
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device()
    ds = TubeletSet(args.tubelets)
    clips, rows = split_rows(ds, args)
    model = Reader(args.model).to(device)
    print(
        f"{args.model}: {model.n_trainable() / 1e6:.2f} M trainable, {len(rows['train'])} train clips, {device}"
    )
    objective = Objective.from_train(ds, rows["train"], device)
    history = train(model, loaders_for(clips, args), objective, args)
    preds = {
        r: predict_clip(model, ds.frames[r], objective)
        for r in rows["val"] + rows["test"]
    }
    metrics = score(preds, ds, rows, Windowing(args.window, args.stride))
    metrics.update(describe(model, args, clips), history=history)
    save(model, preds, ds, args.out)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["test"], indent=1))


if __name__ == "__main__":
    main()
