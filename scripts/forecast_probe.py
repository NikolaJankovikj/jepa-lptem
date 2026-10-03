"""Forecast-trained readout -- POST HOC diagnostic, fixed in PLAN Step 10 before it ran.

The frozen Gate-1 probes read the per-frame forecasts as fading particles
(experiments/core-perframe). That cannot tell "the forecast lost the size
information" from "the information is still there, blurred into a form those
probes never saw". So a fresh probe is trained on each kind of future latent and
asked for the ground truth at that future frame.

Snippets as in the follow-up: context = latent frames 0-7, forecast = 8-31 (+24).
Four sources of the latent at +h, each with its own probe:

- ``model``       -- the predictor's rollout (frozen)
- ``persistence`` -- frame 7, held
- ``ctxmean``     -- the mean of frames 0-7, held
- ``floor``       -- the true encoded future: the best any probe can do

Every probe is the Gate-1 ``AttentiveProbe`` plus a learned horizon embedding, so
the persistence and context-mean probes are *learned extrapolators from the
context*: they can read the current size and the process's look and turn them into
a size at +h. A forecast carries dynamics only if its probe beats theirs.

Test (physics stratum, diffusion and reaction): the reference is the better of
persistence / context mean in mean-radius error at +24 on that process; powered iff
the floor's paired gain over it has a clip-bootstrap CI above 0; "growth is in the
forecasts" iff the model's does too.

The rollout runs under bf16 autocast, the precision the predictor trained in: it
is recomputed every epoch (caching 3200 x 24 forecast frames would take ~40 GB).

Usage (remote, ~2-3 h on one GPU):
    uv run python -u scripts/forecast_probe.py \
        --features data/features/dynamics-v1-perframe --gate1 experiments/gate1-perframe \
        --predictor experiments/predictor-perframe/predictor.pt \
        --out experiments/forecast-probe
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from eval_forecasts import METHODS, load_predictor, snippet_exponent, snippet_truth
from gate1_secondary import split_indices
from gate2_followup import bootstrap_ci, size_test
from torch import nn
from torch.utils.data import DataLoader, Dataset
from train_predictor import autocast, pick_device
from train_probes import WindowCache, attentive_targets

from jepa_lptem.eval.probes import AttentiveProbe

HORIZON = 24
REPORT_HORIZONS = (1, 5, 10, 24)
TEST_PROCESSES = ("diffusion", "reaction")
BASELINES = ("persistence", "ctxmean")


class HorizonProbe(nn.Module):
    """The Gate-1 attentive probe, told how many steps ahead its latent is."""

    def __init__(self, horizon: int = HORIZON, dim: int = 1024, heads: int = 8):
        super().__init__()
        self.h_emb = nn.Parameter(torch.randn(horizon, dim) * 0.02)
        self.probe = AttentiveProbe(dim, out_dim=3, heads=heads)

    def forward(self, tokens: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """``(B, S, D)`` latents, ``(B,)`` 0-based horizons -> ``(B, 3)``."""
        return self.probe(tokens + self.h_emb[h][:, None])


@dataclass
class Readout:
    """What every pass shares: the frozen predictor (with its feature
    standardization and context length), the target z-scoring, one probe per source."""

    model: nn.Module
    stats: tuple[torch.Tensor, torch.Tensor]
    ctx: int
    norm: tuple[torch.Tensor, torch.Tensor]
    probes: dict[str, HorizonProbe]


class Windows(Dataset):
    """One item = a window's context + forecast latents and the GT observables
    ``[count, mean_radius, total_area]`` at the forecast frames."""

    def __init__(self, cache: WindowCache, idx: list[int], ctx: int):
        self.cache, self.idx, self.ctx = cache, idx, ctx

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k: int):
        feats, _, rad, _ = self.cache.load(self.idx[k])
        span = slice(self.ctx, self.ctx + HORIZON)
        return (
            torch.from_numpy(feats[: self.ctx + HORIZON]),
            torch.from_numpy(attentive_targets(rad[:, span]).astype(np.float32)),
        )


def futures(r: Readout, x: torch.Tensor) -> dict:
    """``(B, ctx + HORIZON, S, D)`` raw latents -> each source's ``(B, HORIZON, S, D)``."""
    mean, std = r.stats
    context = x[:, : r.ctx]
    with autocast(x.device.type):
        rolled = r.model.rollout((context - mean) / std, HORIZON)

    def held(frame):
        return frame[:, None].expand(-1, HORIZON, -1, -1)

    return {
        "model": rolled.float() * std + mean,
        "persistence": held(context[:, -1]),
        "ctxmean": held(context.mean(dim=1)),
        "floor": x[:, r.ctx :],
    }


def batch_losses(probes, fut: dict, y: torch.Tensor) -> dict:
    """Per source, z-scored MSE over every forecast frame of the batch."""
    b = y.shape[0]
    h = torch.arange(HORIZON, device=y.device).repeat(b)
    target = y.reshape(-1, y.shape[-1])
    return {
        s: nn.functional.mse_loss(
            probes[s](fut[s].reshape(b * HORIZON, *fut[s].shape[2:]), h), target
        )
        for s in METHODS
    }


def run_epoch(r: Readout, loader, opt=None) -> dict:
    """One pass; trains when ``opt`` is given. Returns the mean loss per source."""
    mu, sd = r.norm
    for p in r.probes.values():
        p.train(opt is not None)
    sums = dict.fromkeys(METHODS, 0.0)
    for x, y in loader:
        x, y = x.to(mu.device), (y.to(mu.device) - mu) / sd
        fut = futures(r, x)
        with torch.set_grad_enabled(opt is not None):
            losses = batch_losses(r.probes, fut, y)
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            sum(losses.values()).backward()
            opt.step()
        for s, v in losses.items():
            sums[s] += v.item()
    return {s: v / max(len(loader), 1) for s, v in sums.items()}


def train(r: Readout, loaders: dict, epochs: int) -> dict:
    """Joint training (the probes share no weights); each source keeps the epoch
    with its lowest val loss. Returns the picked epoch per source."""
    params = [p for m in r.probes.values() for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=1e-3)
    best = {s: (float("inf"), None, 0) for s in METHODS}
    for ep in range(1, epochs + 1):
        tr = run_epoch(r, loaders["train"], opt)
        va = run_epoch(r, loaders["val"])
        for s in METHODS:
            if va[s] < best[s][0]:
                best[s] = (va[s], copy.deepcopy(r.probes[s].state_dict()), ep)
        print(
            f"epoch {ep:2d}  train / val  "
            + "  ".join(f"{s} {tr[s]:.3f}/{va[s]:.3f}" for s in METHODS),
            flush=True,
        )
    for s, (_, state, _) in best.items():
        r.probes[s].load_state_dict(state)
    return {s: best[s][2] for s in METHODS}


# ------------------------------ test scoring ------------------------------------


def score_source(obs: np.ndarray, truth: dict) -> dict:
    """One source's probed ``(HORIZON, 3)`` observables against the truth."""
    return {
        "meanr_err": np.abs(obs[:, 1] - truth["obs"][:, 1]),
        "exponent": snippet_exponent(truth["times"], obs[:, 1]),
    }


def collect(cache, test, r: Readout) -> list[dict]:
    """Every physics-stratum test window, scored per source at the 24 forecast frames."""
    mu, sd = r.norm
    h = torch.arange(HORIZON, device=mu.device)
    for p in r.probes.values():
        p.eval()
    records = []
    for i in test:
        feats, pos, rad, meta = cache.load(i)
        if meta["stratum"] != "physics":
            continue
        truth = snippet_truth(pos, rad, meta, np.arange(r.ctx, r.ctx + HORIZON))
        x = torch.from_numpy(feats[: r.ctx + HORIZON])[None].to(mu.device)
        with torch.no_grad():
            fut = futures(r, x)
            obs = {
                s: (r.probes[s](fut[s][0], h) * sd + mu).cpu().numpy() for s in METHODS
            }
        scores = {s: score_source(obs[s], truth) for s in METHODS}
        records.append({"meta": meta, "truth": truth, "scores": scores})
    return records


def per_horizon(rows, source: str, key: str) -> dict:
    series = np.array([row["scores"][source][key] for row in rows])
    return {f"+{k}": float(series[:, k - 1].mean()) for k in REPORT_HORIZONS}


def process_entry(rows) -> dict:
    clips = [row["meta"]["clip"] for row in rows]
    entry = {
        "n_snippets": len(rows),
        "true_exponent": rows[0]["meta"].get("exponent"),
        "gt_crop_exponent": float(
            np.nanmedian([row["truth"]["exponent_crop"] for row in rows])
        ),
    }
    for s in METHODS:
        exps = [row["scores"][s]["exponent"] for row in rows]
        entry[s] = {
            "mean_radius_mae": per_horizon(rows, s, "meanr_err"),
            "exponent": {
                "median": float(np.nanmedian(exps)),
                "ci": bootstrap_ci(exps, clips, np.nanmedian),
            },
        }
    return entry


def growth_test(rows) -> dict:
    """The fixed reading: paired mean-radius-error gains at +24 over the better
    baseline on this process (picking it on test favours the baselines)."""
    err = {
        s: np.array([row["scores"][s]["meanr_err"][-1] for row in rows])
        for s in METHODS
    }
    ref = min(BASELINES, key=lambda s: err[s].mean())
    clips = [row["meta"]["clip"] for row in rows]
    return {"reference": ref, **size_test(err[ref], err["model"], err["floor"], clips)}


def analyse(records: list[dict]) -> dict:
    procs = sorted({row["meta"]["process"] for row in records})
    by_proc = {
        p: [row for row in records if row["meta"]["process"] == p] for p in procs
    }
    return {
        "by_process": {p: process_entry(rows) for p, rows in by_proc.items()},
        "growth_test": {
            p: growth_test(by_proc[p]) for p in TEST_PROCESSES if p in by_proc
        },
    }


# -------------------------------- reporting -------------------------------------


def print_errors(by_process: dict) -> None:
    print(f"\nmean-radius error (px) by horizon ({' / '.join(METHODS)}):")
    for p, e in by_process.items():
        cells = (
            f"+{k}: "
            + " / ".join(f"{e[s]['mean_radius_mae'][f'+{k}']:.2f}" for s in METHODS)
            for k in REPORT_HORIZONS
        )
        print(f"  {p:12s} " + "   ".join(cells))


def print_exponents(by_process: dict) -> None:
    print(
        f"\ngrowth exponent over +1..+{HORIZON}, median [CI] "
        f"({' / '.join(METHODS)}) | GT-in-crop / true:"
    )
    for p, e in by_process.items():
        cells = (
            f"{e[s]['exponent']['median']:.3f} "
            f"[{e[s]['exponent']['ci'][0]:.2f}, {e[s]['exponent']['ci'][1]:.2f}]"
            for s in METHODS
        )
        print(
            f"  {p:12s} "
            + " / ".join(cells)
            + f" | {e['gt_crop_exponent']:.3f} / {e['true_exponent']}"
        )


def print_growth_test(tests: dict) -> None:
    print(
        f"\ngrowth test at +{HORIZON} (mean-radius error gain over the better baseline):"
    )
    label = {True: "YES, growth in the forecasts", False: "no", None: "untestable"}
    for p, g in tests.items():
        print(
            f"  {p:12s} reference {g['reference']}: floor {g['gain_floor']:+.2f} "
            f"[{g['gain_floor_ci'][0]:+.2f}, {g['gain_floor_ci'][1]:+.2f}]  model "
            f"{g['gain_model']:+.2f} [{g['gain_model_ci'][0]:+.2f}, "
            f"{g['gain_model_ci'][1]:+.2f}]  -> {label[g['pass']]}"
        )


def report(results: dict, picked: dict, out: Path) -> None:
    print_errors(results["by_process"])
    print_exponents(results["by_process"])
    print_growth_test(results["growth_test"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps({"epochs_picked": picked, **results}, indent=2, default=float)
    )
    print(f"\nwrote {out}/metrics.json")


# ---------------------------------- setup ---------------------------------------


def target_norm(cache, train_idx, ctx: int, device: str):
    """Train-split mean/std of the forecast-frame observables (z-scored MSE)."""
    t_all = np.concatenate(
        [
            attentive_targets(cache.load_gt(i)[1][:, ctx : ctx + HORIZON])
            for i in train_idx
        ]
    )
    return tuple(
        torch.as_tensor(a, dtype=torch.float32, device=device)
        for a in (t_all.mean(axis=0), t_all.std(axis=0) + 1e-9)
    )


def make_readout(cache, idx, args, device: str) -> Readout:
    model, (mean, std), ctx = load_predictor(args.predictor, device)
    model.requires_grad_(False)
    dim = cache.load(idx["train"][0])[0].shape[-1]
    return Readout(
        model=model,
        stats=tuple(
            torch.as_tensor(a, dtype=torch.float32, device=device) for a in (mean, std)
        ),
        ctx=ctx,
        norm=target_norm(cache, idx["train"], ctx, device),
        probes={s: HorizonProbe(HORIZON, dim).to(device) for s in METHODS},
    )


def make_loaders(cache, idx, ctx: int, args) -> dict:
    return {
        part: DataLoader(
            Windows(cache, idx[part], ctx),
            batch_size=args.windows_per_batch,
            shuffle=part == "train",
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
        )
        for part in ("train", "val")
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument(
        "--gate1", required=True, type=Path, help="metrics.json with the split"
    )
    ap.add_argument("--predictor", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--windows-per-batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    device = pick_device()
    cache = WindowCache(args.features)
    idx = split_indices(cache, args.gate1 / "metrics.json", ("train", "val", "test"))
    readout = make_readout(cache, idx, args, device)
    print(
        f"windows {', '.join(f'{k} {len(v)}' for k, v in idx.items())}  device {device}"
        f"  ctx {readout.ctx}  horizon {HORIZON}  epochs {args.epochs}  (post hoc)",
        flush=True,
    )
    picked = train(readout, make_loaders(cache, idx, readout.ctx, args), args.epochs)
    print(f"epochs picked on val: {picked}")

    report(analyse(collect(cache, idx["test"], readout)), picked, args.out)


if __name__ == "__main__":
    main()
