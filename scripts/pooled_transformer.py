"""P3: candidate B, a small causal transformer on the pooled state, judged by Gate 2'.

PLAN Step 10, position-free forecasting, fixed 2026-10-11 before any pooled number.
Each latent frame's pooled state (the z-scored frame average, P1) is one token. A
causal transformer (dim 256, depth 4, heads 4) reads the 8 context frames. Its last
output plus a learned horizon embedding predicts the pooled state at +h for every
h = 1..24 directly, as a change from the last context frame (zero-initialised, so the
untrained model is persistence, like the token-level predictor). L1 on the z-scored
pooled state, AdamW 1e-3, cosine, 100 epochs, batch 64, seed 0. The checkpoint is
selected on val L1 at +24. No labels. Then the same evaluation and Gate 2' as P2
(``pooled_forecast.evaluate``), with the linear candidate alongside.

Runs on the Mac (CPU; a few minutes):
    uv run python scripts/pooled_transformer.py \
        --pooled data/features/dynamics-v1-perframe-pooled.npz \
        --gate1 experiments/gate1-perframe --readout experiments/pooled-gate1/readout.npz \
        --out experiments/pooled-forecast
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from pooled_forecast import evaluate, load_readout, print_report
from pooled_gate1 import CTX, HORIZON, load_context
from torch import nn
from train_predictor import cosine_with_warmup

DIM, DEPTH, HEADS = 256, 4, 4
LR, EPOCHS, BATCH, SEED = 1e-3, 100, 64, 0


class PooledForecaster(nn.Module):
    """``(B, CTX, D)`` pooled context -> ``(B, HORIZON, D)`` forecasts of +1..+HORIZON."""

    def __init__(self, enc_dim: int):
        super().__init__()
        self.inp = nn.Linear(enc_dim, DIM)
        self.t_emb = nn.Parameter(torch.zeros(1, CTX, DIM))
        self.h_emb = nn.Parameter(torch.randn(1, HORIZON, DIM) * 0.02)
        layer = nn.TransformerEncoderLayer(
            DIM,
            HEADS,
            4 * DIM,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, DEPTH, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(DIM)
        self.out = nn.Linear(DIM, enc_dim)
        nn.init.zeros_(self.out.weight)  # starts as exact persistence
        nn.init.zeros_(self.out.bias)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        mask = nn.Transformer.generate_square_subsequent_mask(CTX, device=ctx.device)
        h = self.blocks(self.inp(ctx) + self.t_emb, mask=mask, is_causal=True)[:, -1:]
        return ctx[:, -1:] + self.out(self.norm(h + self.h_emb))


def train(z: np.ndarray, split: dict) -> tuple[PooledForecaster, list[dict]]:
    """Fit on train windows; keep the epoch with the lowest val L1 at +24."""
    torch.manual_seed(SEED)
    x = torch.from_numpy(z.astype(np.float32))
    model = PooledForecaster(x.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    tr = torch.from_numpy(split["train"])
    steps = EPOCHS * -(-len(tr) // BATCH)
    sched = cosine_with_warmup(opt, steps, warmup=min(500, steps // 10))
    history, best = [], (float("inf"), None)
    for epoch in range(1, EPOCHS + 1):
        loss = run_epoch(model, x[tr], (opt, sched))
        val_l1 = val_l1_at_24(model, x[torch.from_numpy(split["val"])])
        history.append({"epoch": epoch, "train_l1": loss, "val_l1_24": val_l1})
        if val_l1 < best[0]:
            best = (val_l1, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[1])
    return model, history


def run_epoch(model: PooledForecaster, windows: torch.Tensor, opt) -> float:
    optimizer, sched = opt
    model.train()
    losses = []
    for batch in torch.randperm(len(windows)).split(BATCH):
        w = windows[batch]
        loss = (model(w[:, :CTX]) - w[:, CTX:]).abs().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sched.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def val_l1_at_24(model: PooledForecaster, windows: torch.Tensor) -> float:
    model.eval()
    return float((model(windows[:, :CTX])[:, -1] - windows[:, -1]).abs().mean())


@torch.no_grad()
def forecast_all(model: PooledForecaster, z: np.ndarray) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(z[:, :CTX].astype(np.float32))).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pooled", type=Path, required=True)
    ap.add_argument("--gate1", type=Path, required=True)
    ap.add_argument("--readout", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pooled, ctx = load_context(args.pooled, args.gate1)
    readout = load_readout(args.readout)
    z = (pooled["mean"].astype(np.float32) - readout.x_mu) / readout.x_sd
    model, history = train(z, ctx["split"])
    result = evaluate(readout, z, ctx, extra={"transformer": forecast_all(model, z)})
    print_report(result)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics_p3.json").write_text(
        json.dumps(result | {"history": history}, indent=1)
    )
    torch.save(model.state_dict(), args.out / "transformer.pt")
    print(f"\nwrote {args.out}/metrics_p3.json, transformer.pt")


if __name__ == "__main__":
    main()
