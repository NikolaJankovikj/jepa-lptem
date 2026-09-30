"""Block-causal latent forecaster: rolls V-JEPA 2 latent frames forward in time.

The predictor of the latent world model (Step 9). The input is a context of
encoded latent frames, each a grid of ``n_spatial`` tokens; one transformer over
all of them predicts, for every frame t, the tokens of frame t+1. A token attends
within its own frame and to earlier frames only (V-JEPA 2-AC style, no actions),
so the forecast of t+1 never sees t+1 or later -- tested directly.

Three choices differ from the PLAN draft, all deliberate:

- **It predicts the change, from persistence.** Output = current frame + a learned
  update whose head starts at zero, so the untrained model *is* the persistence
  baseline ("next frame = this frame"). Much of the motion is a martingale that
  nothing can beat (Brownian position), so the model should start at that floor
  and learn only what is predictable on top of it.
- **A narrower working width** (``dim``, projected in and out of the encoder's
  ``enc_dim``), as V-JEPA's own predictors do. Gate 1 showed the data, not the
  compute, is the limit: 3200 training windows do not need a 100M-parameter,
  1024-wide predictor.
- **A learned spatial embedding** beside the temporal one, so attention knows which
  tokens are neighbours without relying on the encoder to have kept it.
"""

from __future__ import annotations

import torch
from torch import nn


class LatentForecaster(nn.Module):
    """``(B, T, S, enc_dim)`` latent frames -> per-frame next-frame forecasts."""

    def __init__(
        self,
        enc_dim: int = 1024,
        dim: int = 512,
        depth: int = 8,
        heads: int = 8,
        n_spatial: int = 256,
        max_t: int = 64,
    ):
        super().__init__()
        self.n_spatial = n_spatial
        self.inp = nn.Linear(enc_dim, dim)
        self.t_emb = nn.Parameter(torch.zeros(1, max_t, 1, dim))
        self.s_emb = nn.Parameter(torch.randn(1, 1, n_spatial, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            4 * dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, enc_dim)
        nn.init.zeros_(self.out.weight)  # starts as exact persistence
        nn.init.zeros_(self.out.bias)

    def _mask(self, n_frames: int, device) -> torch.Tensor:
        frame = torch.arange(n_frames * self.n_spatial, device=device) // self.n_spatial
        return (
            frame[:, None] < frame[None, :]
        )  # True = blocked: the key is a later frame

    def forward_all(self, ctx: torch.Tensor) -> torch.Tensor:
        """``[:, t]`` is the forecast of frame ``t + 1`` from frames ``<= t``."""
        b, t, s, _ = ctx.shape
        x = self.inp(ctx) + self.t_emb[:, :t] + self.s_emb
        h = self.blocks(x.reshape(b, t * s, -1), mask=self._mask(t, ctx.device))
        return ctx + self.out(self.norm(h)).reshape(ctx.shape)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        """Forecast of the frame after the context: ``(B, S, enc_dim)``."""
        return self.forward_all(ctx)[:, -1]

    @torch.no_grad()
    def rollout(self, ctx: torch.Tensor, horizon: int) -> torch.Tensor:
        """Autoregressive forecast ``horizon`` frames ahead: ``(B, horizon, S, enc_dim)``.

        Each forecast is appended and the oldest frame dropped, so the context
        length -- and with it the temporal embedding the model was trained on --
        stays fixed.
        """
        outs = []
        for _ in range(horizon):
            nxt = self(ctx)
            outs.append(nxt)
            ctx = torch.cat([ctx[:, 1:], nxt[:, None]], dim=1)
        return torch.stack(outs, dim=1)
