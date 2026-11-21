"""ConvLSTM pixel-space forecaster: the pixel baseline of PLAN Step 11.

Frames in [0, 1] go through a stride-4 conv encoder, stacked ConvLSTM cells and a
transposed-conv decoder. Like ``LatentForecaster`` it predicts the change from the
current frame with a head that starts at zero, so the untrained model is exactly
persistence and training starts at the floor a Brownian martingale imposes.
"""

from __future__ import annotations

import torch
from torch import nn

KERNEL = 3


class ConvLSTMCell(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.gates = nn.Conv2d(
            channels + hidden, 4 * hidden, KERNEL, padding=KERNEL // 2
        )

    def forward(self, x: torch.Tensor, state: tuple) -> tuple:
        h, c = state
        i, f, o, g = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class PixelForecaster(nn.Module):
    """``(B, T, H, W)`` frames -> the forecast of each next frame, same shape."""

    def __init__(self, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.hidden = hidden
        self.encode = nn.Sequential(
            nn.Conv2d(1, 32, KERNEL, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, hidden, KERNEL, stride=2, padding=1),
            nn.GELU(),
        )
        self.cells = nn.ModuleList(ConvLSTMCell(hidden, hidden) for _ in range(layers))
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(hidden, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )
        nn.init.zeros_(self.decode[-1].weight)  # starts as exact persistence
        nn.init.zeros_(self.decode[-1].bias)

    def init_state(self, frame: torch.Tensor) -> list:
        b, h, w = frame.shape
        zeros = frame.new_zeros(b, self.hidden, h // 4, w // 4)
        return [(zeros, zeros) for _ in self.cells]

    def step(self, frame: torch.Tensor, state: list) -> tuple:
        """One frame in, the next frame's forecast out."""
        x = self.encode(frame[:, None])
        new_state = []
        for cell, s in zip(self.cells, state, strict=True):
            x, c = cell(x, s)
            new_state.append((x, c))
        return (frame + self.decode(x)[:, 0]).clamp(0, 1), new_state

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        state, preds = self.init_state(frames[:, 0]), []
        for t in range(frames.shape[1]):
            pred, state = self.step(frames[:, t], state)
            preds.append(pred)
        return torch.stack(preds, dim=1)

    def rollout(self, ctx: torch.Tensor, horizon: int) -> torch.Tensor:
        """Read the context, then feed each forecast back as the next input."""
        state = self.init_state(ctx[:, 0])
        for t in range(ctx.shape[1]):
            pred, state = self.step(ctx[:, t], state)
        outs = [pred]
        for _ in range(horizon - 1):
            pred, state = self.step(pred, state)
            outs.append(pred)
        return torch.stack(outs, dim=1)


def rollout_loss(
    model: PixelForecaster, clips: torch.Tensor, ctx: int, teacher: float
) -> torch.Tensor:
    """Pixel MSE over every forecast after the context. Past the context each input is
    the true frame with probability ``teacher`` (scheduled sampling), else the model's
    own previous forecast; within the context it is always the true frame."""
    state, loss = model.init_state(clips[:, 0]), 0.0
    frame = clips[:, 0]
    for t in range(clips.shape[1] - 1):
        pred, state = model.step(frame, state)
        if t + 1 >= ctx:
            loss = loss + (pred - clips[:, t + 1]).pow(2).mean()
        own = t + 1 >= ctx and torch.rand(()) >= teacher
        frame = pred if own else clips[:, t + 1]
    return loss / (clips.shape[1] - ctx)
