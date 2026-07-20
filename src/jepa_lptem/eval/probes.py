"""Attentive readout probes: translate V-JEPA latent states into physical numbers.

These are the only trainable pieces of the evaluation, and the novel part of the
method — no library reads V-JEPA latents. They follow the standard V-JEPA
attentive-probe protocol: the encoder stays frozen, a small head is trained on top.

- ``AttentiveProbe``  latent frame -> scalar observables ``[count, mean_r, area]``.
- ``CentroidProbe``   latent frame -> occupancy grid + sub-cell (dy, dx) offsets.
- ``decode_centroids`` occupancy + offsets -> sub-pixel centroids (for tracking/MSD).

In Step 2 we only build them and prove they can fit (``tests/test_probes.py``);
they are trained for real on cached features in Step 7.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class AttentiveProbe(nn.Module):
    """Single learned query cross-attends over latent tokens -> regression head."""

    def __init__(self, dim: int = 1024, out_dim: int = 1, heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # (B, N, D) -> (B, out_dim)
        q = self.query.expand(tokens.size(0), -1, -1)
        pooled, _ = self.attn(q, tokens, tokens)
        return self.head(pooled.squeeze(1))


class CentroidProbe(nn.Module):
    """Per-token head -> occupancy logits + sub-cell (dy, dx) offsets.

    Tokens are assumed to be a ``grid x grid`` spatial layout (row-major). A learned
    positional embedding lets the head express a position-dependent occupancy field.
    The offset channels carry the sub-cell centroid position so that displacements
    smaller than one grid cell survive (raw grid centroids would quantize Brownian
    steps to noise, corrupting MSD exactly where it matters).
    """

    def __init__(self, dim: int = 1024, grid: int = 16):
        super().__init__()
        self.grid = grid
        self.pos = nn.Parameter(torch.randn(1, grid * grid, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 3)  # [occupancy_logit, dy, dx]

    def forward(self, tokens: torch.Tensor):  # (B, grid*grid, D)
        B = tokens.size(0)
        x = self.norm(tokens + self.pos)
        out = self.head(x).reshape(B, self.grid, self.grid, 3)
        occ = out[..., 0]
        off = out[..., 1:]
        return occ, off


def decode_centroids(
    occupancy: np.ndarray,
    offsets: np.ndarray,
    threshold: float = 0.5,
    cell_size: float = 16.0,
) -> np.ndarray:
    """Occupancy probabilities + sub-cell offsets -> ``(K, 2)`` ``(row, col)`` px centroids.

    A cell ``(i, j)`` with ``occupancy > threshold`` yields a centroid at
    ``((i + 0.5 + dy) * cell_size, (j + 0.5 + dx) * cell_size)``, where ``(dy, dx)``
    is that cell's offset. ``occupancy`` is expected as probabilities in ``[0, 1]``
    (apply a sigmoid to logits before calling).
    """
    occupancy = np.asarray(occupancy, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    ii, jj = np.where(occupancy > threshold)
    if ii.size == 0:
        return np.zeros((0, 2))
    dy = offsets[ii, jj, 0]
    dx = offsets[ii, jj, 1]
    rows = (ii + 0.5 + dy) * cell_size
    cols = (jj + 0.5 + dx) * cell_size
    return np.stack([rows, cols], axis=1)
