"""Candidate B starts as persistence and forecasts every horizon at once."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from pooled_transformer import CTX, HORIZON, PooledForecaster


def test_untrained_model_is_persistence_at_every_horizon():
    ctx = torch.randn(3, CTX, 16)
    out = PooledForecaster(16)(ctx)
    assert out.shape == (3, HORIZON, 16)
    assert torch.equal(out, ctx[:, -1:].expand(-1, HORIZON, -1))


def test_a_longer_context_is_persistence_too():
    ctx = torch.randn(2, 16, 8)
    out = PooledForecaster(8, ctx=16)(ctx)
    assert torch.equal(out, ctx[:, -1:].expand(-1, HORIZON, -1))
