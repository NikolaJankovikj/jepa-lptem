"""The post-hoc forecast-trained readout: sources and the horizon-aware probe."""

import sys
from pathlib import Path

import torch
from torch.nn.functional import mse_loss

from jepa_lptem.models.predictor import LatentForecaster

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from forecast_probe import HORIZON, HorizonProbe, Readout, futures

S, D, CTX = 4, 8, 4


def test_futures_build_every_source_from_the_right_frames():
    torch.manual_seed(0)
    x = torch.randn(2, CTX + HORIZON, S, D)
    model = LatentForecaster(enc_dim=D, dim=16, depth=1, heads=2, n_spatial=S).eval()
    stats = (torch.zeros(D), torch.ones(D))
    r = Readout(model=model, stats=stats, ctx=CTX, norm=stats, probes={})
    out = futures(r, x)
    assert all(v.shape == (2, HORIZON, S, D) for v in out.values())
    assert torch.equal(out["floor"], x[:, CTX:])
    assert torch.allclose(
        out["persistence"], x[:, CTX - 1 : CTX].expand_as(out["floor"])
    )
    assert torch.allclose(out["ctxmean"][:, 5], x[:, :CTX].mean(dim=1))
    # an untrained model is persistence by construction
    assert torch.allclose(out["model"], out["persistence"], atol=1e-5)


def test_held_frame_probe_can_still_extrapolate_through_the_horizon():
    """The persistence baseline sees the same latent at every +h; only the horizon
    embedding lets it turn that into a size that grows -- without it the bar the
    forecast must clear would be set artificially low."""
    torch.manual_seed(0)
    probe = HorizonProbe(horizon=HORIZON, dim=D, heads=2)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    tokens = torch.randn(1, S, D).expand(HORIZON, -1, -1)
    h = torch.arange(HORIZON)
    target = (h.float() / HORIZON)[:, None].expand(-1, 3)
    for _ in range(300):
        loss = mse_loss(probe(tokens, h), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    pred = probe(tokens, h)[:, 0].detach()
    assert loss.item() < 1e-3
    assert pred[-1] - pred[0] > 0.8
