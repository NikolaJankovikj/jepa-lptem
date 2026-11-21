"""Step 11: the ConvLSTM pixel-space forecaster and its rollout loss."""

import torch
from torch.nn import init

from jepa_lptem.models.convlstm import PixelForecaster, rollout_loss

B, T, H = 2, 6, 32  # batch, frames, frame side


def small_model() -> PixelForecaster:
    torch.manual_seed(0)
    return PixelForecaster(hidden=8, layers=1)


def test_output_shapes():
    model, ctx = small_model(), torch.rand(B, T, H, H)
    assert model(ctx).shape == (B, T, H, H)
    assert model.rollout(ctx, horizon=5).shape == (B, 5, H, H)


def test_untrained_model_is_exactly_persistence():
    # Like LatentForecaster: the update head starts at zero, so before training every
    # forecast is the last frame, the floor a Brownian martingale imposes.
    model, ctx = small_model(), torch.rand(B, T, H, H)
    assert torch.equal(model(ctx), ctx)
    assert torch.equal(model.rollout(ctx, horizon=3), ctx[:, -1:].expand(B, 3, H, H))


def test_causality():
    # A forecast made after frame t never depends on frames later than t.
    model, ctx = small_model(), torch.rand(B, T, H, H)
    for p in model.parameters():
        init.normal_(p, std=0.1)
    later = ctx.clone()
    later[:, -1] = torch.rand(B, H, H)
    assert torch.equal(model(ctx)[:, :-1], model(later)[:, :-1])


def growing_disks(n: int, frames: int) -> torch.Tensor:
    """Dark disks on grey whose radius grows by one pixel every two frames."""
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(H), indexing="ij")
    clips = torch.full((n, frames, H, H), 0.7)
    gen = torch.Generator().manual_seed(1)
    for i in range(n):
        cy, cx = torch.randint(12, 20, (2,), generator=gen)
        r0 = torch.randint(2, 5, (1,), generator=gen).item()
        for t in range(frames):
            disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= (r0 + t / 2) ** 2
            clips[i, t][disk] = 0.2
    return clips


def test_learns_growth_beyond_persistence():
    # The rollout loss trains the forecaster on its own outputs; on a deterministic
    # growth it must beat holding the last frame by a wide margin at +8.
    torch.manual_seed(0)
    model = PixelForecaster(hidden=16, layers=1)
    clips = growing_disks(32, 16)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(300):
        opt.zero_grad()
        rollout_loss(model, clips, ctx=8, teacher=0.0).backward()
        opt.step()
    ctx, future = clips[:, :8], clips[:, 8:]
    with torch.no_grad():
        model_err = (model.rollout(ctx, 8)[:, -1] - future[:, -1]).pow(2).mean()
    persist_err = (ctx[:, -1] - future[:, -1]).pow(2).mean()
    assert model_err * 5 < persist_err
