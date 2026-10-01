"""Step 9: the block-causal latent forecaster and its training losses."""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import init

from jepa_lptem.models.predictor import LatentForecaster

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

B, T, S, D = 2, 6, 4, 16  # batch, latent frames, tokens per frame, encoder dim


def small_model(**kw) -> LatentForecaster:
    torch.manual_seed(0)
    return LatentForecaster(
        enc_dim=D, dim=32, depth=2, heads=4, n_spatial=S, max_t=16, **kw
    )


def test_output_shapes():
    model, ctx = small_model(), torch.randn(B, T, S, D)
    assert model.forward_all(ctx).shape == (B, T, S, D)
    assert model(ctx).shape == (B, S, D)
    assert model.rollout(ctx, horizon=5).shape == (B, 5, S, D)


def test_untrained_model_is_exactly_persistence():
    # The output head starts at zero and the model predicts a change, so before any
    # training every forecast is "the next frame equals this one" -- the baseline
    # it has to beat is where it starts, not somewhere it must first climb to.
    model, ctx = small_model(), torch.randn(B, T, S, D)
    assert torch.equal(model.forward_all(ctx), ctx)
    assert torch.equal(model.rollout(ctx, horizon=3), ctx[:, -1:].expand(B, 3, S, D))


def test_block_causality():
    # Changing the last context frame may change only the last forecast: the
    # forecast of frame t+1 must never see frame t+1 or anything after it.
    model = small_model()
    init.normal_(model.out.weight)  # a zero head would make this trivially true
    ctx = torch.randn(B, T, S, D)
    later = ctx.clone()
    later[:, -1] += torch.randn(B, S, D)
    with torch.no_grad():
        a, b = model.forward_all(ctx), model.forward_all(later)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)
    assert not torch.allclose(a[:, -1], b[:, -1], atol=1e-3)


def test_rollout_feeds_its_own_forecasts_back():
    model = small_model()
    init.normal_(model.out.weight, std=0.1)
    ctx = torch.randn(B, T, S, D)
    with torch.no_grad():
        first = model(ctx)
        second = model(torch.cat([ctx[:, 1:], first[:, None]], dim=1))
        rolled = model.rollout(ctx, horizon=2)
    assert torch.allclose(rolled[:, 0], first, atol=1e-5)
    assert torch.allclose(rolled[:, 1], second, atol=1e-5)


def test_training_beats_persistence_on_deterministic_dynamics():
    # Every token moves by a fixed velocity each frame: persistence is off by
    # exactly |v|, and a working model + loss must learn to add v.
    from train_predictor import forecast_losses

    torch.manual_seed(1)
    v = torch.randn(1, 1, S, D)
    model = small_model()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(300):
        x = torch.randn(8, 1, S, D) + v * torch.arange(T + 2).view(1, -1, 1, 1)
        loss, _ = forecast_losses(model, x, ctx=T, rollout=2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    x = torch.randn(8, 1, S, D) + v * torch.arange(T + 2).view(1, -1, 1, 1)
    with torch.no_grad():
        _, parts = forecast_losses(model, x, ctx=T, rollout=2)
    assert parts["model_l1_1"].mean() < 0.2 * parts["persistence_l1_1"].mean()
    assert parts["model_l1_2"].mean() < 0.2 * parts["persistence_l1_2"].mean()


def test_eval_starts_stay_inside_the_window():
    from train_predictor import eval_starts

    starts = eval_starts(n_frames=32, length=10, n=3)
    assert starts.tolist() == [0, 11, 22]  # first, middle, last possible
    assert eval_starts(n_frames=10, length=10, n=3).tolist() == [0]


def test_standardize_round_trips():
    from train_predictor import standardize, unstandardize

    x = np.random.default_rng(0).normal(3.0, 2.0, (5, 4, D)).astype(np.float32)
    mean, std = x.mean(axis=(0, 1)), x.std(axis=(0, 1))
    z = standardize(x, mean, std)
    assert abs(float(z.mean())) < 1e-5
    assert np.allclose(unstandardize(z, mean, std), x, atol=1e-5)


def test_context_mean_beats_persistence_on_pure_noise():
    # A static scene under fresh per-frame noise: the next frame is unpredictable
    # beyond its mean, so averaging the context beats holding the last frame. This
    # is why persistence alone is too weak a bar for "learned dynamics".
    from train_predictor import forecast_losses
    torch.manual_seed(2)
    x = torch.randn(1, 1, S, D) + 0.5 * torch.randn(64, T + 2, S, D)
    with torch.no_grad():
        _, parts = forecast_losses(small_model(), x, ctx=T, rollout=2)
    for k in (1, 2):
        assert parts[f"ctxmean_l1_{k}"].mean() < 0.8 * parts[f"persistence_l1_{k}"].mean()
        assert torch.allclose(parts[f"model_l1_{k}"], parts[f"persistence_l1_{k}"])  # untrained


def test_init_restores_weights_stats_and_architecture(tmp_path):
    from argparse import Namespace

    from train_predictor import load_init, save_checkpoint

    model = small_model()
    stats = (np.full(D, 2.0, np.float32), np.full(D, 3.0, np.float32))
    trained = Namespace(ctx=6, dim=32, depth=2, heads=4)
    save_checkpoint(tmp_path / "p.pt", model, stats, trained, epoch=7, processes=["static"])

    cli = Namespace(ctx=8, dim=512, depth=8, heads=8)  # CLI defaults must not win
    state, (mean, std) = load_init(tmp_path / "p.pt", cli)
    assert vars(cli) == vars(trained)
    assert np.array_equal(mean, stats[0]) and np.array_equal(std, stats[1])
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
