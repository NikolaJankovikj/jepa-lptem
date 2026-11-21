"""Thesis figures, from the experiments' metrics files (seconds, on the Mac).

    uv run python thesis/figures.py --clips <dir>

--clips is a folder of freshly generated clips for the simulator panel:
    uv run python scripts/generate_dynamics_dataset.py --out <dir> --n-videos 10 --n-frames 100
"""

import argparse
import json
import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
OUT = ROOT / "thesis/figures"
AML = runpy.run_path(str(ROOT / "docs/aml/figures.py"))  # encoder-comparison panels, reused

# Validated categorical slots 1-5 (dataviz validator, light); markers carry identity too,
# because slots 3-5 sit below 3:1 contrast on white.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
INK, MUTED, GRID, REF = AML["INK"], AML["MUTED"], AML["GRID"], "#8a8983"
HORIZONS = ["+1", "+5", "+10", "+24"]
STEPS = [int(h) for h in HORIZONS]
TOKEN_PX = 16
CROP = 256


def load(rel: str) -> dict:
    return json.loads((EXP / rel).read_text())


def style(ax, title: str = "") -> None:
    """Recessive grid and spines, the encoder-comparison figure's look."""
    ax.set_title(title, fontsize=8, color=INK)
    ax.grid(color=GRID, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED)


def line(ax, xy: tuple, i: int, label: str) -> None:
    ax.plot(*xy, color=SERIES[i], marker=MARKERS[i], markersize=4, linewidth=2, label=label)


def legend_above(fig, ax, ncols: int) -> None:
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncols=ncols,
               frameon=False, fontsize=7)


def growth_panels(draw, ncols: int, name: str) -> None:
    """Diffusion and reaction side by side, error against horizon; draw(ax, proc) adds the lines."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), layout="constrained", sharey=True)
    for ax, proc in zip(axes, ("diffusion", "reaction"), strict=True):
        draw(ax, proc)
        ax.set_xlabel("horizon (latent steps)")
        style(ax, proc)
    axes[0].set_ylabel("mean-radius MAE (px)")
    legend_above(fig, axes[0], ncols)
    fig.savefig(OUT / name)


def fig_simulator(clips: Path) -> None:
    """One physics-stratum clip per process: first and last frame, the model's 256² crop."""
    procs = ["diffusion", "reaction", "static", "ripening", "coalescence"]
    fig, axes = plt.subplots(2, 5, figsize=(7.0, 3.1), layout="constrained")
    for col, proc in enumerate(procs):
        clip = np.load(next(clips.glob(f"*_{proc}_physics.npz")))
        frames = clip["frames"]
        c = (frames.shape[1] - CROP) // 2
        for row, t in enumerate((0, len(frames) - 1)):
            axes[row, col].imshow(frames[t, c : c + CROP, c : c + CROP], cmap="gray")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[0, col].set_title(f"{proc}\n{float(clip['beam_flux']):.0f} e/px", fontsize=8, color=INK)
    axes[0, 0].set_ylabel("first frame", fontsize=8)
    axes[1, 0].set_ylabel("last frame", fontsize=8)
    fig.savefig(OUT / "fig_simulator.pdf")


def fig_lookahead() -> None:
    """Feature change of the context when the window is re-encoded without its future."""
    cos = np.array(load("leak-check/leak.json")["cosine_by_position"])
    x = np.arange(len(cos))
    fig, ax = plt.subplots(figsize=(3.4, 2.4), layout="constrained")
    line(ax, (x, 1 - cos), 0, "whole-window encoding")
    line(ax, (x, np.zeros(len(x))), 1, "per-frame encoding")
    ax.set_xlabel("context latent frame (7 = next to the forecast)")
    ax.set_ylabel("1 − cosine to the honest encoding")
    style(ax)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "fig_lookahead.pdf")


def fig_persistence() -> None:
    """Fraction of particles whose centre stays in its patch, by patch size."""
    files = {TOKEN_PX: "token-persistence/metrics.json"}
    files |= {p: f"region-persistence/persistence-{p}.json" for p in (32, 64, 128)}
    fig, ax = plt.subplots(figsize=(3.4, 2.5), layout="constrained")
    for i, (px, rel) in enumerate(files.items()):
        steps = load(rel)["all"]
        hs = sorted(steps, key=int)
        label = f"{px} px" + (" (one token)" if px == TOKEN_PX else "")
        line(ax, ([int(h) for h in hs], [steps[h]["same_cell"] for h in hs]), i, label)
    ax.axhline(1.0, color=REF, linestyle="--", linewidth=1)
    ax.text(24, 0.97, f"{CROP} px (whole crop)", ha="right", va="top", fontsize=7, color=MUTED)
    ax.set_xlabel("horizon (latent steps)")
    ax.set_ylabel("centre still in its patch")
    ax.set_ylim(0, 1.05)
    style(ax)
    legend_above(fig, ax, 2)
    fig.savefig(OUT / "fig_persistence.pdf")


def fig_pooled_forecast() -> None:
    """Mean-radius error vs horizon of the position-free forecasts and their baselines."""
    mae = load("pooled-forecast/metrics.json")["mae_px"]
    methods = [("persistence", "persistence"), ("ctxmean", "context mean"),
               ("pooled_trend", "pooled trend"), ("linear", "linear forecaster (A)")]

    def draw(ax, proc):
        for i, (key, label) in enumerate(methods):
            line(ax, (STEPS, [mae[proc][key][h] for h in HORIZONS]), i, label)
        ax.plot(STEPS, [mae[proc]["floor"][h] for h in HORIZONS], color=REF, linestyle="--",
                linewidth=1.5, label="floor (true future latents)")

    growth_panels(draw, 3, "fig_pooled_forecast.pdf")


def fig_dose() -> None:
    """The linear forecaster's error at low and high dose, context 16, with each bin's floor."""
    split = load("snr/metrics.json")["forecasts"]["context_16"]["beam_flux"]

    def draw(ax, proc):
        for i, dose in enumerate(("low", "high")):
            m = split[dose][proc]["mae_px"]
            line(ax, (STEPS, [m["linear"][h] for h in HORIZONS]), i, f"{dose} dose, forecast")
            ax.plot(STEPS, [m["floor"][h] for h in HORIZONS], color=SERIES[i], linestyle="--",
                    linewidth=1.2, label=f"{dose} dose, floor")

    growth_panels(draw, 4, "fig_dose.pdf")


def ci_band(ax, by_lag: dict, i: int) -> None:
    lags = sorted(by_lag, key=int)
    ax.fill_between([int(lag) for lag in lags], [by_lag[lag]["ci"][0] for lag in lags],
                    [by_lag[lag]["ci"][1] for lag in lags], color=SERIES[i], alpha=0.15, linewidth=0)


def fig_merges() -> None:
    """AUC of each signal, merge patches against moving-particle patches, by lag."""
    auc = load("events/metrics.json")["test"]["token"]["auc"]
    signals = [("model", "forecaster's +1 error"), ("persistence", "raw latent change"),
               ("ctxmean", "distance from context average"), ("pixel", "brightness change (pixels)")]
    fig, ax = plt.subplots(figsize=(3.6, 2.7), layout="constrained")
    for i, (key, label) in enumerate(signals):
        lags = sorted(auc[key], key=int)
        line(ax, ([int(lag) for lag in lags], [auc[key][lag]["auc"] for lag in lags]), i, label)
    ci_band(ax, auc["model"], 0)
    ax.axhline(0.5, color=REF, linestyle="--", linewidth=1)
    ax.axvline(0, color=GRID, linewidth=1)
    ax.set_xticks([-2, -1, 0, 1])
    ax.set_xlabel("latent steps from the merge")
    ax.set_ylabel("AUC")
    style(ax)
    legend_above(fig, ax, 2)
    fig.savefig(OUT / "fig_merges.pdf")


# Movie file prefix -> the paper it comes from (Section 4.3).
MOVIE_PAPERS = {
    "samem": "SAM-EM nanorods", "aggregative": "Woehl 2014", "am1c20824": "Son 2022",
    "am8b03688": "Robertson 2018", "nn5b06632": "Welch 2016",
    "cm2c03226": "Dachraoui 2023 (HAADF)", "ja9b05225": "Wang 2019", "nn6b05270": "Kim 2016",
}


def movie_name(file: str) -> str:
    stem = file.split(".")[0]
    paper = MOVIE_PAPERS[stem.split("_")[0]]
    return f"{paper}, movie {stem[-1]}" if stem.startswith("ja9b05225") else paper


def fig_containment() -> None:
    """Feature-space containment of each real movie against held-out synthetic clips."""
    c = load("real/metrics.json")["containment"]
    movies = sorted(c["movies"].items(), key=lambda kv: kv[1]["ratio"])
    held = list(c["held_out_ratios"].values())
    y = np.arange(len(movies))
    fig, ax = plt.subplots(figsize=(4.6, 2.9), layout="constrained")
    ax.scatter([m["ratio"] for _, m in movies], y, color=SERIES[1], marker="s", s=24, zorder=3,
               label="real movie")
    ax.scatter(held, np.full(len(held), -1.0), color=SERIES[0], marker="o", s=12, alpha=0.6,
               label="held-out synthetic clip")
    ax.axvline(c["limit"], color=REF, linestyle="--", linewidth=1)
    ax.set_yticks([-1, *y], ["synthetic", *(movie_name(n) for n, _ in movies)], fontsize=7)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])
    ax.minorticks_off()
    ax.set_xlabel(f"containment ratio (log scale; limit {c['limit']:.2f}, dashed)")
    style(ax)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.savefig(OUT / "fig_containment.pdf")


def fig_data_efficiency() -> None:
    runs = json.loads(AML["METRICS"].read_text())["runs"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), layout="constrained")
    for ax, (stat, title) in zip(axes, AML["PANELS"], strict=True):
        AML["plot_panel"](ax, runs, stat, title)
    legend_above(fig, axes[0], 3)
    fig.savefig(OUT / "fig_data_efficiency.pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "text.color": INK,
                         "axes.labelcolor": INK})
    fig_simulator(args.clips)
    for make in (fig_lookahead, fig_persistence, fig_pooled_forecast, fig_dose, fig_merges,
                 fig_containment, fig_data_efficiency):
        make()


if __name__ == "__main__":
    main()
